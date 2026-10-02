import os
from pathlib import Path

# Определяем структуру файлов проекта
files = {
    "custom_diffusion/config.py": '''import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_SIZE = 64
UPSCALED_IMAGE_SIZE = 256
BATCH_SIZE = 4
MAX_LENGTH = 77
TEXT_EMBED_DIM = 768
TIMESTEPS = 1000
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
''',

    "custom_diffusion/models/__init__.py": '''# Package initialization
''',

    "custom_diffusion/models/embed.py": '''import torch
import torch.nn as nn
import math
from typing import List
from transformers import CLIPTextModel, CLIPTokenizer
from custom_diffusion.config import CLIP_MODEL_NAME, MAX_LENGTH

class TextConditioner(nn.Module):
    """Извлекает текстовые эмбеддинги из замороженной модели CLIP."""
    def __init__(self, model_name: str = CLIP_MODEL_NAME):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
    def encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        device = next(self.text_encoder.parameters()).device
        input_ids = inputs.input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
            embeddings = outputs.last_hidden_state  # [B, seq_len, 768]
        return embeddings

class SinusoidalTimeEmbedding(nn.Module):
    """Синусоидальное кодирование временного шага t."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb
''',

    "custom_diffusion/models/attention.py": '''import torch
import torch.nn as nn
import math

class SpatialCrossAttention(nn.Module):
    """Модуль Cross-Attention (Запросы Q — фичи картинки, Ключи K и Значения V — текст)."""
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], context: [B, seq_len, context_dim]
        B, C, H, W = x.shape
        N = H * W
        
        # Проекция изображения в вектор последин
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        
        q = self.to_q(x_flat).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Вычисление внимания
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.to_out(out)
        
        # Возвращаем исходную размерность карты тензоров [B, C, H, W]
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)
''',

    "custom_diffusion/models/blocks.py": '''import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual блок с подмешиванием вектора времени t."""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        time_proj = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + time_proj
        h = self.conv2(self.act2(self.norm2(h)))
        return self.shortcut(x) + h

class Downsample(nn.Module):
    """Уменьшение пространственного разрешения в 2 раза."""
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)

class Upsample(nn.Module):
    """Увеличение пространственного разрешения в 2 раза."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
''',

    "custom_diffusion/models/unet.py": '''import torch
import torch.nn as nn
from custom_diffusion.models.embed import SinusoidalTimeEmbedding
from custom_diffusion.models.attention import SpatialCrossAttention
from custom_diffusion.models.blocks import ResBlock, Downsample, Upsample
from custom_diffusion.config import TEXT_EMBED_DIM

class BaseUNet(nn.Module):
    """Базовый U-Net (64x64) с Downsampling, Upsampling и Cross-Attention."""
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.res1 = ResBlock(base_channels, base_channels, time_dim)
        self.down1 = Downsample(base_channels)
        self.res2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.attn2 = SpatialCrossAttention(base_channels * 2, TEXT_EMBED_DIM)
        
        # Bottleneck
        self.mid_res1 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.mid_attn = SpatialCrossAttention(base_channels * 2, TEXT_EMBED_DIM)
        self.mid_res2 = ResBlock(base_channels * 2, base_channels * 2, time_dim)
        
        # Decoder
        self.up1 = Upsample(base_channels * 2)
        self.res3 = ResBlock(base_channels * 4, base_channels, time_dim)
        self.attn3 = SpatialCrossAttention(base_channels, TEXT_EMBED_DIM)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        
        h1 = self.conv_in(x)
        h1 = self.res1(h1, t_emb)
        
        h2 = self.down1(h1)
        h2 = self.res2(h2, t_emb)
        h2 = h2 + self.attn2(h2, text_emb)
        
        # Bottleneck
        h_mid = self.mid_res1(h2, t_emb)
        h_mid = h_mid + self.mid_attn(h_mid, text_emb)
        h_mid = self.mid_res2(h_mid, t_emb)
        
        # Decoder с Skip Connections
        h_up = self.up1(h_mid)
        h_up = torch.cat([h_up, h1], dim=1)
        h_out = self.res3(h_up, t_emb)
        h_out = h_out + self.attn3(h_out, text_emb)
        
        return self.conv_out(h_out)

class UpscalerUNet(nn.Module):
    """U-Net Апскейлер (256x256). Вход — 6 каналов (Шум 256x256 + Апскейленный низкорез 64x64)."""
    def __init__(self, in_channels: int = 6, out_channels: int = 3, base_channels: int = 64):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.res1 = ResBlock(base_channels, base_channels, time_dim)
        self.down1 = Downsample(base_channels)
        self.res2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.attn = SpatialCrossAttention(base_channels * 2, TEXT_EMBED_DIM)
        
        self.up1 = Upsample(base_channels * 2)
        self.res3 = ResBlock(base_channels * 3, base_channels, time_dim)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        
        h1 = self.conv_in(x)
        h1 = self.res1(h1, t_emb)
        
        h2 = self.down1(h1)
        h2 = self.res2(h2, t_emb)
        h2 = h2 + self.attn(h2, text_emb)
        
        h_up = self.up1(h2)
        h_up = torch.cat([h_up, h1], dim=1)
        h_out = self.res3(h_up, t_emb)
        
        return self.conv_out(h_out)
''',

    "custom_diffusion/utils/__init__.py": '''# Package initialization
''',

    "custom_diffusion/utils/scheduler.py": '''import torch
import torch.nn as nn
from custom_diffusion.config import TIMESTEPS

class DDPMNoiseScheduler:
    """Точная математика прямой (q_sample) и обратной (p_sample) диффузии Ho et al."""
    def __init__(self, num_timesteps: int = TIMESTEPS, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Константы для прямой диффузии
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Константы для обратной диффузии
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Прямой процесс: зашумление x_0 до x_t."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor):
        """Обратный процесс: удаление шума из x_t до x_{t-1}."""
        device = x.device
        t_idx = t.item() if isinstance(t, torch.Tensor) else t
        t_tensor = torch.full((x.shape[0],), t_idx, device=device, dtype=torch.long)
        
        with torch.no_grad():
            pred_noise = model(x, t_tensor, text_emb)
            
        beta = self.betas[t_idx]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t_idx]
        sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas[t_idx])
        
        # Среднее значение p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alpha * (x - beta / sqrt_one_minus_alpha * pred_noise)
        
        if t_idx == 0:
            return model_mean
        else:
            posterior_var = self.posterior_variance[t_idx]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var) * noise
''',

    "custom_diffusion/train.py": '''import torch
import torch.nn as nn
from custom_diffusion.models.embed import TextConditioner
from custom_diffusion.models.unet import BaseUNet, UpscalerUNet
from custom_diffusion.utils.scheduler import DDPMNoiseScheduler
from custom_diffusion.config import DEVICE, BASE_IMAGE_SIZE, UPSCALED_IMAGE_SIZE

def train_base_step(model: BaseUNet, scheduler: DDPMNoiseScheduler, conditioner: TextConditioner, images: torch.Tensor, prompts: list, optimizer: torch.optim.Optimizer):
    model.train()
    optimizer.zero_grad()
    
    text_emb = conditioner.encode_prompt(prompts)
    B = images.shape[0]
    t = torch.randint(0, scheduler.num_timesteps, (B,), device=DEVICE)
    
    noisy_images, noise = scheduler.q_sample(images, t)
    pred_noise = model(noisy_images, t, text_emb)
    
    loss = nn.functional.mse_loss(pred_noise, noise)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_upscaler_step(model: UpscalerUNet, scheduler: DDPMNoiseScheduler, conditioner: TextConditioner, high_res_images: torch.Tensor, low_res_images: torch.Tensor, prompts: list, optimizer: torch.optim.Optimizer):
    model.train()
    optimizer.zero_grad()
    
    text_emb = conditioner.encode_prompt(prompts)
    B, C, H, W = high_res_images.shape
    
    # 1. Апскейлим низкорез до 256x256
    low_res_upscaled = nn.functional.interpolate(low_res_images, size=(H, W), mode="bicubic", align_corners=False)
    
    # 2. Зашумляем высокую картинку
    t = torch.randint(0, scheduler.num_timesteps, (B,), device=DEVICE)
    noisy_high_res, noise = scheduler.q_sample(high_res_images, t)
    
    # 3. Конкатенируем в 6 каналов
    input_tensor = torch.cat([noisy_high_res, low_res_upscaled], dim=1)
    
    pred_noise = model(input_tensor, t, text_emb)
    loss = nn.functional.mse_loss(pred_noise, noise)
    loss.backward()
    optimizer.step()
    return loss.item()
''',

    "custom_diffusion/generate.py": '''import torch
import torch.nn as nn
from custom_diffusion.models.embed import TextConditioner
from custom_diffusion.models.unet import BaseUNet, UpscalerUNet
from custom_diffusion.utils.scheduler import DDPMNoiseScheduler
from custom_diffusion.config import DEVICE, BASE_IMAGE_SIZE, UPSCALED_IMAGE_SIZE

def generate(prompt: str, base_model: BaseUNet, upscaler_model: UpscalerUNet, scheduler: DDPMNoiseScheduler, conditioner: TextConditioner, steps: int = 50) -> torch.Tensor:
    base_model.eval()
    upscaler_model.eval()
    
    text_emb = conditioner.encode_prompt([prompt])
    B = 1
    
    print(f"--- 1/3 Генерация базового изображения 64x64 по промпту: '{prompt}' ---")
    x_64 = torch.randn((B, 3, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), device=DEVICE)
    
    step_ratio = scheduler.num_timesteps // steps
    timesteps = list(range(scheduler.num_timesteps - 1, -1, -step_ratio))
    
    for t in timesteps:
        x_64 = scheduler.p_sample(base_model, x_64, t, text_emb)
        
    print("--- 2/3 Апскейл и подготовка к 256x256 ---")
    x_64_upscaled = nn.functional.interpolate(x_64, size=(UPSCALED_IMAGE_SIZE, UPSCALED_IMAGE_SIZE), mode="bicubic", align_corners=False)
    x_256 = torch.randn((B, 3, UPSCALED_IMAGE_SIZE, UPSCALED_IMAGE_SIZE), device=DEVICE)
    
    print("--- 3/3 Генерация высокого разрешения 256x256 ---")
    for t in timesteps:
        # Конкатенация шума 256x256 и подтянутого низкореза 64x64 -> 6 каналов
        conditioned_input = torch.cat([x_256, x_64_upscaled], dim=1)
        x_256 = scheduler.p_sample(upscaler_model, conditioned_input, t, text_emb)
        
    print("Готово! Изображение 256x256 успешно сгенерировано.")
    return x_256
''',

    "run_demo.py": '''import torch
from custom_diffusion.config import DEVICE, BASE_IMAGE_SIZE, UPSCALED_IMAGE_SIZE
from custom_diffusion.models.embed import TextConditioner
from custom_diffusion.models.unet import BaseUNet, UpscalerUNet
from custom_diffusion.utils.scheduler import DDPMNoiseScheduler
from custom_diffusion.train import train_base_step, train_upscaler_step
from custom_diffusion.generate import generate

def main():
    print(f"Запуск проекта на устройстве: {DEVICE}")
    
    # 1. Инициализация моделей
    conditioner = TextConditioner().to(DEVICE)
    base_model = BaseUNet().to(DEVICE)
    upscaler_model = UpscalerUNet().to(DEVICE)
    scheduler = DDPMNoiseScheduler().to(DEVICE)
    
    opt_base = torch.optim.AdamW(base_model.parameters(), lr=1e-4)
    opt_upscale = torch.optim.AdamW(upscaler_model.parameters(), lr=1e-4)
    
    # 2. Тестовый шаг обучения (проверка отсутствия ошибок размерностей)
    print("\\n[Тест] Запуск демо-шага обучения...")
    dummy_high = torch.randn((1, 3, UPSCALED_IMAGE_SIZE, UPSCALED_IMAGE_SIZE), device=DEVICE)
    dummy_low = torch.randn((1, 3, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), device=DEVICE)
    prompt = ["a photo of a red car"]
    
    loss1 = train_base_step(base_model, scheduler, conditioner, dummy_low, prompt, opt_base)
    loss2 = train_upscaler_step(upscaler_model, scheduler, conditioner, dummy_high, dummy_low, prompt, opt_upscale)
    print(f"Base Loss: {loss1:.4f} | Upscaler Loss: {loss2:.4f}")
    
    # 3. Запуск генерации
    print("\\n[Тест] Запуск инференса (генерации)...")
    res = generate("a fantasy landscape, digital art", base_model, upscaler_model, scheduler, conditioner, steps=10)
    print(f"Выходной тензор картинки: {res.shape}")

if __name__ == "__main__":
    main()
'''
}

# Автоматически создаем папки и пишем код в файлы
print("Разворачиваем структуру проекта custom_diffusion...")
for path, content in files.items():
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Создан файл: {path}")

print("\nВсё готово! Теперь можно запускать скрипт проверки:")
print("python run_demo.py")
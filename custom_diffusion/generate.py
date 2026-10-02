import os
import torch
import torch.nn as nn
from datetime import datetime
from PIL import Image
from custom_diffusion.models.embed import TextConditioner
from custom_diffusion.models.unet import BaseUNet, UpscalerUNet
from custom_diffusion.utils.scheduler import DDPMNoiseScheduler
from custom_diffusion.config import DEVICE, BASE_IMAGE_SIZE, UPSCALED_IMAGE_SIZE

@torch.no_grad()
def generate(
    prompt: str, 
    base_model: BaseUNet, 
    upscaler_model: UpscalerUNet, 
    scheduler: DDPMNoiseScheduler, 
    conditioner: TextConditioner, 
    steps: int = 50,
    cfg_scale: float = 7.5
) -> torch.Tensor:
    base_model.eval()
    upscaler_model.eval()
    conditioner.eval()
    
    B = 1
    use_cfg = cfg_scale > 1.0
    
    # 1. Подготовка текстовых эмбеддингов для Base Model
    if use_cfg:
        cond_text_emb = conditioner.encode_prompt([prompt])
        uncond_text_emb = conditioner.encode_prompt([""])
        text_emb = torch.cat([uncond_text_emb, cond_text_emb], dim=0) # [2, seq_len, dim]
    else:
        text_emb = conditioner.encode_prompt([prompt]) # [1, seq_len, dim]
    
    timesteps = list(range(scheduler.num_timesteps - 1, -1, -1))
    
    # ----------------------------------------------------
    # 1. Генерация Базового изображения 64x64
    # ----------------------------------------------------
    print(f"--- 1/2 Генерация 64x64 (Base CFG: {cfg_scale}) ---")
    x_64 = torch.randn((B, 3, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), device=DEVICE)
    
    for t in timesteps:
        t_tensor = torch.full((B,), t, device=DEVICE, dtype=torch.long)
        
        if use_cfg:
            x_in = torch.cat([x_64] * 2, dim=0)       # [2, 3, 64, 64]
            t_in = torch.cat([t_tensor] * 2, dim=0)   # [2]
            
            noise_pred = base_model(x_in, t_in, text_emb)
            noise_uncond, noise_cond = noise_pred.chunk(2)
            pred_noise = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            pred_noise = base_model(x_64, t_tensor, text_emb)
        
        # Шаг деноизинга DDPM
        beta = scheduler.betas[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = torch.sqrt(1.0 / scheduler.alphas[t])
        
        model_mean = sqrt_recip_alpha * (x_64 - beta / sqrt_one_minus_alpha * pred_noise)
        
        # Для базовой модели тоже убираем шум на последних шагах t <= 10
        if t > 10:
            noise = torch.randn_like(x_64)
            posterior_var = scheduler.posterior_variance[t]
            x_64 = model_mean + torch.sqrt(posterior_var) * noise
        else:
            x_64 = model_mean

    # ----------------------------------------------------
    # 2. Генерация высокого разрешения 256x256 (Upscaler)
    # ----------------------------------------------------
    print("--- 2/2 Апскейл до 256x256 (Upscaler CFG: 1.0) ---")
    
    # Ограничиваем базовый кадр нормальным диапазоном [-1, 1]
    x_64_conditioned = x_64.clamp(-1.0, 1.0)
    x_64_upscaled = nn.functional.interpolate(
        x_64_conditioned, size=(UPSCALED_IMAGE_SIZE, UPSCALED_IMAGE_SIZE), mode="bicubic", align_corners=False
    )
    x_256 = torch.randn((B, 3, UPSCALED_IMAGE_SIZE, UPSCALED_IMAGE_SIZE), device=DEVICE)
    
    cond_text_emb_single = conditioner.encode_prompt([prompt]) if use_cfg else text_emb
    
    for t in timesteps:
        t_tensor = torch.full((B,), t, device=DEVICE, dtype=torch.long)
        
        # Мягкий шум на кондишн (0.15 вместо глубокого зашумления)
        noise_cond = torch.randn_like(x_64_upscaled)
        x_64_noisy = x_64_upscaled + 0.15 * noise_cond

        conditioned_input = torch.cat([x_256, x_64_noisy], dim=1) # [1, 6, 256, 256]
        
        pred_noise = upscaler_model(conditioned_input, t_tensor, cond_text_emb_single)
        
        beta = scheduler.betas[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = torch.sqrt(1.0 / scheduler.alphas[t])
        
        model_mean = sqrt_recip_alpha * (x_256 - beta / sqrt_one_minus_alpha * pred_noise)
        
        # Отключаем шум при t <= 10, чтобы убрать зернистость
        if t > 10:
            noise = torch.randn_like(x_256)
            posterior_var = scheduler.posterior_variance[t]
            x_256 = model_mean + torch.sqrt(posterior_var) * noise
        else:
            x_256 = model_mean

        x_256 = x_256.clamp(-2.0, 2.0)

    # ----------------------------------------------------
    # 3. Сохранение результатов
    # ----------------------------------------------------
    x_64_final = x_64.clamp(-1.0, 1.0)
    x_256_final = x_256.clamp(-1.0, 1.0)

    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    clean_prompt = "".join(c for c in prompt if c.isalnum() or c in (" ", "_", "-")).strip()[:30]
    clean_prompt = clean_prompt.replace(" ", "_") if clean_prompt else "generation"
    
    output_dir = os.path.join("outputs", date_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    base_file_path = os.path.join(output_dir, f"{time_str}_{clean_prompt}_64x64.png")
    upscale_file_path = os.path.join(output_dir, f"{time_str}_{clean_prompt}_256x256.png")
    
    img_64_tensor = (x_64_final.squeeze(0).cpu() + 1.0) / 2.0 * 255.0
    Image.fromarray(img_64_tensor.permute(1, 2, 0).byte().numpy()).save(base_file_path)
    
    img_256_tensor = (x_256_final.squeeze(0).cpu() + 1.0) / 2.0 * 255.0
    Image.fromarray(img_256_tensor.permute(1, 2, 0).byte().numpy()).save(upscale_file_path)
    
    print(f"\n[Успех] Сохранено в: {output_dir}")
    print(f"  └ Base (64x64): {os.path.basename(base_file_path)}")
    print(f"  └ Upscale (256x256): {os.path.basename(upscale_file_path)}")
    
    return x_256_final
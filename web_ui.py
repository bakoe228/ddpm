import torch
import torch.nn as nn
import gradio as gr
from PIL import Image

from custom_diffusion.config import DEVICE, BASE_IMAGE_SIZE, UPSCALED_IMAGE_SIZE
from custom_diffusion.models.embed import TextConditioner
from custom_diffusion.models.unet import BaseUNet, UpscalerUNet
from custom_diffusion.utils.scheduler import DDPMNoiseScheduler

# 1. Загрузка моделей и весов
print("Инициализация моделей и загрузка чекпоинтов...")
conditioner = TextConditioner().to(DEVICE)
base_model = BaseUNet().to(DEVICE)
upscaler_model = UpscalerUNet().to(DEVICE)
scheduler = DDPMNoiseScheduler().to(DEVICE)

base_model.load_state_dict(torch.load("base_unet_checkpoint.pt", map_location=DEVICE, weights_only=True))
upscaler_model.load_state_dict(torch.load("upscaler_unet_checkpoint.pt", map_location=DEVICE, weights_only=True))

base_model.eval()
upscaler_model.eval()
conditioner.eval()
print("Модели успешно загружены!")

# 2. Функция генерации для веб-интерфейса
@torch.no_grad()
def generate_images(prompt: str, steps: int = 50):
    if not prompt.strip():
        return None, None, "Введите текстовый промпт!"

    text_emb = conditioner.encode_prompt([prompt])
    B = 1
    
    # 1. Генерация Base 64x64
    x_64 = torch.randn((B, 3, BASE_IMAGE_SIZE, BASE_IMAGE_SIZE), device=DEVICE)
    step_ratio = max(1, scheduler.num_timesteps // steps)
    timesteps = list(range(scheduler.num_timesteps - 1, -1, -step_ratio))
    
    for t in timesteps:
        x_64 = scheduler.p_sample(base_model, x_64, t, text_emb)
        x_64 = x_64.clamp(-2.0, 2.0)
        
    x_64 = x_64.clamp(-1.0, 1.0)
    
    # Конвертация Base 64x64 в PIL Image
    img_64_tensor = (x_64.squeeze(0).cpu() + 1.0) / 2.0 * 255.0
    img_64 = Image.fromarray(img_64_tensor.permute(1, 2, 0).byte().numpy())

    # 2. Генерация Upscaler 256x256
    x_64_upscaled = nn.functional.interpolate(x_64, size=(UPSCALED_IMAGE_SIZE, UPSCALED_IMAGE_SIZE), mode="bicubic", align_corners=False)
    x_256 = torch.randn((B, 3, UPSCALED_IMAGE_SIZE, UPSCALED_IMAGE_SIZE), device=DEVICE)
    
    for t in timesteps:
        conditioned_input = torch.cat([x_256, x_64_upscaled], dim=1)
        t_tensor = torch.full((B,), t, device=DEVICE, dtype=torch.long)
        
        pred_noise = upscaler_model(conditioned_input, t_tensor, text_emb)
        
        beta = scheduler.betas[t]
        sqrt_one_minus_alpha = scheduler.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = torch.sqrt(1.0 / scheduler.alphas[t])
        
        model_mean = sqrt_recip_alpha * (x_256 - beta / sqrt_one_minus_alpha * pred_noise)
        
        if t == 0:
            x_256 = model_mean
        else:
            posterior_var = scheduler.posterior_variance[t]
            noise = torch.randn_like(x_256)
            x_256 = model_mean + torch.sqrt(posterior_var) * noise
            
        x_256 = x_256.clamp(-2.0, 2.0)
            
    # Конвертация Upscale 256x256 в PIL Image
    img_256_tensor = (x_256.squeeze(0).cpu().clamp(-1, 1) + 1.0) / 2.0 * 255.0
    img_256 = Image.fromarray(img_256_tensor.permute(1, 2, 0).byte().numpy())

    return img_64, img_256, f"Успешно сгенерировано по промпту: '{prompt}'"

# 3. Верстка веб-интерфейса Gradio
with gr.Blocks(title="Custom Diffusion Generator") as demo:
    gr.Markdown("# 🎨 Custom Diffusion Model (64x64 Base → 256x256 Upscaler)")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Текстовый промпт", 
                placeholder="например: a red car on a street, sunny day...",
                lines=2
            )
            steps_slider = gr.Slider(
                minimum=10, 
                maximum=100, 
                value=50, 
                step=5, 
                label="Количество шагов диффузии (Steps)"
            )
            generate_btn = gr.Button("🚀 Сгенерировать", variant="primary")
            status_text = gr.Textbox(label="Статус", interactive=False)
            
        with gr.Column():
            with gr.Row():
                output_64 = gr.Image(label="Базовый кадр (64x64)", type="pil")
                output_256 = gr.Image(label="Апскейл (256x256)", type="pil")

    generate_btn.click(
        fn=generate_images,
        inputs=[prompt_input, steps_slider],
        outputs=[output_64, output_256, status_text]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
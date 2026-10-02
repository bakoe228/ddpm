import torch
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

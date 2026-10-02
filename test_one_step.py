import torch
from PIL import Image
from custom_diffusion.config import DEVICE
from custom_diffusion.models.unet import BaseUNet
from custom_diffusion.models.embed import TextConditioner
from custom_diffusion.utils.scheduler import DDPMNoiseScheduler

base_model = BaseUNet().to(DEVICE)
base_model.load_state_dict(torch.load("base_unet_checkpoint.pt", map_location=DEVICE))
base_model.eval()

conditioner = TextConditioner().to(DEVICE)
scheduler = DDPMNoiseScheduler().to(DEVICE)

# Берём случайный шум и шаг t = 500
x = torch.randn((1, 3, 64, 64), device=DEVICE)
t = torch.tensor([500], device=DEVICE, dtype=torch.long)
text_emb = conditioner.encode_prompt(["a black cat on a sofa"])

with torch.no_grad():
    pred_noise = base_model(x, t, text_emb)

print("Мин предсказанного шума:", pred_noise.min().item())
print("Макс предсказанного шума:", pred_noise.max().item())
print("Среднее предсказанного шума:", pred_noise.mean().item())
import torch
from PIL import Image
from custom_diffusion.config import DEVICE
from custom_diffusion.models.embed import TextConditioner
from custom_diffusion.models.unet import BaseUNet, UpscalerUNet
from custom_diffusion.utils.scheduler import DDPMNoiseScheduler
from custom_diffusion.generate import generate

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print(f"Запуск генерации на устройстве: {DEVICE}")
    
    # 1. Загрузка моделей
    conditioner = TextConditioner().to(DEVICE)
    base_model = BaseUNet().to(DEVICE)
    upscaler_model = UpscalerUNet().to(DEVICE)
    scheduler = DDPMNoiseScheduler().to(DEVICE)
    
    print(f"Параметров в Base UNet: {count_parameters(base_model):,}")
    print(f"Параметров в Upscaler UNet: {count_parameters(upscaler_model):,}")

    # 2. Загрузка чекпоинтов
    print("Загружаем обученные веса...")
    base_model.load_state_dict(torch.load("base_unet_checkpoint.pt", map_location=DEVICE))
    upscaler_model.load_state_dict(torch.load("upscaler_unet_checkpoint.pt", map_location=DEVICE))

    # 3. Интерактивный цикл ввода
    while True:
        prompt = input("\nВведите промпт (или 'exit' для выхода): ").strip()
        if prompt.lower() in ["exit", "quit", "выход"]:
            print("Завершение работы.")
            break
            
        if not prompt:
            continue

        print(f"Генерация по запросу: '{prompt}'...")
        res = generate(prompt, base_model, upscaler_model, scheduler, conditioner, steps=50)

        # Сохранение в файл
        img_tensor = res.squeeze(0).cpu().clamp(-1, 1)
        img_tensor = (img_tensor + 1) / 2.0 * 255.0
        img_array = img_tensor.permute(1, 2, 0).byte().numpy()

        image = Image.fromarray(img_array)
        image.save("result_256.png")
        print("Готово! Картинка сохранена в 'result_256.png'")

if __name__ == "__main__":
    main()
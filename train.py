import os
import gc
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from custom_diffusion.config import DEVICE
from custom_diffusion.models.embed import TextConditioner
from custom_diffusion.models.unet import BaseUNet, UpscalerUNet
from custom_diffusion.utils.scheduler import DDPMNoiseScheduler
from custom_diffusion.train import train_base_step, train_upscaler_step


class CustomFolderDataset(Dataset):
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.samples = []
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    img_path = os.path.join(root, file)
                    txt_path = os.path.splitext(img_path)[0] + '.txt'
                    if os.path.exists(txt_path):
                        self.samples.append((img_path, txt_path))
                        
        print(f"Загружено валидных пар картинка+текст: {len(self.samples)}")

        self.transform_base = T.Compose([
            T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop((256, 256)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        attempt = 0
        while attempt < 5:
            current_idx = (idx + attempt) % len(self.samples)
            img_path, txt_path = self.samples[current_idx]
            try:
                with Image.open(img_path) as img:
                    image = img.convert('RGB')
                    
                with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    caption = f.read().strip()

                if not caption:
                    caption = "a quality detailed photo"

                return {
                    "image": self.transform_base(image),
                    "caption": caption
                }
            except Exception:
                attempt += 1

        raise RuntimeError("Не удалось прочитать ни одно изображение после 5 попыток.")


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}ч {m:02d}мин {s:02d}сек"


def resize_gpu(tensor: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(tensor, size=(size, size), mode='bicubic', align_corners=False)


def main():
    parser = argparse.ArgumentParser(description="DDPM Training Script")
    parser.add_argument("--data_dir", type=str, default="./data", help="Путь к датасету")
    parser.add_argument("--save_dir", type=str, default=".", help="Папка для сохранения чекпоинтов")
    parser.add_argument("--epochs", type=int, default=1, help="Количество эпох")
    parser.add_argument("--batch_size", type=int, default=8, help="Размер батча")
    parser.add_argument("--mode", type=str, default=None, choices=["1", "2", "3"], help="Режим обучения (1, 2 или 3)")
    args = parser.parse_args()

    # Относительный путь от корня проекта по умолчанию
    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(data_dir)

    if not os.path.exists(data_dir):
        print(f"Ошибка: Папка {data_dir} не найдена!")
        return

    # Директория для хранения чекпоинтов
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    base_ckpt_path = os.path.join(save_dir, "base_unet_checkpoint.pt")
    upscaler_ckpt_path = os.path.join(save_dir, "upscaler_unet_checkpoint.pt")

    choice = args.mode
    if not choice:
        print("\n--- Выберите режим обучения ---")
        print("1 — Только Base UNet (64x64)")
        print("2 — Только Upscaler UNet (256x256)")
        print("3 — Оба этапа подряд (Base -> Upscaler)")
        choice = input("Введите цифру (1, 2 или 3): ").strip()

    dataset = CustomFolderDataset(data_dir)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=2,
        pin_memory=True if DEVICE == "cuda" else False,
        persistent_workers=True
    )

    conditioner = TextConditioner().to(DEVICE)
    base_model = BaseUNet().to(DEVICE)
    upscaler_model = UpscalerUNet().to(DEVICE)
    scheduler = DDPMNoiseScheduler().to(DEVICE)

    opt_base = torch.optim.AdamW(base_model.parameters(), lr=3e-5, weight_decay=1e-2)
    opt_upscale = torch.optim.AdamW(upscaler_model.parameters(), lr=3e-5, weight_decay=1e-2)

    total_start_time = time.perf_counter()

    # 1. Base UNet (64x64)
    if choice in ("1", "3"):
        if os.path.exists(base_ckpt_path):
            print(f"\nЗагрузка существующего чекпоинта Base UNet из {base_ckpt_path}...")
            base_model.load_state_dict(torch.load(base_ckpt_path, map_location=DEVICE))

        print("\nСтарт Этапа 1: Обучение Base UNet (64x64)...")
        base_epochs = args.epochs

        for epoch in range(base_epochs):
            epoch_start = time.perf_counter()
            running_loss = 0.0
            
            pbar = tqdm(
                dataloader, 
                desc=f"Base UNet | Эпоха {epoch+1}/{base_epochs}", 
                unit="batch"
            )
            
            for batch in pbar:
                imgs_gpu = batch["image"].to(DEVICE)
                low_res = resize_gpu(imgs_gpu, 64)
                
                with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                    loss = train_base_step(
                        base_model, scheduler, conditioner, low_res, batch["caption"], opt_base
                    )
                running_loss += loss
                pbar.set_postfix({"Loss": f"{loss:.4f}"})

            epoch_elapsed = time.perf_counter() - epoch_start
            avg_loss = running_loss / len(dataloader)
            print(f" Эпоха {epoch+1}/{base_epochs} завершена за {format_time(epoch_elapsed)} | Средний Base Loss: {avg_loss:.4f}")
            
            torch.save(base_model.state_dict(), base_ckpt_path)
            print(f"Сохранён чекпоинт: {base_ckpt_path}")
            torch.cuda.empty_cache()
            gc.collect()

    # 2. Upscaler UNet (256x256)
    if choice in ("2", "3"):
        if os.path.exists(upscaler_ckpt_path):
            print(f"\nЗагрузка существующего чекпоинта Upscaler UNet из {upscaler_ckpt_path}...")
            upscaler_model.load_state_dict(torch.load(upscaler_ckpt_path, map_location=DEVICE))

        print("\nСтарт Этапа 2: Обучение Upscaler UNet (256x256)...")
        upscale_epochs = args.epochs

        for epoch in range(upscale_epochs):
            epoch_start = time.perf_counter()
            running_loss = 0.0
            
            pbar = tqdm(
                dataloader, 
                desc=f"Upscaler UNet | Эпоха {epoch+1}/{upscale_epochs}", 
                unit="batch"
            )
            
            for batch in pbar:
                high_res = batch["image"].to(DEVICE)
                low_res = resize_gpu(high_res, 64)

                with torch.amp.autocast('cuda', enabled=(DEVICE == "cuda")):
                    loss = train_upscaler_step(
                        upscaler_model, scheduler, conditioner, high_res, low_res, batch["caption"], opt_upscale
                    )
                running_loss += loss
                pbar.set_postfix({"Loss": f"{loss:.4f}"})

            epoch_elapsed = time.perf_counter() - epoch_start
            avg_loss = running_loss / len(dataloader)
            print(f" Эпоха {epoch+1}/{upscale_epochs} завершена за {format_time(epoch_elapsed)} | Средний Upscaler Loss: {avg_loss:.4f}")
            
            torch.save(upscaler_model.state_dict(), upscaler_ckpt_path)
            print(f"Сохранён чекпоинт: {upscaler_ckpt_path}")
            torch.cuda.empty_cache()
            gc.collect()

    total_elapsed = time.perf_counter() - total_start_time
    print(f"\n[Успех] Завершено! Общее время работы: {format_time(total_elapsed)}")

if __name__ == "__main__":
    main()

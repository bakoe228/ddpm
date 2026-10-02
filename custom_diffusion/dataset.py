import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class DiffusionDataset(Dataset):
    def __init__(self, image_dir="dataset"):
        self.image_dir = image_dir
        
        # Поддерживаемые расширения картинок
        valid_extensions = (".png", ".jpg", ".jpeg", ".webp")
        
        # Сканируем папку и ищем все файлы картинок
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(valid_extensions)
        ]
        
        # Инициализируем трансформ один раз при создании датасета
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # 1. Загружаем изображение
        high_res_img = Image.open(img_path).convert("RGB")
        
        # 2. Масштабируем до размеров 64x64 и 256x256
        low_res_img = high_res_img.resize((64, 64), Image.BICUBIC)
        high_res_img = high_res_img.resize((256, 256), Image.BICUBIC)
        
        # 3. Ищем парный .txt файл с описанием (например, photo.jpg -> photo.txt)
        txt_name = os.path.splitext(img_name)[0] + ".txt"
        txt_path = os.path.join(self.image_dir, txt_name)
        
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        else:
            # Запасной вариант, если .txt забыли создать
            caption = "a quality detailed photo"

        return {
            "high_res": self.transform(high_res_img),
            "low_res": self.transform(low_res_img),
            "caption": caption
        }
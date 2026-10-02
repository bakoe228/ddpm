import os
import argparse
import torch
import gc
from PIL import Image
from tqdm import tqdm

import transformers.dynamic_module_utils
transformers.dynamic_module_utils.check_imports = lambda filename: []

import transformers.configuration_utils
if not hasattr(transformers.configuration_utils.PretrainedConfig, "forced_bos_token_id"):
    setattr(transformers.configuration_utils.PretrainedConfig, "forced_bos_token_id", None)

from transformers import AutoProcessor, AutoModelForCausalLM

parser = argparse.ArgumentParser(description="Generate Captions using Florence-2")
parser.add_argument("--img_dir", type=str, default="./data", help="Путь к изображениям")
parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = os.path.abspath(args.img_dir)
MODEL_ID = 'microsoft/Florence-2-large'
BATCH_SIZE = args.batch_size

print(f"Загрузка Florence-2-large на {DEVICE} (BATCH={BATCH_SIZE})...")

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
    attn_implementation="sdpa"
).to(DEVICE)

model.eval()

valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')

all_image_paths = []
for root, _, files in os.walk(IMG_DIR):
    for file in files:
        if file.lower().endswith(valid_extensions):
            img_path = os.path.join(root, file)
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(txt_path):
                all_image_paths.append(img_path)

print(f"Осталось обработать: {len(all_image_paths)}")

if not all_image_paths:
    print("Все картинки уже размечены!")
    exit()

PROMPT = "<MORE_DETAILED_CAPTION>"

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

batches = list(chunk_list(all_image_paths, BATCH_SIZE))

for step, batch_paths in enumerate(tqdm(batches, desc="Florence-2 Turbo Captioning")):
    images = []
    valid_paths = []
    
    for path in batch_paths:
        try:
            with Image.open(path) as raw_img:
                images.append(raw_img.convert('RGB'))
                valid_paths.append(path)
        except Exception:
            pass

    if not images:
        continue

    try:
        inputs = processor(text=[PROMPT] * len(images), images=images, return_tensors="pt", padding=True).to(DEVICE, torch.float16)
        
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=96,
                num_beams=1,
                do_sample=False,
                early_stopping=False,
                use_cache=True
            )
            
        generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=False)
        
        for idx, gen_text in enumerate(generated_texts):
            parsed_answer = processor.post_process_generation(
                gen_text, 
                task=PROMPT, 
                image_size=(images[idx].width, images[idx].height)
            )
            caption = parsed_answer[PROMPT]
            
            txt_path = os.path.splitext(valid_paths[idx])[0] + ".txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(caption.strip())

    except Exception as e:
        print(f"\nОшибка батча: {e}")

    if step > 0 and step % 100 == 0:
        torch.cuda.empty_cache()
        gc.collect()

print("\n[Успех] Все картинки размечены!")

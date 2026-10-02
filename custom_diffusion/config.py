import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_IMAGE_SIZE = 64
UPSCALED_IMAGE_SIZE = 256
BATCH_SIZE = 4
MAX_LENGTH = 77
TEXT_EMBED_DIM = 512
TIMESTEPS = 1000
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

import torch
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
        
        # Убрали use_safetensors=True, чтобы загружать стандартный bin-файл без ошибок
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
    def encode_prompt(self, prompts: List[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts,
            padding="max_length",  # Фиксируем seq_len до MAX_LENGTH для корректной работы CFG
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        device = next(self.text_encoder.parameters()).device
        input_ids = inputs.input_ids.to(device)
        
        with torch.no_grad():
            outputs = self.text_encoder(input_ids)
            embeddings = outputs.last_hidden_state  # Форма [B, MAX_LENGTH, 768]
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
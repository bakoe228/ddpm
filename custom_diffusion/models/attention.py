import torch
import torch.nn as nn
import math

class SpatialCrossAttention(nn.Module):
    """Модуль Cross-Attention (Запросы Q — фичи картинки, Ключи K и Значения V — текст)."""
    def __init__(self, query_dim: int, context_dim: int, heads: int = 8):
        super().__init__()
        self.heads = heads
        self.head_dim = query_dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], context: [B, seq_len, context_dim]
        B, C, H, W = x.shape
        N = H * W
        
        # Проекция изображения в вектор последин
        x_flat = x.permute(0, 2, 3, 1).reshape(B, N, C)
        
        q = self.to_q(x_flat).reshape(B, N, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(context).reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(context).reshape(B, -1, self.heads, self.head_dim).transpose(1, 2)
        
        # Вычисление внимания
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)
        out = self.to_out(out)
        
        # Возвращаем исходную размерность карты тензоров [B, C, H, W]
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)

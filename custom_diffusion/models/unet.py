import torch
import torch.nn as nn
from custom_diffusion.models.embed import SinusoidalTimeEmbedding
from custom_diffusion.models.attention import SpatialCrossAttention
from custom_diffusion.models.blocks import ResBlock, Downsample, Upsample
from custom_diffusion.config import TEXT_EMBED_DIM

class BaseUNet(nn.Module):
    """Базовый U-Net (64x64) с увеличенной глубиной и каналами."""
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_channels: int = 128):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder Level 1 (64x64 -> 128 ch)
        self.res1 = ResBlock(base_channels, base_channels, time_dim)
        self.down1 = Downsample(base_channels)
        
        # Encoder Level 2 (32x32 -> 256 ch)
        self.res2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.attn2 = SpatialCrossAttention(base_channels * 2, TEXT_EMBED_DIM)
        self.down2 = Downsample(base_channels * 2)
        
        # Bottleneck (16x16 -> 512 ch)
        self.mid_res1 = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid_attn = SpatialCrossAttention(base_channels * 4, TEXT_EMBED_DIM)
        self.mid_res2 = ResBlock(base_channels * 4, base_channels * 4, time_dim)
        
        # Decoder Level 2 (16x16 -> 32x32)
        self.up2 = Upsample(base_channels * 4)
        self.res_up2 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        self.attn_up2 = SpatialCrossAttention(base_channels * 2, TEXT_EMBED_DIM)
        
        # Decoder Level 1 (32x32 -> 64x64)
        self.up1 = Upsample(base_channels * 2)
        self.res_up1 = ResBlock(base_channels * 2 + base_channels, base_channels, time_dim)
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        
        # Conv In & Encoder
        h1 = self.conv_in(x)
        h1 = self.res1(h1, t_emb)
        
        h2 = self.down1(h1)
        h2 = self.res2(h2, t_emb)
        h2 = h2 + self.attn2(h2, text_emb)
        
        h3 = self.down2(h2)
        
        # Bottleneck
        h_mid = self.mid_res1(h3, t_emb)
        h_mid = h_mid + self.mid_attn(h_mid, text_emb)
        h_mid = self.mid_res2(h_mid, t_emb)
        
        # Decoder
        h_up = self.up2(h_mid)
        h_up = torch.cat([h_up, h2], dim=1)
        h_up = self.res_up2(h_up, t_emb)
        h_up = h_up + self.attn_up2(h_up, text_emb)
        
        h_up = self.up1(h_up)
        h_up = torch.cat([h_up, h1], dim=1)
        h_out = self.res_up1(h_up, t_emb)
        
        return self.conv_out(h_out)

class UpscalerUNet(nn.Module):
    """U-Net Апскейлер (256x256). Вход — 6 каналов."""
    def __init__(self, in_channels: int = 6, out_channels: int = 3, base_channels: int = 128):
        super().__init__()
        time_dim = base_channels * 4
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder
        self.res1 = ResBlock(base_channels, base_channels, time_dim)
        self.down1 = Downsample(base_channels)
        
        self.res2 = ResBlock(base_channels, base_channels * 2, time_dim)
        self.attn2 = SpatialCrossAttention(base_channels * 2, TEXT_EMBED_DIM)
        self.down2 = Downsample(base_channels * 2)
        
        # Bottleneck
        self.mid_res = ResBlock(base_channels * 2, base_channels * 4, time_dim)
        self.mid_attn = SpatialCrossAttention(base_channels * 4, TEXT_EMBED_DIM)
        
        # Decoder
        self.up2 = Upsample(base_channels * 4)
        self.res_up2 = ResBlock(base_channels * 4 + base_channels * 2, base_channels * 2, time_dim)
        
        self.up1 = Upsample(base_channels * 2)
        self.res_up1 = ResBlock(base_channels * 2 + base_channels, base_channels, time_dim)
        
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(t)
        
        h1 = self.conv_in(x)
        h1 = self.res1(h1, t_emb)
        
        h2 = self.down1(h1)
        h2 = self.res2(h2, t_emb)
        h2 = h2 + self.attn2(h2, text_emb)
        
        h3 = self.down2(h2)
        
        h_mid = self.mid_res(h3, t_emb)
        h_mid = h_mid + self.mid_attn(h_mid, text_emb)
        
        h_up = self.up2(h_mid)
        h_up = torch.cat([h_up, h2], dim=1)
        h_up = self.res_up2(h_up, t_emb)
        
        h_up = self.up1(h_up)
        h_up = torch.cat([h_up, h1], dim=1)
        h_out = self.res_up1(h_up, t_emb)
        
        return self.conv_out(h_out)
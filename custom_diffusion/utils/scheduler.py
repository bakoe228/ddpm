import torch
import torch.nn as nn
from custom_diffusion.config import TIMESTEPS

class DDPMNoiseScheduler:
    """Точная математика прямой (q_sample) и обратной (p_sample) диффузии Ho et al."""
    def __init__(self, num_timesteps: int = TIMESTEPS, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = nn.functional.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Константы для прямой диффузии
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Константы для обратной диффузии
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None):
        """Прямой процесс: зашумление x_0 до x_t."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor):
        """Обратный процесс: удаление шума из x_t до x_{t-1}."""
        device = x.device
        t_idx = t.item() if isinstance(t, torch.Tensor) else t
        t_tensor = torch.full((x.shape[0],), t_idx, device=device, dtype=torch.long)
        
        with torch.no_grad():
            pred_noise = model(x, t_tensor, text_emb)
            
        beta = self.betas[t_idx]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t_idx]
        sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas[t_idx])
        
        # Среднее значение p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alpha * (x - beta / sqrt_one_minus_alpha * pred_noise)
        
        if t_idx == 0:
            return model_mean
        else:
            posterior_var = self.posterior_variance[t_idx]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var) * noise

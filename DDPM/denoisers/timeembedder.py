import torch
import math

def sinusoidal_time_embed(t, dim):
    # Uses sinusoidal embedding to embed t to a vector of size dim
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / half))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, 2*half]
    if dim % 2 == 1:  # pad if odd
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb
import torch
import torch.nn as nn
import math
from utils.model_utils.init_utils import init

def sinusoidal_time_embed(t, dim):
    """
    Compute sinusoidal (Fourier) embeddings of a timestep or noise level.

    This is the same idea as positional encodings in Transformers:
    map a scalar input `t` into a higher-dimensional vector of sinusoids
    with different frequencies.

    Args:
        t (torch.Tensor): 1D tensor of shape [B], containing timesteps or scalars.
        dim (int): Dimension of the output embedding.

    Returns:
        torch.Tensor: Sinusoidal embeddings of shape [B, dim].
                      If `dim` is odd, the result is padded by one column of zeros.
    """
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * (-math.log(10000.0) / half))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [B, 2*half]
    if dim % 2 == 1:  # pad if odd
        emb = torch.nn.functional.pad(emb, (0,1))
    return emb

class TimeEmbed(nn.Module):
    """
    Time embedding module: maps timesteps into a learned vector representation.

    Pipeline:
        - Use sinusoidal_time_embed to compute Fourier features of t
        - Pass through a 2-layer MLP with SiLU nonlinearity
        - Output a time embedding vector of size time_dim_out

    Args:
        time_dim_in (int): Input dimension to the MLP (size of sinusoidal embedding).
        time_dim_out (int): Output dimension of the time embedding.
        hidden (int): Hidden dimension of the intermediate MLP layer.

    Shape:
        Input:  t (torch.Tensor), shape [B], scalar timestep per batch element
        Output: embedding (torch.Tensor), shape [B, time_dim_out]
    """
    def __init__(self, time_dim_in: int, time_dim_out: int, hidden: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim_in, hidden),
            nn.SiLU(),
            nn.Linear(hidden, time_dim_out),
        )
        self._init_weights()

    @torch.no_grad()
    def _init_weights(self):
        for m in self.mlp:
            init(m, nonlin="relu", scheme="kaiming")
        

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = sinusoidal_time_embed(t, self.mlp[0].in_features)  # time_dim_in
        return self.mlp(base)  # [B, time_dim_out]
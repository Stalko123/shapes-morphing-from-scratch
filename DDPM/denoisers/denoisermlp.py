import torch
import torch.nn as nn
from typing import Sequence, Union, Tuple
from DDPM.denoisers.timeembedder import sinusoidal_time_embed


def _get_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    # default
    return nn.SiLU()


def _get_norm(kind: str, dim: int) -> nn.Module:
    kind = (kind or "none").lower()
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        # BatchNorm1d expects [B, F] and normalizes over batch
        return nn.BatchNorm1d(dim)
    return nn.Identity()


class DenoiserMLP(nn.Module):
    """
    MLP denoiser:
      - Flattens image, concats sinusoidal time embedding
      - Block order: Linear -> Norm -> Activation -> Dropout (per hidden layer)
      - Predicts noise
    """
    def __init__(
        self,
        img_shape: Tuple[int, int, int],            # (C, H, W)
        hidden_sizes: Sequence[int] = (1024, 1024),
        time_dim: int = 128,
        activation: str = "silu",
        norm: str = "layer",                        # 'none' | 'layer' | 'batch'
        dropout: Union[float, Sequence[float]] = 0.0,
    ):
        super().__init__()
        C, H, W = img_shape
        self.img_shape = img_shape
        in_img = C * H * W
        self.time_dim = time_dim
        input_dim = in_img + time_dim

        # normalize dropout config
        if isinstance(dropout, (int, float)):
            dropouts = [float(dropout)] * len(hidden_sizes)
        else:
            dropouts = list(dropout)
            assert len(dropouts) == len(hidden_sizes), f"DenoiserMLP error : dropout list must match hidden_sizes ({len(hidden_sizes)}), got {len(dropouts)}"

        act = _get_activation(activation)

        layers = []
        prev = input_dim
        for i, width in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, width))
            layers.append(_get_norm(norm, width))
            layers.append(act.__class__())
            if dropouts[i] > 0:
                layers.append(nn.Dropout(dropouts[i]))
            prev = width

        layers.append(nn.Linear(prev, in_img))
        self.net = nn.Sequential(*layers)


    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, C, H, W]
        t:   [B] (long)
        returns ε_pred with same shape as x_t
        """
        B = x_t.size(0)
        x_flat = x_t.view(B, -1)
        #t_emb = sinusoidal_time_embed(t, self.time_dim)  # [B, time_dim]
        t_emb = t.unsqueeze(1)
        input = torch.cat([x_flat, t_emb], dim=1)
        eps = self.net(input).view_as(x_t)
        return eps
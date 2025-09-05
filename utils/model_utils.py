import torch
import math
import torch.nn as nn

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


def get_activation(name: str) -> nn.Module:
    """
    Retrieve an activation function module by name.

    Supported names:
        - "relu": nn.ReLU
        - "gelu": nn.GELU
        - "tanh": nn.Tanh
        - "silu" (default): nn.SiLU

    Args:
        name (str): Name of the activation function (case-insensitive).
                    Defaults to "silu" if None or unrecognized.

    Returns:
        nn.Module: The corresponding activation function module.
    """
    name = (name or "silu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    # default
    return nn.SiLU()


def get_norm1d(kind: str, dim: int) -> nn.Module:
    """
    Retrieve a 1D normalization layer (for vectors of shape [B, F]).

    Supported kinds:
        - "layer": nn.LayerNorm(dim)
        - "batch": nn.BatchNorm1d(dim)
        - "none":  nn.Identity()

    Args:
        kind (str): Type of normalization. Case-insensitive.
        dim (int): Feature dimension.

    Returns:
        nn.Module: The corresponding normalization layer.
    """
    kind = (kind or "none").lower()
    if kind == "layer":
        return nn.LayerNorm(dim)
    if kind == "batch":
        # BatchNorm1d expects [B, F] and normalizes over batch
        return nn.BatchNorm1d(dim)
    return nn.Identity()


def get_norm2d(kind: str, num_channels: int, groups: int) -> nn.Module:
    """
    Retrieve a 2D normalization layer (for tensors of shape [B, C, H, W]).

    Supported kinds:
        - "none":      nn.Identity()
        - "batch2d":   nn.BatchNorm2d(num_channels)
        - "instance":  nn.InstanceNorm2d(num_channels, affine=True)
        - "layer2d":   nn.GroupNorm(1, num_channels)  # channels-first "LayerNorm"
        - "group" (default): nn.GroupNorm(groups, num_channels)

    Args:
        kind (str): Type of normalization. Case-insensitive.
        num_channels (int): Number of channels in the feature maps.
        groups (int): Number of groups for GroupNorm. Clamped to [1, num_channels].

    Returns:
        nn.Module: The corresponding normalization layer.
    """
    kind = (kind or "group").lower()
    if kind == "none":
        return nn.Identity()
    if kind == "batch2d":
        return nn.BatchNorm2d(num_channels)
    if kind == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if kind == "layer2d":
        # channels-first layer normalization
        return nn.GroupNorm(1, num_channels)
    # default: group norm
    return nn.GroupNorm(max(1, min(groups, num_channels)), num_channels)


def conv3x3(in_ch: int, out_ch: int, k: int = 3, stride: int = 1) -> nn.Conv2d:
    """
    Create a 2D convolution layer with "same" padding for odd kernel sizes.

    This is a convenience wrapper around nn.Conv2d with:
        - kernel size = k (default 3)
        - stride = stride (default 1)
        - padding = k // 2, so that spatial resolution is preserved when stride=1

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        k (int, optional): Kernel size. Default is 3.
        stride (int, optional): Stride. Default is 1.

    Returns:
        nn.Conv2d: The convolution layer.
    """
    pad = k // 2  # "same" padding for odd kernels
    return nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=stride, padding=pad)


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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        base = sinusoidal_time_embed(t, self.mlp[0].in_features)  # time_dim_in
        return self.mlp(base)  # [B, time_dim_out]

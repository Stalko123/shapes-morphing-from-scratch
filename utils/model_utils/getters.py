import torch.nn as nn

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


def conv3x3(in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1) -> nn.Conv2d:
    """
    Create a 2D convolution layer with "same" padding for odd kernel sizes.

    This is a convenience wrapper around nn.Conv2d with:
        - kernel size = k (default 3)
        - stride = stride (default 1)
        - padding = k // 2, so that spatial resolution is preserved when stride=1

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        k : Kernel size.
        stride (int, optional): Stride. Default is 1.

    Returns:
        nn.Conv2d: The convolution layer.
    """
    pad = kernel_size // 2  # "same" padding for odd kernels
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad)
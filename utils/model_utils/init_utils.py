import torch
import torch.nn as nn

def init_norm(m):
    if getattr(m, "weight", None) is not None:
            nn.init.ones_(m.weight)
    if getattr(m, "bias", None) is not None:
        nn.init.zeros_(m.bias)
        

def _make_bilinear_kernel(k: int):
    """
    2D bilinear upsampling kernel of size k x k.
    """
    factor = (k + 1) // 2
    if k % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = torch.arange(k, dtype=torch.float32)
    filt = (1 - torch.abs(og - center) / factor)
    kernel = filt[:, None] * filt[None, :]
    return kernel

def init_convtranspose_bilinear(m: nn.ConvTranspose2d):
    """
    Initialize ConvTranspose2d as bilinear upsampler (inplace).
    """
    k = m.kernel_size[0]
    assert m.kernel_size[0] == m.kernel_size[1], "Use square kernels for bilinear init."
    weight = torch.zeros_like(m.weight)
    bilinear = _make_bilinear_kernel(k)
    # fill the (oc, ic) channel pairs with the same bilinear kernel
    for oc in range(m.out_channels):
        ic = oc if oc < m.in_channels else (oc % m.in_channels)
        weight[oc, ic, :, :] = bilinear
    with torch.no_grad():
        m.weight.copy_(weight)
        if m.bias is not None:
            m.bias.zero_()


def init(m: nn.Module, scheme: str, nonlin: str) -> None:
    """
    Initialize Linear / Conv / Norm / ConvTranspose layers with a unified scheme.
    - scheme: "auto" | "kaiming" | "xavier" | "lecun" | "orthogonal" | "normal"
    - nonlin: activation name driving the gain ("relu" works for ReLU/SiLU/GELU)

    Biases default to zeros (safe for all).
    """
    # norm layers: gamma=1, beta=0
    if isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        init_norm(m)
        return

    is_linear = isinstance(m, nn.Linear)
    is_conv = isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                             nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d))
    if not (is_linear or is_conv):
        return

   
    if isinstance(m, nn.ConvTranspose2d):
        kH, kW = m.kernel_size
        sH, sW = m.stride
        if sH > 1 and sW > 1 and kH == kW:
            init_convtranspose_bilinear(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            return

    if scheme == "auto":
        scheme = "kaiming" if nonlin.lower() in ("relu", "silu", "swish", "gelu") else "xavier"

    if scheme == "kaiming":
        # fan_in keeps forward activations well-scaled for ReLU-like nonlins
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif scheme == "xavier":
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(
            "tanh" if nonlin.lower() == "tanh" else "linear"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif scheme == "lecun":
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif scheme == "orthogonal":
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain(
            "relu" if nonlin.lower() in ("relu", "silu", "swish", "gelu") else "linear"))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif scheme == "normal":
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    else:
        # fallback: Kaiming fan_in
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
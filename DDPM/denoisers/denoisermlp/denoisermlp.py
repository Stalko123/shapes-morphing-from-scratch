import torch
import torch.nn as nn
from typing import Sequence, Union, Tuple, Literal
import math
from utils.model_utils import sinusoidal_time_embed, get_activation, get_norm1d


class DenoiserMLP(nn.Module):
    """
    MLP denoiser:
      - Flattens image, concats sinusoidal time embedding
      - Block order: Linear -> Norm -> Activation -> Dropout (per hidden layer)
      - Predicts noise
    """
    def __init__(
        self,
        img_shape: Tuple[int, int, int],                                    # (C, H, W) : image shape
        hidden_sizes: Sequence[int],                                        # widths of the hidden layers
        time_dim: int = 128,                                                # dimension of time embedding
        activation: Literal["silu", "relu", "gelu", "tanh"] = "silu",       # activation function
        norm: Literal["none", "layer", "batch"] = "layer",                  # 'none' | 'layer' | 'batch'
        dropout: Union[float, Sequence[float]] = 0.0,                       # dropout layers
        init_scheme: str = "auto",                                          # initialization scheme
    ):
        super().__init__()
        C, H, W = img_shape
        self.img_shape = img_shape
        in_img = C * H * W
        self.time_dim = time_dim
        input_dim = in_img + time_dim
        self.init_scheme = (init_scheme or "auto").lower()
        self.activation = activation

        # normalize dropout config
        if isinstance(dropout, (int, float)):
            dropouts = [float(dropout)] * len(hidden_sizes)
        else:
            dropouts = list(dropout)
            assert len(dropouts) == len(hidden_sizes), f"DenoiserMLP error : dropout list must match hidden_sizes ({len(hidden_sizes)}), got {len(dropouts)}"

        act = get_activation(activation)

        layers = []
        prev = input_dim
        for i, width in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev, width))
            layers.append(get_norm1d(norm, width))
            layers.append(act.__class__())
            if dropouts[i] > 0:
                layers.append(nn.Dropout(dropouts[i]))
            prev = width

        layers.append(nn.Linear(prev, in_img))
        self.net = nn.Sequential(*layers)


    def _apply_init(self):
        """
        Initialize Linear/Norm layers according to self.init_scheme and self.activation_name.
        - SiLU/GELU/ReLU -> Kaiming fan_in (relu gain) by default
        - tanh -> Xavier
        - lecun -> 1/fan_in (aka LeCun normal)
        - orthogonal/normal supported explicitly
        """
        # map activation to a PyTorch nonlinearity string for gain
        act = self.activation
        if act in ("relu", "gelu", "silu", "swish"):
            nonlin = "relu"
        elif act == "tanh":
            nonlin = "tanh"
        else:
            nonlin = "linear"

        for m in self.net.modules():
            if isinstance(m, nn.Linear):

                scheme = self.init_scheme

                if scheme == "kaiming" or (scheme == "auto" and nonlin == "relu"):
                    # fan_in to preserve forward activation variance for ReLU-like
                    nn.init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity=nonlin)
                    if m.bias is not None:
                        # match PyTorch default bias init bound
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                        nn.init.uniform_(m.bias, -bound, bound)

                elif scheme == "xavier" or (scheme == "auto" and act == "tanh"):
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain(nonlin))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif scheme == "lecun":
                    nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_in", nonlinearity="linear")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif scheme == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain(nonlin))
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif scheme == "normal":
                    # small normal
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                else:
                    # fallback: kaiming
                    nn.init.kaiming_uniform_(m.weight, a=0.0, mode="fan_in", nonlinearity=nonlin)
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                        nn.init.uniform_(m.bias, -bound, bound)

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)


    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, C, H, W]
        t:   [B] (long)
        returns ε_pred with same shape as x_t
        """
        B = x_t.size(0)
        x_flat = x_t.view(B, -1)
        t_emb = sinusoidal_time_embed(t, self.time_dim)  # [B, time_dim]
        input = torch.cat([x_flat, t_emb], dim=1)
        eps = self.net(input).view_as(x_t)
        return eps
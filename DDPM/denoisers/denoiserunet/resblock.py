import torch
import torch.nn as nn
from utils.model_utils.getters import get_activation, get_norm2d, conv3x3
from typing import Literal


class ResBlock(nn.Module):
    """
    Norm-Act-Conv -> (FiLM with time) -> Norm-Act-Conv -> + skip
    - FiLM (scale_shift) after first Norm: y = (1+gamma)*y + beta
    - Zero-init second conv for stability.
    """
    def __init__(
        self,
        in_ch: int,                                                             # number of input channels
        out_ch: int,                                                            # number of output channels
        norm: Literal["none","batch2d","group","instance","layer2d"],           # normalizaton 
        activation: Literal["silu", "relu", "gelu", "tanh"],                    # activation
        groups: int,                                                            # number of groups for group norm
        kernel_size: int = 3,                                                   # kernel size
        time_dim: int = 0,                                                      # if > 0, expect FiLM from time
        dropout: float = 0.0,                                                   # dropout
    ):
        super().__init__()
        self.act = get_activation(activation)
        self.norm1 = get_norm2d(norm, in_ch, groups)
        self.conv1 = conv3x3(in_ch, out_ch)

        self.use_time = time_dim > 0
        if self.use_time:
            self.to_film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_dim, 2 * out_ch)  # gamma, beta
            )

        self.norm2 = get_norm2d(norm, out_ch, groups)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = conv3x3(out_ch, out_ch)

        # zero-init second conv (helps diffusion training)
        nn.init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            nn.init.zeros_(self.conv2.bias)

        # skip for channel change
        self.skip = (nn.Identity() if in_ch == out_ch
                     else nn.Conv2d(in_ch, out_ch, kernel_size=1))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        if self.use_time and t_emb is not None:
            # FiLM after first norm/conv path
            gamma, beta = self.to_film(t_emb).chunk(2, dim=1)   # [B,C]
            gamma = gamma[:, :, None, None] # expends
            beta  = beta[:, :, None, None]  # expends
            h = (1.0 + gamma) * h + beta

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)
import torch.nn as nn
from typing import Literal


class Upsample(nn.Module):
    def __init__(
            self,
            ch: int,                                                               # number of channels
            method: Literal["convtranspose","nearest_conv"] = "nearest_conv"       # up-sampling method
        ):
        super().__init__()
        if method == "convtranspose":
            self.up = nn.ConvTranspose2d(ch, ch, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode="nearest")
            self.mix = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

        self.method = method

    def forward(self, x):
        x = self.up(x)
        if self.method == "nearest_conv":
            x = self.mix(x)
        return x
import torch.nn as nn
from typing import Literal

class Downsample(nn.Module):
    def __init__(
            self, ch: int,                                          # number of input channels
            method: Literal["stride","pool","avgpool"] = "stride"   # down sampling method
        ):
        super().__init__()
        if method == "stride":
            self.op = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
        elif method == "pool":
            self.op = nn.Sequential(nn.MaxPool2d(2))
        else:  # "avgpool"
            self.op = nn.Sequential(nn.AvgPool2d(2))

    def forward(self, x): return self.op(x)
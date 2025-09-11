import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils.getters import get_norm2d

class Attention2d(nn.Module):
    """
    Multi-head self-attention on (H,W) tokens.
    """
    def __init__(self, channels: int, num_heads: int = 4, norm: str = "group", groups: int = 32):
        super().__init__()
        assert channels % num_heads == 0, "Attention2d: channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = get_norm2d(norm, channels, groups)

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        returns: [B, C, H, W]
        """
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)

        qkv = self.qkv(x)                               # [B, 3C, H, W]
        q, k, v = qkv.chunk(3, dim=1)                   # each [B, C, H, W]

        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)

        attn = torch.einsum("bhdk,bhdq->bhkq", q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)

        # [B, H, D, HW]
        out = torch.einsum("bhkq,bhdq->bhdk", attn, v)
        out = out.contiguous().view(b, c, h, w)                    # merge heads

        out = self.proj(out)
        return x_in + out                                          # residual

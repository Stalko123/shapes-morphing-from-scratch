import torch
import torch.nn as nn
from typing import Sequence, Union, Tuple, Literal
import math
from utils.model_utils import get_activation, get_norm1d, TimeEmbed, init


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation: for a hidden vector h (B, D) and a cond vector c (B, C),
    produce gamma, beta in (B, D) and return gamma * h + beta. We parameterize:
        [gamma_delta, beta] = Linear(c) -> (B, 2D)
        gamma = 1 + gamma_delta            
    """
    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.to_gamma_beta = nn.Linear(cond_dim, 2 * hidden_dim)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gb = self.to_gamma_beta(cond)                 # (B, 2D)
        gamma_delta, beta = gb.chunk(2, dim=-1)       # (B, D), (B, D)
        gamma = 1.0 + gamma_delta
        return gamma * h + beta


class DenoiserMLP(nn.Module):
    """
    MLP denoiser with FiLM conditioning by time embedding:
      - Flatten image
      - Per hidden layer: Linear -> Norm -> FiLM(time) -> Activation -> (Dropout)
      - Final Linear predicts noise, reshaped to image
    """
    def __init__(
        self,
        img_shape: Tuple[int, int, int],                          # (C, H, W)
        hidden_sizes: Sequence[int],                              # widths of hidden layers
        time_base_dim: int = 128,                                 # sinusoidal input dim
        time_output_dim: int = 256,                               # time embedding output dim
        time_hidden: int = 512,                                   # time MLP hidden
        activation: Literal["silu", "relu", "gelu", "tanh"] = "silu",
        norm: Literal["none", "layer", "batch"] = "layer",
        dropout: Union[float, Sequence[float]] = 0.0,
        init_scheme: str = "auto",
    ):
        super().__init__()
        C, H, W = img_shape
        self.img_shape = img_shape
        in_img = C * H * W

        self.init_scheme = (init_scheme or "auto").lower()
        self.activation_name = activation

        # Time embedding network
        self.time_embed = TimeEmbed(
            time_dim_in=time_base_dim,
            time_dim_out=time_output_dim,
            hidden=time_hidden,
        )
        self.cond_dim = time_output_dim

        # normalize dropout config
        if isinstance(dropout, (int, float)):
            dropouts = [float(dropout)] * len(hidden_sizes)
        else:
            dropouts = list(dropout)
            assert len(dropouts) == len(hidden_sizes), \
                f"dropout list must match hidden_sizes ({len(hidden_sizes)}), got {len(dropouts)}"

        act = get_activation(activation)

        self.in_proj = nn.Linear(in_img, hidden_sizes[0]) if hidden_sizes else nn.Identity()
        self.in_norm = get_norm1d(norm, hidden_sizes[0]) if hidden_sizes else nn.Identity()
        self.in_film = FiLM(self.cond_dim, hidden_sizes[0]) if hidden_sizes else nn.Identity()
        self.in_act = act.__class__() if hidden_sizes else nn.Identity()
        self.in_drop = nn.Dropout(dropouts[0]) if hidden_sizes and dropouts[0] > 0 else nn.Identity()

        blocks = []
        films  = []
        norms  = []
        acts   = []
        drops  = []

        for i in range(1, len(hidden_sizes)):
            in_d, out_d = hidden_sizes[i-1], hidden_sizes[i]
            blocks.append(nn.Linear(in_d, out_d))
            norms.append(get_norm1d(norm, out_d))
            films.append(FiLM(self.cond_dim, out_d))
            acts.append(act.__class__())
            drops.append(nn.Dropout(dropouts[i]) if dropouts[i] > 0 else nn.Identity())

        self.blocks = nn.ModuleList(blocks)
        self.norms  = nn.ModuleList(norms)
        self.films  = nn.ModuleList(films)
        self.acts   = nn.ModuleList(acts)
        self.drops  = nn.ModuleList(drops)

        # Output head
        last_width = hidden_sizes[-1] if hidden_sizes else in_img
        self.out_proj = nn.Linear(last_width, in_img)

        self._apply_init()


    def _apply_init(self):
        act = self.activation_name
        nonlin = "relu" if act in ("relu", "gelu", "silu", "swish") else ("tanh" if act == "tanh" else "linear")
        for m in self.modules():
            init(m, self.init_scheme, nonlin)


    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, C, H, W]
        t:   [B] (int/long or float)
        returns ε_pred with shape [B, C, H, W]
        """
        B = x_t.size(0)
        x = x_t.view(B, -1)

        t_emb = self.time_embed(t)

        if isinstance(self.in_proj, nn.Identity):
            h = x
        else:
            h = self.in_proj(x)
            h = self.in_norm(h)
            h = self.in_film(h, t_emb)
            h = self.in_act(h)
            h = self.in_drop(h)

        for lin, norm, film, act, drop in zip(self.blocks, self.norms, self.films, self.acts, self.drops):
            h = lin(h)
            h = norm(h)
            h = film(h, t_emb)
            h = act(h)
            h = drop(h)

        eps = self.out_proj(h).view_as(x_t)
        return eps
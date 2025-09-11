import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union, Literal
from utils.model_utils.getters import get_activation, get_norm2d
from utils.model_utils.init_utils import init
from utils.model_utils.time_embedding import TimeEmbed
from .attention2d import Attention2d
from .resblock import ResBlock
from .downsample import Downsample
from .upsample import Upsample


class DenoiserUNet(nn.Module):
    """
    UNet denoiser:
      - Down path: [num_res_blocks per stage] -> Downsample
      - Bottleneck: n ResBlocks
      - Up path: Upsample -> concat skip -> [num_res_blocks per stage]
      - Time conditioning: sinusoidal -> MLP -> FiLM in ResBlocks
    """
    def __init__(
        self,
        img_shape: Tuple[int, int, int],                                        # (C, H, W) : image shape
        base_channels: int,                                                     # start depth
        channel_mults: Sequence[int],                                           # per-stage depth multipliers
        num_res_blocks: Union[int, Sequence[int]],                              # per-stage residual depth 
        upsample: Literal["convtranspose","nearest_conv"],                      # up-sampling method
        norm: Literal["none","batch2d","group","instance","layer2d"],           # normalization kind
        groups: int,                                                            # groups for groupnorm
        num_res_blocks_in_bottleneck: int = 2,                                  # number of residual blocks in bottleneck
        stem_kernel: int = 3,
        head_kernel: int = 3,
        downsample:  Literal["stride","pool","avgpool"] = "stride",             # down-sampling
        activation: Literal["silu", "relu", "gelu", "tanh"] = "silu",           # activation
        time_base_dim: int = 128,                                               # input dim for sinusoidal embedding
        time_output_dim: int = 256,                                             # time embedding output dim
        time_hidden: int = 512,                                                 # time MLP hidden
        dropout: float = 0.0,                                                   # dropout inside resblocks
        init_scheme: str = "auto",                                              # weight initialization scheme
        attn_stages: Sequence[bool] = None,                                     # where to use attention
        attn_num_heads: int = 4,                                                # number of attention heads
        attn_in_bottleneck: bool = True,                                        # use attention in bottleneck
    ):
        super().__init__()
        C, _, _ = img_shape
        self.in_channels = C
        self.out_channels = C
        self.activation_name = activation
        self.norm = norm
        self.groups = groups
        self.num_res_blocks_in_bottleneck = num_res_blocks_in_bottleneck
        self.stem_kernel = stem_kernel
        self.head_kernel = head_kernel
        self.init_scheme = init_scheme
        self.attn_num_heads = attn_num_heads
        self.attn_in_bottleneck = attn_in_bottleneck

        # normalize configs
        S = len(channel_mults)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * S
        assert len(num_res_blocks) == S
        assert stem_kernel % 2 == 1 and head_kernel % 2 == 1, "DenoiserUNet error : choose odd kernel sizes"

        if attn_stages is None:
            # default : attention on the last 1-2 (lowest-res) stages
            attn_stages = [False] * S
            if S >= 1:
                attn_stages[-1] = True
            if S >= 2:
                attn_stages[-2] = True
        assert len(attn_stages) == S
        self.attn_stages = attn_stages

        # time embedding
        self.time_base_dim = time_base_dim
        self.time_dim_out = time_output_dim
        self.time_embed = TimeEmbed(time_dim_in=self.time_base_dim,
                                    time_dim_out=time_output_dim,
                                    hidden=time_hidden)

        # input
        ch = base_channels * channel_mults[0]
        if norm == "group":
            assert groups <= ch and ch%groups == 0, "DenoiserUNet error : groups need to divide the base number of channels"
        self.in_conv = nn.Conv2d(C, ch, kernel_size=stem_kernel, stride=1, padding=stem_kernel//2)

        # encoder
        enc_blocks = nn.ModuleList()
        enc_channels = []
        for s, mult in enumerate(channel_mults):
            stage_ch = base_channels * mult
            # within-stage: keep channels constant
            if ch != stage_ch:
                enc_blocks.append(ResBlock(ch, stage_ch, norm, activation, groups,
                                           time_dim=time_output_dim, dropout=dropout))
                ch = stage_ch
            # residual stack
            for _ in range(num_res_blocks[s]):
                enc_blocks.append(ResBlock(ch, ch, norm, activation, groups, time_dim=time_output_dim, dropout=dropout))

            if self.attn_stages[s]:
                enc_blocks.append(Attention2d(ch, num_heads=self.attn_num_heads, norm=norm, groups=groups))

            enc_channels.append(ch)

            # downsample except last stage
            if s < S - 1:
                enc_blocks.append(Downsample(ch, method=downsample))

        self.encoder = enc_blocks

        # bottleneck
        self.bottleneck = nn.ModuleList()
        for _ in range(num_res_blocks_in_bottleneck):
            self.bottleneck.append(ResBlock(ch, ch, norm, activation, groups, time_dim=time_output_dim, dropout=dropout))
        if self.attn_in_bottleneck:
            self.bottleneck.append(Attention2d(ch, num_heads=self.attn_num_heads, norm=norm, groups=groups))

        # decoder
        dec_blocks = nn.ModuleList()
        for s in reversed(range(S)):
            # upsample except immediately after last stage (which is the bottom)
            if s < S - 1:
                dec_blocks.append(Upsample(ch, method=upsample))
            # after upsample, concat skip from encoder stage s -> channels double
            skip_ch = enc_channels[s]
            dec_blocks.append(nn.Conv2d(ch + skip_ch, skip_ch, kernel_size=1))  # channel mixer
            ch = skip_ch
            # residual stack
            for _ in range(num_res_blocks[s]):
                dec_blocks.append(ResBlock(ch, ch, norm, activation, groups, time_dim=time_output_dim, dropout=dropout))
            if self.attn_stages[s]:
                dec_blocks.append(Attention2d(ch, num_heads=self.attn_num_heads, norm=norm, groups=groups))
        self.decoder = dec_blocks

        # output head
        self.out_norm = get_norm2d(norm, ch, groups)
        self.out_act = get_activation(activation)
        self.out_conv = nn.Conv2d(ch, self.out_channels, kernel_size=head_kernel, stride=1, padding=head_kernel//2)

        self._init_weights()

    
    def _init_weights(self):
        act = self.activation_name
        nonlin = "relu" if act in ("relu", "gelu", "silu", "swish") else ("tanh" if act == "tanh" else "linear")
        for m in self.modules():
            init(m, self.init_scheme, nonlin)
    
    def _is_stage_block(self, m: nn.Module) -> bool:
        return isinstance(m, (ResBlock, Attention2d))
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x_t: [B, C, H, W]
        t:   [B] (long or float); embedded via sinusoidal_time_embed
        """
        # time embedding shared across blocks (each block has its own projector)
        t_emb = self.time_embed(t)  # [B, time_dim]

        # stem
        h = self.in_conv(x_t)

        # encoder pass with skip saves
        skips = []
        i = 0
        while i < len(self.encoder):
            block = self.encoder[i]
            if self._is_stage_block(block):
                while i < len(self.encoder) and self._is_stage_block(self.encoder[i]):
                    m = self.encoder[i]
                    h = m(h, t_emb) if isinstance(m, ResBlock) else m(h)
                    i += 1
                skips.append(h)
                if i < len(self.encoder) and isinstance(self.encoder[i], Downsample):
                    h = self.encoder[i](h)
                    i += 1
            else:
                h = block(h)
                i += 1

        # bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb) if isinstance(block, ResBlock) else block(h)

        # decoder pass
        j = 0
        s = len(skips) - 1
        while j < len(self.decoder):
            if isinstance(self.decoder[j], Upsample):
                h = self.decoder[j](h)
                j += 1
            # merge skip
            assert s >= 0
            skip = skips[s]
            s -= 1
            # concat then 1x1 mix to stage channels
            h = torch.cat([h, skip], dim=1)
            h = self.decoder[j](h)  # 1x1 mixer
            j += 1
            # residual stack at this stage
            while j < len(self.decoder) and self._is_stage_block(self.decoder[j]):
                m = self.decoder[j]
                h = m(h, t_emb) if isinstance(m, ResBlock) else m(h)
                j += 1

        # head
        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)
        return out
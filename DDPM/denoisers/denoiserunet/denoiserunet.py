import torch
import torch.nn as nn
from typing import Sequence, Tuple, Union, Literal
from model_utils import TimeEmbed, get_activation, get_norm2d, conv3x3
from resblock import ResBlock
from downsample import Downsample
from upsample import Upsample


class DenoiserUNet(nn.Module):
    """
    UNet denoiser:
      - Down path: [num_res_blocks per stage] -> Downsample
      - Bottleneck: 2 ResBlocks
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
        kernel_size: Union[int, Sequence[int]] = 3,                             # conv kernel(s)
        downsample:  Literal["stride","pool","avgpool"] = "stride",             # down-sampling
        activation: Literal["silu", "relu", "gelu", "tanh"] = "silu",           # activation
        time_dim: int = 256,                                                    # time embedding output dim
        time_hidden: int = 512,                                                 # time MLP hidden
        dropout: float = 0.0,                                                   # dropout inside resblocks
    ):
        super().__init__()
        C, _, _ = img_shape
        self.in_channels = C
        self.out_channels = C
        self.activation = activation
        self.norm = norm
        self.groups = groups
        self.num_res_blocks_in_bottleneck = num_res_blocks_in_bottleneck

        # normalize configs
        S = len(channel_mults)
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * S
        assert len(num_res_blocks) == S
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * (2*S + 2)

        # time embedding: sinusoidal -> MLP to time_dim
        self.time_base_dim = 128  # input dim for sinusoidal_time_embed
        self.time_embed = TimeEmbed(time_dim_in=self.time_base_dim,
                                    time_dim_out=time_dim,
                                    hidden=time_hidden)

        # input
        self.in_conv = conv3x3(C, base_channels * channel_mults[0], k=3, stride=1)

        # encoder
        enc_blocks = nn.ModuleList()
        enc_channels = []
        ch = base_channels * channel_mults[0]
        for s, mult in enumerate(channel_mults):
            stage_ch = base_channels * mult
            # within-stage: keep channels constant (convert if needed)
            if ch != stage_ch:
                enc_blocks.append(ResBlock(ch, stage_ch, norm, activation, groups, kernel_size=3,
                                           time_dim=time_dim, dropout=dropout))
                ch = stage_ch
            # residual stack
            for _ in range(num_res_blocks[s]):
                enc_blocks.append(ResBlock(ch, ch, norm, activation, groups,
                                           kernel_size=3, time_dim=time_dim, dropout=dropout))
            enc_channels.append(ch)  # save for skip
            # downsample except last stage
            if s < S - 1:
                enc_blocks.append(Downsample(ch, method=downsample))
        self.encoder = enc_blocks

        # bottleneck
        self.bottleneck = nn.ModuleList()
        for _ in range(num_res_blocks_in_bottleneck):
            self.bottleneck.append(ResBlock(ch, ch, norm, activation, groups, kernel_size=3,
                             time_dim=time_dim, dropout=dropout))

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
                dec_blocks.append(ResBlock(ch, ch, norm, activation, groups,
                                           kernel_size=3, time_dim=time_dim, dropout=dropout))
        self.decoder = dec_blocks

        # output head
        self.out_norm = get_norm2d(norm, ch, groups)
        self.out_act = get_activation(activation)
        self.out_conv = conv3x3(ch, self.out_channels, k=3, stride=1)

    
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
            if isinstance(block, ResBlock):
                h = block(h, t_emb)
                i += 1
                # keep consuming consecutive resblocks until a downsample or next stage
                while i < len(self.encoder) and isinstance(self.encoder[i], ResBlock):
                    h = self.encoder[i](h, t_emb)
                    i += 1
                # end of stage; save skip before optional Downsample
                skips.append(h)
                if i < len(self.encoder) and isinstance(self.encoder[i], Downsample):
                    h = self.encoder[i](h)
                    i += 1
            else:
                h = block(h)
                i += 1

        # bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb)

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
            while j < len(self.decoder) and isinstance(self.decoder[j], ResBlock):
                h = self.decoder[j](h, t_emb)
                j += 1

        # head
        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)
        return out

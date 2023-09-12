import torch
from torch.nn import functional as F
from torch import nn
from typing import Union, List, Tuple, Optional
from .modules import TimeEmbedding, AttentionBlock, ResNetBlock, Upsample, Downsample


class UNet(nn.Module):
    def __init__(
        self,
        channels:int,
        out_channels:int,
        ch_mult:Union[List[int], Tuple[int]],
        num_res_blocks:int,
        att_levels:Union[List[int], Tuple[int]],
        num_groups:int,
        resample_with_conv:Optional[bool]=True,
        p:Optional[float]=.0
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.att_levels = att_levels
        self.ch_mult = ch_mult
        self.resample_with_conv = resample_with_conv
        self.p = p
        self.embedding = TimeEmbedding()
        self.embed_proj1 = nn.Linear(channels, channels*4)
        self.embed_proj2 = nn.Linear(channels*4, channels*4)
        self.conv_in = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.conv_out = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=3, padding=1)

        # Downsampling layers
        self.downsample_layers = nn.ModuleList()
        num_resolutions = len(ch_mult)
        for i_level in range(num_resolutions):
            for i_block in range(num_res_blocks):
                mult = None
                if i_level == 0:
                    if i_block == 0:
                        mult = 1
                    else:
                        mult = ch_mult[0]
                else:
                    if i_block == 0:
                        mult = ch_mult[i_level-1]
                    else:
                        mult = ch_mult[i_level]

                #print(f"in_channels = {mult*channels}")
                if i_level in att_levels:
                    self.downsample_layers.append(
                        nn.Sequential(
                            ResNetBlock(
                                in_channels=channels*mult,
                                out_channels=channels*ch_mult[i_level],
                                embed_dim=channels*4,
                                num_groups=num_groups,
                                p=p
                            ),
                            AttentionBlock(
                                in_dim=channels*ch_mult[i_level],
                                att_dim=channels*ch_mult[i_level],
                                num_groups=num_groups
                            )
                        )
                    )
                else:
                    self.downsample_layers.append(
                        ResNetBlock(
                            in_channels=channels*mult,
                            out_channels=channels*ch_mult[i_level],
                            embed_dim=channels*4,
                            num_groups=num_groups,
                            p=p
                        )
                    )
            if i_level != num_resolutions-1:
                self.downsample_layers.append(
                    Downsample(
                        in_channels=channels*ch_mult[i_level],
                        out_channels=channels*ch_mult[i_level]
                    )
                )

        # Middle Layers
        self.middle_resblock1 = ResNetBlock(
            in_channels=channels*ch_mult[-1],
            out_channels=channels*ch_mult[-1],
            embed_dim=channels*4,
            num_groups=num_groups,
            p=p
        )
        self.middle_resblock2 = ResNetBlock(
            in_channels=channels*ch_mult[-1],
            out_channels=channels*ch_mult[-1],
            embed_dim=channels*4,
            num_groups=num_groups,
            p=p
        )
        self.middle_attblock = AttentionBlock(
            in_dim=channels*ch_mult[-1],
            att_dim=channels*ch_mult[-1],
            num_groups=num_groups
        )

        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                in_channels = None
                if i_level == num_resolutions-1:
                    if i_block == self.num_res_blocks:
                        in_channels = channels*ch_mult[i_level-1]
                    else:
                        in_channels = channels*ch_mult[i_level]

                    in_channels += channels*ch_mult[i_level]
                else:
                    if i_block == 0:
                        in_channels = channels*ch_mult[i_level+1]
                    if i_block == self.num_res_blocks:
                        if i_level == 0:
                            in_channels = channels
                        else:
                            in_channels = channels*ch_mult[i_level-1]
                    if 0 < i_block < self.num_res_blocks:
                        in_channels = channels*ch_mult[i_level]

                    in_channels += channels*ch_mult[i_level]

                if i_level in att_levels:
                    self.upsample_layers.append(
                        nn.Sequential(
                            ResNetBlock(
                                in_channels=in_channels,
                                out_channels=channels*ch_mult[i_level],
                                embed_dim=channels*4,
                                num_groups=num_groups,
                                p=p
                            ),
                            AttentionBlock(
                                in_dim=in_channels,
                                att_dim=in_channels,
                                num_groups=num_groups
                            )
                        )
                    )
                else:
                    self.upsample_layers.append(
                        ResNetBlock(
                            in_channels=in_channels,
                            out_channels=channels*ch_mult[i_level],
                            embed_dim=channels*4,
                            num_groups=num_groups,
                            p=p
                        )
                    )

            if i_level != 0:
                self.upsample_layers.append(
                    Upsample(
                        in_channels=channels*ch_mult[i_level],
                        out_channels=channels*ch_mult[i_level]
                    )
                )

    def swish(self, x:torch.Tensor) -> torch.Tensor:
        return x*F.sigmoid(x)

    def forward(self, x:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        # Timestep embedding
        embed = self.embedding(t, self.channels)
        embed = self.embed_proj1(embed)
        embed = self.embed_proj2(self.swish(embed))
        assert embed.shape == (B, self.channels*4)

        # Downsampling
        num_resolutions = len(self.ch_mult)
        h = self.conv_in(x)
        hs = [h]
        for layer in self.downsample_layers:
            h = layer(hs[-1], embed)
            hs.append(h)

        # Middle
        h = self.middle_resblock1(hs[-1], embed)
        h = self.middle_attblock(h)
        h = self.middle_resblock2(h, embed)

        # Upsampling
        for i, layer in enumerate(self.upsample_layers):
            if (i+1) % (self.num_res_blocks+2) != 0:
                # ResNetBlock layer
                h = layer(torch.cat((h, hs.pop()), dim=1), embed)
            else:
                # Upsample layer
                #print(type(layer))
                h = layer(h)

        assert not hs

        # End
        h = self.swish(self.group_norm(h))
        h = self.conv_out(h)
        expected_shape = (x.shape[0], self.out_channels) + tuple(x.shape[2:])
        assert tuple(h.shape) == expected_shape, f"Error: expected h to have shape {expected_shape} but got {tuple(h.shape)}"
        return h
import torch
import torchvision
import math
from torch.nn import functional as F
from torchvision.transforms.functional import resize
from torch import nn
from typing import Optional


class Upsample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor, embed=None) -> torch.Tensor:
        B, C, H, W = x.shape
        x = resize(x, (H*2, W*2), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
        x = self.conv(x)
        assert x.shape == (B, C, H*2, W*2), f"Shape error: after upsampling of shape {(B,C,H,W)} we except shape {(B,C,2*H,2*W)} but got {x.shape}"
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels:int, out_channels:int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x:torch.Tensor, embed=None) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.conv(x)
        assert x.shape == (B, C, H//2, W//2), f"Shape error: after downsampling of shape {(B,C,H,W)} we except shape {(B,C,H//2,W//2)} but got {x.shape}"
        return x

class TimeEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, timesteps:torch.Tensor, embed_dim:int) -> torch.Tensor:
        assert len(timesteps.shape) == 1
        half_dim = embed_dim // 2
        pos = torch.exp(torch.arange(half_dim, device=timesteps.device) * -math.log(10000) / (half_dim-1))
        amp = timesteps[:,None]*pos[None,:]
        embeddings = torch.cat((torch.sin(amp), torch.cos(amp)), dim=-1)
        if embed_dim % 2 == 1:
            embeddings = F.pad(embeddings, pad=(0, 1, 0, 0), mode='constant', value=0)

        assert embeddings.shape == (timesteps.shape[0], embed_dim)
        return embeddings

class Nin(nn.Linear):
    def __init__(self, in_features:int, out_features:int, bias:Optional[bool]=True, **kwargs):
        super().__init__(in_features, out_features, bias, **kwargs)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.einsum('bijc,kc->bijk', x.transpose(1,3), self.weight)
        if self.bias is not None:
            x += self.bias
        return x.transpose(1,3)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, embed_dim:int, num_groups:int, conv_shortcut:Optional[bool]=False, p:Optional[float]=.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p = p
        self.group_norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.group_norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        if conv_shortcut:
            self.shortcut = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        else:
            self.shortcut = Nin(in_features=in_channels, out_features=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.proj = nn.Linear(embed_dim, out_channels)
        self.dropout = nn.Dropout(p)

    def swish(self, x:torch.Tensor) -> torch.Tensor:
        return x*F.sigmoid(x)

    def forward(self, x:torch.Tensor, embed:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        #print(f"in_channels = {self.in_channels}, x_shape = {x.shape}")
        h = x
        h = self.swish(self.group_norm1(h))
        h = self.conv1(h)
        #print(f'After conv : {h.shape}')
        h += self.swish(self.proj(embed))[:,:,None,None]
        h = self.swish(self.group_norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if C != self.out_channels:
            x = self.shortcut(x)

        assert x.shape == h.shape, f"Error: x and h must have same shape. But got {x.shape} and {h.shape}"
        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, in_dim:int, att_dim:int, num_groups:int):
        super().__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.v_nin = Nin(in_features=in_dim, out_features=att_dim)
        self.q_nin = Nin(in_features=in_dim, out_features=att_dim)
        self.k_nin = Nin(in_features=in_dim, out_features=att_dim)
        self.proj_out_nin = Nin(in_features=in_dim, out_features=in_dim)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_dim)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.q_nin(h)
        k = self.k_nin(h)
        v = self.v_nin(h)

        w = torch.einsum('bchw,bcHW->bhwHW', q, k) * (C**(-0.5))
        w = w.view((B, H, W, H*W))
        w = F.softmax(w, dim=-1)
        w = w.view((B, H, W, H, W))

        h = torch.einsum('bhwHW,bcHW->bchw', w, v)
        h = self.proj_out_nin(h)
        assert h.shape == x.shape
        return x + h
"""PyNET-lite, Mobile U-Net ISP and compact hybrid residual refiner."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .blocks import ConvBlock,DSConv

def count_parameters(model): return sum(p.numel() for p in model.parameters())

class MobileUNetISP(nn.Module):
    """Pure learned packed-Bayer to full-resolution RGB baseline."""
    def __init__(self,base=16):
        super().__init__()
        self.e1=ConvBlock(4,base); self.e2=ConvBlock(base,base*2); self.e3=ConvBlock(base*2,base*3)
        self.down=nn.AvgPool2d(2); self.mid=ConvBlock(base*3,base*4)
        self.d3=ConvBlock(base*7,base*3); self.d2=ConvBlock(base*5,base*2); self.d1=ConvBlock(base*3,base)
        self.head=nn.Sequential(nn.Conv2d(base,12,3,padding=1),nn.PixelShuffle(2),nn.Sigmoid())
    def forward(self,packed):
        a=self.e1(packed); b=self.e2(self.down(a)); c=self.e3(self.down(b)); z=self.mid(self.down(c))
        up=lambda x,size:F.interpolate(x,size=size,mode="bilinear",align_corners=False)
        y=self.d3(torch.cat([up(z,c.shape[-2:]),c],1)); y=self.d2(torch.cat([up(y,b.shape[-2:]),b],1))
        y=self.d1(torch.cat([up(y,a.shape[-2:]),a],1))
        return self.head(y)

class PyNETLite(nn.Module):
    """Three-scale pyramid refiner for packed RAW, with progressive RGB heads."""
    def __init__(self,base=12):
        super().__init__(); self.coarse=nn.Sequential(DSConv(4,base),DSConv(base,base*2,stride=2),DSConv(base*2,base*2))
        self.mid=nn.Sequential(DSConv(base*2+4,base*2),DSConv(base*2,base))
        self.fine=nn.Sequential(DSConv(base+4,base),DSConv(base,base))
        self.head=nn.Sequential(nn.Conv2d(base,12,3,padding=1),nn.PixelShuffle(2),nn.Sigmoid())
    def forward(self,packed):
        half=F.avg_pool2d(packed,2,ceil_mode=True)
        coarse=self.coarse(half)
        mid=F.interpolate(coarse,size=half.shape[-2:],mode="bilinear",align_corners=False)
        mid=self.mid(torch.cat([mid,half],1))
        fine=self.fine(torch.cat([F.interpolate(mid,size=packed.shape[-2:],mode="bilinear",align_corners=False),packed],1))
        return self.head(fine)

class HybridResidualRefiner(nn.Module):
    """8-channel raw-conditioned refiner: linear RGB + packed RAW + noise map."""
    def __init__(self,base=16,residual_scale=.25):
        super().__init__(); self.residual_scale=float(residual_scale)
        self.e1=ConvBlock(8,base); self.e2=ConvBlock(base,base*2); self.e3=ConvBlock(base*2,base*3)
        self.down=nn.AvgPool2d(2); self.mid=ConvBlock(base*3,base*4)
        self.d3=ConvBlock(base*7,base*3); self.d2=ConvBlock(base*5,base*2); self.d1=ConvBlock(base*3,base)
        self.head=nn.Conv2d(base,3,3,padding=1)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
    def forward(self,classical_linear_rgb,packed_raw,noise_map):
        h,w=classical_linear_rgb.shape[-2:]
        raw=F.interpolate(packed_raw,size=(h,w),mode="bilinear",align_corners=False)
        if noise_map.ndim==1: noise_map=noise_map[:,None,None,None]
        if noise_map.ndim==3: noise_map=noise_map[:,None]
        noise_map=F.interpolate(noise_map,size=(h,w),mode="bilinear",align_corners=False) if noise_map.shape[-2:]!=(h,w) else noise_map
        features=torch.cat([classical_linear_rgb,raw,noise_map],dim=1)
        a=self.e1(features); b=self.e2(self.down(a)); c=self.e3(self.down(b)); z=self.mid(self.down(c))
        up=lambda x,size:F.interpolate(x,size=size,mode="bilinear",align_corners=False)
        y=self.d3(torch.cat([up(z,c.shape[-2:]),c],1)); y=self.d2(torch.cat([up(y,b.shape[-2:]),b],1)); y=self.d1(torch.cat([up(y,a.shape[-2:]),a],1))
        return torch.clamp(classical_linear_rgb+self.residual_scale*torch.tanh(self.head(y)),0,1)

MODEL_REGISTRY={"mobile_unet":MobileUNetISP,"pynet_lite":PyNETLite,"hybrid_refiner":HybridResidualRefiner}

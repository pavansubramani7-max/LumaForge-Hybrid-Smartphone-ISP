"""Luma-only unsharp and edge-aware sharpening."""
from __future__ import annotations
import numpy as np
from .colorspace import rgb_to_ycbcr, ycbcr_to_rgb
from .denoise import _box_mean

def unsharp_mask(rgb, amount=.25, radius=1.2, edge_aware=True):
    x=np.asarray(rgb,np.float32); ycc=rgb_to_ycbcr(x); y=ycc[...,0]
    r=max(1,int(round(radius*2))); blur=_box_mean(y,r)
    detail=y-blur
    if edge_aware:
        edge=np.clip(np.abs(detail)*8,0,1)
        detail*=np.clip(1-edge*.65,0,1)
    ycc[...,0]=np.clip(y+float(amount)*detail,0,1)
    return ycbcr_to_rgb(ycc)

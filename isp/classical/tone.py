"""Gamma, sRGB transfer, filmic/Reinhard/ACES tone mapping and local contrast."""
from __future__ import annotations
import numpy as np
from .colorspace import linear_to_srgb

def tone_map(rgb, method="aces", exposure=1.0, gamma=2.2):
    x=np.maximum(np.asarray(rgb,np.float32)*float(exposure),0)
    if method=="reinhard": x=x/(1+x)
    elif method=="filmic":
        x=np.maximum(x-.004,0); x=(x*(6.2*x+.5))/(x*(6.2*x+1.7)+.06)
    elif method=="aces": x=(x*(2.51*x+.03))/(x*(2.43*x+.59)+.14)
    elif method=="gamma": x=np.clip(x,0,1)**(1/float(gamma))
    elif method!="srgb": raise ValueError(f"Unknown tone curve: {method}")
    return linear_to_srgb(np.clip(x,0,1))

def local_contrast(rgb, clip_limit=2.0, tile_grid=(8,8)):
    """CLAHE on luma when OpenCV exists, otherwise bounded local-contrast normalization."""
    x=np.asarray(rgb,np.float32)
    try:
        import cv2
        ycc=cv2.cvtColor(np.uint8(np.clip(x,0,1)*255),cv2.COLOR_RGB2YCrCb)
        ycc[...,0]=cv2.createCLAHE(clipLimit=clip_limit,tileGridSize=tile_grid).apply(ycc[...,0])
        return cv2.cvtColor(ycc,cv2.COLOR_YCrCb2RGB).astype(np.float32)/255
    except ImportError:
        lum=x@np.array([.2126,.7152,.0722],np.float32)
        from .denoise import _box_mean
        mean=_box_mean(lum,8); var=np.maximum(_box_mean(lum*lum,8)-mean*mean,1e-4)
        adjusted=np.clip((lum-mean)/np.sqrt(var)*.08+lum,0,1)
        ratio=adjusted/np.maximum(lum,1e-4)
        return np.clip(x*ratio[...,None],0,1)

"""Gray-world, shades-of-gray, white-patch, max-RGB and manual WB."""
from __future__ import annotations
import numpy as np

def estimate_gains(rgb, method="gray_world", power=6, percentile=99.5):
    x=np.maximum(np.asarray(rgb,np.float32),1e-6)
    if method=="manual": return np.ones(3,np.float32)
    if method=="gray_world": stats=x.mean(axis=(0,1))
    elif method=="shades_of_gray": stats=np.mean(x**power,axis=(0,1))**(1/power)
    elif method in {"white_patch","max_rgb"}: stats=np.percentile(x,percentile,axis=(0,1))
    else: raise ValueError(f"Unknown WB method: {method}")
    target=float(np.exp(np.mean(np.log(np.maximum(stats,1e-6)))))
    return np.clip(target/np.maximum(stats,1e-6),.4,3.5).astype(np.float32)

def apply_white_balance(rgb, gains):
    return np.maximum(np.asarray(rgb,np.float32)*np.asarray(gains,np.float32).reshape(1,1,3),0)

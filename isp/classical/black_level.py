"""Black/white level normalization and hot/dead-pixel correction."""
from __future__ import annotations
import numpy as np

def normalize(raw, black_level=0, white_level=None):
    x=np.asarray(raw); y=x.astype(np.float32)
    if white_level is None: white_level=float(np.iinfo(x.dtype).max) if np.issubdtype(x.dtype,np.integer) else 1.
    if np.issubdtype(x.dtype,np.floating) and np.nanmax(y)<=1.5 and white_level>1: white_level=1.
    return np.clip((y-black_level)/max(float(white_level)-float(black_level),1e-8),0,1)

def correct_dead_pixels(mosaic, threshold=0.2):
    """Replace isolated extreme CFA outliers with the same-colour 3x3 median."""
    x=np.asarray(mosaic,np.float32); h,w=x.shape
    out=x.copy()
    for py in (0,1):
        for px in (0,1):
            plane=x[py::2,px::2]
            if min(plane.shape)<3: continue
            padded=np.pad(plane,1,mode="reflect")
            neighbors=np.stack([padded[dy:dy+plane.shape[0],dx:dx+plane.shape[1]]
                                for dy in range(3) for dx in range(3) if (dy,dx)!=(1,1)],axis=0)
            median=np.median(neighbors,axis=0)
            out_plane=out[py::2,px::2]
            out_plane[np.abs(plane-median)>threshold]=median[np.abs(plane-median)>threshold]
    return out

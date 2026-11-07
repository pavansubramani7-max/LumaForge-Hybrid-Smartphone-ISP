"""Pattern-aware bilinear and gradient-directed Bayer demosaicing."""
from __future__ import annotations
import numpy as np
from isp.raw_utils import BAYER_PATTERNS

def _box_interp(values, mask, radius=1):
    h,w=values.shape; pad=np.pad(values*mask,radius,mode="reflect"); mp=np.pad(mask.astype(np.float32),radius,mode="reflect")
    num=np.zeros((h,w),np.float32); den=np.zeros((h,w),np.float32)
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            num+=pad[radius+dy:radius+dy+h,radius+dx:radius+dx+w]
            den+=mp[radius+dy:radius+dy+h,radius+dx:radius+dx+w]
    return num/np.maximum(den,1e-8)

def bilinear_demosaic(mosaic, pattern="RGGB"):
    x=np.asarray(mosaic,np.float32)
    if x.ndim!=2: raise ValueError("Bayer input must be a 2D mosaic")
    if pattern not in BAYER_PATTERNS: raise ValueError(f"Unsupported CFA pattern: {pattern}")
    grid=BAYER_PATTERNS[pattern]; h,w=x.shape; channels=[]
    for color in "RGB":
        mask=np.zeros((h,w),bool)
        for py in range(2):
            for px in range(2):
                if grid[py][px]==color: mask[py::2,px::2]=True
        interp=_box_interp(x,mask,1); interp[mask]=x[mask]; channels.append(interp)
    return np.stack(channels,axis=-1)

def edge_directed_demosaic(mosaic, pattern="RGGB"):
    """Gradient-correct green interpolation; chroma uses same-colour interpolation."""
    x=np.asarray(mosaic,np.float32); base=bilinear_demosaic(x,pattern)
    grid=BAYER_PATTERNS[pattern]; out=base.copy(); h,w=x.shape
    # On R/B sites, choose the lower-gradient axis for green reconstruction.
    p=np.pad(x,1,mode="reflect")
    horizontal=(p[1:-1,0:-2]+p[1:-1,2:])/2
    vertical=(p[0:-2,1:-1]+p[2:,1:-1])/2
    gh=np.abs(p[1:-1,0:-2]-p[1:-1,2:]); gv=np.abs(p[0:-2,1:-1]-p[2:,1:-1])
    green=(gh<=gv)*horizontal+(gv<gh)*vertical
    for py in range(2):
        for px in range(2):
            if grid[py][px] in "RB": out[py::2,px::2,1]=green[py::2,px::2]
    return out

def malvar_he_cutler_demosaic(mosaic, pattern="RGGB"):
    """Malvar-He-Cutler 5x5 Bayer interpolation with reflected image borders."""
    x=np.asarray(mosaic,np.float32)
    if x.ndim!=2: raise ValueError("Bayer input must be a 2D mosaic")
    if pattern not in BAYER_PATTERNS: raise ValueError(f"Unsupported CFA pattern: {pattern}")
    if min(x.shape)<5: return bilinear_demosaic(x,pattern)
    grid=BAYER_PATTERNS[pattern]; h,w=x.shape
    kernels={
        "green":np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]],np.float32)/8,
        "color_h":np.array([[0,0,.5,0,0],[0,-1,0,-1,0],[-1,4,5,4,-1],[0,-1,0,-1,0],[0,0,.5,0,0]],np.float32)/8,
        "color_cross":np.array([[0,0,-1.5,0,0],[0,2,0,2,0],[-1.5,0,6,0,-1.5],[0,2,0,2,0],[0,0,-1.5,0,0]],np.float32)/8,
    }
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(x,2,mode="reflect"),(5,5))
    estimates={name:np.sum(windows*kernel,axis=(-1,-2)) for name,kernel in kernels.items()}
    estimates["color_v"]=np.sum(windows*kernels["color_h"].T,axis=(-1,-2))
    out=np.zeros((h,w,3),np.float32)
    for py in range(2):
        for px in range(2):
            measured=grid[py][px]; yy=slice(py,None,2); xx=slice(px,None,2)
            out[yy,xx,"RGB".index(measured)]=x[yy,xx]
            if measured in "RB":
                out[yy,xx,1]=estimates["green"][yy,xx]
                other="B" if measured=="R" else "R"
                out[yy,xx,"RGB".index(other)]=estimates["color_cross"][yy,xx]
            else:
                horizontal=grid[py][1-px]=="R"
                red_est=estimates["color_h" if horizontal else "color_v"][yy,xx]
                blue_est=estimates["color_v" if horizontal else "color_h"][yy,xx]
                out[yy,xx,0]=red_est; out[yy,xx,2]=blue_est
    return np.clip(out,0,1.5)

DEMOSAICERS={"bilinear":bilinear_demosaic,"malvar_he_cutler":malvar_he_cutler_demosaic,"edge_directed":edge_directed_demosaic}

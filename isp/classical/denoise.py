"""Bilateral, non-local means, guided and Haar-wavelet denoising."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter

def _box_mean(x, radius=2):
    x=np.asarray(x,np.float32); h,w=x.shape[:2]; r=radius; size=2*r+1
    pad=np.pad(x,((r,r),(r,r))+(((0,0),) if x.ndim==3 else ()),mode="reflect")
    out=np.zeros_like(x,np.float32)
    for dy in range(size):
        for dx in range(size): out+=pad[dy:dy+h,dx:dx+w]
    return out/(size*size)

def gaussian_denoise(rgb, sigma=.5):
    x=np.asarray(rgb,np.float32)
    image=Image.fromarray(np.uint8(np.clip(x,0,1)*255))
    return np.asarray(image.filter(ImageFilter.GaussianBlur(float(sigma))),np.float32)/255

def bilateral_denoise(rgb, sigma_color=.08, radius=2):
    x=np.asarray(rgb,np.float32); h,w=x.shape[:2]; r=radius
    padded=np.pad(x,((r,r),(r,r),(0,0)),mode="reflect")
    accum=np.zeros_like(x); norm=np.zeros((h,w,1),np.float32); spatial=2*(r/1.5)**2+1e-8
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            neighbor=padded[r+dy:r+dy+h,r+dx:r+dx+w]
            ds=np.exp(-(dx*dx+dy*dy)/spatial)
            dc=np.exp(-np.sum((neighbor-x)**2,axis=-1,keepdims=True)/(2*sigma_color*sigma_color+1e-8))
            weight=ds*dc; accum+=neighbor*weight; norm+=weight
    return accum/np.maximum(norm,1e-8)

def non_local_means(rgb, h_strength=.08, patch_radius=1, search_radius=3):
    """Compact NLM approximation using patch SSD and shifted neighborhoods."""
    x=np.asarray(rgb,np.float32); height,width=x.shape[:2]; r=search_radius
    padded=np.pad(x,((r+patch_radius,r+patch_radius),(r+patch_radius,r+patch_radius),(0,0)),mode="reflect")
    center=padded[r+patch_radius:r+patch_radius+height,r+patch_radius:r+patch_radius+width]
    accum=center.copy(); weights=np.ones((height,width,1),np.float32)
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            if dy==0 and dx==0: continue
            neighbor=padded[r+patch_radius+dy:r+patch_radius+dy+height,r+patch_radius+dx:r+patch_radius+dx+width]
            distance=np.mean((center-neighbor)**2,axis=-1,keepdims=True)
            weight=np.exp(-np.maximum(distance-2*h_strength*h_strength,0)/(h_strength*h_strength+1e-8))
            accum+=neighbor*weight; weights+=weight
    return accum/weights

def guided_filter(rgb, radius=4, eps=.01):
    x=np.asarray(rgb,np.float32); gray=x@np.array([.2126,.7152,.0722],np.float32)
    mean_i=_box_mean(gray,radius); mean_p=_box_mean(x,radius)
    cov=np.stack([_box_mean(gray*x[...,c],radius)-mean_i*mean_p[...,c] for c in range(3)],axis=-1)
    var=_box_mean(gray*gray,radius)-mean_i*mean_i
    a=cov/(var[...,None]+eps); b=mean_p-a*mean_i[...,None]
    return np.clip(_box_mean(a,radius)*gray[...,None]+_box_mean(b,radius),0,1)

def wavelet_shrink(rgb, threshold=.025):
    """One-level orthonormal Haar detail shrinkage, with odd-edge padding."""
    x=np.asarray(rgb,np.float32); h,w=x.shape[:2]; ph=h%2; pw=w%2
    p=np.pad(x,((0,ph),(0,pw),(0,0)),mode="reflect")
    a=p[0::2,0::2]; b=p[0::2,1::2]; c=p[1::2,0::2]; d=p[1::2,1::2]
    ll=(a+b+c+d)*.5; lh=(a-b+c-d)*.5; hl=(a+b-c-d)*.5; hh=(a-b-c+d)*.5
    shrink=lambda band: np.sign(band)*np.maximum(np.abs(band)-threshold,0)
    lh,hl,hh=map(shrink,(lh,hl,hh))
    out=np.empty_like(p); out[0::2,0::2]=(ll+lh+hl+hh)*.5; out[0::2,1::2]=(ll-lh+hl-hh)*.5
    out[1::2,0::2]=(ll+lh-hl-hh)*.5; out[1::2,1::2]=(ll-lh-hl+hh)*.5
    return np.clip(out[:h,:w],0,1)

def denoise(rgb, method="bilateral", strength=.08):
    methods={"bilateral":lambda:bilateral_denoise(rgb,strength),"nlm":lambda:non_local_means(rgb,strength),
             "guided":lambda:guided_filter(rgb,eps=max(strength**2,1e-4)),"wavelet":lambda:wavelet_shrink(rgb,strength),
             "gaussian":lambda:gaussian_denoise(rgb)}
    if method not in methods: raise ValueError(f"Unknown denoiser: {method}")
    return methods[method]()

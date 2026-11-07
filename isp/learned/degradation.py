"""Camera-inspired Poisson-Gaussian, illumination, WB and vignetting degradation."""
from __future__ import annotations
import numpy as np

def degrade_packed(packed,sigma=.02,exposure_range=(.5,1.5),wb_error=.12,vignetting=.25,seed=None,cct_kelvin=5500):
    """Degrade HWC packed RGGB sensor planes while preserving the 4-plane layout."""
    rng=np.random.default_rng(seed); x=np.clip(np.asarray(packed,np.float32),0,1).copy()
    exposure=float(rng.uniform(*exposure_range)); x*=exposure
    # R, G1, G2, B gains approximate CCT / sensor WB errors.
    cct=float(np.clip(cct_kelvin,2500,9000)); warm_shift=(5500-cct)/6500
    red=float(rng.uniform(1-wb_error,1+wb_error))*(1+.35*warm_shift)
    blue=float(rng.uniform(1-wb_error,1+wb_error))*(1-.35*warm_shift)
    x[...,0]*=red; x[...,1:3]*=rng.uniform(.97,1.03); x[...,3]*=blue
    h,w=x.shape[:2]; yy,xx=np.mgrid[-1:1:complex(h),-1:1:complex(w)]
    mask=np.clip(1-vignetting*(xx*xx+yy*yy),.55,1); x*=mask[...,None]
    photons=max(50.,1/max(float(sigma)**2,1e-6)); shot=rng.poisson(np.clip(x,0,1)*photons)/photons
    read=float(rng.uniform(.15,1.0))*sigma
    return np.clip(shot+rng.normal(0,read,x.shape),0,1).astype(np.float32)

def degrade_rgb(rgb,noise_sigma=.02,blur_sigma=0,jpeg_quality=None,seed=None):
    """Synthetic display-RGB degradation for robustness experiments."""
    rng=np.random.default_rng(seed); x=np.asarray(rgb,np.float32).copy()
    x*=float(rng.uniform(.35,1.4)); x*=rng.uniform(.85,1.15,size=(1,1,3))
    x=np.clip(x+rng.normal(0,noise_sigma,x.shape),0,1)
    if blur_sigma>0:
        from PIL import Image,ImageFilter
        x=np.asarray(Image.fromarray(np.uint8(x*255)).filter(ImageFilter.GaussianBlur(blur_sigma)),np.float32)/255
    if jpeg_quality is not None:
        from io import BytesIO
        from PIL import Image
        buf=BytesIO(); Image.fromarray(np.uint8(x*255)).save(buf,format="JPEG",quality=int(jpeg_quality)); buf.seek(0)
        x=np.asarray(Image.open(buf).convert("RGB"),np.float32)/255
    return x

"""Per-frame noise, illumination and exposure estimator with an explicit policy."""
from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
from isp.raw_utils import normalize_raw
from isp.classical.demosaic import bilinear_demosaic

@dataclass
class ISPPolicy:
    noise_sigma: float
    brightness: float
    cct_kelvin: float
    wb_gains: tuple[float,float,float]
    denoise_strength: float
    refiner_alpha: float
    sharpen_amount: float
    ccm_preset: str
    tone_method: str
    def to_dict(self): return asdict(self)

def estimate_noise_mad(mosaic):
    x=np.asarray(mosaic,np.float32); vals=[]
    for py in (0,1):
        for px in (0,1):
            p=x[py::2,px::2]
            if min(p.shape)<3: continue
            q=np.pad(p,1,mode="reflect")
            lap=4*p-q[:-2,1:-1]-q[2:,1:-1]-q[1:-1,:-2]-q[1:-1,2:]
            vals.append(lap.ravel())
    if not vals: return 0.
    v=np.concatenate(vals); med=np.median(v)
    return float(np.clip(np.median(np.abs(v-med))/(.6745*np.sqrt(20)),0,.25))

def estimate_cct(rgb):
    x=np.maximum(np.asarray(rgb,np.float32),1e-6); means=x.mean(axis=(0,1)); r,g,b=means
    # McCamy approximation from a simple RGB-to-XYZ transform; report as coarse estimate.
    xyz=np.array([.4124*r+.3576*g+.1805*b,.2126*r+.7152*g+.0722*b,.0193*r+.1192*g+.9505*b])
    xx=xyz[0]/max(xyz.sum(),1e-8); yy=xyz[1]/max(xyz.sum(),1e-8)
    n=(xx-.3320)/max(.1858-yy,1e-6)
    cct=float(449*n**3+3525*n**2+6823.3*n+5520.33)
    return float(np.clip(cct,2500,9000))

def estimate_policy(raw,pattern="RGGB",black_level=0,white_level=None):
    x=normalize_raw(raw,black_level,white_level)
    rgb=bilinear_demosaic(x,pattern)
    sigma=estimate_noise_mad(x); brightness=float(np.mean(rgb))
    means=np.maximum(rgb.mean(axis=(0,1)),1e-4); neutral=float(np.exp(np.mean(np.log(means))))
    gains=np.clip(neutral/means,.6,2.5)
    cct=estimate_cct(rgb)
    alpha=float(np.clip(.9-1.4*sigma,.45,.9))
    denoise=float(np.clip(.5+6*sigma,.5,2.0))
    sharpen=float(np.clip(.28-1.2*sigma,.05,.3))
    preset="A" if cct<3800 else "F11" if cct<5200 else "D65"
    tone="filmic" if brightness<.2 else "aces"
    return ISPPolicy(sigma,brightness,cct,tuple(float(g) for g in gains),denoise,alpha,sharpen,preset,tone)

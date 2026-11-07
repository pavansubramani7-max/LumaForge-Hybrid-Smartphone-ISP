"""Configurable classical ISP with per-stage timing and intermediate images."""
from __future__ import annotations
from dataclasses import dataclass,field
import time
from typing import Any
import numpy as np
from isp.raw_utils import normalize_raw, guess_bayer_pattern
from .black_level import correct_dead_pixels
from .white_balance import estimate_gains, apply_white_balance
from .demosaic import DEMOSAICERS
from .ccm import apply_ccm
from .colorspace import rgb_to_ycbcr,ycbcr_to_rgb
from .denoise import denoise as denoise_image
from .tone import tone_map,local_contrast
from .sharpen import unsharp_mask

DEFAULT_CCM=np.array([[1.48,-.32,-.16],[-.12,1.25,-.13],[-.02,-.44,1.46]],np.float32)

@dataclass
class PipelineResult:
    image: np.ndarray
    linear_rgb: np.ndarray
    intermediates: dict[str,np.ndarray]=field(default_factory=dict)
    timings_ms: dict[str,float]=field(default_factory=dict)
    metadata: dict[str,Any]=field(default_factory=dict)

class ClassicalISP:
    def __init__(self,config:dict|None=None): self.config=config or {}
    def run(self,image,input_kind="raw",pattern="auto",black_level=0,white_level=None,
            wb_method="gray_world",wb_gains=None,demosaic_method="bilinear",ccm=None,
            denoise_method="bilateral",noise_sigma=.04,tone_method="aces",exposure=1.,
            sharpen_amount=.2,use_clahe=False,enabled=None):
        enabled=enabled or {}; timings={}; stages={}; metadata={}
        def timed(name,fn):
            start=time.perf_counter(); value=fn(); timings[name]=(time.perf_counter()-start)*1000; stages[name]=np.asarray(value).copy(); return value
        x=np.asarray(image)
        if input_kind.lower() in {"raw","bayer","mosaic"}:
            if x.ndim!=2: raise ValueError(f"Bayer RAW must be a 2D mosaic, got {x.shape}")
            if pattern=="auto":
                result=guess_bayer_pattern(x); pattern=result["pattern"] if result["pattern"]!="unknown" else "RGGB"; metadata["bayer_guess"]=result
            if enabled.get("black_level",True): x=timed("black_level",lambda:normalize_raw(x,black_level,white_level))
            if enabled.get("dead_pixel",True): x=timed("dead_pixel",lambda:correct_dead_pixels(x))
            demosaicer=DEMOSAICERS.get(demosaic_method)
            if demosaicer is None: raise ValueError(f"Unknown demosaic method {demosaic_method}")
            rgb=timed("demosaic",lambda:demosaicer(x,pattern))
            metadata["bayer_pattern"]=pattern
        else:
            rgb=x.astype(np.float32)
            if np.issubdtype(x.dtype,np.integer): rgb/=float(np.iinfo(x.dtype).max)
            if rgb.ndim==2: rgb=np.repeat(rgb[...,None],3,axis=-1)
            if rgb.ndim!=3 or rgb.shape[-1]<3: raise ValueError(f"RGB input must be HxWx3, got {rgb.shape}")
            rgb=rgb[...,:3]
            # Direct RGB inputs are display-referred; linearize before physical stages.
            from .colorspace import srgb_to_linear
            rgb=srgb_to_linear(rgb)
            metadata["demosaic_skipped"]=True
        stages.setdefault("input",np.asarray(image).copy())
        gains=np.asarray(wb_gains if wb_gains is not None else estimate_gains(rgb,wb_method),np.float32)
        if enabled.get("white_balance",True): rgb=timed("white_balance",lambda:apply_white_balance(rgb,gains))
        matrix=np.asarray(ccm if ccm is not None else self.config.get("ccm",DEFAULT_CCM),np.float32)
        if enabled.get("ccm",True): rgb=timed("ccm",lambda:apply_ccm(rgb,matrix))
        if enabled.get("denoise",True):
            def denoise_stage():
                ycc=rgb_to_ycbcr(np.clip(rgb,0,1)); strength=max(float(noise_sigma),.005)
                ycc[...,0]=denoise_image(ycc[...,:1].repeat(3,axis=-1),denoise_method,strength)[...,0]
                ycc[...,1:]=denoise_image(ycc[...,1:].repeat(3,axis=-1),denoise_method,strength*1.5)[...,:2]
                return ycbcr_to_rgb(ycc)
            rgb=timed("denoise",denoise_stage)
        linear=rgb.copy(); stages["linear_rgb"]=linear.copy()
        if enabled.get("tone",True): rgb=timed("tone",lambda:tone_map(rgb,tone_method,exposure))
        if use_clahe and enabled.get("clahe",True): rgb=timed("clahe",lambda:local_contrast(rgb))
        if enabled.get("sharpen",True): rgb=timed("sharpen",lambda:unsharp_mask(rgb,sharpen_amount))
        stages["output"]=np.clip(rgb,0,1).copy()
        return PipelineResult(stages["output"],linear,stages,timings,metadata)

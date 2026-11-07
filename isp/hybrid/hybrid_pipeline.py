"""Traditional, pure-learned and hybrid RAW-to-RGB processing modes."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
from isp.raw_utils import normalize_raw,pack_bayer
from isp.learned.infer import infer_hybrid_refiner,infer_pure_bayer
from isp.classical.pipeline import ClassicalISP
from isp.classical.ccm import interpolate_illuminant_ccm
from isp.classical.tone import tone_map
from isp.classical.sharpen import unsharp_mask
from .adaptive_controller import estimate_policy
from .fusion import confidence_weighted_fusion

@dataclass
class HybridResult:
    traditional: np.ndarray
    learned: np.ndarray|None
    hybrid: np.ndarray
    policy: dict
    timings_ms: dict
    confidence: np.ndarray|None=None

class HybridISPPipeline:
    def __init__(self,classical:ClassicalISP|None=None,learned_model=None,pure_model=None,device="cpu",config=None):
        self.classical=classical or ClassicalISP(); self.learned_model=learned_model; self.pure_model=pure_model
        self.device=torch.device(device); self.config=config or {}
        if "ccm_presets" not in self.config:
            import json
            preset_path=Path(__file__).resolve().parents[2]/"configs"/"ccm_presets.yaml"
            try:
                import yaml
                self.config["ccm_presets"]=yaml.safe_load(preset_path.read_text(encoding="utf-8"))["illuminants"]
            except (ImportError, OSError, KeyError, ValueError):
                self.config["ccm_presets"]={}

    def process_raw(self,raw,pattern="RGGB",black_level=0,white_level=None,adaptive=True,mode="hybrid"):
        raw=np.asarray(raw); policy=estimate_policy(raw,pattern,black_level,white_level) if adaptive else None
        policy_values=policy.to_dict() if policy else {}
        cfg=self.config; gains=policy.wb_gains if policy else cfg.get("wb_gains",(1.5,1.,1.8))
        sigma=policy.noise_sigma if policy else cfg.get("noise_sigma",.02)
        ccm=cfg.get("ccm")
        if ccm is None and policy and cfg.get("ccm_presets"):
            ccm=interpolate_illuminant_ccm(cfg["ccm_presets"],policy.cct_kelvin)
        enabled={**cfg.get("enabled_stages",{}),"tone":False,"sharpen":False}
        stages=self.classical.run(raw,"raw",pattern,black_level,white_level,wb_method="manual",wb_gains=gains,
            demosaic_method=cfg.get("demosaic_method","bilinear"),ccm=ccm,
            denoise_method=cfg.get("denoise_method","bilateral"),noise_sigma=sigma,
            tone_method=cfg.get("tone_method",policy.tone_method if policy else "aces"),
            exposure=cfg.get("exposure",1.),sharpen_amount=policy.sharpen_amount if policy else cfg.get("sharpen_amount",.2),
            enabled=enabled)
        base_linear=stages.linear_rgb
        tone_enabled=cfg.get("enabled_stages",{}).get("tone",True); sharp_enabled=cfg.get("enabled_stages",{}).get("sharpen",True)
        baseline_display=tone_map(base_linear,cfg.get("tone_method",policy.tone_method if policy else "aces"),cfg.get("exposure",1.)) if tone_enabled else np.clip(base_linear,0,1)
        traditional=unsharp_mask(baseline_display,policy.sharpen_amount if policy else cfg.get("sharpen_amount",.2)) if sharp_enabled else baseline_display
        timings={**stages.timings_ms}; timings["traditional_total_ms"]=sum(timings.values())
        if mode=="traditional": return HybridResult(traditional,None,traditional,policy_values,timings)
        packed=pack_bayer(normalize_raw(raw,black_level,white_level),pattern).astype(np.float32)
        device=self.device
        learned=None; conf=None; hybrid_linear=base_linear
        if mode=="learned" and self.pure_model is not None:
            model=self.pure_model.to(device).eval(); t=time.perf_counter()
            pred=infer_pure_bayer(model,packed,device,cfg.get("tile_size",256),cfg.get("tile_overlap",32))
            timings["learned_ms"]=(time.perf_counter()-t)*1000; learned=pred
            out=tone_map(learned,cfg.get("tone_method","aces"),cfg.get("exposure",1.))
            return HybridResult(traditional,out,out,policy_values,timings)
        if self.learned_model is not None:
            model=self.learned_model.to(device).eval(); t=time.perf_counter()
            pred=infer_hybrid_refiner(model,base_linear,packed,sigma,device,cfg.get("tile_size",512),cfg.get("tile_overlap",64))
            timings["refiner_ms"]=(time.perf_counter()-t)*1000; learned=pred
            alpha=policy.refiner_alpha if policy else cfg.get("fusion_alpha",.85)
            alpha=float(np.clip(alpha*cfg.get("fusion_alpha_scale",1.),0.,1.))
            hybrid_linear,conf=confidence_weighted_fusion(base_linear,learned,alpha,sigma)
        hybrid=tone_map(hybrid_linear,cfg.get("tone_method",policy.tone_method if policy else "aces"),cfg.get("exposure",1.)) if tone_enabled else np.clip(hybrid_linear,0,1)
        if sharp_enabled: hybrid=unsharp_mask(hybrid,policy.sharpen_amount if policy else cfg.get("sharpen_amount",.2))
        timings["hybrid_total_ms"]=sum(timings.values())
        return HybridResult(traditional,learned,hybrid,policy_values,timings,conf)

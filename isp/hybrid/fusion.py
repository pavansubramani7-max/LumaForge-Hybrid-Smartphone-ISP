"""Confidence-weighted fusion between classical and learned reconstructions."""
from __future__ import annotations
import numpy as np

def confidence_weighted_fusion(classical,learned,alpha=.85,noise_sigma=0):
    c=np.asarray(classical,np.float32); l=np.asarray(learned,np.float32)
    if c.shape!=l.shape: raise ValueError(f"Fusion shapes differ: {c.shape} vs {l.shape}")
    # Reduce learned influence in locally unstable/high-noise flat regions.
    high=np.abs(l-c).mean(axis=-1,keepdims=True)
    confidence=np.exp(-high/(.08+4*float(noise_sigma)))
    weight=np.clip(float(alpha)*confidence,0,1)
    return np.clip((1-weight)*c+weight*l,0,1),weight[...,0]

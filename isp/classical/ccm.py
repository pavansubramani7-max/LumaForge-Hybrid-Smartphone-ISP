"""Camera colour correction matrix application, interpolation and chart fitting."""
from __future__ import annotations
import numpy as np

def apply_ccm(rgb, matrix):
    matrix=np.asarray(matrix,np.float32)
    if matrix.shape!=(3,3): raise ValueError("CCM must be 3x3")
    return np.maximum(np.asarray(rgb,np.float32)@matrix.T,0)

def interpolate_ccm(presets, cct_kelvin):
    """Interpolate matrices by reciprocal temperature between ordered presets."""
    points=sorted((int(k),np.asarray(v,np.float32)) for k,v in presets.items())
    if not points: raise ValueError("At least one CCM preset is required")
    cct=float(np.clip(cct_kelvin,points[0][0],points[-1][0]))
    for (t0,m0),(t1,m1) in zip(points,points[1:]):
        if t0<=cct<=t1:
            q=(1/cct-1/t0)/(1/t1-1/t0)
            return (1-q)*m0+q*m1
    return points[0][1] if cct<=points[0][0] else points[-1][1]

def interpolate_illuminant_ccm(presets, cct_kelvin):
    """Interpolate named illuminant matrices using standard nominal CCT anchors."""
    anchors={"A":2856,"F11":4000,"D65":6504}
    numeric={}
    for name,matrix in presets.items():
        key=str(name).upper()
        temperature=anchors.get(key,int(key) if key.isdigit() else None)
        if temperature is not None: numeric[temperature]=matrix
    return interpolate_ccm(numeric,cct_kelvin)

def fit_ccm(camera_rgb, reference_rgb, regularization=1e-4):
    """Ridge least-squares 3x3 camera-to-reference matrix from chart patches."""
    x=np.asarray(camera_rgb,np.float64); y=np.asarray(reference_rgb,np.float64)
    if x.shape!=y.shape or x.ndim!=2 or x.shape[1]!=3 or len(x)<3:
        raise ValueError("CCM fitting expects matching Nx3 camera/reference patch arrays")
    mat=np.linalg.solve(x.T@x+regularization*np.eye(3),x.T@y)
    return mat.T.astype(np.float32)

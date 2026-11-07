"""Robust camera noise estimation using high-pass MAD and local variance."""
import numpy as np

def mad_sigma(image):
    x=np.asarray(image,np.float32)
    if x.ndim==3: x=x@np.array([.2126,.7152,.0722],np.float32)
    hp=4*x-np.roll(x,1,0)-np.roll(x,-1,0)-np.roll(x,1,1)-np.roll(x,-1,1)
    return float(np.median(np.abs(hp-np.median(hp)))/(.6745*np.sqrt(20)))

def patch_variance_sigma(image,patch=16):
    x=np.asarray(image,np.float32)
    if x.ndim==3: x=x@np.array([.2126,.7152,.0722],np.float32)
    h,w=x.shape; vals=[]
    for y in range(0,h-patch+1,patch):
        for xx in range(0,w-patch+1,patch):
            block=x[y:y+patch,xx:xx+patch]
            if np.std(block)<.05: vals.append(float(np.var(block)))
    return float(np.sqrt(np.median(vals))) if vals else mad_sigma(x)

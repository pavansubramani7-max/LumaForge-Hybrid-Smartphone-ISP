"""Small sRGB/XYZ/Lab/YCbCr/HSV conversion helpers."""
from __future__ import annotations
import numpy as np

def srgb_to_linear(rgb):
    x=np.clip(np.asarray(rgb,np.float32),0,1)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def linear_to_srgb(rgb):
    x=np.maximum(np.asarray(rgb,np.float32),0)
    return np.clip(np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055),0,1)

def srgb_to_xyz(rgb):
    x=srgb_to_linear(rgb)
    return x@np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]],np.float32).T

def xyz_to_lab(xyz):
    x=np.asarray(xyz,np.float32)/np.array([.95047,1,1.08883],np.float32)
    d=6/29; f=np.where(x>d**3,np.cbrt(x),x/(3*d*d)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)

def srgb_to_lab(rgb): return xyz_to_lab(srgb_to_xyz(rgb))

def rgb_to_ycbcr(rgb):
    x=np.asarray(rgb,np.float32); r,g,b=x[...,0],x[...,1],x[...,2]
    return np.stack([.299*r+.587*g+.114*b,-.168736*r-.331264*g+.5*b+.5,.5*r-.418688*g-.081312*b+.5],axis=-1)

def ycbcr_to_rgb(ycbcr):
    y,cb,cr=np.moveaxis(np.asarray(ycbcr,np.float32),-1,0); cb=cb-.5; cr=cr-.5
    return np.clip(np.stack([y+1.402*cr,y-.344136*cb-.714136*cr,y+1.772*cb],axis=-1),0,1)

def rgb_to_hsv(rgb):
    x=np.clip(np.asarray(rgb,np.float32),0,1); mx=x.max(-1); mn=x.min(-1); delta=mx-mn
    h=np.zeros_like(mx); nz=delta>1e-8
    r,g,b=x[...,0],x[...,1],x[...,2]
    mask=nz&(mx==r); h[mask]=((g-b)[mask]/delta[mask])%6
    mask=nz&(mx==g); h[mask]=(b-r)[mask]/delta[mask]+2
    mask=nz&(mx==b); h[mask]=(r-g)[mask]/delta[mask]+4
    h=h/6; s=np.where(mx>0,delta/mx,0)
    return np.stack([h,s,mx],axis=-1)

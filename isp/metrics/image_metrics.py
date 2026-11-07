"""PSNR, SSIM, MS-SSIM, Delta E76/CIEDE2000 and optional LPIPS."""
from __future__ import annotations
import numpy as np
from isp.classical.colorspace import srgb_to_lab
from src.isp.metrics import psnr,ssim

def delta_e_76(pred,target):
    return float(np.mean(np.linalg.norm(srgb_to_lab(pred)-srgb_to_lab(target),axis=-1)))

def delta_e_2000(pred,target):
    l1=srgb_to_lab(pred).astype(np.float64); l2=srgb_to_lab(target).astype(np.float64)
    L1,a1,b1=np.moveaxis(l1,-1,0); L2,a2,b2=np.moveaxis(l2,-1,0)
    C1=np.sqrt(a1*a1+b1*b1); C2=np.sqrt(a2*a2+b2*b2); Cbar=(C1+C2)/2
    G=.5*(1-np.sqrt(Cbar**7/(Cbar**7+25**7+1e-20)))
    ap1=(1+G)*a1; ap2=(1+G)*a2
    Cp1=np.sqrt(ap1*ap1+b1*b1); Cp2=np.sqrt(ap2*ap2+b2*b2)
    hp1=np.mod(np.degrees(np.arctan2(b1,ap1)),360); hp2=np.mod(np.degrees(np.arctan2(b2,ap2)),360)
    dL=L2-L1; dC=Cp2-Cp1; dh=hp2-hp1
    dh=np.where(Cp1*Cp2==0,0,np.where(dh>180,dh-360,np.where(dh< -180,dh+360,dh)))
    dH=2*np.sqrt(Cp1*Cp2)*np.sin(np.radians(dh/2))
    Lbar=(L1+L2)/2; Cbarp=(Cp1+Cp2)/2
    hdiff=np.abs(hp1-hp2); hsum=hp1+hp2
    hbar=np.where(Cp1*Cp2==0,hsum,np.where(hdiff<=180,hsum/2,np.where(hsum<360,(hsum+360)/2,(hsum-360)/2)))
    T=1-.17*np.cos(np.radians(hbar-30))+.24*np.cos(np.radians(2*hbar))+.32*np.cos(np.radians(3*hbar+6))-.20*np.cos(np.radians(4*hbar-63))
    Sl=1+.015*(Lbar-50)**2/np.sqrt(20+(Lbar-50)**2); Sc=1+.045*Cbarp; Sh=1+.015*Cbarp*T
    Rt=-2*np.sqrt(Cbarp**7/(Cbarp**7+25**7+1e-20))*np.sin(np.radians(60*np.exp(-((hbar-275)/25)**2)))
    value=np.sqrt((dL/Sl)**2+(dC/Sc)**2+(dH/Sh)**2+Rt*(dC/Sc)*(dH/Sh))
    return float(np.mean(value))

def ms_ssim(pred,target,levels=4):
    x=np.asarray(pred,np.float32); y=np.asarray(target,np.float32); scores=[]
    for _ in range(levels):
        scores.append(max(0.,ssim(x,y)))
        if min(x.shape[:2])<32: break
        h=x.shape[0]//2*2; w=x.shape[1]//2*2; x=x[:h,:w]; y=y[:h,:w]
        x=(x[::2,::2]+x[1::2,::2]+x[::2,1::2]+x[1::2,1::2])*.25
        y=(y[::2,::2]+y[1::2,::2]+y[::2,1::2]+y[1::2,1::2])*.25
    weights=np.array([.4,.3,.2,.1][:len(scores)]); weights/=weights.sum()
    return float(np.prod(np.maximum(scores,1e-8)**weights))

def lpips_alex(pred,target,model=None):
    try: import lpips
    except ImportError: return None
    import torch
    net=model or lpips.LPIPS(net="alex")
    x=torch.from_numpy(np.asarray(pred,np.float32).transpose(2,0,1)[None]*2-1)
    y=torch.from_numpy(np.asarray(target,np.float32).transpose(2,0,1)[None]*2-1)
    with torch.inference_mode(): return float(net(x,y).item())

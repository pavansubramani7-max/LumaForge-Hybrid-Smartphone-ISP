"""Memory-bounded overlapping-tile inference for pure and hybrid models."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

def _starts(length,tile,overlap):
    step=max(8,tile-overlap)
    return [0] if length<=tile else list(range(0,length,step))

def _weights(h,w,top,left,bottom,right,H,W,overlap):
    wy=np.ones(h,np.float32); wx=np.ones(w,np.float32)
    ry=min(overlap,h//2); rx=min(overlap,w//2)
    if top>0 and ry: wy[:ry]=np.linspace(.05,1,ry,dtype=np.float32)
    if bottom<H and ry: wy[-ry:]=np.linspace(1,.05,ry,dtype=np.float32)
    if left>0 and rx: wx[:rx]=np.linspace(.05,1,rx,dtype=np.float32)
    if right<W and rx: wx[-rx:]=np.linspace(1,.05,rx,dtype=np.float32)
    return (wy[:,None]*wx[None,:])[...,None]

def infer_pure_bayer(model,packed,device="cpu",tile_size=256,overlap=32):
    """Infer packed HWC4 Bayer to 2x full-resolution HWC RGB."""
    x=np.asarray(packed,np.float32); h,w=x.shape[:2]; tile=max(32,int(tile_size//8*8)); overlap=max(0,int(overlap//8*8))
    H,W=2*h,2*w; out=np.zeros((H,W,3),np.float32); weights=np.zeros((H,W,1),np.float32)
    param=next(model.parameters()); model=model.to(device).eval()
    with torch.inference_mode():
        for top in _starts(h,tile,overlap):
            for left in _starts(w,tile,overlap):
                bot=min(top+tile,h); right=min(left+tile,w); patch=x[top:bot,left:right]
                ph,pw=(-patch.shape[0])%8,(-patch.shape[1])%8
                tensor=torch.from_numpy(patch.transpose(2,0,1).copy()[None]).to(device=device,dtype=param.dtype)
                if ph or pw: tensor=F.pad(tensor,(0,pw,0,ph),mode="replicate")
                pred=model(tensor)[...,:patch.shape[0]*2,:patch.shape[1]*2].squeeze(0).permute(1,2,0).float().cpu().numpy()
                y0,y1=top*2,bot*2; x0,x1=left*2,right*2
                weight=_weights(y1-y0,x1-x0,y0,x0,y1,x1,H,W,overlap*2)
                out[y0:y1,x0:x1]+=pred*weight; weights[y0:y1,x0:x1]+=weight
    return out/np.maximum(weights,1e-8)

def infer_hybrid_refiner(model,base_hwc,packed_hwc,noise_sigma,device="cpu",tile_size=512,overlap=64):
    """Run 8-input-channel residual model over full-resolution RGB tiles."""
    base=np.asarray(base_hwc,np.float32); raw=np.asarray(packed_hwc,np.float32); H,W=base.shape[:2]
    tile=max(32,int(tile_size//8*8)); overlap=max(0,int(overlap//8*8)); out=np.zeros_like(base); weights=np.zeros((H,W,1),np.float32)
    param=next(model.parameters()); model=model.to(device).eval()
    # Starts are even so each full-resolution crop keeps its Bayer-cell alignment.
    for_size=lambda n,t,o: _starts(n,t,o)
    with torch.inference_mode():
        for top in for_size(H,tile,overlap):
            top-=top%2
            for left in for_size(W,tile,overlap):
                left-=left%2; bottom=min(top+tile,H); right=min(left+tile,W)
                ph, pw=bottom-top,right-left
                raw_patch=raw[top//2:(bottom+1)//2,left//2:(right+1)//2]
                base_patch=base[top:bottom,left:right]
                x=torch.from_numpy(base_patch.transpose(2,0,1).copy()[None]).to(device=device,dtype=param.dtype)
                r=torch.from_numpy(raw_patch.transpose(2,0,1).copy()[None]).to(device=device,dtype=param.dtype)
                n=torch.full((1,1,ph,pw),float(noise_sigma),device=device,dtype=param.dtype)
                pred=model(x,r,n).squeeze(0).permute(1,2,0).float().cpu().numpy()
                weight=_weights(ph,pw,top,left,bottom,right,H,W,overlap)
                out[top:bottom,left:right]+=pred*weight; weights[top:bottom,left:right]+=weight
    return out/np.maximum(weights,1e-8)

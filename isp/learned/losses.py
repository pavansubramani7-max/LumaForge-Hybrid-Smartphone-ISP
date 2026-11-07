"""Reconstruction, multi-scale structure, colour and edge-aware ISP losses."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def _rgb_to_lab(rgb):
    x=torch.clamp(rgb,0,1); x=torch.where(x<=.04045,x/12.92,((x+.055)/1.055).pow(2.4))
    m=x.new_tensor([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]])
    xyz=torch.einsum("bchw,kc->bkhw",x,m)/x.new_tensor([.95047,1.,1.08883])[None,:,None,None]
    d=6/29; f=torch.where(xyz>d**3,xyz.clamp_min(1e-10).pow(1/3),xyz/(3*d*d)+4/29)
    l=116*f[:,1:2]-16; a=500*(f[:,0:1]-f[:,1:2]); b=200*(f[:,1:2]-f[:,2:3])
    return torch.cat([l,a,b],dim=1)

def _ssim_loss(x,y):
    c1=.01**2; c2=.03**2
    mux=F.avg_pool2d(x,7,1,3); muy=F.avg_pool2d(y,7,1,3)
    vx=(F.avg_pool2d(x*x,7,1,3)-mux*mux).clamp_min(0)
    vy=(F.avg_pool2d(y*y,7,1,3)-muy*muy).clamp_min(0)
    cov=F.avg_pool2d(x*y,7,1,3)-mux*muy
    score=((2*mux*muy+c1)*(2*cov+c2))/((mux*mux+muy*muy+c1)*(vx+vy+c2))
    return 1-score.mean()

def multiscale_ssim_loss(x,y,levels=3):
    losses=[]
    for level in range(levels):
        losses.append(_ssim_loss(x,y)/(2**level))
        if min(x.shape[-2:])<16: break
        x=F.avg_pool2d(x,2,ceil_mode=True); y=F.avg_pool2d(y,2,ceil_mode=True)
    return sum(losses)

class CompositeISPLoss(nn.Module):
    def __init__(self,l1=1.,ssim=.15,lab=.05,gradient=.05,vgg=0.):
        super().__init__(); self.weights=(l1,ssim,lab,gradient,vgg); self.vgg=None
        if vgg:
            try:
                from torchvision.models import vgg16, VGG16_Weights
                net=vgg16(weights=VGG16_Weights.DEFAULT).features[:16].eval()
                for p in net.parameters(): p.requires_grad=False
                self.vgg=net
            except Exception as exc: raise RuntimeError("VGG perceptual loss needs torchvision and its VGG16 weights") from exc
    def forward(self,pred,target):
        w1,ws,wl,wg,wv=self.weights
        pieces={"l1":F.l1_loss(pred,target),"ms_ssim":multiscale_ssim_loss(pred,target),
                "lab":F.l1_loss(_rgb_to_lab(pred),_rgb_to_lab(target)),
                "gradient":F.l1_loss(pred[:,:,:,1:]-pred[:,:,:,:-1],target[:,:,:,1:]-target[:,:,:,:-1])+
                           F.l1_loss(pred[:,:,1:,:]-pred[:,:,:-1,:],target[:,:,1:,:]-target[:,:,:-1,:])}
        if self.vgg is not None:
            mean=pred.new_tensor([.485,.456,.406])[None,:,None,None]; std=pred.new_tensor([.229,.224,.225])[None,:,None,None]
            pieces["vgg"]=F.l1_loss(self.vgg((pred-mean)/std),self.vgg((target-mean)/std))
        else: pieces["vgg"]=pred.new_zeros(())
        total=w1*pieces["l1"]+ws*pieces["ms_ssim"]+wl*pieces["lab"]+wg*pieces["gradient"]+wv*pieces["vgg"]
        return total,pieces

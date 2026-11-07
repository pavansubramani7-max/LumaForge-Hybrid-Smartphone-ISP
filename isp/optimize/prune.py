"""Magnitude pruning utilities with an optional recovery fine-tune callback."""
from __future__ import annotations
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils import prune

def prune_model(model,amount=.3,structured=False):
    if not 0<amount<1: raise ValueError("Pruning amount must be between 0 and 1")
    layers=[(m,"weight") for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    if not layers: raise ValueError("No Conv2d or Linear weights to prune")
    if structured:
        for module,name in layers: prune.ln_structured(module,name,amount=amount,n=1,dim=0); prune.remove(module,name)
    else:
        prune.global_unstructured(layers,pruning_method=prune.L1Unstructured,amount=amount)
        for module,name in layers: prune.remove(module,name)
    return model

def zero_fraction(model):
    params=[p.detach() for name,p in model.named_parameters() if name.endswith("weight")]
    total=sum(p.numel() for p in params); zeros=sum(int((p==0).sum()) for p in params)
    return zeros/max(total,1)

def save_pruned(model,path,metadata=None):
    payload={"model":model.state_dict(),"pruning_sparsity":zero_fraction(model),"metadata":metadata or {}}
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)
    return Path(path).stat().st_size

"""Reduced precision conversion and checkpoint export helpers."""
from pathlib import Path
import torch

def convert_precision(model,precision="fp32",device="cpu"):
    if precision=="fp32": return model.float()
    if precision=="fp16": return model.half()
    if precision=="bfloat16": return model.to(dtype=torch.bfloat16)
    raise ValueError(f"Unsupported precision {precision}")

def save_precision_checkpoint(checkpoint_path,output_path,precision="fp16"):
    state=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    key="model" if isinstance(state,dict) and "model" in state else None
    weights=state[key] if key else state
    dtype={"fp16":torch.float16,"bfloat16":torch.bfloat16}.get(precision)
    if dtype is None: raise ValueError("Checkpoint export precision must be fp16 or bfloat16")
    compact={k:(v.to(dtype) if torch.is_floating_point(v) else v) for k,v in weights.items()}
    if key: state[key]=compact; state["precision"]=precision
    else: state={"model":compact,"precision":precision}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True); torch.save(state,output_path)
    return Path(output_path).stat().st_size

"""Dynamic INT8 and FX post-training static INT8 quantization."""
from __future__ import annotations
import torch
from torch import nn

def quantize_dynamic(model,modules=(nn.Linear,)):
    """Dynamic INT8 is effective for eligible Linear layers; Conv-heavy models use PTQ below."""
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(),set(modules),dtype=torch.qint8)

def quantize_static_fx(model,calibration_batches,example_inputs,forward_batch,backend=None,max_batches=100):
    """Calibrate and convert a Conv model with FX graph-mode static INT8 quantization."""
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_fx,convert_fx
    supported=[name for name in torch.backends.quantized.supported_engines if name!="none"]
    backend=backend or ("x86" if "x86" in supported else supported[0] if supported else None)
    if backend not in supported: raise RuntimeError(f"Requested quantized backend {backend!r} unsupported; available: {supported}")
    torch.backends.quantized.engine=backend
    mapping=get_default_qconfig_mapping(backend)
    example=tuple(x.cpu() for x in example_inputs) if isinstance(example_inputs,(tuple,list)) else (example_inputs.cpu(),)
    prepared=prepare_fx(model.cpu().eval(),mapping,example)
    with torch.inference_mode():
        for index,batch in enumerate(calibration_batches):
            if index>=max_batches: break
            values=forward_batch(batch)
            if not isinstance(values,(tuple,list)): values=(values,)
            prepared(*[value.cpu() if torch.is_tensor(value) else value for value in values])
    return convert_fx(prepared).eval()

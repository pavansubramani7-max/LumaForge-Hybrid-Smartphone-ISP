"""TorchScript/ONNX export and ONNX Runtime inference benchmark."""
from pathlib import Path
import torch

def export_torchscript(model,sample,output_path):
    model.eval(); traced=torch.jit.trace(model,sample,strict=False); traced.save(str(output_path)); return Path(output_path).stat().st_size

def export_onnx(model,sample,output_path,opset=17):
    model.eval(); torch.onnx.export(model,sample,str(output_path),opset_version=opset,
        input_names=[f"input_{i}" for i in range(len(sample))] if isinstance(sample,tuple) else ["input"],
        output_names=["output"],dynamic_axes=None)
    return Path(output_path).stat().st_size

def benchmark_onnx(path,inputs,runs=50,warmup=10,threads=1):
    try: import onnxruntime as ort
    except ImportError as exc: raise RuntimeError("Install onnxruntime to benchmark exported ONNX models") from exc
    options=ort.SessionOptions(); options.intra_op_num_threads=threads; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
    feeds={meta.name:value for meta,value in zip(session.get_inputs(),inputs)}
    for _ in range(warmup): session.run(None,feeds)
    import time,numpy as np
    times=[]
    for _ in range(runs):
        start=time.perf_counter(); session.run(None,feeds); times.append((time.perf_counter()-start)*1000)
    return {"mean_ms":float(np.mean(times)),"p50_ms":float(np.percentile(times,50)),"p95_ms":float(np.percentile(times,95)),"fps":1000/float(np.mean(times))}

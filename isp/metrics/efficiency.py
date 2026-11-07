"""Model footprint and inference efficiency measurements."""
from __future__ import annotations
import time
from pathlib import Path
import torch

def process_memory_mb():
    """Return process RSS and OS high-water RSS in MiB where the platform exposes them."""
    try:
        import psutil
        info=psutil.Process().memory_info()
        return {"rss_mb":info.rss/1024**2,"peak_rss_mb":getattr(info,"peak_wset",None)/1024**2
                if getattr(info,"peak_wset",None) else None}
    except ImportError:
        pass
    if __import__("sys").platform=="win32":
        try:
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD),
                    ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                    ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t),
                    ("PrivateUsage",ctypes.c_size_t)]
            counters=Counters(); counters.cb=ctypes.sizeof(counters)
            kernel=ctypes.WinDLL("kernel32",use_last_error=True); psapi=ctypes.WinDLL("psapi",use_last_error=True)
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
            if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
                return {"rss_mb":counters.WorkingSetSize/1024**2,
                        "peak_rss_mb":counters.PeakWorkingSetSize/1024**2}
        except (AttributeError,OSError):
            pass
    return {"rss_mb":None,"peak_rss_mb":None}

def model_stats(model,checkpoint=None,input_shape=(1,4,128,128)):
    params=sum(p.numel() for p in model.parameters())
    bytes_on_disk=Path(checkpoint).stat().st_size if checkpoint and Path(checkpoint).exists() else None
    macs=None
    try:
        from thop import profile
        macs=int(profile(model,inputs=(torch.zeros(input_shape),),verbose=False)[0])
    except Exception: pass
    return {"parameters":params,"fp32_parameter_mb":params*4/1024**2,"checkpoint_mb":bytes_on_disk/1024**2 if bytes_on_disk else None,"macs":macs}

def benchmark_callable(fn,warmup=10,runs=50,device=None):
    def synchronize():
        if device is not None and torch.device(device).type=="cuda": torch.cuda.synchronize(device)
    for _ in range(warmup): fn()
    synchronize()
    samples=[]
    for _ in range(runs):
        synchronize(); start=time.perf_counter(); fn(); synchronize(); samples.append((time.perf_counter()-start)*1000)
    values=torch.tensor(samples,dtype=torch.float32)
    return {"runs":runs,"mean_ms":float(values.mean()),"p50_ms":float(torch.quantile(values,.5)),"p95_ms":float(torch.quantile(values,.95)),"fps":1000/float(values.mean())}

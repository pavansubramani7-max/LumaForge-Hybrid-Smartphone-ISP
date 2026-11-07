"""Measure ISP checkpoint size, parameters, memory and tiled CPU/GPU latency.

This is an engineering benchmark, not a phone benchmark.  It deliberately uses
the project's real ``infer_image`` tiling path so the 1080p measurement has a
bounded model activation footprint.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.isp.model import TinyISPUNet, infer_image, parameter_count  # noqa: E402


def memory_snapshot(process):
    """Return portable process RSS fields where the host exposes them."""
    if process is not None:
        info = process.memory_info()
        return {"rss_mb": info.rss / 2**20}
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        handle = kernel32.GetCurrentProcess()
        get_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        ok = get_info(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return {
                "rss_mb": counters.WorkingSetSize / 2**20,
                "peak_rss_mb": counters.PeakWorkingSetSize / 2**20,
            }
    return {}


def measure_case(model, shape, device, warmup, runs, tile_size, overlap):
    height, width = shape
    # Fixed seed makes input generation repeatable without implying image quality.
    rng = np.random.default_rng(42)
    features = rng.random((height, width, 4), dtype=np.float32)
    for _ in range(warmup):
        infer_image(model, features, device=device, tile_size=tile_size, overlap=overlap)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    tracemalloc.start()
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        infer_image(model, features, device=device, tile_size=tile_size, overlap=overlap)
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    _, python_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "resolution": f"{width}x{height}",
        "warmup_runs": warmup,
        "timed_runs": runs,
        "tile_size": tile_size,
        "overlap": overlap,
        "latency_ms": {
            "mean": float(statistics.mean(timings)),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
        },
        # tracemalloc measures Python allocations only; do not mislabel it as RSS.
        "python_peak_tracemalloc_mb": python_peak / 2**20,
    }
    if device == "cuda":
        result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/isp_tiny.pt")
    parser.add_argument("--out", default="outputs/efficiency_benchmark.json")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU intra-op threads; 1 gives reproducible mobile-like host timing")
    parser.add_argument("--sizes", nargs="+", default=("256", "512", "1080p"),
                        choices=("256", "512", "1080p"))
    args = parser.parse_args()
    if args.warmup < 0 or args.runs < 1:
        parser.error("--warmup must be >= 0 and --runs must be >= 1")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is false.")

    if args.threads < 1:
        parser.error("--threads must be >= 1")
    torch.set_num_threads(args.threads)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = state.get("model", state) if isinstance(state, dict) else state
    model = TinyISPUNet().to(device).eval()
    model.load_state_dict(weights)
    sizes = {"256": (256, 256), "512": (512, 512), "1080p": (1080, 1920)}

    try:
        import psutil
        process = psutil.Process()
    except ImportError:
        process = None
    memory_before = memory_snapshot(process)
    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_size_mb": checkpoint.stat().st_size / 2**20,
        "parameters": parameter_count(model),
        "device": device,
        "precision": "fp32",
        "torch_intra_op_threads": args.threads,
        "measurement_note": "Host-process RSS includes Python/PyTorch; tracemalloc is Python allocations only. Not a handset measurement.",
        "host_memory_before": memory_before or None,
        "cases": [],
    }
    for key in args.sizes:
        report["cases"].append(measure_case(
            model, sizes[key], device, args.warmup, args.runs,
            args.tile_size, args.overlap,
        ))
        memory_after = memory_snapshot(process)
        if memory_after:
            report["host_memory_after_case"] = memory_after
    if device == "cuda":
        report["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20

    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()

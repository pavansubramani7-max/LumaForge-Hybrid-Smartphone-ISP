"""Compare FP32 and exported FP16 TinyISP checkpoints on a paired test split."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate import resolve_test_root  # noqa: E402
from src.isp.classical import run_classical  # noqa: E402
from src.isp.metrics import delta_e_76, psnr, ssim  # noqa: E402
from src.isp.model import TinyISPUNet, infer_image, parameter_count  # noqa: E402


def evaluate_checkpoint(path: Path, split: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyISPUNet().eval()
    model.load_state_dict(checkpoint.get("model", checkpoint))
    precision = checkpoint.get("precision", "fp32")
    if precision == "fp16":
        model.half()
    elif precision == "bfloat16":
        model.to(dtype=torch.bfloat16)

    config = {"black": 0, "white": None, "wb": (1.5, 1.0, 1.8),
              "exposure": 1.25, "denoise": 0.08, "sharpen": 0.25,
              "noise_feature": 0.08}
    rows, times = [], []
    raw_dir, rgb_dir = split / "raw", split / "rgb"
    raw_paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in {".npy", ".png", ".tif", ".tiff"})
    for raw_path in raw_paths:
        target_path = next((rgb_dir / (raw_path.stem + ext) for ext in (".png", ".tif", ".tiff", ".jpg", ".jpeg")
                            if (rgb_dir / (raw_path.stem + ext)).exists()), None)
        if target_path is None:
            continue
        raw = np.load(raw_path) if raw_path.suffix.lower() == ".npy" else np.asarray(Image.open(raw_path))
        target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.float32) / 255.0
        white = config["white"] or (float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0)
        base = run_classical(raw, "rggb", config["black"], white, wb=config["wb"],
                             exposure=config["exposure"], denoise_strength=config["denoise"],
                             sharpen_amount=config["sharpen"])
        feature = np.concatenate([base, np.full((*base.shape[:2], 1), config["noise_feature"], np.float32)], axis=-1)
        start = time.perf_counter()
        output = infer_image(model, feature, device="cpu")
        times.append((time.perf_counter() - start) * 1000)
        if output.shape != target.shape:
            raise ValueError(f"Image dimensions differ for {raw_path.name}: {output.shape} vs {target.shape}")
        rows.append({"image": raw_path.name, "psnr_db": psnr(output, target),
                     "ssim": ssim(output, target), "delta_e76": delta_e_76(output, target)})
    if not rows:
        raise SystemExit(f"No aligned RAW/RGB pairs found in {split}")
    return {
        "checkpoint": str(path.resolve()), "precision": precision,
        "checkpoint_size_bytes": path.stat().st_size,
        "checkpoint_size_mb": path.stat().st_size / 2**20,
        "parameters": parameter_count(model), "scenes": len(rows),
        "mean_metrics": {key: float(np.mean([row[key] for row in rows]) )
                         for key in ("psnr_db", "ssim", "delta_e76")},
        "median_latency_ms_per_scene": float(np.median(times)),
        "per_image": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", default="models/isp_tiny.pt")
    parser.add_argument("--fp16", default="outputs/isp_tiny_fp16.pt")
    parser.add_argument("--data", default=None, help="paired test split or dataset root")
    parser.add_argument("--out", default="outputs/fp16_comparison.json")
    args = parser.parse_args()
    def project_path(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path
    split = resolve_test_root(args.data)
    report = {"data_root": str(split), "synthetic_dataset": split.parent.name == "synthetic_pairs",
              "device": "cpu", "runs_per_scene": 1,
              "note": "One inference per scene; synthetic validation is not camera-quality evidence.",
              "variants": [evaluate_checkpoint(project_path(args.fp32), split),
                           evaluate_checkpoint(project_path(args.fp16), split)]}
    out = project_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    main()

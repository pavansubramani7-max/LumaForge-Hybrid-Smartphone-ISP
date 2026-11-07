"""Run labeled synthetic robustness and classical-stage ablation studies."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isp.classical.pipeline import ClassicalISP  # noqa: E402
from isp.learned.degradation import degrade_packed  # noqa: E402
from isp.metrics.image_metrics import delta_e_2000, ms_ssim  # noqa: E402
from isp.raw_utils import normalize_raw, pack_bayer, unpack_bayer  # noqa: E402
from src.isp.classical import run_classical  # noqa: E402
from src.isp.metrics import delta_e_76, psnr, ssim  # noqa: E402
from src.isp.model import TinyISPUNet, infer_image  # noqa: E402


def load_pairs(split):
    raw_dir, rgb_dir = split / "raw", split / "rgb"
    if not raw_dir.is_dir() or not rgb_dir.is_dir():
        raise SystemExit(f"Expected paired folders raw/ and rgb/ in {split}")
    pairs = []
    for raw_path in sorted(raw_dir.iterdir()):
        if raw_path.suffix.lower() not in {".npy", ".png", ".tif", ".tiff"}:
            continue
        target_path = next((rgb_dir / (raw_path.stem + ext) for ext in
                            (".png", ".tif", ".tiff", ".jpg", ".jpeg")
                            if (rgb_dir / (raw_path.stem + ext)).exists()), None)
        if target_path:
            raw = np.load(raw_path) if raw_path.suffix.lower() == ".npy" else np.asarray(Image.open(raw_path))
            target = np.asarray(Image.open(target_path).convert("RGB"), np.float32) / 255.0
            pairs.append((raw_path.name, raw, target))
    if not pairs:
        raise SystemExit(f"No aligned RAW/RGB image pairs found in {split}")
    return pairs


def score(output, target):
    if output.shape != target.shape:
        raise ValueError(f"Prediction/target dimensions differ: {output.shape} vs {target.shape}")
    return {"psnr_db": psnr(output, target), "ssim": ssim(output, target),
            "ms_ssim": ms_ssim(output, target), "delta_e76": delta_e_76(output, target),
            "delta_e2000": delta_e_2000(output, target)}


def run_studies(split, checkpoint, out_dir):
    pairs = load_pairs(split)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = TinyISPUNet(base=int(ckpt.get("base", 24))).eval()
    model.load_state_dict(ckpt.get("model", ckpt))
    cfg = {"black": 0, "white": None, "wb": (1.5, 1.0, 1.8), "exposure": 1.25,
           "denoise": 0.08, "sharpen": 0.25, "noise_feature": 0.08}
    robustness, ablation = [], []

    # Controlled sensor degradations: one factor changes at a time with fixed seeds.
    sweeps = {
        "noise_sigma": [0.005, 0.02, 0.05, 0.10],
        "exposure_gain": [0.35, 0.60, 1.00, 1.35],
        "cct_kelvin": [2500, 4500, 6500, 9000],
        "wb_error": [0.0, 0.10, 0.20, 0.40],
    }
    parameter_names = {"noise_sigma": "sigma", "exposure_gain": "exposure",
                       "cct_kelvin": "cct_kelvin", "wb_error": "wb_error"}
    for image_index, (name, raw, target) in enumerate(pairs):
        white = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0
        clean = normalize_raw(raw, 0, white)
        packed = pack_bayer(clean, "RGGB")
        for factor, levels in sweeps.items():
            for level_index, value in enumerate(levels):
                params = {"sigma": 0.02, "exposure": 1.0, "cct_kelvin": 5500, "wb_error": 0.1}
                params[parameter_names[factor]] = value
                stressed = degrade_packed(packed, params["sigma"],
                    (params["exposure"], params["exposure"]), params["wb_error"],
                    vignetting=0.0, seed=42 + image_index * 100 + level_index,
                    cct_kelvin=params["cct_kelvin"])
                mosaic = unpack_bayer(stressed, "RGGB")
                base = run_classical(mosaic, "rggb", 0, 1.0, wb=cfg["wb"],
                    exposure=cfg["exposure"], denoise_strength=max(params["sigma"], .01),
                    sharpen_amount=cfg["sharpen"])
                features = np.concatenate([base, np.full((*base.shape[:2], 1),
                    params["sigma"], np.float32)], axis=-1)
                hybrid = infer_image(model, features, device="cpu")
                for method, output in (("traditional", base), ("hybrid", hybrid)):
                    robustness.append({"image": name, "factor": factor, "value": value,
                                       "pipeline": method, **score(output, target)})

        # Classical ablation: remove one stage at a time against the same synthetic target.
        pipeline = ClassicalISP()
        full = pipeline.run(clean, "raw", "RGGB", 0, 1.0).image
        variants = [("full pipeline", None)] + [(f"without {stage}", stage) for stage in
            ("white_balance", "ccm", "denoise", "tone", "sharpen")]
        for label, removed in variants:
            enabled = None if removed is None else {key: True for key in
                ("black_level", "dead_pixel", "white_balance", "ccm", "denoise", "tone", "sharpen")}
            if removed:
                enabled[removed] = False
                output = pipeline.run(clean, "raw", "RGGB", 0, 1.0, enabled=enabled).image
            else:
                output = full
            ablation.append({"image": name, "variant": label, **score(output, target)})

    def summarize(rows, grouping):
        result = []
        groups = {}
        for row in rows:
            key = tuple(row[field] for field in grouping)
            groups.setdefault(key, []).append(row)
        for key, group in groups.items():
            summary = dict(zip(grouping, key))
            for metric in ("psnr_db", "ssim", "ms_ssim", "delta_e76", "delta_e2000"):
                values = [r[metric] for r in group]
                summary[metric + "_mean"] = float(np.mean(values))
                summary[metric + "_std"] = float(np.std(values))
            summary["n"] = len(group)
            result.append(summary)
        return result

    out_dir.mkdir(parents=True, exist_ok=True)
    reports = {
        "robustness": {"dataset": str(split.resolve()), "synthetic": True, "scenes": len(pairs),
                       "perceptual_metric": "MS-SSIM (LPIPS package/weights unavailable)",
                       "note": "Controlled synthetic sensor corruption; not real-camera robustness evidence.",
                       "summary": summarize(robustness, ("factor", "value", "pipeline")),
                       "per_image": robustness},
        "ablation": {"dataset": str(split.resolve()), "synthetic": True, "scenes": len(pairs),
                     "perceptual_metric": "MS-SSIM (LPIPS package/weights unavailable)",
                     "note": "Classical-stage ablation on generated pairs; not camera-calibrated evidence.",
                     "summary": summarize(ablation, ("variant",)), "per_image": ablation},
    }
    for name, report in reports.items():
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        flat = report["summary"]
        with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
            fields = list(flat[0]) if flat else []
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(flat)
        print(f"{name}: {len(report['summary'])} summary rows; saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/synthetic_pairs/test", help="paired test split")
    parser.add_argument("--checkpoint", default="models/isp_tiny.pt")
    parser.add_argument("--out", default="outputs/synthetic_studies")
    args = parser.parse_args()
    def path(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p
    run_studies(path(args.data), path(args.checkpoint), path(args.out))


if __name__ == "__main__":
    main()

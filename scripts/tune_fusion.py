"""Select modular hybrid fusion strength on validation scenes only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.dataset import _read_raw, discover_pairs  # noqa: E402
from isp.classical.pipeline import ClassicalISP  # noqa: E402
from isp.evaluation.benchmark import _load_model, _score  # noqa: E402
from isp.hybrid.hybrid_pipeline import HybridISPPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/synthetic_pairs")
    parser.add_argument("--split", default="val")
    parser.add_argument("--out", default="outputs/fusion_tuning.json")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--scales", nargs="+", type=float,
                        default=[0.0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    args = parser.parse_args()
    def project_path(value):
        path = Path(value).expanduser()
        return path if path.is_absolute() else ROOT / path

    root = project_path(args.data_root).resolve()
    split_file = root / "splits.json"
    manifest = json.loads(split_file.read_text(encoding="utf-8"))
    selected = set(manifest.get(args.split, []))
    pairs = [p for p in discover_pairs(root) if p.key in selected and p.raw_path.is_relative_to(root)]
    if not pairs:
        raise SystemExit(f"No {args.split} paired RAW/RGB scenes found under {root}")
    model_dir = project_path(args.models_dir)
    refiner_load = _load_model(model_dir / "hybrid_refiner.pt", "cpu")
    mobile_load = _load_model(model_dir / "mobile_unet.pt", "cpu")
    refiner = refiner_load[0] if refiner_load else None
    mobile = mobile_load[0] if mobile_load else None
    if refiner is None:
        raise SystemExit("models/hybrid_refiner.pt is required for fusion tuning")
    candidates = []
    for scale in args.scales:
        if not 0 <= scale <= 1:
            parser.error("fusion scales must be between 0 and 1")
        pipe = HybridISPPipeline(ClassicalISP(), refiner, mobile, "cpu",
            {"tile_size": 512, "tile_overlap": 64, "wb_gains": (1.5, 1., 1.8),
             "fusion_alpha_scale": scale})
        rows = []
        for pair in pairs:
            raw, metadata = _read_raw(pair.raw_path)
            if raw.ndim != 2:
                continue
            target = np.asarray(Image.open(pair.rgb_path).convert("RGB"), np.float32) / 255.
            meta_path = root / "dataset_metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            pattern = metadata.get("bayer_pattern", meta.get("bayer_pattern", "RGGB"))
            white = meta.get("white_level")
            if white is None:
                white = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.
            output = pipe.process_raw(raw, pattern, meta.get("black_level", 0), white, adaptive=True)
            rows.append({"key": pair.key, "traditional": _score(output.traditional, target),
                         "hybrid": _score(output.hybrid, target)})
        metrics = ("psnr_db", "ssim", "ms_ssim", "delta_e76", "delta_e2000")
        traditional = {m: float(np.mean([r["traditional"][m] for r in rows])) for m in metrics}
        hybrid = {m: float(np.mean([r["hybrid"][m] for r in rows])) for m in metrics}
        improved = {m: (hybrid[m] > traditional[m] if m in {"psnr_db", "ssim", "ms_ssim"}
                        else hybrid[m] < traditional[m]) for m in metrics}
        # Equal-weight normalized signed gain; used only to rank validation candidates.
        gains = [(hybrid[m] - traditional[m]) / max(abs(traditional[m]), 1e-8)
                 * (1 if m in {"psnr_db", "ssim", "ms_ssim"} else -1) for m in metrics]
        candidates.append({"scale": scale, "scenes": len(rows), "traditional": traditional,
                           "hybrid": hybrid, "improved_metrics": improved,
                           "improved_count": sum(improved.values()), "mean_normalized_gain": float(np.mean(gains)),
                           "per_scene": rows})

    winners = [c for c in candidates if all(c["improved_metrics"].values())]
    pool = winners or candidates
    selected = max(pool, key=lambda c: (c["improved_count"], c["mean_normalized_gain"]))
    report = {"dataset": str(root), "split": args.split, "synthetic": bool(
        json.loads((root / "dataset_metadata.json").read_text(encoding="utf-8")).get("synthetic", False)),
        "selection_rule": "Choose a validation fusion scale improving all five metrics if available; otherwise maximize number improved then mean normalized gain.",
        "selected_scale": selected["scale"], "all_metrics_improved_on_validation": all(selected["improved_metrics"].values()),
        "selected_validation_result": selected, "candidates": candidates}
    output = project_path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"selected_scale={selected['scale']} all_metrics_improved={report['all_metrics_improved_on_validation']} scenes={selected['scenes']}")
    print(f"saved={output.resolve()}")


if __name__ == "__main__":
    main()

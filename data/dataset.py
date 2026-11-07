"""Dataset discovery, pairing, split generation and paired RAW/RGB loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import re
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from isp.raw_utils import normalize_raw,augment_packed,unpack_bayer

RAW_EXTS = {".npy", ".png", ".tif", ".tiff", ".dng", ".raw"}
RGB_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
RAW_DIR_WORDS = {"raw", "raws", "mosaic", "mosaics", "input", "inputs", "noisy"}
RGB_DIR_WORDS = {"rgb", "gt", "target", "targets", "reference", "canon", "dslr", "clean"}
_PAIR_WORDS = re.compile(r"(?i)(?:[_ .-](?:raw|rgb|gt|target|reference|input|noisy|clean|canon|dslr|huawei|p20))+$")


@dataclass(frozen=True)
class PairRecord:
    key: str
    raw_path: Path
    rgb_path: Path


def normalized_key(path: Path) -> str:
    name = path.stem
    previous = None
    while previous != name:
        previous, name = name, _PAIR_WORDS.sub("", name)
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _candidate_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        suffix = path.suffix.lower()
        parent_tokens = {p.lower() for p in path.relative_to(root).parts[:-1]}
        name_tokens = set(re.split(r"[^a-z0-9]+", path.stem.lower()))
        raw_named = bool(name_tokens & {"raw", "mosaic", "noisy", "input"})
        rgb_named = bool(name_tokens & {"rgb", "gt", "target", "reference", "clean", "canon", "dslr"})
        if suffix in RAW_EXTS and (parent_tokens & RAW_DIR_WORDS or raw_named or suffix in {".npy", ".dng", ".raw"}):
            yield "raw", path
        elif suffix in RGB_EXTS and (parent_tokens & RGB_DIR_WORDS or rgb_named or suffix in {".jpg", ".jpeg"}):
            yield "rgb", path
        elif suffix in {".png", ".tif", ".tiff"}:
            try:
                mode = Image.open(path).mode
            except Exception:
                continue
            if mode in {"1", "L", "I", "I;16", "I;16B", "I;16L"}:
                yield "raw", path
            else:
                yield "rgb", path


def discover_pairs(root: str | Path) -> list[PairRecord]:
    """Pair files by normalized basename, preferring explicit raw/RGB directories."""
    root = Path(root).expanduser().resolve()
    raw_map: dict[str, list[Path]] = {}
    rgb_map: dict[str, list[Path]] = {}
    for kind, path in _candidate_files(root):
        key = normalized_key(path)
        if not key:
            continue
        (raw_map if kind == "raw" else rgb_map).setdefault(key, []).append(path)
    result = []
    for key in sorted(raw_map.keys() & rgb_map.keys()):
        raws = sorted(raw_map[key]); rgbs = sorted(rgb_map[key])
        # Pair only unambiguous normalized names; never silently Cartesian-pair duplicates.
        if len(raws) == len(rgbs) == 1:
            result.append(PairRecord(key, raws[0], rgbs[0]))
    return result


def _read_raw(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    suffix = path.suffix.lower()
    metadata: dict[str, Any] = {"path": str(path)}
    if suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
    elif suffix == ".dng":
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("DNG support needs rawpy. Install it with `pip install rawpy`.") from exc
        with rawpy.imread(str(path)) as camera_raw:
            raw = camera_raw.raw_image_visible.copy()
            pattern = camera_raw.raw_pattern
            if pattern is not None:
                metadata["raw_pattern_indices"] = pattern.tolist()
            metadata["black_level_per_channel"] = list(camera_raw.black_level_per_channel)
            metadata["white_level"] = int(camera_raw.white_level)
            metadata["bayer_pattern"] = guess_bayer_pattern(pattern=pattern)
    elif suffix == ".raw":
        raise ValueError(f"Unheadered .raw file needs width, height, dtype and Bayer pattern in config: {path.name}")
    else:
        raw = np.asarray(Image.open(path))
    metadata.update({"shape": list(raw.shape), "dtype": str(raw.dtype)})
    if raw.ndim not in (2, 3):
        raise ValueError(f"RAW array must be 2D mosaic or 3D packed data, got {raw.shape} in {path}")
    if raw.ndim == 2:
        metadata["bayer_pattern_guess"] = guess_bayer_pattern(mosaic=raw)
    elif raw.ndim == 3 and (raw.shape[-1] == 4 or raw.shape[0] == 4):
        metadata["layout"] = "packed4"
    else:
        raise ValueError(f"Unsupported RAW array layout {raw.shape} in {path}; expected mosaic or packed-4")
    return raw, metadata


def guess_bayer_pattern(mosaic: np.ndarray | None = None, pattern=None) -> dict[str, Any]:
    """Use rawpy metadata when available; otherwise return a documented heuristic guess."""
    mappings = {
        "RGGB": [[0, 1], [1, 2]], "BGGR": [[2, 1], [1, 0]],
        "GRBG": [[1, 0], [2, 1]], "GBRG": [[1, 2], [0, 1]],
    }
    if pattern is not None:
        arr = np.asarray(pattern).tolist()
        for name, mapping in mappings.items():
            if arr == mapping:
                return {"pattern": name, "confidence": 1.0, "method": "raw metadata"}
    if mosaic is None or np.asarray(mosaic).ndim != 2:
        return {"pattern": "unknown", "confidence": 0.0, "method": "metadata unavailable"}
    x = np.asarray(mosaic, np.float32)
    planes = [x[0::2,0::2], x[0::2,1::2], x[1::2,0::2], x[1::2,1::2]]
    if min(p.size for p in planes) < 4:
        return {"pattern": "unknown", "confidence": 0.0, "method": "image too small"}
    correlations = []
    for a, b in ((1, 2), (0, 3)):
        aa, bb = planes[a].ravel(), planes[b].ravel()
        count = min(len(aa), len(bb), 20000)
        aa, bb = aa[:count], bb[:count]
        if count > 2 and float(np.std(aa)) > 1e-8 and float(np.std(bb)) > 1e-8:
            corr = float(np.corrcoef(aa, bb)[0, 1])
        else:
            corr = 0.0
        correlations.append(corr if np.isfinite(corr) else 0.0)
    green_pair = (1, 2) if correlations[0] >= correlations[1] else (0, 3)
    remaining = [i for i in range(4) if i not in green_pair]
    means = [float(np.mean(planes[i])) for i in remaining]
    red_idx, blue_idx = (remaining[0], remaining[1]) if means[0] >= means[1] else (remaining[1], remaining[0])
    arrangement = [None] * 4
    arrangement[red_idx], arrangement[blue_idx] = "R", "B"
    arrangement[green_pair[0]] = arrangement[green_pair[1]] = "G"
    name = "".join(arrangement)
    score_gap = abs(correlations[0] - correlations[1])
    confidence = float(np.clip(.35 + .4 * score_gap, .35, .75))
    return {"pattern": name if name in mappings else "unknown", "confidence": confidence,
            "method": "heuristic; verify against sensor metadata"}


def _pack_mosaic(mosaic: np.ndarray, pattern: str) -> np.ndarray:
    if mosaic.shape[0] % 2 or mosaic.shape[1] % 2:
        mosaic = mosaic[:mosaic.shape[0]//2*2, :mosaic.shape[1]//2*2]
    grid = {"RGGB": ((0,0),(0,1),(1,0),(1,1)),
            "BGGR": ((1,1),(0,1),(1,0),(0,0)),
            "GRBG": ((0,1),(0,0),(1,1),(1,0)),
            "GBRG": ((1,0),(0,0),(1,1),(0,1))}.get(pattern, ((0,0),(0,1),(1,0),(1,1)))
    planes = [mosaic[y::2, x::2] for y, x in grid]
    return np.stack(planes, axis=-1)


class PairedRawRGBDataset(Dataset):
    """Paired RAW/RGB dataset; split IDs are read from a generated splits.json."""
    def __init__(self, data_root: str | Path, split: str, splits_file: str | Path = "splits.json",
                 patch_size: int | None = 128, bayer_pattern: str = "auto", augment: bool = False,
                 black_level=0,white_level=None,noise_sigma=0,wb_gains=(1.5,1.,1.8),ccm=None):
        self.root = Path(data_root).expanduser().resolve()
        self.split, self.patch_size, self.bayer_pattern, self.augment = split, patch_size, bayer_pattern, augment
        self.black_level,self.white_level,self.noise_sigma=black_level,white_level,noise_sigma
        self.wb_gains=np.asarray(wb_gains,np.float32)
        self.ccm=np.asarray(ccm if ccm is not None else [[1.48,-.32,-.16],[-.12,1.25,-.13],[-.02,-.44,1.46]],np.float32)
        self.records = discover_pairs(self.root)
        self.dataset_metadata={}
        for metadata_name in ("dataset_metadata.json","metadata.json"):
            metadata_path=self.root/metadata_name
            if metadata_path.exists():
                self.dataset_metadata=json.loads(metadata_path.read_text(encoding="utf-8")); break
        split_path = Path(splits_file)
        if not split_path.is_absolute():
            local = self.root / split_path
            split_path = local if local.exists() else self.root.parent / split_path
        if not split_path.exists():
            make_splits(self.root, split_path)
        split_map = json.loads(split_path.read_text(encoding="utf-8"))
        wanted = set(split_map.get(split, []))
        self.records = [r for r in self.records if r.key in wanted]
        if not self.records:
            raise ValueError(f"No paired images in split '{split}' under {self.root}. Inspect the dataset and split manifest.")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        raw, meta = _read_raw(record.raw_path)
        pattern_info = meta.get("bayer_pattern_guess", meta.get("bayer_pattern", {}))
        configured_pattern=self.dataset_metadata.get("bayer_pattern")
        pattern = self.bayer_pattern if self.bayer_pattern != "auto" else (configured_pattern or pattern_info.get("pattern", "RGGB"))
        if raw.ndim == 2:
            raw_float = normalize_raw(raw,self.black_level,self.white_level)
            packed = _pack_mosaic(raw_float, pattern)
        else:
            arr = normalize_raw(raw,self.black_level,self.white_level)
            if arr.shape[-1] == 4: packed = arr
            else: packed = np.moveaxis(arr, 0, -1)
        rgb = np.asarray(Image.open(record.rgb_path).convert("RGB"), np.float32) / 255.0
        h, w = packed.shape[:2]
        # Packed Bayer is half resolution; RGB target is normally twice its dimensions.
        if rgb.shape[:2] == (h * 2, w * 2):
            pass
        elif raw.ndim == 2 and rgb.shape[:2] == raw.shape:
            rgb = rgb[::2, ::2]
        else:
            raise ValueError(f"RAW/RGB geometry does not align for {record.key}: packed={packed.shape}, RGB={rgb.shape}")
        size = min(self.patch_size or min(h,w), h, w)
        size = max(8, size // 8 * 8)
        if self.augment:
            top = np.random.randint(0, max(1, h-size+1)); left = np.random.randint(0, max(1, w-size+1))
        else:
            top, left = max(0,(h-size)//2), max(0,(w-size)//2)
        packed = packed[top:top+size,left:left+size]
        rgb = rgb[top*2:(top+size)*2,left*2:(left+size)*2]
        if self.augment:
            op=np.random.choice(["identity","hflip","vflip","rot90","rot180","rot270"])
            packed,pattern=augment_packed(packed,pattern,op)
            rgb=np.rot90(rgb,{"rot90":1,"rot180":2,"rot270":3}.get(op,0))
            if op=="hflip": rgb=np.fliplr(rgb)
            if op=="vflip": rgb=np.flipud(rgb)
            if self.noise_sigma>0:
                from isp.learned.degradation import degrade_packed
                packed=degrade_packed(packed,self.noise_sigma)
        from isp.classical.demosaic import bilinear_demosaic
        from isp.classical.white_balance import apply_white_balance
        from isp.classical.ccm import apply_ccm
        from isp.classical.colorspace import srgb_to_linear
        packed=np.clip(packed,0,1).astype(np.float32)
        mosaic=unpack_bayer(packed,pattern)
        baseline=np.clip(apply_ccm(apply_white_balance(bilinear_demosaic(mosaic,pattern),self.wb_gains),self.ccm),0,1)
        target_linear=srgb_to_linear(rgb)
        return {"raw": torch.from_numpy(packed.transpose(2,0,1).copy()),
                "baseline":torch.from_numpy(baseline.transpose(2,0,1).copy()),
                "rgb": torch.from_numpy(target_linear.transpose(2,0,1).copy()),
                "noise_sigma":torch.tensor(float(self.noise_sigma),dtype=torch.float32),"key": record.key}


def make_splits(data_root: str | Path, output: str | Path = "splits.json", seed=42,
                train=.8, val=.1, test=.1):
    if abs(train + val + test - 1.0) > 1e-6:
        raise ValueError("Train/validation/test fractions must sum to 1")
    records = discover_pairs(data_root)
    if len(records) < 3:
        raise ValueError(f"Need at least 3 unambiguous pairs to split; found {len(records)}")
    keys = [r.key for r in records]
    rng = np.random.default_rng(seed); rng.shuffle(keys)
    n_train = max(1, int(round(len(keys) * train)))
    n_val = max(1, int(round(len(keys) * val)))
    if n_train + n_val >= len(keys): n_train, n_val = len(keys)-2, 1
    manifest = {"seed": seed, "data_root": str(Path(data_root).resolve()),
                "counts": {"total":len(keys), "train":n_train, "val":n_val, "test":len(keys)-n_train-n_val},
                "train":keys[:n_train], "val":keys[n_train:n_train+n_val], "test":keys[n_train+n_val:]}
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

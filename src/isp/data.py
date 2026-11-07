"""Safe ZIP intake and validation for paired RAW/RGB ISP datasets."""
from __future__ import annotations
from pathlib import Path, PurePosixPath
import os
import zipfile


def safe_extract_zip(fileobj, destination: str | Path, max_uncompressed_bytes=40 * 1024**3):
    """Extract a user ZIP without path traversal, symlinks, or oversized archives."""
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(fileobj) as archive:
        entries = archive.infolist()
        for info in entries:
            rel = PurePosixPath(info.filename.replace("\\", "/"))
            if rel.is_absolute() or (rel.parts and ":" in rel.parts[0]) or any(part in {"..", ""} for part in rel.parts):
                raise ValueError(f"Unsafe path in ZIP: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"ZIP symlink is not supported: {info.filename}")
            total += info.file_size
        if total > max_uncompressed_bytes:
            raise ValueError(f"Expanded ZIP is too large ({total / 1024**3:.1f} GiB limit 40 GiB)")
        for info in entries:
            if info.is_dir():
                continue
            rel = PurePosixPath(info.filename.replace("\\", "/"))
            target = (root / Path(*rel.parts)).resolve()
            if root not in target.parents:
                raise ValueError(f"Unsafe path in ZIP: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as src, target.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    dst.write(chunk)
    return root


def inspect_pair_layout(root: str | Path):
    """Inspect the documented split/raw+rgb structure and report pairs per split."""
    root = Path(root)
    report = {}
    exts = {".npy", ".png", ".tif", ".tiff"}
    for split in ("train", "val", "test"):
        raw_dir, rgb_dir = root / split / "raw", root / split / "rgb"
        if not raw_dir.is_dir() or not rgb_dir.is_dir():
            report[split] = {"pairs": 0, "valid": False, "message": "Expected split/raw and split/rgb folders"}
            continue
        raws = {p.stem: p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
        rgbs = {p.stem: p for p in rgb_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff", ".jpg", ".jpeg"}}
        common = sorted(raws.keys() & rgbs.keys())
        report[split] = {"pairs": len(common), "valid": bool(common),
                         "raw_without_rgb": len(raws.keys() - rgbs.keys()),
                         "rgb_without_raw": len(rgbs.keys() - raws.keys()),
                         "message": "Ready" if common else "No matching filename stems"}
    return report


def find_pair_root(root: str | Path):
    """Find the root containing train/raw and val/raw, including a ZIP top folder."""
    root = Path(root)
    for current, dirs, _files in os.walk(root):
        candidate = Path(current)
        if (candidate / "train" / "raw").is_dir() and (candidate / "val" / "raw").is_dir():
            return candidate
    return root

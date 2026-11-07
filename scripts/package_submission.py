"""Create a reviewable P2 source/demo bundle without the unrelated fraud archive."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/LumaForge_P2_Submission.zip")
    args = parser.parse_args()
    output = Path(args.out).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    include_roots = ["app", ".streamlit", "isp", "src", "configs", "tests"]
    include_files = ["README.md", "requirements.txt", "pyproject.toml", "Dockerfile",
                     ".dockerignore", "app.py", "models/isp_tiny.pt", "models/isp_tiny_fp16.pt",
                     "scripts/benchmark_efficiency.py", "scripts/compare_fp16.py",
                     "scripts/run_synthetic_studies.py", "scripts/package_submission.py",
                     "scripts/evaluate.py", "scripts/train.py", "scripts/tune_fusion.py", "scripts/make_synthetic_dataset.py",
                     "scripts/inspect_dataset.py", ".github/workflows/ci.yml",
                     "data/synthetic_pairs"]
    report_paths = ["outputs/final_audit_report.md", "outputs/evaluation.json",
                    "outputs/efficiency_benchmark_256_512.json",
                    "outputs/efficiency_benchmark_1080p.json",
                    "outputs/fp16_comparison.json", "outputs/isp_tiny_fp16.pt",
                    "outputs/fusion_tuning.json", "outputs/synthetic_studies",
                    "outputs/tuned_synthetic_benchmark"]
    files = set()
    for root in include_roots:
        directory = ROOT / root
        if directory.exists():
            files.update(p for p in directory.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})
    for name in include_files + report_paths:
        path = ROOT / name
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(p for p in path.rglob("*") if p.is_file()
                         and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"})

    missing = [name for name in ("README.md", "app.py", "models/isp_tiny.pt")
               if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit("Cannot package submission; required files are missing: " + ", ".join(missing))
    with ZipFile(output, "w", ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())
    print(f"Created {output.resolve()} ({output.stat().st_size:,} bytes; {len(files)} files)")
    print("Excluded data/incoming_archive6: unrelated FraudxAI tables are not ISP training data.")


if __name__ == "__main__":
    main()

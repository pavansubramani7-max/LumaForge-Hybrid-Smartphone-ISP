# HybridISP Studio — Computer Vision Hackathon P2

HybridISP Studio is a runnable Streamlit prototype for Bayer RAW-to-RGB reconstruction. It combines a configurable classical ISP, compact learned ISP models, an adaptive noise/illumination controller, hybrid fusion, evaluation tools, and CPU-friendly training/export workflows.

## Data status — read before interpreting scores

The uploaded `archive (6).zip` was extracted to `data/incoming_archive6/` and inspected. It contains six FraudxAI payment transaction CSV tables and documentation, with **zero image files and zero RAW/RGB pairs**. It cannot train or evaluate an image signal processor and is kept separate from ISP code. No external image dataset was downloaded. To make the demo runnable, this project includes a tiny generated Bayer/RGB dataset under `data/synthetic_pairs/`. Any checkpoint or score trained on it is explicitly synthetic and is not evidence of real-camera or institute-dataset performance.

When the camera dataset is available, set `data.data_root` in `configs/default.yaml` or use `--data_root`. The preferred layout is paired aligned mosaics and RGB targets (`.npy`, TIFF, or grayscale PNG RAW; RGB PNG/TIFF), with scene-disjoint splits. DNG support requires optional `rawpy` and suitable metadata. Run the inspector before training:

```powershell
python scripts/inspect_dataset.py --data_root .\data\your_camera_pairs --out_dir outputs\inspection
```

## Run the app

From the project root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Run that command from the repository root. `app.py` is the main upload, training and evaluation interface. Uploading an image starts processing automatically and displays the four-column comparison: **Visualized RAW Image | Reconstructed RGB Image | Huawei P20’s ISP Photo | Canon Photo**. Huawei-style and Canon-style previews are generated automatically from the reconstructed RGB; they are explicitly simulations, not actual device captures. Add a real aligned RGB target in the optional reference section to compute quality metrics. The complete review can be inspected and downloaded as JSON, HTML or PDF. For the multipage dashboard (Overview, ISP Playground, pipeline comparison, robustness, efficiency, ablation and report export), use `python -m streamlit run app/streamlit_app.py` instead.

The Streamlit interface uses a wine-colour theme and Times New Roman typography. The dependency-free PDF report uses the standard Times-Roman serif font.

To keep working while the real dataset is unavailable, train a quick synthetic demo in the Overview page, or run these commands:

```powershell
python scripts/make_synthetic_dataset.py --out data/synthetic_pairs
python -m isp.learned.train --data_root data/synthetic_pairs --models mobile_unet hybrid_refiner --quick
python scripts/run_benchmark.py --data_root data/synthetic_pairs --device cpu --out_dir outputs/synthetic_benchmark
```

`--quick` is for a smoke/demo checkpoint. For actual adaptation, omit it and set epochs/patch size/batch size in `configs/default.yaml`:

```powershell
python -m isp.learned.train --config configs/default.yaml --data_root .\data\your_camera_pairs --models mobile_unet hybrid_refiner
```

The trainer creates a deterministic 80/10/10 scene split in `splits.json` and writes checkpoints in `models/`. Training time depends on image count, patch size, CPU/GPU, and selected model. Do not report synthetic demo measurements as camera-dataset results.

## Project components

- `isp/raw_utils.py`: Bayer normalization, CFA patterns, 4-channel packing and raw loading.
- `isp/classical/`: black level, WB, demosaicing, CCM, denoising, tone/gamma, sharpening, and timed pipeline.
- `isp/learned/`: Mobile U-Net ISP, PyNET-lite, residual refiner, degradation augmentation, losses, training and tiled inference.
- `isp/hybrid/`: adaptive control, confidence fusion, and hybrid orchestration.
- `isp/optimize/`: dynamic/static quantization, pruning, FP16/BF16 export, distillation, TorchScript/ONNX export.
- `isp/metrics/`, `isp/evaluation/`: image quality, efficiency, benchmark, robustness, and report generation.
- `app/`: multipage Streamlit dashboard.
- `scripts/`: data inspection/generation, training entry points, and benchmark launcher.

## Useful commands

```powershell
python -m unittest discover -s tests -v
python scripts/inspect_dataset.py --data_root .\data\synthetic_pairs --out_dir outputs\synthetic_inspection
python scripts/run_benchmark.py --data_root .\data\synthetic_pairs --device cpu --out_dir results
python scripts/evaluate.py
python scripts/benchmark_efficiency.py
python scripts/compare_fp16.py
python scripts/run_synthetic_studies.py
python scripts/tune_fusion.py
python scripts/run_benchmark.py --data_root data/synthetic_pairs --fusion_alpha_scale 0.5 --out_dir outputs/tuned_synthetic_benchmark
python scripts/package_submission.py
```

`benchmark_efficiency.py` defaults to the rubric's 10 warm-up and 50 timed runs at 256×256, 512×512 and tiled 1080p. The 1080p CPU case is slow; use `--sizes 256 512` for a quicker host sweep. `compare_fp16.py` evaluates the FP16 export against FP32 on paired test images. Both commands default to the bundled synthetic split/checkpoint and must be rerun with camera data before making camera-quality claims.

`run_synthetic_studies.py` writes robustness sweeps and classical-stage ablations under `outputs/synthetic_studies/`; its results are explicitly synthetic-only. `package_submission.py` creates a source, demo-checkpoint and report ZIP under `outputs/`. Docker and GitHub Actions definitions are included; Docker could not be built on the current Windows environment because Docker is not installed.

`tune_fusion.py` selects a fusion scale using only the validation split. Re-run `run_benchmark.py` with the selected value to evaluate the test split; the bundled scale 0.5 improved all five tracked metrics on validation, but the test result still had a small PSNR regression, so do not describe that modular pipeline as a consistent all-metric win.

For container use, run `docker build -t lumaforge-isp .` and `docker run --rm -p 8501:8501 lumaforge-isp`. For a quick source bundle, run `python scripts/package_submission.py`.

`scripts/evaluate.py` defaults to the bundled `data/synthetic_pairs/test` split. For another paired dataset, pass either its test split or dataset root, for example `python scripts/evaluate.py --data .\data\your_camera_pairs\test`.

Optional capabilities (DNG, LPIPS, ONNX Runtime, FLOPs, plots) depend on optional packages and are reported unavailable when they are not installed. Measurements are actual local runs; CPU timing is not a mobile-phone latency claim. The project reports whether a dataset is synthetic and does not claim the hybrid beats the baseline when measured results do not support that claim.

## Current demo limitations

The bundled generated dataset is deliberately small. Its model is suitable for exercising the application, not for a research-quality camera result. The default CCM/noise recipe is generic; credible colour Delta E, robustness, and device efficiency require real paired scenes, sensor calibration, a fixed target colour space, and measurements on the target phone. Static quantization and hardware-sensitive paths should be measured on the deployment runtime before making performance claims.

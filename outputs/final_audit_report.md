# HybridISP Studio — Final Build and Audit

Audit date: 2026-11-07  
Project root: `C:\Users\pavan\Documents\Codex\2026-11-06\problem-statement-2-probid-p2-intelligent`

## Executive result

The software prototype is built and its unit suite passes. **The full research/hackathon acceptance criteria are not met**, because the provided archive is a payment-fraud dataset with no camera images or RAW/RGB pairs. Camera-specific model training and real-camera quality claims cannot be completed from that archive. This report separates verified software capabilities from dataset-blocked claims.

## Acceptance audit

| Requirement | Status | Evidence / limitation |
|---|---|---|
| Streamlit image upload, automatic processing and final review export | PASS | Root `app.py` processes uploads without a separate run action, uses the requested four columns, automatically generates Huawei-style and Canon-style simulated previews, and exports JSON, HTML and PDF review reports. A separate real aligned reference gates quality scores. |
| Bayer RAW and RGB ingestion | PARTIAL | Root UI accepts 2D mosaics and expands row-major 4-plane NPY input; RGGB/BGGR/GRBG/GBRG are selectable. DNG decoding requires optional `rawpy`, and a true DNG sensor-CFA/black/white metadata path is not fully automatic. |
| Classical ISP stages | PASS / PARTIAL | Modular pipeline includes black/white normalization, WB, demosaicing, CCM, denoising, tone mapping, gamma and sharpening. Actual sensor calibration and colour-chart validation are absent. |
| Learned ISP models and training workflow | PASS for synthetic demo | Live app TinyISP: 50-epoch run (best epoch 32), 921,003 parameters, 3.70 MB. Modular Hybrid Residual Refiner: 80 cumulative epochs, 17,379 parameters, 0.162 MB candidate; Mobile U-Net: 30 epochs, 18,008 parameters, 0.165 MB. All were trained on synthetic pairs. |
| Fine-tuning on supplied camera dataset | BLOCKED | Uploaded ZIP contains six FraudxAI transaction CSV tables and documentation; zero image files and zero aligned camera pairs. No external dataset was downloaded. |
| Adaptive noise and illumination handling | PASS at prototype level | Controller estimates noise, brightness, coarse CCT and WB gains and selects bounded processing policy; unit tests cover policy output and adaptive CCM path. |
| Traditional vs learned vs hybrid metrics | PARTIAL | Benchmark exists for three held-out synthetic scenes. LPIPS was unavailable (`null`). Results do not show the hybrid beating the baseline on all metrics. |
| Optimization: FP16, INT8, pruning, distillation | PARTIAL | FP16 export, INT8 quantization, pruning and distillation implementations exist; optimization smoke tests pass. A full accuracy/latency Pareto evaluation of all variants was not produced. PyTorch emitted quantization deprecation and numerical-accuracy warnings. |
| Efficiency: latency, memory, FLOPs, target-phone timing | PARTIAL | Efficiency utilities exist, but this audit did not verify the required 256/512/1080p warm-up + 50-run sweep, peak-memory table, or measurements on a target phone. |
| Robustness and ablation dashboards | PARTIAL | Streamlit pages and stress-test logic exist. No real-camera degradation/recovery study is available without paired data. |
| PDF/HTML/CSV report export | PARTIAL | Benchmark report export exists; per-upload image review exports JSON/HTML/PDF and output PNGs. Reports correctly avoid quality claims without aligned ground truth. |
| Docker, CI, style/lint acceptance | NOT VERIFIED | The repository has a `pyproject.toml` and unit tests. Docker and CI workflow files were not found during this audit. |

## Measured benchmark (synthetic only)

Held-out generated scenes: **3**, CPU, synthetic dataset. This modular benchmark predates the separate root-app image-path correction measured below.

| Pipeline | PSNR ↑ dB | SSIM ↑ | CIEDE2000 ↓ |
|---|---:|---:|---:|
| Traditional | 11.9243 | 0.3798 | 22.8478 |
| Hybrid | 11.8953 | 0.3874 | 22.4007 |
| Pure learned | 9.6240 | 0.2481 | 28.1170 |

Hybrid vs traditional: PSNR **−0.0290 dB**, SSIM **+0.0076**, CIEDE2000 **−0.4471**. These generated scenes are too few and too synthetic to support camera-quality conclusions. LPIPS values are unavailable in the saved benchmark.

Full measurements: `outputs/final_synthetic_benchmark/benchmark.json` and `outputs/final_synthetic_benchmark/benchmark.csv`.

## Verification run

- `python -m unittest discover -s tests -v`: **36 tests passed** (117.2 seconds). PyTorch quantization warnings were emitted.
- `python -m compileall -q app isp data scripts src`: passed.
- RAW pipeline smoke check: 64×64 16-bit Bayer mosaic produced finite 64×64×3 output; PDF report exporter produced a PDF signature.
- Continued the synthetic Hybrid Residual Refiner for 50 additional epochs from its prior 30-epoch checkpoint. Validation loss decreased 16.2%; the candidate remains separate because held-out PSNR declined while SSIM/MS-SSIM/colour error improved. See `outputs/training_followup_report.md`.
- Trained the TinyISP checkpoint used by the root app for 50 epochs on the same generated pairs; best validation L1 was 0.10710 at epoch 32. On three generated test scenes, PSNR/SSIM/CIE76 improved from 14.6957/0.6208/28.2080 for the prior checkpoint to 16.3242/0.7289/21.8612. CPU hybrid median latency was 187.3 ms at the dataset's 128×128 scene resolution. See `outputs/strong_synthetic_training/eval_trained_isp_tiny.json`.
- Streamlit version: 1.66.1.
- Live Chrome page at `localhost:8504` was inspected. The uploaded 447×447 JPEG was detected as RGB; all four requested panels, both automatic style previews, the report expander, and PDF download action were visible. The learned stage ran. No app exception was visible. Generated camera-style images are marked as simulations and are not used as ground truth.
- Streamlit UI uses Times New Roman CSS and the wine palette from `.streamlit/config.toml`; the dependency-free PDF uses Times-Roman.
- Frontend visual audit: fixed Material Symbols being overridden by the global Times font (which exposed strings such as `keyboard_double_arrow` and `upload`); constrained the sidebar and content width responsively; reduced oversized title/heading scale; restyled info notices and uploader controls to match the wine palette. Refreshed the live `localhost:8504` page and confirmed it is rendering with no visible app exception.
- ISP controls now use persistent explicit widget keys and two-decimal formatting. Verified intended defaults: WB R/G/B = 1.50/1.00/1.80, exposure = 1.25, denoise = 0.08, and detail recovery = 0.25. Already-rendered RGB receives neutral WB gains by default when adaptive WB is off; Bayer RAW uses the displayed sensor gains.

## Run instructions

From the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The multipage dashboard can be launched with `python -m streamlit run app/streamlit_app.py`.

## Required next input for a full research-grade completion

Provide the intended camera dataset containing paired, spatially aligned Bayer RAW and target RGB images, plus CFA pattern, black/white levels and target colour-space metadata. Then retrain and rerun scene-disjoint evaluation, robustness, optimization and efficiency sweeps. Until then, synthetic-trained checkpoints are demo artifacts only and no statement that HybridISP is superior on smartphone images is warranted.

## Final readiness check (2026-11-07)

## Problem-statement rubric audit

| Rubric area | Implementation present | Audit status | Evidence and limit |
|---|---|---|---|
| A. Classical CV | Yes | Implemented; sensor calibration partial | Bayer normalization, all four CFA demosaicing patterns, WB, CCM, bilateral denoise, tone mapping/gamma, sharpening and colour-space utilities exist; unit tests cover key stages. No target-camera calibration or colour-chart validation. |
| B. Deep learning | Yes | Demo trained; camera adaptation blocked | Tiny U-Net, Mobile U-Net, PyNET-lite and residual refiner code plus training/finetune scripts exist. Active TinyISP checkpoint has 921,003 parameters / 3.70 MB and was trained on synthetic pairs only. |
| C. Hybrid ISP | Yes | Implemented and measured only on synthetic data | Classical front end, learned refinement, fusion and traditional/learned/hybrid benchmark exist. App uses learned refinement for Bayer RAW; it intentionally skips the Bayer-trained checkpoint on rendered RGB. |
| Adaptive noise/illumination | Yes | Prototype implemented | Controller estimates noise, brightness, coarse colour temperature and WB; the Robustness Lab can sweep controlled degradations. No results on real camera captures. |
| RAW-to-RGB | Yes | Supported with metadata limits | 2D Bayer, four-channel packed NPY and DNG where `rawpy` is installed. CFA is selected by user; DNG metadata-to-pipeline wiring is not fully automatic. |
| Model optimization | Code exists | Partial evidence | FP16, INT8, pruning and distillation utilities are implemented and smoke-tested. No complete accuracy/latency/Pareto report across all variants; active FP32 checkpoint exceeds the requested <0.5M parameter target. |
| Quality metrics | Yes | Partial | PSNR, SSIM, CIE76, CIEDE2000 and MS-SSIM are implemented. LPIPS is unavailable in saved runs. Current evaluation covers only 3 synthetic test scenes. |
| Efficiency metrics | Yes | Partial | Model size, parameter count and CPU latency are saved for the synthetic run. Peak memory is `null`; required 256/512/1080p and target-phone benchmarks were not completed. |
| Robustness and ablation | Dashboard pages present | Partial | Robustness and classical-stage ablation logic is implemented in Streamlit pages; no standalone `isp/evaluation/robustness.py` or `ablation.py` modules and no real-image results. Both require RAW input; ablation also requires an aligned reference. |
| Deliverable / visual comparison | Yes | Demo, not camera-validated | Streamlit upload, four-panel output and JSON/HTML/PDF review exports work. Huawei/Canon-style outputs are simulated previews, not real camera photos or ground truth. |

**Rubric verdict:** software/demo artifacts exist for most stages, but the camera-quality deliverable is **not fully satisfied**. The supplied FraudxAI archive has zero image pairs; it cannot support real-camera training, target-device efficiency or camera-quality claims.

- Supplied archive inspection: 9 files (six CSVs, README and checksum files), 0 image pairs. It is not an ISP training dataset. The live Dataset & Training page reports 0 train/validation/test pairs and correctly disables Start training.
- Available training data: only the generated demo set, 30 synthetic RAW/RGB pairs (24 train / 3 validation / 3 test). The saved TinyISP checkpoint exists at `models/isp_tiny.pt` (921,003 parameters; 3,704,373 bytes, FP32); this is demo training, not institute-data training.
- Output/metric audit: synthetic test benchmark uses only 3 scenes. Hybrid mean PSNR is 11.8953 dB vs 11.9243 dB traditional (−0.0290 dB); SSIM is 0.3874 vs 0.3798; CIEDE2000 is 22.4007 vs 22.8478. Thus Hybrid does not improve every required quality metric. LPIPS is unavailable. Camera previews generated by the app are simulations and are not references.
- Current code check: targeted Python compilation passed; full regression suite passed with 39 tests (latest run: 80.844 seconds). Existing PyTorch quantization deprecation/numerical-accuracy warnings remain.
- Image-path correction: the root app now uses normalized bilinear demosaicing for all four selectable Bayer patterns; four-plane row-major packed NPY mosaics are unpacked instead of truncated to RGB; integer and float rendered RGB values are normalized, decoded from sRGB, processed in linear light, and encoded back without applying the RAW ACES curve twice. Checkpoint ISP metadata no longer overrides the active UI sliders. The Bayer-trained checkpoint is skipped on rendered RGB uploads.
- Live-site recheck after the code reload: `localhost:8504` displays the ISP Studio with no Streamlit traceback; the Bayer-pattern selector and automatic white-level expansion control are present.
- Final live review: before the last caption correction, the active 447×447 RGB upload showed all four output panels, the visual-review disclaimer, and PDF report action without errors. The caption now accurately says the Bayer-trained model is skipped on RGB; after refreshing to apply that text update, the uploader cleared as expected, so upload the image again to see its panels.
- Regression and current root-app synthetic evaluation after the image-path correction: **39 tests passed**; RGB and 4-channel Bayer smoke inputs produced finite outputs at expected dimensions. On the same 3 synthetic held-out scenes, the root app's classical output measured 13.1239 dB PSNR / 0.6369 SSIM / 38.4805 mean CIE76; hybrid measured 15.3253 dB / 0.7197 / 26.7335. The no-argument `python scripts/evaluate.py` completes from both the repository root and a different working directory using `data/synthetic_pairs/test`; observed median CPU latency ranged from 159–186 ms at 128×128 across runs. Results are synthetic-only; LPIPS and real phone latency remain unmeasured. See `outputs/evaluation.json` for the latest run, including dataset and checkpoint provenance.
- Evaluation path fix: app defaults now prefer the bundled synthetic dataset when no valid uploaded paired root exists; the evaluation control falls back to a real on-disk test split. CLI accepts a split or its dataset root via `--data` or `--data_root`, resolves project-relative paths regardless of current directory, and reports an actionable message for missing/unpaired folders. Three dedicated path tests pass.
- Training/data recheck: default synthetic layout is valid with 24 train, 3 validation and 3 test pairs. A synthetic ZIP round-trip extracted safely and was discovered at its nested root with 24/3 train/validation pairs. A two-epoch CPU training smoke produced `outputs/training_smoke/isp_tiny.pt` and a 2-row history (`best val L1=0.17789`) without replacing `models/isp_tiny.pt`; the active checkpoint remains epoch 32 with best validation L1 0.10710, trained on the same synthetic data. Eight dataset, ZIP safety and evaluation path tests passed. The supplied fraud archive still has zero image pairs and cannot be used for ISP training.
- Training dashboard now displays the selected data root and checkpoint provenance, warns when the existing model came from another data root, and explains why training is disabled when aligned pairs are absent. After refreshing the live app, it loaded without a Streamlit traceback; file uploads clear on a full browser refresh and must be selected again.
- Verdict: runnable demo build, **not a complete or validated camera ISP submission yet**. The blocker is the absent paired Bayer RAW / RGB camera dataset; code cannot safely manufacture real sensor pairs or honest camera-quality results from the fraud tables.

## Follow-up work (2026-11-07)

- Added `scripts/benchmark_efficiency.py`, which measures the actual tiled model inference path at 256×256, 512×512 and 1080p, with configurable warm-up and timed runs (defaults: 10 and 50). It records parameter count, checkpoint size, latency mean/p50/p95, Python allocation peak, and host process RSS/peak RSS on Windows (or psutil-enabled platforms). CUDA allocation is included when available. This is a host benchmark, not a handset estimate.
- Smoke-ran the new profiler at 256×256 with one timed run: FP32 model, 921,003 parameters, checkpoint 3.533 MiB, latency 251.9 ms; host RSS rose from 217.5 MiB to a 304.0 MiB process peak during the run. This verifies the measurement path, but **does not satisfy** the full 10+50-run multi-resolution report. Run `python scripts/benchmark_efficiency.py` for the rubric-default sweep; the 1080p case can take several minutes on CPU.
- Completed the full configured 10-warm-up/50-run CPU measurements at 256×256 and 512×512. 256: mean 262.7 ms, p50 261.6 ms, p95 280.3 ms. 512: mean 958.8 ms, p50 968.5 ms, p95 1,053.1 ms. Process RSS peaked at 527.8 MiB during the run; Python allocation peaks were 3.29 MiB and 13.04 MiB respectively. Details are in `outputs/efficiency_benchmark_256_512.json`. The rubric's 1080p measurement remains pending.
- Ran one 1080p (1920×1080) tiled FP32 CPU smoke measurement: 9,695 ms, with process peak RSS 597.0 MiB. It is a single run without warm-up, not the rubric's full 10+50 measurement. See `outputs/efficiency_benchmark_1080p_smoke.json`.
- Exported an FP16 checkpoint to `outputs/isp_tiny_fp16.pt` (1,859,445 bytes / 1.773 MiB) and compared FP32 vs FP16 on the same three synthetic test scenes. Mean PSNR/SSIM/CIE76 were 15.3253 / 0.719690 / 26.73347 for FP32 and 15.3264 / 0.719672 / 26.73249 for FP16. These metrics are effectively unchanged within this tiny synthetic set. Single-inference median timing was 79.2 ms FP32 vs 14,734.6 ms FP16 on CPU; FP16 is therefore a storage experiment only on this host and is not recommended for CPU inference. Full per-scene output is `outputs/fp16_comparison.json`.

## Delivery and synthetic study follow-up

- Added `scripts/run_synthetic_studies.py` to run four controlled raw-domain stress sweeps (read/shot noise, exposure, colour temperature, WB error) through Traditional and Hybrid modes, plus six classical-stage ablation variants. It writes per-image rows and mean ± standard deviation CSV/JSON under `outputs/synthetic_studies/`. These are synthetic-only, 3-scene results. The app's camera-image studies remain unrun because no camera RAW/target pair was supplied.
- The study reports include PSNR, SSIM, MS-SSIM (the available perceptual proxy), CIE76 and CIEDE2000. LPIPS package/weights are unavailable in this environment, so no LPIPS result is claimed.
- Added Dockerfile, `.dockerignore`, GitHub Actions CI workflow, and `scripts/package_submission.py`. Docker is not installed on this host, so the image build was not locally verified; the workflow has not run on GitHub. Final source ZIP generation and test-suite check remain pending after the full 1080p benchmark completes.
- Completed the 1080p 10-warm-up/50-run tiled CPU benchmark using 8 intra-op threads: mean 9,995.5 ms, p50 9,747.6 ms, p95 11,386.0 ms; process peak RSS 613.4 MiB; Python allocation peak 63.8 MiB. Full measurements are in `outputs/efficiency_benchmark_1080p.json`. This is host CPU performance and is not a mobile-device latency estimate.
- Added MS-SSIM and CIEDE2000 to the root evaluation output; LPIPS remains null because the optional package and pretrained weights are unavailable. On the three synthetic scenes, Hybrid measured 15.3253 dB PSNR, 0.71969 SSIM, 0.81885 MS-SSIM, and 15.9137 mean Delta E 2000. Classical measured 13.1239 dB, 0.63689, 0.76264, and 18.4516 respectively. These values remain demo-only.
- Completed synthetic robustness sweeps and classical-stage ablation. At read/shot sigma 0.005→0.10, hybrid mean PSNR fell 15.56→14.17 dB and SSIM 0.72→0.54 across three scenes. Ablation found disabling tone mapping strongly degraded results (12.08→7.71 dB PSNR), while disabling CCM improved the generated-set PSNR (12.08→13.60 dB); this shows the generic CCM needs calibration and is not universally beneficial. Full details: `outputs/synthetic_studies/robustness.{json,csv}` and `ablation.{json,csv}`.
- Re-ran delivery validation after changes: `python -m compileall -q app isp data scripts src` passed; `python -m unittest discover -s tests -v` passed **39 tests** in 53.4 seconds. CI workflow YAML parses locally. Docker image build and remote GitHub Actions execution remain unverified because Docker is absent and no remote CI run was triggered.
- Added a repeatable source/demo/report ZIP creator. The final package is generated at `outputs/LumaForge_P2_Submission.zip`; it excludes the unrelated FraudxAI archive and carries the synthetic dataset/checkpoints with explicit synthetic-only reports.
- Latest primary-app paired test comparison has three generated test scenes. Traditional → Hybrid: PSNR 13.1239 → 15.3253 dB (+2.2014 dB); SSIM 0.63689 → 0.71969 (+0.08280); MS-SSIM 0.76264 → 0.81885 (+0.05621); CIE76 38.4805 → 26.7335 (−11.7470); CIEDE2000 18.4516 → 15.9137 (−2.5379). Each of those three scenes individually improved on these metrics. LPIPS is null; MS-SSIM is the available perceptual proxy. Results are saved with per-scene values, standard deviations, and deltas in `outputs/evaluation.json`.
- This synthetic result does not erase the older modular benchmark result (`outputs/final_synthetic_benchmark/benchmark.json`), where the Hybrid variant was 0.029 dB below Traditional PSNR while slightly improving the other reported metrics. The repository currently has two evaluation paths and they produce different results; neither supports a general real-camera improvement claim. Treat the root-app comparison as the current app's result and the modular table as a separate legacy experiment.
- Final data inventory confirms the only non-synthetic imagery is absent: `data/incoming_archive6/` contains six FraudxAI CSVs and documentation; `data/synthetic_pairs/` contains 30 generated pairs (24/3/3). Thus camera-data training and camera-quality validation remain blocked, not completed.
- Added validation-only fusion tuning (`scripts/tune_fusion.py`). Scale 0.5 improved all five metrics on the three synthetic validation scenes. On the held-out synthetic test scenes, the tuned modular hybrid improved SSIM/MS-SSIM/Delta E but remained 0.0107 dB below Traditional PSNR. The result is in `outputs/tuned_synthetic_benchmark/`; tuning does not resolve the cross-scene all-metric generalization gap.
- The current root-app evaluation separately improves all five tracked metrics on its three synthetic test scenes (see `outputs/evaluation.json`). Since this differs from the modular benchmark, the repo does not yet demonstrate consistent results across both implementations. Do not claim a general hybrid win until the evaluation paths are unified and the result is reproduced on real camera pairs.
- The active model still exceeds the requested <0.5M parameter and <2 MB INT8 targets. Reducing its width or quantizing it without measuring quality would be an unvalidated tradeoff; the next optimization step must produce a compact candidate and score it on the held-out split.

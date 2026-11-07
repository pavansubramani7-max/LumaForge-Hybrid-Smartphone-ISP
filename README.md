# LumaForge — Hybrid Smartphone ISP

LumaForge is a Streamlit image-processing project for reconstructing display-ready RGB images from Bayer RAW sensor measurements. Its computer-vision approach combines a classical image signal processing (ISP) pipeline with compact learned restoration models, adaptive heuristics, and tools for paired-image evaluation. The project also includes a photo-editing studio; its creative looks are a separate post-processing feature and are not part of the RAW reconstruction model. 

**PROTOTYPE LINK**  https://lumaforge-hybrid-smartphone-isp-2ytsamtbpkoxqogspaqsdp.streamlit.app/

> **Evidence and dataset status.** The supplied `archive (6).zip` was inspected and contains FraudxAI transaction CSV files, not photographs or sensor data. It contains no Bayer RAW, RGB targets, or aligned RAW/RGB training pairs. The included demo pairs are generated procedurally and exist to exercise software paths. Results from those pairs are synthetic-only and do not establish camera quality or superiority over a phone ISP.

## 1. What the application does

The project offers two Streamlit entry points:

| Entry point | Intended use | Processing implementation |
| --- | --- | --- |
| `app.py` | Main upload studio, automatic image review, train/evaluate controls, image export | Lightweight pipeline in `src/isp/classical.py`; optional `src/isp/model.py` checkpoint for Bayer RAW |
| `app/streamlit_app.py` | Multi-page technical dashboard: overview, ISP playground, comparisons, robustness, efficiency, ablations and reports | Modular pipeline in `isp/classical/`, `isp/learned/`, `isp/hybrid/`, and `isp/evaluation/` |

The uploaded file is inspected for dimensions, array type, channel count and storage depth. A single-channel mosaic can be processed as Bayer RAW when the user selects the RAW input path and supplies the correct pattern and sensor levels. RGB/JPEG input is treated as already demosaiced and display-referred; Bayer demosaicing and camera-to-RGB colour calibration cannot be recovered from an ordinary JPEG.

The main studio presents four columns: a visualized input, reconstructed RGB, and two automatically rendered camera-style previews. The Huawei P20 and Canon panels are deterministic **style simulations**, not actual photos captured by those cameras and not ground truth. For measured quality scores, supply a spatially aligned RGB reference. The report records when a reference was absent and does not claim an objective quality score in that case.

## 2. Computer-vision problem formulation

A Bayer sensor measures only one colour component at each photosite. For a common RGGB colour filter array (CFA), the 2 × 2 sampling tile is:

```text
R G
G B
```

For a scene radiance field \(S\), the sensor samples one component per pixel and adds sensor effects. A simplified measurement model is:

\[
y_p = \operatorname{clip}\bigl(g_p\,S_{c(p)}(p) + n_{shot}(p) + n_{read}(p),\; b,\; w\bigr),
\]

where \(c(p)\) is the CFA colour at pixel \(p\), \(g_p\) includes exposure and shading, \(b\) and \(w\) are black and saturation levels, shot noise is signal-dependent, and read noise is approximately signal-independent. Reconstruction must infer the two missing colour channels at every pixel while correcting sensor scaling, colour response, noise and display encoding.

### RAW normalization and CFA handling

`isp/raw_utils.py` supports `.npy`, grayscale PNG/TIFF and optional DNG via `rawpy`. It normalizes sensor values as

\[
x = \operatorname{clip}\left(\frac{y-b}{w-b},0,1\right),
\]

using supplied black/white levels when known. The known camera metadata should take precedence over statistical guessing. The Bayer-pattern estimator is intentionally labelled heuristic and has modest confidence; scene colours and texture can mislead it. Headerless `.raw` files need external width, height, bit depth, levels and CFA metadata. Four-plane packing stores the R, G1, G2 and B samples at half width and half height, which reduces network input dimensions while preserving the sensor samples.

The RAW visualization panel is a false-colour display of the sparse CFA planes. Its purpose is to show the input mosaic; it is not a demosaiced reconstruction or evidence that missing colours were measured.

### Demosaicing: sparse sampling to dense RGB

`isp/classical/demosaic.py` implements three interpolation strategies:

* **Bilinear / normalized convolution:** for each colour plane, interpolate the masked sensor samples and divide by the interpolated mask. Normalizing the numerator by the mask avoids darkening around missing samples and reflected borders. Original measured CFA samples are copied back exactly.
* **Edge-directed green interpolation:** at red/blue sites, estimate horizontal and vertical gradients and interpolate green along the lower-gradient direction. This reduces colour zippering near some edges compared with isotropic interpolation; the chroma planes use the bilinear estimates.
* **Malvar–He–Cutler (MHC):** apply 5 × 5 linear filters specialized for green, red/blue-at-green and opposite-colour-at-red/blue reconstruction. It uses local spatial correlation and can recover sharper detail than basic bilinear interpolation, but it is still a fixed linear method and can show artifacts on strong edges or unusual CFA data.

The compact main-app baseline uses its own pattern-aware bilinear implementation in `src/isp/classical.py`. The modular dashboard exposes the three demosaicers above. For good results, Bayer pattern and RAW levels must be correct; a wrong pattern produces systematic colour errors that downstream filters cannot reliably repair.

### White balance and colour correction

White balance (WB) compensates for the illuminant and sensor channel sensitivities. The classical module offers:

* **Gray-world:** equalizes channel means, assuming average scene reflectance is neutral.
* **Shades-of-gray:** equalizes channel Minkowski norms, interpolating between average and bright-pixel assumptions.
* **White-patch / max-RGB:** estimates the illuminant from high channel percentiles, assuming some scene region is close to white.
* **Manual gains:** uses selected red, green and blue gains or metadata-provided as-shot gains.

These are scene assumptions, not universal illuminant detectors. A monochromatic scene, coloured light, or large saturated object can bias automatic WB. The optional adaptive controller computes bounded gray-world-like gains as a per-image heuristic.

After WB, a 3 × 3 colour correction matrix (CCM) maps camera RGB to a target RGB basis:

\[
\mathbf{r}_{out}=\max(\mathbf{M}\,\mathbf{r}_{in},0).
\]

The code can interpolate named illuminant matrices and fit a matrix with ridge-regularized least squares from matched colour-chart patches. A generic default matrix is only an approximate starting point. Accurate colour requires a camera-specific calibration, known illuminant, a defined target space (normally linear sRGB/XYZ), and a valid chart or aligned colour reference. The generic matrix must not be described as camera-calibrated.

### Denoising and edge preservation

The modular classical path supports Gaussian smoothing, bilateral filtering, a compact non-local-means approximation, guided filtering and one-level Haar wavelet shrinkage. These methods encode different priors:

* Gaussian smoothing suppresses high spatial frequencies but blurs edges.
* Bilateral filtering weights neighbours by both spatial distance and colour similarity, smoothing within regions while reducing smoothing across strong edges.
* Non-local means averages similar patches beyond immediate neighbours, at greater computational cost.
* Guided filtering uses a local linear model guided by image luminance to preserve boundaries.
* Wavelet shrinkage thresholds high-frequency Haar coefficients, assuming small coefficients are more likely noise than structure.

Noise strength is a control, not a calibrated sensor-noise distribution. The modular pipeline applies stronger filtering to chroma in YCbCr than to luma, because chroma noise is often less perceptually objectionable and preserving luminance helps retain edge detail. Excessive filtering removes texture; weak filtering leaves noise. The right setting depends on sensor gain, exposure and noise calibration.

### Tone mapping, transfer functions and sharpening

Sensor-linear values are not directly suitable for display. The code supports fitted ACES-style, filmic and Reinhard tone curves plus the piecewise sRGB transfer function. Tone mapping compresses scene dynamic range into display range; it is distinct from demosaicing and colour correction. Direct RGB uploads are decoded from sRGB to linear light before physical-like operations and encoded for display afterwards in the modular pipeline.

Optional CLAHE (contrast-limited adaptive histogram equalization) operates on luminance for local contrast. It can reveal detail in shadows, but may amplify noise or create a locally unnatural appearance. Unsharp masking adds a scaled high-frequency residual to luminance:

\[
Y_{sharp}=Y + \lambda(Y-G_\sigma * Y),
\]

where \(G_\sigma * Y\) is a local blur and \(\lambda\) is sharpening strength. Luma-only sharpening avoids directly amplifying chroma noise. The edge-aware attenuation in the modular function moderates sharpening near large residuals to reduce halos; it does not guarantee artifact-free output.

## 3. Learned image restoration

`isp/learned/models.py` contains three models used by the modular dashboard and training scripts:

1. **MobileUNetISP:** encoder/decoder with skip connections. It accepts four packed Bayer planes and outputs full-resolution RGB. Multi-scale features provide both broad colour/illumination context and local texture detail.
2. **PyNETLite:** a compact multi-scale pyramid that progressively refines features from coarse to fine before producing RGB.
3. **HybridResidualRefiner:** receives an 8-channel feature tensor: classical linear RGB (3), resized packed RAW planes (4), and a noise-level map (1). It predicts a bounded correction to the classical estimate. The residual formulation makes the classical reconstruction an explicit starting point and constrains the network to learn corrections.

The modular trainer uses paired RAW/RGB data and combines pixel L1, multi-scale SSIM-style structure loss, Lab colour difference loss, and horizontal/vertical gradient loss. These terms encourage colour proximity, local structure and edge consistency. VGG perceptual loss is implemented as an optional term but the included training command disables it (`vgg=0`) to avoid external pretrained-weight dependencies. The optimizer is AdamW with gradient clipping, cosine learning-rate scheduling, exponential moving average weights, and CUDA automatic mixed precision when a CUDA device is available. CPU is supported without AMP.

The model adapts only to the data and target rendering it sees. Paired images must be geometrically aligned, correctly exposed/registered, in a consistent colour space, and split by **scene** rather than by crop to avoid leakage. Patch training increases sample variety but does not create new scenes. CFA-safe geometric augmentation repacks the transformed mosaic into semantic colour planes. Synthetic Poisson/read noise and WB/exposure/vignetting changes are optional training degradations; their parameters are approximations, not a measured camera sensor model.

The lightweight main app uses a separate residual U-Net in `src/isp/model.py`, which refines the main app's classical RGB output using the RGB estimate plus a noise scalar map. Tiled inference with overlapping regions and feathered blending bounds memory and reduces visible tile seams. It is loaded only for Bayer RAW when a compatible checkpoint exists. The app deliberately skips this synthetic RAW model for already-rendered RGB inputs, because that is an unvalidated input domain.

## 4. Hybrid ISP and adaptive control

The modular hybrid path runs deterministic, physically interpretable stages before the learned refiner and display rendering after it:

```text
Bayer mosaic
  → black/white normalization and dead-pixel correction
  → estimate noise / brightness / coarse illuminant
  → white balance → demosaic → CCM → light denoising
  → learned RAW-conditioned residual refinement
  → confidence-weighted blend with the classical linear estimate
  → tone map + sRGB encoding → luma sharpening
```

Keeping tone mapping and sharpening after the refiner lets the network work in a more linear, well-conditioned representation. Confidence fusion reduces the learned contribution where the learned output disagrees strongly with the classical estimate. In the implementation, confidence is `exp(-mean(abs(learned-classical))/(0.08 + 4*noise_sigma))`, multiplied by a configured blend amount. This is a discrepancy heuristic, **not** a probability calibrated against ground truth. A learned model can be wrong while agreeing with the baseline, and a useful correction can disagree with it.

The adaptive controller estimates a noise proxy from the median absolute deviation (MAD) of Laplacian responses on the four CFA parity planes; mean demosaiced brightness; gray-world-like WB gains; and a rough CCT using a simple RGB-to-XYZ transform and McCamy approximation. It maps those estimates through bounded rules to choose denoise strength, refiner blend, sharpening, an illuminant CCM preset and tone curve. These are lightweight heuristics to demonstrate adaptive control. CCT is not measured from a calibrated illuminant sensor, and the MAD estimate is not a validated photon-transfer noise calibration.

The dashboard can compare traditional, pure learned and hybrid processing for Bayer input when the relevant checkpoints are available. With RGB input, demosaicing is skipped and the current dashboard uses the classical RGB path; it does not claim a trained RAW model is a valid pure-learned RGB baseline.

## 5. Input data, pairing and reproducibility

### Supported paired dataset layout

`data/dataset.py` discovers unambiguous matching basenames in common RAW and target folders. A typical structure is:

```text
my_pairs/
├── raw/                 # Bayer .npy/.dng/gray .png or .tif
└── rgb/                 # aligned .png/.jpg/.tif target RGB
```

or:

```text
my_pairs/
├── scene_001_raw.npy
└── scene_001_rgb.png
```

For mosaic RAW, each target should correspond to the same scene and be registered to the RAW reconstruction. For DNG, install `rawpy`; for headerless `.raw`, provide metadata and adapt the loader. Set `data.data_root`, Bayer pattern, black level and white level in `configs/default.yaml`, or pass `--data_root` on commands that support it. Pattern inference is a fallback; confirm it from camera metadata.

The synthetic generator creates 30 simple rendered scenes by default: 24 train, 3 validation and 3 test. It constructs gradients, shapes and detail bars, converts an RGB target to linear light, mosaics RGGB samples, and adds Poisson-like shot noise and Gaussian read noise. This is useful for smoke tests and pipeline mechanics. It is not a natural-image dataset and its small test split cannot support broad quality conclusions.

### Inspect and generate data

```powershell
python scripts/make_synthetic_dataset.py --out data/synthetic_pairs
python scripts/inspect_dataset.py --data_root data/synthetic_pairs --output outputs/dataset_sample_grid.png --json_output outputs/dataset_inspection.json
```

The inspector reports candidate pairs, shape/dtype and pattern evidence and saves a RAW-preview/target grid. A heuristic pattern guess should be confirmed from metadata. The dataset loader writes a seeded 80/10/10 split manifest (`splits.json`) when one is not present. Keep test scenes held out until final evaluation; do not tune fusion or hyperparameters on the test set.

## 6. Evaluation concepts and honest interpretation

`scripts/evaluate.py` and the modular benchmark compare outputs with aligned RGB targets. Metrics answer different questions:

| Metric | What it measures | Better direction | Important caveat |
| --- | --- | --- | --- |
| PSNR (dB) | Pixelwise mean squared error on normalized RGB | Higher | Strongly penalizes small spatial shifts; not a perceptual score |
| SSIM | Local luminance, contrast and structural similarity | Higher | Depends on windowing and implementation; not a colour-accuracy metric |
| MS-SSIM | Structural similarity across downsampled scales | Higher | Complements but does not replace full-resolution inspection |
| CIE76 ΔE | Euclidean distance in CIELAB | Lower | Simple colour difference; does not model all perceptual nonuniformity |
| CIEDE2000 ΔE | Perceptually adjusted Lab colour difference | Lower | Meaningful only with appropriate colour management and matched images |
| LPIPS (AlexNet) | Deep-feature perceptual distance | Lower | Optional dependency; unavailable when `lpips` is not installed |

Report the number of independent scenes, mean and standard deviation, image alignment, target rendering/colour space, and whether a metric was unavailable. Never compare metrics across different crops, resizing, target colour spaces or registration conditions. A lower ΔE against a Canon-rendered target means closer to that target rendering, not universally more accurate scene colour.

Efficiency tools report parameter counts, checkpoint size, latency and available memory information. Latency depends on image resolution, CPU/GPU, thread count, warm-up, tiling, precision and runtime. Host CPU figures are not phone latency. RAM process readings and CUDA allocated memory are different measurements. Quantization, pruning, distillation and reduced precision are implemented as tools/paths; a performance claim requires exporting the artifact, running it on a named runtime/device and measuring both speed and quality after conversion.

Robustness sweeps apply controlled synthetic changes such as noise, exposure, white-balance/CCT error or blur. They test behavior under the implemented perturbation recipe, not real-world robustness unless the perturbations and camera data are representative. Ablations show how metrics change when selected stages are disabled for the given evaluation set; they do not prove causal benefit on other cameras.

## 7. Installation and run

Use Python 3.10+ and run commands from the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. To run the technical multi-page dashboard instead:

```powershell
python -m streamlit run app/streamlit_app.py
```

## 8. Train and evaluate

Run a brief synthetic smoke training job:

```powershell
python -m isp.learned.train --config configs/default.yaml --data_root data/synthetic_pairs --models mobile_unet hybrid_refiner --quick
```

Run configured training on your own paired camera data:

```powershell
python -m isp.learned.train --config configs/default.yaml --data_root .\data\your_camera_pairs --models mobile_unet hybrid_refiner
```

Use enough independent training and validation scenes, fix the target colour space and RAW metadata, and inspect the saved split manifest and training history. The `--quick` option reduces epochs, patch size and channel count; it is for a fast demonstration and not a strong camera model. Example benchmark and efficiency commands:

```powershell
python scripts/run_benchmark.py --data_root data/synthetic_pairs --device cpu --out_dir outputs/synthetic_benchmark
python scripts/run_synthetic_studies.py
python scripts/benchmark_efficiency.py --sizes 256 512
python scripts/evaluate.py --data .\data\your_camera_pairs\test
```

Use `python scripts/benchmark_efficiency.py --help` for the full-resolution/tiled options; 1080p timing is substantially slower. Results are written to the selected output directory. Some metrics and exports depend on optional packages and can report unavailable rather than fabricating a value.

## 9. Photo studio and generated looks

The main app includes multiple-photo selection, a library of World/Era/Cinema/Light looks, intensity and effect controls, undo/redo/reset, before/after viewing, synchronized zoom/pan and PNG downloads. Looks are deterministic colour and texture operations applied after ISP reconstruction. They do not simulate the physical capture conditions of a country, historical era, film stock, cinema camera or lighting setup. Keep the untreated reconstruction for ISP evaluation; evaluate creative grades separately.

The main upload workflow can export the reconstructed RGB and an automatic JSON/HTML/PDF review. Camera-style panels are clearly marked as simulated. Reference-based metrics are only attached when the user supplies a size-matched aligned target; the main app currently reports PSNR, SSIM and mean CIE76 ΔE for that optional reference.

## 10. Repository map

```text
app.py                         Main Streamlit photo studio and review
app/                           Multi-page dashboard, components, looks and reports
src/isp/classical.py           Main-app compact classical ISP
src/isp/model.py               Main-app tiled residual U-Net and inference
isp/raw_utils.py               Bayer loading, normalization, packing and CFA helpers
isp/classical/                 Modular demosaic, WB, CCM, denoise, tone, colour and sharpen stages
isp/learned/                   Models, data wrapper, degradation, losses, training and inference
isp/hybrid/                    Adaptive controller, fusion and modular hybrid pipeline
isp/metrics/                   Image-quality and efficiency metrics
isp/evaluation/                Benchmark, robustness and ablation workflows
isp/optimize/                  Precision, quantization, pruning, distillation and export tools
data/dataset.py                Pair discovery, scene split and paired RAW/RGB loader
scripts/                       Dataset tools, synthetic generation and evaluation launchers
configs/                       Pipeline, model, training and CCM configuration
models/                        Checkpoints when generated or supplied
outputs/                       Local reports, plots and benchmark artifacts
```

## 11. Known limits and responsible claims

* The supplied FraudxAI archive is unrelated to computer vision and was not used to train this ISP.
* The bundled training pairs are synthetic, simple and few in number. A model trained on them is a software demo, not a production camera reconstruction model.
* No claim of real camera adaptation, state-of-the-art output, or consistent improvement across PSNR, SSIM, ΔE and LPIPS is justified without a sufficiently large aligned camera test set and measured results.
* Generic WB and CCM defaults are not sensor calibration. RAW metadata and camera-specific colour calibration matter.
* The main app, modular research pipeline, and creative editing layer are separate components. The Huawei/Canon preview images and creative presets are generated transformations, not captured references.
* Performance numbers are specific to the machine, model artifact, image size, precision and runtime used. They do not establish mobile-device performance.

For a defensible camera-quality study, collect scene-disjoint paired Bayer RAW and target RGB, document sensor levels and target colour rendering, calibrate WB/CCM with a chart, include varied lighting and noise conditions, freeze the test split, report per-scene distributions, compare against a documented conventional ISP, and benchmark the exported model on the intended device.

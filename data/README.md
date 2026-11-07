# Dataset placement

Place the institute-provided dataset under this directory or pass another folder with `--data_root`.
Do not download additional datasets. `python scripts/inspect_dataset.py --data_root ./data` scans common
RAW/target layouts, reports candidate pairs and writes a visualization to `outputs/dataset_sample_grid.png`.

Supported direct RAW inputs: 2D `.npy`, 16-bit grayscale `.png`/`.tif`, and `.dng` when `rawpy` is
installed. `.raw` requires a sidecar or config specifying dimensions, bit depth and Bayer pattern.
Targets should be aligned RGB PNG/JPEG/TIFF images with matching filename stems. The loader also
accepts 4-channel packed RAW arrays. If folder naming or filenames differ, the inspector reports what
it found so the mapping can be configured rather than silently pairing unrelated images.

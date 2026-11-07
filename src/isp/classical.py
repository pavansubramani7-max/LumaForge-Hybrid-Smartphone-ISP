"""Small, dependency-light classical ISP stages for RGGB Bayer and RGB inputs."""
from __future__ import annotations

import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load RGB image or single-channel Bayer mosaic (PNG/TIFF/NPY)."""
    if path.lower().endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def normalize_raw(raw: np.ndarray, black: float = 0.0, white: float | None = None) -> np.ndarray:
    x = np.asarray(raw, dtype=np.float32)
    if white is None:
        white = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0
    elif not np.issubdtype(raw.dtype, np.integer) and float(np.nanmax(x)) <= 1.5 and white > 1:
        white = 1.0
    return np.clip((x - black) / max(white - black, 1e-8), 0, 1)


def demosaic_rggb(mosaic: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Normalized-convolution bilinear demosaic for a Bayer CFA pattern."""
    m = np.asarray(mosaic, np.float32)
    if m.ndim != 2:
        raise ValueError("Bayer input must be a 2D mosaic")
    patterns = {
        "RGGB": ((0, 1, 1), (1, 2, 1)),
        "BGGR": ((2, 1, 1), (1, 0, 1)),
        "GRBG": ((1, 0, 1), (2, 1, 1)),
        "GBRG": ((1, 2, 1), (0, 1, 1)),
    }
    pattern = pattern.upper()
    if pattern not in patterns:
        raise ValueError(f"Unsupported Bayer pattern: {pattern}")
    tile = patterns[pattern]
    # Bilinear weights depend on the sampling colour: red/blue use diagonal
    # weights and green uses the four-neighbour cross. Normalize at borders.
    from numpy.lib.stride_tricks import sliding_window_view
    result = []
    for c in range(3):
        parity = np.zeros_like(m)
        for y in range(2):
            for x in range(2):
                if tile[y][x] == c:
                    parity[y::2, x::2] = 1
        if c == 1:
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32)
        else:
            kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
        sparse = parity * m
        sparse_windows = sliding_window_view(np.pad(sparse, 1, mode="reflect"), (3, 3))
        mask_windows = sliding_window_view(np.pad(parity, 1, mode="reflect"), (3, 3))
        num = np.sum(sparse_windows * kernel, axis=(-1, -2))
        den = np.sum(mask_windows * kernel, axis=(-1, -2))
        interp = num / np.maximum(den, 1e-8)
        interp[parity > 0] = m[parity > 0]
        result.append(interp)
    return np.stack(result, axis=-1)


def unpack_bayer_4ch(packed: np.ndarray) -> np.ndarray:
    """Expand row-major packed Bayer planes (H/2,W/2,4) to a 2D mosaic."""
    packed = np.asarray(packed)
    if packed.ndim != 3 or packed.shape[-1] != 4:
        raise ValueError("Packed Bayer input must have shape (H, W, 4)")
    h, w, _ = packed.shape
    mosaic = np.empty((h * 2, w * 2), dtype=packed.dtype)
    mosaic[0::2, 0::2] = packed[..., 0]
    mosaic[0::2, 1::2] = packed[..., 1]
    mosaic[1::2, 0::2] = packed[..., 2]
    mosaic[1::2, 1::2] = packed[..., 3]
    return mosaic


def white_balance(rgb: np.ndarray, gains=(2.0, 1.0, 1.6)) -> np.ndarray:
    return np.maximum(np.asarray(rgb, np.float32) * np.asarray(gains, np.float32), 0)


def color_matrix(rgb: np.ndarray, matrix=None) -> np.ndarray:
    # Approximate camera-to-sRGB matrix; replace with metadata/calibration when available.
    if matrix is None:
        matrix = np.array([[1.48, -0.32, -0.16], [-0.12, 1.25, -0.13], [-0.02, -0.44, 1.46]], np.float32)
    return np.maximum(np.asarray(rgb, np.float32) @ np.asarray(matrix, np.float32).T, 0)


def denoise(rgb: np.ndarray, strength: float = 0.08) -> np.ndarray:
    """Edge-preserving bilateral filter when OpenCV exists; Gaussian fallback."""
    try:
        import cv2
        x = np.asarray(rgb, np.float32)
        return cv2.bilateralFilter(x, d=5, sigmaColor=strength, sigmaSpace=2.0)
    except ImportError:
        from PIL import ImageFilter
        image = Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255))
        return np.asarray(image.filter(ImageFilter.GaussianBlur(0.45)), np.float32) / 255


def tone_map(rgb: np.ndarray, exposure: float = 1.0, gamma: float = 2.2) -> np.ndarray:
    x = np.maximum(np.asarray(rgb, np.float32) * exposure, 0)
    # ACES fitted curve: preserves midtone contrast while compressing highlights.
    x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    x = np.clip(x, 0, 1)
    if abs(float(gamma) - 2.2) < 1e-6:
        return linear_to_srgb(x)
    return np.power(x, 1 / float(gamma))


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Decode display-referred sRGB to linear light before ISP operations."""
    x = np.clip(np.asarray(rgb, np.float32), 0, 1)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Encode linear-light RGB with the standard piecewise sRGB OETF."""
    x = np.clip(np.asarray(rgb, np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def sharpen(rgb: np.ndarray, amount: float = 0.25) -> np.ndarray:
    try:
        import cv2
        x = np.asarray(rgb, np.float32)
        blur = cv2.GaussianBlur(x, (0, 0), 1.0)
        return np.clip(x + amount * (x - blur), 0, 1)
    except ImportError:
        return np.asarray(rgb, np.float32)


def estimate_adaptive_controls(image: np.ndarray, input_kind="rgb", black=0, white=None, pattern="RGGB"):
    """Estimate neutral gains, exposure and noise from this frame (bounded heuristics)."""
    x = normalize_raw(image, black, white) if input_kind == "rggb" else np.asarray(image, np.float32)
    if x.ndim == 2:
        rgb = demosaic_rggb(x, pattern)
    else:
        if np.issubdtype(image.dtype, np.integer): x = x / float(np.iinfo(image.dtype).max)
        rgb = x[..., :3]
    means = np.maximum(np.mean(rgb, axis=(0, 1)), 1e-4)
    neutral = float(np.exp(np.mean(np.log(means))))
    gains = np.clip(neutral / means, 0.65, 2.5)
    lum = rgb @ np.array([.2126, .7152, .0722], np.float32)
    p95 = max(float(np.percentile(lum, 95)), 1e-3)
    exposure = float(np.clip(.78 / p95, .6, 1.8))
    # Robust high-pass MAD; heuristic proxy, not a calibrated sensor noise model.
    hp = lum - np.pad(lum, ((1, 1), (1, 1)), mode="reflect")[1:-1, :-2]
    sigma = float(np.clip(1.4826 * np.median(np.abs(hp - np.median(hp))), .005, .2))
    return tuple(float(v) for v in gains), exposure, sigma


def run_classical(image: np.ndarray, input_kind="rgb", black=0, white=None,
                  wb=(2.0, 1.0, 1.6), exposure=1.0, denoise_strength=0.08,
                  sharpen_amount=0.25, pattern="RGGB") -> np.ndarray:
    is_bayer_raw = input_kind == "rggb"
    if is_bayer_raw:
        rgb = demosaic_rggb(normalize_raw(image, black, white), pattern)
        rgb = white_balance(rgb, wb)
        rgb = color_matrix(rgb)
    else:
        rgb = np.asarray(image, np.float32)
        if np.issubdtype(image.dtype, np.integer):
            rgb /= float(np.iinfo(image.dtype).max)
        else:
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
            peak = float(np.max(rgb)) if rgb.size else 0.0
            if peak > 1.5:
                scale = 255.0 if peak <= 255 else (65535.0 if peak <= 65535 else peak)
                rgb /= scale
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        elif rgb.shape[-1] == 1:
            rgb = np.repeat(rgb, 3, axis=-1)
        if not np.isfinite(rgb).all():
            rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
        # Uploaded RGB files are already sRGB encoded; process them in linear
        # light, then tone-map and encode once on the way back out.
        rgb = srgb_to_linear(rgb)
        rgb = white_balance(rgb, wb)
    rgb = denoise(rgb, denoise_strength)
    if is_bayer_raw:
        rgb = tone_map(rgb, exposure)
    else:
        # A JPEG/PNG is already tone-mapped. Keep its tone curve and apply only
        # linear-light exposure, clipping at the display gamut before encoding.
        rgb = linear_to_srgb(np.clip(rgb * exposure, 0, 1))
    return sharpen(rgb, sharpen_amount)


def save_rgb(path: str, rgb: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5), "RGB").save(path)

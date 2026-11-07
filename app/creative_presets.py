"""Creative color looks layered after the existing ISP output."""
from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter


def _look(description: str, **values: Any) -> dict[str, Any]:
    return {"description": description, "temperature": 0.0, "saturation": 1.0,
            "contrast": 1.0, "fade": 0.0, "sepia": 0.0, "grain": 0.0,
            "scratches": 0.0, "halation": 0.0, "vignette": 0.0,
            "softness": 0.0, "shadow_tint": (0.0, 0.0, 0.0),
            "highlight_tint": (0.0, 0.0, 0.0), **values}


LOOKS: dict[str, dict[str, dict[str, Any]]] = {
    "WORLD": {
        "INDIA": _look("Golden warmth, earthy color and gentle highlight glow.", temperature=.55, saturation=1.12, contrast=1.05, halation=.04),
        "JAPAN": _look("Soft neutrals, muted color and bright, delicate highlights.", temperature=.04, saturation=.78, contrast=.92, fade=.05, softness=.08),
        "ITALY": _look("Mediterranean gold with nostalgic warmth.", temperature=.48, saturation=1.06, contrast=1.02, fade=.06),
        "ICELAND": _look("Cool blue color with quiet, cinematic contrast.", temperature=-.55, saturation=.78, contrast=1.04, vignette=.08),
        "MOROCCO": _look("Desert amber and sandy, earthy shadows.", temperature=.70, saturation=.92, contrast=1.08, fade=.05),
        "BRAZIL": _look("Tropical greens, vibrant color and warm sunlight.", temperature=.25, saturation=1.28, contrast=1.04),
        "SWITZERLAND": _look("Crisp whites, natural greens and clean contrast.", temperature=.02, saturation=1.04, contrast=1.12),
        "NORWAY": _look("Cool shadows, restrained color and a moody finish.", temperature=-.48, saturation=.75, contrast=1.08, vignette=.12),
    },
    "ERA": {
        "1910s": _look("Monochrome, soft focus and an antique fade.", saturation=0.0, contrast=.88, sepia=.22, fade=.16, softness=.18, scratches=.10),
        "1920s": _look("Monochrome contrast with fine film grain.", saturation=0.0, contrast=1.13, sepia=.10, grain=.12, scratches=.07),
        "1930s": _look("Early-color inspired hues with gently faded color.", saturation=.74, contrast=.91, fade=.14, grain=.04),
        "1940s": _look("Muted vintage color and restrained contrast.", saturation=.78, contrast=.94, fade=.10, grain=.035),
        "1950s": _look("Rich, warm film color with rounded highlights.", temperature=.22, saturation=1.14, contrast=1.08, halation=.04),
        "1960s": _look("Bold film color with visible texture.", temperature=.10, saturation=1.2, contrast=1.10, grain=.055),
        "1970s": _look("Warm, grainy tones with lifted shadows.", temperature=.40, saturation=.92, contrast=.91, fade=.17, grain=.10, vignette=.10),
        "1980s": _look("Saturated retro color and punchy contrast.", temperature=.05, saturation=1.30, contrast=1.14),
        "1990s": _look("Natural color with relaxed film-camera contrast.", temperature=.02, saturation=.96, contrast=.98, grain=.025),
        "2000s": _look("Cool early-digital character with crisp edges.", temperature=-.12, saturation=1.02, contrast=1.10),
        "2010s": _look("Clean, crisp digital color with balanced contrast.", saturation=1.04, contrast=1.06),
        "2020s": _look("Modern HDR-inspired detail and controlled highlights.", saturation=1.06, contrast=1.02, fade=.025, halation=.025),
    },
    "CINEMA": {
        "HOLLYWOOD": _look("Teal-shadow and warm-highlight inspired cinematic contrast.", temperature=.08, saturation=1.08, contrast=1.15, vignette=.12, halation=.04, shadow_tint=(-.025, .006, .035), highlight_tint=(.035, .012, -.025)),
        "KOLLYWOOD": _look("Expressive warm color, rich saturation and dramatic contrast.", temperature=.35, saturation=1.22, contrast=1.16),
        "SANDALWOOD": _look("Earthy greens and browns with deep, restrained shadows.", temperature=.16, saturation=.86, contrast=1.16, vignette=.09),
        "TOLLYWOOD": _look("Vibrant color and punchy highlights.", temperature=.12, saturation=1.25, contrast=1.18, halation=.035),
        "K-CINEMA": _look("Muted atmosphere and softly rolled highlights.", temperature=-.12, saturation=.76, contrast=.94, fade=.07, softness=.04),
        "EUROPEAN CINEMA": _look("Natural color, gentle contrast and film restraint.", temperature=.04, saturation=.82, contrast=.96, fade=.04, grain=.02),
        "DOCUMENTARY": _look("Neutral, realistic color with modest contrast.", saturation=.94, contrast=1.01),
        "ANIME-INSPIRED": _look("Vivid color separation with clean, controlled saturation.", temperature=.06, saturation=1.22, contrast=1.12),
    },
    "LIGHT": {
        "MORNING": _look("Soft, bright light with a hint of warmth.", temperature=.18, saturation=1.02, contrast=.96, halation=.025),
        "DAYLIGHT": _look("Neutral whites and balanced color.", temperature=0.0, saturation=1.0, contrast=1.0),
        "AFTERNOON": _look("Warm sunlight and richer natural shadows.", temperature=.30, saturation=1.08, contrast=1.05),
        "GOLDEN HOUR": _look("Amber highlights and gentle golden contrast.", temperature=.62, saturation=1.12, contrast=.98, halation=.06),
        "EVENING": _look("Orange warmth and a more atmospheric shadow range.", temperature=.48, saturation=1.04, contrast=1.12, vignette=.07),
        "BLUE HOUR": _look("Cool blue color and deep, quiet tones.", temperature=-.62, saturation=.80, contrast=1.10, vignette=.08),
        "NIGHT": _look("Cool shadows, controlled color and a moody finish.", temperature=-.38, saturation=.82, contrast=1.16, vignette=.16),
    },
}


def apply_look(image: np.ndarray, preset: dict[str, Any] | None, intensity: float = 1.0,
               adjustments: dict[str, float] | None = None) -> np.ndarray:
    """Apply a deterministic creative grade to a fresh ISP output, never a prior look."""
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 3 or source.shape[-1] < 3:
        raise ValueError(f"Creative looks require an H×W×3 image; received {source.shape}")
    source = np.clip(np.nan_to_num(source[..., :3], nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    if not preset or intensity <= 0:
        return source.copy()
    amount = float(np.clip(intensity, 0.0, 1.0))
    cfg = {**preset, **(adjustments or {})}
    out = source.copy()
    lum = out @ np.array([.2126, .7152, .0722], dtype=np.float32)
    sat = float(cfg.get("saturation", 1.0))
    out = lum[..., None] + (out - lum[..., None]) * (1.0 + (sat - 1.0) * amount)
    temp = float(cfg.get("temperature", 0.0)) * amount
    out *= np.array([1.0 + .20 * temp, 1.0 - .025 * abs(temp), 1.0 - .18 * temp], dtype=np.float32)
    shadow_tint = np.asarray(cfg.get("shadow_tint", (0.0, 0.0, 0.0)), dtype=np.float32)
    highlight_tint = np.asarray(cfg.get("highlight_tint", (0.0, 0.0, 0.0)), dtype=np.float32)
    if np.any(shadow_tint) or np.any(highlight_tint):
        out += amount * ((1.0 - lum[..., None]) * shadow_tint + lum[..., None] * highlight_tint)
    contrast = 1.0 + (float(cfg.get("contrast", 1.0)) - 1.0) * amount
    out = .5 + (out - .5) * contrast
    fade = float(cfg.get("fade", 0.0)) * amount
    out = out * (1.0 - fade) + (.12 + .72 * out) * fade
    sepia = float(cfg.get("sepia", 0.0)) * amount
    if sepia:
        sepia_rgb = out @ np.array([[.393, .769, .189], [.349, .686, .168], [.272, .534, .131]], dtype=np.float32).T
        out = out * (1.0 - sepia) + sepia_rgb * sepia
    vignette = float(cfg.get("vignette", 0.0)) * amount
    if vignette:
        h, w = out.shape[:2]
        yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        radius = np.clip((xx * xx + yy * yy) / 2.0, 0.0, 1.0)
        out *= (1.0 - vignette * radius)[..., None]
    grain = float(cfg.get("grain", 0.0)) * amount
    scratches = float(cfg.get("scratches", 0.0)) * amount
    if grain or scratches:
        rng = np.random.default_rng(7341)
        noise = rng.normal(0.0, grain * .22, size=out.shape[:2]).astype(np.float32)
        if scratches:
            h, w = out.shape[:2]
            for x in rng.choice(w, size=max(1, int(w * scratches * .012)), replace=False):
                y0 = int(rng.integers(0, max(1, h // 3)))
                y1 = int(rng.integers(max(y0 + 1, 2 * h // 3), h + 1))
                noise[y0:y1, x:min(w, x + 1)] -= scratches * .20
        out += noise[..., None]
    halation = float(cfg.get("halation", 0.0)) * amount
    if halation:
        glow_mask = np.clip((out.max(axis=-1, keepdims=True) - .72) / .28, 0, 1)
        glow = Image.fromarray(np.uint8(glow_mask[..., 0] * 255)).filter(ImageFilter.GaussianBlur(max(1, min(out.shape[:2]) * .012)))
        glow_array = np.asarray(glow, dtype=np.float32)[..., None] / 255.0
        out[..., 0] += glow_array[..., 0] * halation * .65
        out[..., 1] += glow_array[..., 0] * halation * .18
    softness = float(cfg.get("softness", 0.0)) * amount
    out = np.clip(out, 0, 1)
    if softness:
        blurred = np.asarray(Image.fromarray(np.uint8(out * 255)).filter(
            ImageFilter.GaussianBlur(max(.2, softness * 2.0))), dtype=np.float32) / 255.0
        out = out * (1.0 - softness) + blurred * softness
    return np.clip(out, 0.0, 1.0).astype(np.float32)

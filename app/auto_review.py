"""Automatic upload inspection, ISP comparison, and per-image report exports."""
from __future__ import annotations

import hashlib
import io
import json
import time
from datetime import datetime, timezone

import numpy as np
import streamlit as st
from PIL import Image

from app.components import camera_style_preview, get_image, process_modes, score_images, to_image, visualized_raw
from app.state import ROOT
from isp.evaluation.report import build_pdf_report


def _image_png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(to_image(array)).save(buffer, format="PNG")
    return buffer.getvalue()


def _review_html(report: dict) -> str:
    import html

    scores = report.get("reference_metrics")
    score_rows = "<p>No aligned ground-truth image was provided; quality metrics are therefore not reported.</p>"
    if scores:
        score_rows = "<table><tr><th>Pipeline</th><th>Metric</th><th>Value</th></tr>" + "".join(
            f"<tr><td>{html.escape(name)}</td><td>{html.escape(metric)}</td><td>{float(value):.5f}</td></tr>"
            for name, metrics in scores.items() for metric, value in metrics.items()
        ) + "</table>"
    details = html.escape(json.dumps(report, indent=2, default=str))
    return ("<!doctype html><meta charset='utf-8'><title>HybridISP Image Review</title>"
            "<style>body{font:16px 'Times New Roman',Times,serif;max-width:900px;margin:40px auto;color:#2b1720}"
            "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:8px;text-align:left}"
            "pre{white-space:pre-wrap;background:#f2f4f5;padding:16px}</style>"
            f"<h1>HybridISP automatic image review</h1><p>{html.escape(report['created_utc'])}</p>"
            f"<h2>Assessment</h2><p>{html.escape(report['assessment'])}</p><h2>Reference metrics</h2>"
            f"{score_rows}<h2>Capture and pipeline details</h2><pre>{details}</pre>")


def render_auto_review() -> None:
    st.title("Automatic Image Review")
    st.caption("Upload an image in the sidebar. Inspection and ISP comparison run automatically.")
    image, kind = get_image()
    if image is None:
        st.info("Upload a Bayer mosaic (.npy, 16-bit PNG/TIFF or DNG) or an RGB image in the sidebar to begin.")
        return

    source_name = st.session_state.get("input_name", "uploaded image")
    source = np.asarray(image)
    if source.ndim not in (2, 3):
        st.error(f"Cannot process image with shape {source.shape}. Expected a 2D Bayer mosaic or 2D/3D image.")
        return
    pattern = st.session_state.get("bayer_pattern", "RGGB")
    is_bayer = kind == "raw"
    props = {
        "file": source_name, "detected_type": "2D Bayer RAW mosaic" if is_bayer else "Rendered RGB/grayscale image",
        "shape": list(source.shape), "width": int(source.shape[1]), "height": int(source.shape[0]),
        "channels": 1 if source.ndim == 2 else int(source.shape[2]), "dtype": str(source.dtype),
        "observed_bit_depth": int(source.dtype.itemsize * 8), "bayer_pattern": pattern if is_bayer else None,
        "pattern_source": "user-selected; Bayer pattern is not reliably inferable from pixel values alone" if is_bayer else None,
        "value_range": [float(np.nanmin(source)), float(np.nanmax(source))],
    }
    st.subheader("Automatic inspection")
    cols = st.columns(4)
    cols[0].metric("Detected input", "Bayer RAW" if is_bayer else "RGB image")
    cols[1].metric("Resolution", f"{props['width']} × {props['height']}")
    cols[2].metric("Storage", f"{props['dtype']} · {props['observed_bit_depth']} bit")
    cols[3].metric("Bayer pattern", pattern if is_bayer else "Not applicable")
    st.caption(f"Value range: {props['value_range'][0]:.4g} to {props['value_range'][1]:.4g}. Bit depth above is storage depth; effective sensor bit depth may be lower.")

    # Key results to the actual upload and effective settings so reruns do not redo inference.
    key_hasher = hashlib.sha256()
    key_hasher.update(source.tobytes())
    key_hasher.update(json.dumps({"kind": kind, "pattern": pattern,
        "black": st.session_state.get("black_level", 0), "white": st.session_state.get("white_level", 65535),
        "adaptive": st.session_state.get("adaptive", True), "device": st.session_state.get("device", "auto"),
        "precision": st.session_state.get("precision", "fp32"), "tile": st.session_state.get("tile_size", 512)}, sort_keys=True).encode())
    cache_key = key_hasher.hexdigest()
    stored = st.session_state.get("auto_review_result")
    if not stored or stored.get("key") != cache_key:
        try:
            with st.spinner("Inspecting capture and running Traditional, Learned, and Hybrid ISP…"):
                started = time.perf_counter()
                result = process_modes(source, kind, pattern=pattern,
                    black=st.session_state.get("black_level", 0), white=st.session_state.get("white_level", 65535),
                    adaptive=st.session_state.get("adaptive", True))
                stored = {"key": cache_key, "result": result, "wall_ms": (time.perf_counter() - started) * 1000}
                st.session_state["auto_review_result"] = stored
        except Exception as exc:
            st.error(f"Automatic ISP processing failed: {type(exc).__name__}: {exc}")
            st.info("Check RAW format, Bayer pattern, black/white levels, or install the optional DNG decoder if required.")
            return
    result = stored["result"]

    st.subheader("Pipeline output comparison")
    input_display = visualized_raw(source, pattern, st.session_state.get("black_level", 0),
                                   st.session_state.get("white_level", 65535)) if is_bayer else source
    if not is_bayer:
        if input_display.ndim == 2: input_display = np.repeat(input_display[..., None], 3, axis=-1)
        if np.issubdtype(source.dtype, np.integer): input_display = input_display.astype(np.float32) / np.iinfo(source.dtype).max
        else: input_display = np.clip(input_display.astype(np.float32), 0, 1)
    huawei_preview = camera_style_preview(result["hybrid"], "huawei")
    canon_preview = camera_style_preview(result["hybrid"], "canon")
    panels = [("Visualized RAW Image", input_display, None),
              ("Reconstructed RGB Image", result["hybrid"], None),
              ("Huawei P20’s ISP Photo", huawei_preview, "Automatically generated style preview; not a device capture."),
              ("Canon Photo", canon_preview, "Automatically generated style preview; not a Canon capture or ground truth.")]
    columns = st.columns(4)
    for col, (label, output, note) in zip(columns, panels):
        with col:
            st.markdown(f"**{label}**")
            st.image(to_image(output), use_container_width=True)
            if note: st.caption(note)

    policy = result.get("policy") or {}
    if policy:
        st.subheader("Adaptive capture assessment")
        st.json(policy)
    else:
        st.info("Adaptive RAW noise/illuminant estimates are available for Bayer mosaics. For rendered RGB, automatic input classification and ISP processing were completed.")
    st.markdown("**Measured processing timings (this run)**")
    st.json({"end_to_end_wall_ms": round(stored["wall_ms"], 2), **{k: round(float(v), 3) for k, v in result.get("timings", {}).items()}})

    with st.expander("Optional real aligned RGB reference for measured quality metrics"):
        st.caption("Do not use the generated camera-style previews as ground truth.")
        reference_file = st.file_uploader("Aligned RGB reference", type=["png", "jpg", "jpeg", "tif", "tiff"], key="auto_review_reference")
    reference = np.asarray(Image.open(reference_file).convert("RGB"), dtype=np.float32) / 255.0 if reference_file else None
    metrics = None
    if reference is not None:
        outputs = {"Traditional": result["traditional"], "Hybrid": result["hybrid"]}
        if result["learned"] is not None:
            outputs["Learned"] = result["learned"]
        if any(np.asarray(out).shape != reference.shape for out in outputs.values()):
            st.error(f"Reference dimensions must exactly match pipeline outputs. Reference: {reference.shape}; output: {result['hybrid'].shape}.")
        else:
            metrics = {name: score_images(out, reference) for name, out in outputs.items()}
            st.subheader("Measured quality against aligned reference")
            st.json(metrics)
    else:
        st.warning("No PSNR, SSIM, Delta E, or LPIPS values are shown because this upload has no aligned ground-truth RGB. Visual output alone cannot establish reconstruction accuracy.")

    assessment = "Image was processed successfully. Camera-style panels are simulations; no reference-based quality claim is available." if metrics is None else "Image was processed and evaluated against the supplied aligned RGB reference. Metrics below are computed from this run."
    report = {"title": "HybridISP automatic image review", "created_utc": datetime.now(timezone.utc).isoformat(),
              "source": props, "runtime": {"device": result.get("device"), "model_metadata": result.get("meta"),
              "wall_ms": stored["wall_ms"], "stage_timings_ms": result.get("timings")},
              "adaptive_policy": policy,
              "camera_style_previews": {"huawei": "simulated style preview", "canon": "simulated style preview"},
              "canon_reference_photo": reference_file.name if reference_file else None,
              "reference_metrics": metrics, "assessment": assessment,
              "limitations": ["A Bayer pattern cannot be confidently inferred from arbitrary pixel statistics; the selected pattern is recorded.",
                              "Storage bit depth does not establish the effective sensor ADC bit depth.",
                              "Model checkpoint provenance is recorded in metadata; synthetic-trained checkpoints are not evidence of real-camera superiority."]}
    st.subheader("Final review")
    st.write(assessment)
    pdf_lines = ["HybridISP Studio - Automatic Image Review", f"Created: {report['created_utc']}",
                 f"File: {source_name}", f"Detected: {props['detected_type']}",
                 f"Resolution: {props['width']} x {props['height']}", f"dtype: {props['dtype']}",
                 f"Assessment: {assessment}", f"Runtime device: {result.get('device')}",
                 f"End-to-end latency: {stored['wall_ms']:.2f} ms"]
    if is_bayer:
        pdf_lines.append(f"Bayer pattern: {pattern} (selected in app)")
    if metrics:
        for name, values in metrics.items():
            pdf_lines.extend(f"{name} {metric}: {value:.5f}" for metric, value in values.items())
    else:
        pdf_lines.append("Reference metrics: unavailable (no aligned ground truth supplied)")
    pdf_lines.extend(["Huawei P20 column: simulated style preview, not a real device capture.",
                      "Canon column: simulated style preview, not ground truth."])
    with st.expander("Review complete image report", expanded=True):
        st.json(report)
        st.download_button("Download final review (JSON)", json.dumps(report, indent=2, default=str),
                           file_name="hybridisp_image_review.json", mime="application/json")
        st.download_button("Download final review (HTML)", _review_html(report),
                           file_name="hybridisp_image_review.html", mime="text/html")
        st.download_button("Download final review (PDF)", build_pdf_report(pdf_lines),
                           file_name="hybridisp_image_review.pdf", mime="application/pdf")
    for label, output in (("traditional", result["traditional"]), ("hybrid", result["hybrid"]), ("learned", result["learned"])):
        if output is not None:
            st.download_button(f"Download {label} output (PNG)", _image_png(output),
                               file_name=f"{label}_isp.png", mime="image/png", key=f"auto_review_{label}_png")

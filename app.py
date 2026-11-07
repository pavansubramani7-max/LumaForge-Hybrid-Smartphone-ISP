from __future__ import annotations
import hashlib, importlib, io, json, subprocess, sys, time, zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
from src.isp import classical as classical_isp
classical_isp = importlib.reload(classical_isp)
run_classical = classical_isp.run_classical
estimate_adaptive_controls = classical_isp.estimate_adaptive_controls
unpack_bayer_4ch = classical_isp.unpack_bayer_4ch
from src.isp.metrics import psnr, ssim, delta_e_76
from src.isp.data import safe_extract_zip, inspect_pair_layout, find_pair_root
from isp.evaluation.report import build_pdf_report
from app.components import camera_style_preview, to_image, visualized_raw
from app.creative_studio import render_creative_studio, render_upload_gallery
from app.theme import apply_theme

st.set_page_config(page_title="LumaForge Photo Studio", page_icon="📷", layout="wide")
apply_theme()
st.markdown("<div class='lf-brand'><span class='lf-mark'>L</span><div><div class='lf-eyebrow'>LUMAFORGE · IMAGE STUDIO</div><h1 style='margin:.1rem 0'>Hybrid Smartphone ISP</h1></div></div>", unsafe_allow_html=True)
st.caption("RAW reconstruction · adaptive classical imaging · learned restoration · creative color looks")

with st.sidebar:
    st.header("Image input")
    uploads = st.file_uploader("RGB image, Bayer mosaic, DNG, or NPY", type=["png", "jpg", "jpeg", "tif", "tiff", "npy", "dng"], key="source_image_upload", accept_multiple_files=True)
    if uploads:
        active_index = int(st.session_state.get("lf_active_upload", 0))
        if active_index >= len(uploads):
            active_index = 0
            st.session_state["lf_active_upload"] = 0
        upload = uploads[active_index]
        st.caption(f"{len(uploads)} photo{'s' if len(uploads) != 1 else ''} loaded · active: {upload.name}")
    else:
        upload = None
    kind = st.selectbox("Input type", ["Auto-detect", "Bayer RAW mosaic", "RGB image"])
    pattern = st.selectbox("Bayer pattern", ["RGGB", "BGGR", "GRBG", "GBRG"], index=0, key="isp_bayer_pattern")
    st.divider()
    st.header("Sensor and ISP controls")
    black = st.number_input("Black level", min_value=0.0, value=0.0, step=1.0)
    white = st.number_input("White level", min_value=1.0, value=4095.0, step=1.0, key="isp_white_level")
    auto_white = st.checkbox("Auto-expand white level if RAW exceeds it", value=True, key="isp_auto_white")
    adaptive = st.checkbox("Estimate WB, exposure, noise from frame", value=False)
    gain_cols = st.columns(3)
    gain_r = gain_cols[0].slider("WB red", 0.5, 3.5, 1.50, 0.05, format="%.2f", key="isp_wb_red")
    gain_g = gain_cols[1].slider("WB green", 0.5, 3.5, 1.00, 0.05, format="%.2f", key="isp_wb_green")
    gain_b = gain_cols[2].slider("WB blue", 0.5, 3.5, 1.80, 0.05, format="%.2f", key="isp_wb_blue")
    gains = (gain_r, gain_g, gain_b)
    exposure = st.slider("Exposure / tone", 0.5, 2.5, 1.25, 0.05, format="%.2f", key="isp_exposure")
    noise = st.slider("Denoising strength", 0.0, 0.3, 0.08, 0.01, format="%.2f", key="isp_denoise")
    sharp = st.slider("Detail recovery", 0.0, 0.8, 0.25, 0.05, format="%.2f", key="isp_detail")
    st.sidebar.caption("RAW WB defaults: R 1.50 · G 1.00 · B 1.80. Rendered RGB uses neutral WB unless you change these gains or enable adaptive estimation.")

studio, training, evaluation = st.tabs(["LumaForge Studio", "Dataset & Training", "Evaluation"])

with studio:
    render_upload_gallery(uploads or [], int(st.session_state.get("lf_active_upload", 0)))
    st.subheader("Your photo, developed and styled")
    if not upload:
        st.info("Upload an image in the sidebar. For RAW reconstruction, use a single-channel RGGB mosaic and enter the sensor black/white levels.")
    else:
        data = upload.getvalue()
        suffix = Path(upload.name).suffix.lower()
        if suffix == ".npy":
            raw = np.load(io.BytesIO(data), allow_pickle=False)
        elif suffix == ".dng":
            try:
                import rawpy
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".dng") as dng_file:
                    dng_file.write(data); dng_file.flush()
                    with rawpy.imread(dng_file.name) as dng: raw = dng.raw_image_visible.copy()
            except ImportError:
                st.error("DNG decoding requires rawpy. Install it with `pip install rawpy` and restart the app.")
                st.stop()
        else:
            raw = np.asarray(Image.open(io.BytesIO(data)))
        packed_bayer = suffix == ".npy" and raw.ndim == 3 and raw.shape[-1] == 4 and kind != "RGB image"
        if packed_bayer:
            raw = unpack_bayer_4ch(raw)
            st.caption("Detected 4-channel packed Bayer RAW; expanded it as a row-major RGGB mosaic.")
        elif raw.ndim == 3 and raw.shape[-1] >= 3:
            raw = raw[..., :3]
        if kind == "Auto-detect":
            input_kind = "rggb" if (suffix in {".npy", ".dng"} or (raw.ndim == 2 and raw.dtype.itemsize > 1)) else "rgb"
        else:
            input_kind = "rggb" if kind == "Bayer RAW mosaic" else "rgb"
        if input_kind == "rggb" and auto_white and np.issubdtype(raw.dtype, np.integer):
            observed_max = float(np.nanmax(raw))
            if observed_max > white * 1.05:
                inferred = min(int(np.iinfo(raw.dtype).max), (1 << int(np.ceil(np.log2(observed_max + 1)))) - 1)
                if inferred > white:
                    st.info(f"RAW peak {observed_max:.0f} exceeds the selected white level; using {inferred} to avoid clipping. Turn off auto-expand to keep the manual value.")
                    white = float(inferred)
        # Sensor as-shot gains are a RAW default; avoid imposing a strong camera
        # correction on already-rendered RGB unless the user changes the sliders.
        if input_kind == "rgb" and not adaptive and np.allclose(gains, (1.50, 1.00, 1.80)):
            gains = (1.0, 1.0, 1.0)
        if input_kind == "rggb" and raw.ndim != 2:
            if raw.ndim == 3 and raw.shape[-1] >= 3:
                st.warning("This upload has three color channels, so it is an RGB image rather than a single-channel Bayer mosaic. Processing it as RGB.")
                input_kind = "rgb"
                gains = (1.0, 1.0, 1.0)
            else:
                st.error(f"RGGB Bayer input must be a 2D single-channel mosaic; received shape {raw.shape}. Select RGB image or upload a Bayer mosaic.")
                st.stop()
        if adaptive:
            gains, exposure, estimated_noise = estimate_adaptive_controls(raw, input_kind, black, white if input_kind == "rggb" else None, pattern=pattern)
            noise = max(noise, estimated_noise)
            st.caption(f"Frame estimates · WB {tuple(round(v, 2) for v in gains)} · exposure {exposure:.2f}× · noise proxy {estimated_noise:.3f}")
        st.subheader("Automatic image inspection")
        info_cols = st.columns(5)
        info_cols[0].metric("Detected format", "2D Bayer RAW" if input_kind == "rggb" else "RGB image")
        info_cols[1].metric("Resolution", f"{raw.shape[1]} × {raw.shape[0]}")
        info_cols[2].metric("Array type", str(raw.dtype))
        info_cols[3].metric("Storage depth", f"{raw.dtype.itemsize * 8} bit")
        info_cols[4].metric("Channel count", str(1 if raw.ndim == 2 else raw.shape[-1]))
        caption = f"Observed values: {float(np.nanmin(raw)):.4g}–{float(np.nanmax(raw)):.4g}. Storage depth is not a measurement of effective sensor ADC depth."
        if input_kind == "rggb":
            caption += f" Bayer arrangement: {pattern} (selected in this app)."
        st.caption(caption)
        started = time.perf_counter()
        baseline = run_classical(raw, input_kind, black=black, white=white if input_kind == "rggb" else None,
                                 wb=gains, exposure=exposure, denoise_strength=noise, sharpen_amount=sharp, pattern=pattern)
        classical_ms = (time.perf_counter() - started) * 1000
        hybrid = baseline
        learned_ms = None
        fp16 = st.checkbox("Prefer FP16 model checkpoint when available", value=False)
        checkpoint = ROOT / "models" / ("isp_tiny_fp16.pt" if fp16 and (ROOT / "models" / "isp_tiny_fp16.pt").exists() else "isp_tiny.pt")
        state = None
        if input_kind == "rggb" and checkpoint.exists():
            try:
                import torch
                from src.isp.model import TinyISPUNet, infer_image
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = TinyISPUNet().to(device)
                state = torch.load(checkpoint, map_location=device, weights_only=True)
                model.load_state_dict(state.get("model", state)); model.eval()
                if state.get("precision") == "fp16" and device == "cuda": model.half()
                elif state.get("precision") == "fp16" and device == "cpu":
                    st.warning("FP16 inference selected on CPU; using full precision for compatibility.")
                # Keep the sidebar controls authoritative. Checkpoint ISP metadata
                # records training provenance; it must not silently replace UI values.
                features = np.concatenate([baseline, np.full((*baseline.shape[:2], 1), noise, np.float32)], axis=-1)
                t0 = time.perf_counter(); hybrid = infer_image(model, features, device)
                learned_ms = (time.perf_counter() - t0) * 1000
                epoch = state.get("epoch", "unknown")
                best_loss = state.get("best_val_l1")
                provenance = "synthetic training data" if state.get("synthetic_data") else "dataset provenance not recorded in checkpoint"
                st.caption(f"Learned restoration ran on the selected Bayer RAW output. Checkpoint: epoch {epoch}; {provenance}." + (f" Best validation L1: {best_loss:.4f}." if best_loss is not None else ""))
            except Exception as exc:
                st.error(f"Could not run trained model: {exc}")
        elif input_kind == "rggb":
            st.warning(f"Learned checkpoint not found at {checkpoint}. Run the training step or use the project checkpoint before comparing learned output.")
        else:
            st.info("This checkpoint was trained for synthetic Bayer RAW reconstruction, so it is skipped for already-rendered RGB to avoid unvalidated colour or detail changes.")
        raw_view = raw.astype(np.float32)
        if input_kind == "rggb" and raw.ndim == 2:
            raw_view = visualized_raw(raw, pattern, black, white)
        elif raw.ndim == 2:
            if np.issubdtype(raw.dtype, np.integer):
                raw_view /= float(np.iinfo(raw.dtype).max)
            else:
                peak = float(np.nanmax(raw_view)) if raw_view.size else 1.0
                if peak > 1.5:
                    raw_view /= 255.0 if peak <= 255 else (65535.0 if peak <= 65535 else peak)
            raw_view = np.repeat(np.clip(raw_view, 0, 1)[..., None], 3, axis=-1)
        elif np.issubdtype(raw.dtype, np.integer):
            raw_view = raw_view / float(np.iinfo(raw.dtype).max)
        else:
            peak = float(np.nanmax(raw_view)) if raw_view.size else 1.0
            if peak > 1.5:
                raw_view /= 255.0 if peak <= 255 else (65535.0 if peak <= 65535 else peak)
            raw_view = np.clip(raw_view, 0, 1)
        selected = hybrid if learned_ms is not None else baseline
        edited = render_creative_studio(raw_view, selected, upload.name, input_kind,
                                        classical_ms + (learned_ms or 0.0))
        huawei_preview = camera_style_preview(selected, "huawei")
        canon_preview = camera_style_preview(selected, "canon")
        columns = st.columns(4, gap="small")
        with columns[0]:
            st.markdown("**Visualized RAW Image**")
            st.image(raw_view, clamp=True, use_container_width=True)
            if input_kind != "rggb": st.caption("RGB input shown; no Bayer mosaic was supplied.")
        with columns[1]:
            st.markdown("**Reconstructed RGB Image**")
            reconstruction = hybrid if learned_ms is not None else baseline
            st.image(np.uint8(np.clip(reconstruction,0,1)*255), use_container_width=True)
            if learned_ms is None:
                reason = ("RAW-trained learned refinement is skipped for rendered RGB to avoid an unvalidated out-of-domain change."
                          if input_kind == "rgb" else "Learned checkpoint is unavailable for this Bayer RAW input.")
                st.caption(f"Conventional ISP output. {reason}")
        with columns[2]:
            st.markdown("**Huawei P20’s ISP Photo**")
            st.image(to_image(huawei_preview), use_container_width=True)
            st.caption("Automatically generated Huawei-style preview; not a device capture.")
        with columns[3]:
            st.markdown("**Canon Photo**")
            st.image(to_image(canon_preview), use_container_width=True)
            st.caption("Automatically generated Canon-style preview; not a Canon capture or ground truth.")
        metrics = st.columns(3)
        metrics[0].metric("Classical path", f"{classical_ms:.1f} ms", help="This host only; excludes mobile camera I/O")
        metrics[1].metric("Learned refinement", f"{learned_ms:.1f} ms" if learned_ms is not None else ("RAW only" if input_kind == "rgb" else "Checkpoint unavailable"))
        metrics[2].metric("Input shape", f"{raw.shape[1]} × {raw.shape[0]}")
        buffer = io.BytesIO(); Image.fromarray(np.uint8(np.clip(selected,0,1)*255)).save(buffer, format="PNG")
        st.download_button("Download reconstructed RGB", buffer.getvalue(), "lumaforge_rgb.png", "image/png")
        huawei_buffer=io.BytesIO(); Image.fromarray(to_image(huawei_preview)).save(huawei_buffer,format="PNG")
        canon_buffer=io.BytesIO(); Image.fromarray(to_image(canon_preview)).save(canon_buffer,format="PNG")
        preview_downloads=st.columns(2)
        preview_downloads[0].download_button("Download Huawei-style preview",huawei_buffer.getvalue(),"huawei_style_preview.png","image/png")
        preview_downloads[1].download_button("Download Canon-style preview",canon_buffer.getvalue(),"canon_style_preview.png","image/png")
        with st.expander("Optional ground-truth reference for measured quality metrics"):
            st.caption("This must be a real aligned RGB target. Generated camera-style previews are not used as ground truth.")
            canon_reference_file = st.file_uploader("Aligned Canon/ground-truth RGB", type=["png","jpg","jpeg","tif","tiff"], key="canon_reference_photo")
        canon_photo = np.asarray(Image.open(canon_reference_file).convert("RGB"), np.float32) / 255 if canon_reference_file else None
        target = canon_photo
        if target is not None:
            if target.shape != selected.shape:
                st.error(f"Canon reference dimensions {target.shape[1]}×{target.shape[0]} do not match output {selected.shape[1]}×{selected.shape[0]}; align the pair before scoring.")
            else:
                c1,c2,c3=st.columns(3)
                c1.metric("PSNR", f"{psnr(selected,target):.3f} dB")
                c2.metric("SSIM", f"{ssim(selected,target):.5f}")
                c3.metric("Mean CIE76 ΔE", f"{delta_e_76(selected,target):.3f}")

        # Create a factual report from this upload and processing run; reference scores
        # are attached only when the user supplied a spatially aligned target.
        report = {
            "title": "HybridISP automatic image review", "source_file": upload.name,
            "detected_type": "2D Bayer RAW" if input_kind == "rggb" else "RGB image",
            "resolution": {"width": int(raw.shape[1]), "height": int(raw.shape[0])},
            "shape": list(raw.shape), "dtype": str(raw.dtype), "storage_bits": int(raw.dtype.itemsize * 8),
            "value_range": [float(np.nanmin(raw)), float(np.nanmax(raw))],
            "bayer_pattern": pattern if input_kind == "rggb" else None,
            "sensor_levels": ({"black": float(black), "white": float(white)} if input_kind == "rggb" else None),
            "adaptive_estimates": ({"white_balance_gains": list(map(float, gains)), "exposure": float(exposure),
                                    "noise_proxy": float(estimated_noise)} if adaptive and input_kind == "rggb" else None),
            "classical_latency_ms": float(classical_ms), "learned_latency_ms": learned_ms,
            "checkpoint_used": str(checkpoint) if learned_ms is not None else None,
            "checkpoint_epoch": int(state.get("epoch")) if learned_ms is not None and state.get("epoch") is not None else None,
            "checkpoint_synthetic_data": bool(state.get("synthetic_data")) if learned_ms is not None else None,
            "camera_style_previews": {"huawei": "simulated style preview", "canon": "simulated style preview"},
            "canon_reference_photo": canon_reference_file.name if canon_reference_file else None,
            "reference_scores": None,
            "assessment": ("Metrics are measured against the supplied aligned RGB reference." if canon_photo is not None and canon_photo.shape == selected.shape
                           else "No aligned ground-truth RGB was supplied; this is a visual review, not a measured quality claim."),
        }
        if canon_photo is not None and canon_photo.shape == selected.shape:
            report["reference_scores"] = {"PSNR_dB": float(psnr(selected,target)),
                                           "SSIM": float(ssim(selected,target)),
                                           "mean_CIE76_DeltaE": float(delta_e_76(selected,target))}
        st.subheader("Final image review")
        st.write(report["assessment"])
        report_json = json.dumps(report, indent=2)
        html_report = ("<!doctype html><meta charset='utf-8'><title>HybridISP image review</title>"
                       "<style>body{font:16px 'Times New Roman',Times,serif;max-width:850px;margin:40px auto;color:#2b1720}pre{white-space:pre-wrap;background:#f2e8eb;padding:18px}</style>"
                       f"<h1>HybridISP automatic image review</h1><p>{report['assessment']}</p><pre>{report_json}</pre>")
        pdf_lines = ["HybridISP Studio - Automatic Image Review", f"File: {upload.name}",
                     f"Detected: {report['detected_type']}", f"Resolution: {raw.shape[1]} x {raw.shape[0]}",
                     f"dtype: {raw.dtype} ({raw.dtype.itemsize * 8}-bit storage)",
                     f"Classical latency: {classical_ms:.2f} ms", report["assessment"]]
        if report["reference_scores"]:
            pdf_lines.extend(f"{name}: {value:.5f}" for name, value in report["reference_scores"].items())
        pdf_lines.extend(["Huawei P20 column: simulated style preview, not a real device capture.",
                          "Canon column: simulated style preview, not ground truth."])
        with st.expander("Review complete image report", expanded=True):
            st.json(report)
            st.download_button("Download final review (JSON)", report_json, "hybridisp_image_review.json", "application/json")
            st.download_button("Download final review (HTML)", html_report, "hybridisp_image_review.html", "text/html")
            st.download_button("Download final review (PDF)", build_pdf_report(pdf_lines), "hybridisp_image_review.pdf", "application/pdf")

with training:
    st.subheader("Paired RAW/RGB data")
    st.write("Expected archive layout: `train/raw`, `train/rgb`, `val/raw`, and `val/rgb`. Matching file stems identify aligned image pairs. Keep complete scenes or capture bursts in only one split.")
    archive = st.file_uploader("Upload dataset ZIP", type=["zip"], key="dataset_zip")
    if archive and st.button("Safely extract and inspect ZIP", type="primary"):
        try:
            archive_id=hashlib.sha256(archive.getvalue()).hexdigest()[:12]
            destination = ROOT / "data" / "uploads" / archive_id
            safe_extract_zip(io.BytesIO(archive.getvalue()), destination)
            pair_root = find_pair_root(destination)
            st.session_state["dataset_root"] = str(pair_root)
            st.success(f"Extracted dataset root: {pair_root}")
        except (ValueError, zipfile.BadZipFile) as exc:
            st.error(f"Could not extract dataset: {exc}")
    default_dataset_root = ROOT / "data" / "synthetic_pairs"
    if not default_dataset_root.exists():
        default_dataset_root = ROOT / "data" / "pairs"
    dataset_root = Path(st.session_state.get("dataset_root", default_dataset_root))
    layout = inspect_pair_layout(dataset_root)
    st.caption(f"Selected dataset root: {dataset_root}")
    st.json(layout)
    ready = all(layout[s]["valid"] and layout[s].get("raw_without_rgb",0)==0 and layout[s].get("rgb_without_raw",0)==0 for s in ("train", "val"))
    if not ready:
        missing=[s for s in ("train","val") if not layout[s]["valid"]]
        unmatched=[s for s in ("train","val") if layout[s].get("raw_without_rgb",0) or layout[s].get("rgb_without_raw",0)]
        details=[]
        if missing: details.append(f"no aligned pairs in {', '.join(missing)}")
        if unmatched: details.append(f"unmatched RAW/RGB filenames in {', '.join(unmatched)}")
        st.warning("Training is disabled: " + "; ".join(details) + ".")
        st.caption("Upload a ZIP with matching Bayer RAW and RGB files under train/raw, train/rgb, val/raw and val/rgb. CSV-only datasets cannot train this image model.")
    c1,c2,c3=st.columns(3)
    epochs=c1.number_input("Epochs", min_value=1, max_value=500, value=30)
    batch=c2.number_input("Batch size", min_value=1, max_value=64, value=4)
    patch=c3.number_input("Patch size", min_value=64, max_value=1024, value=256, step=32)
    amp=st.checkbox("Mixed precision on CUDA (AMP)", value=True)
    st.markdown("**Sensor recipe saved with the model**")
    raw_black=st.number_input("Training black level", min_value=0.0, value=0.0, step=1.0)
    metadata_path=dataset_root/"dataset_metadata.json"
    try:
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    except (OSError, ValueError):
        metadata={}
    default_train_white=float(metadata.get("white_level") or 4095.0)
    raw_white=st.number_input("Training white level", min_value=1.0, value=default_train_white, step=1.0, key="training_white_level")
    wbcols=st.columns(3)
    train_r=wbcols[0].number_input("R gain",.1,5.0,1.5,.05); train_g=wbcols[1].number_input("G gain",.1,5.0,1.0,.05); train_b=wbcols[2].number_input("B gain",.1,5.0,1.8,.05)
    train_exposure=st.number_input("Training exposure",.1,4.0,1.25,.05)
    if st.button("Start training", disabled=not ready, type="primary"):
        log_path=ROOT/"work"/"training.log"; log_path.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,"scripts/train.py","--data",str(dataset_root),"--epochs",str(int(epochs)),"--batch",str(int(batch)),"--patch",str(int(patch)),"--out","models/isp_tiny.pt","--black",str(raw_black),"--white",str(raw_white),"--red-gain",str(train_r),"--green-gain",str(train_g),"--blue-gain",str(train_b),"--exposure",str(train_exposure)]
        if amp: command.append("--amp")
        log_file=log_path.open("w",encoding="utf-8")
        process=subprocess.Popen(command,cwd=ROOT,stdout=log_file,stderr=subprocess.STDOUT)
        st.session_state["train_pid"]=process.pid; st.session_state["train_log"]=str(log_path)
        st.success(f"Training started (process {process.pid}). Use Refresh below to check progress.")
    if st.session_state.get("train_pid"):
        if st.button("Refresh training status"):
            try:
                process=subprocess.Popen(["tasklist","/FI",f"PID eq {st.session_state['train_pid']}","/FO","CSV"],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
                status=process.communicate()[0]
                active=str(st.session_state["train_pid"]) in status
            except Exception:
                active=False
            st.info("Training process is running." if active else "Training process has ended. Check the log and model checkpoint.")
            log=Path(st.session_state.get("train_log",""))
            if log.exists(): st.code(log.read_text(encoding="utf-8",errors="replace")[-12000:])
    checkpoint=ROOT/"models"/"isp_tiny.pt"
    if checkpoint.exists():
        st.success(f"Checkpoint available · {checkpoint.stat().st_size/1024**2:.2f} MiB")
        try:
            import torch
            checkpoint_info=torch.load(checkpoint,map_location="cpu",weights_only=True)
            trained_root=checkpoint_info.get("data_root")
            trained_on=("synthetic demo pairs" if checkpoint_info.get("synthetic_data") else
                        (f"dataset at {trained_root}" if trained_root else "unknown data provenance"))
            epoch_info=checkpoint_info.get("epoch","unknown")
            st.caption(f"Checkpoint provenance: {trained_on} · epoch {epoch_info}.")
            if trained_root and Path(trained_root).resolve()!=dataset_root.resolve():
                st.warning(f"This existing checkpoint was trained from `{trained_root}`; it is not a result from the currently selected dataset `{dataset_root}`.")
        except Exception as exc:
            st.warning(f"Checkpoint exists but its training metadata could not be read: {exc}")
    history_path=ROOT/"outputs"/"training_history.csv"
    if history_path.exists(): st.download_button("Download training history CSV",history_path.read_bytes(),"training_history.csv","text/csv")
    if checkpoint.exists() and st.button("Export FP16 checkpoint"):
        result=subprocess.run([sys.executable,"scripts/export_fp16.py",str(checkpoint),"--out","models/isp_tiny_fp16.pt"],cwd=ROOT,capture_output=True,text=True)
        if result.returncode==0: st.success(result.stdout.strip())
        else: st.error(result.stderr or "FP16 export failed")

with evaluation:
    st.subheader("Reproducible held-out evaluation")
    st.write("The evaluation script compares the conventional pipeline with the trained hybrid on aligned test pairs. It reports PSNR, SSIM, mean CIE76 Delta E, median and p95 latency, parameter count, checkpoint size, and CUDA peak allocation when available.")
    default_data_root=Path(st.session_state.get("dataset_root", default_dataset_root))
    default_test_root=default_data_root/"test"
    if not (default_test_root/"raw").is_dir() or not (default_test_root/"rgb").is_dir():
        synthetic_test_root=ROOT/"data"/"synthetic_pairs"/"test"
        if (synthetic_test_root/"raw").is_dir() and (synthetic_test_root/"rgb").is_dir():
            default_test_root=synthetic_test_root
            st.caption("No paired test split was found under the selected dataset root; defaulting to the bundled synthetic test split.")
    test_root=st.text_input("Test split folder", str(default_test_root), key="evaluation_test_root")
    if st.button("Run evaluation"):
        with st.spinner("Evaluating held-out scenes…"):
            result=subprocess.run([sys.executable,"scripts/evaluate.py","--data",test_root,"--checkpoint",str(ROOT/"models"/"isp_tiny.pt")],cwd=ROOT,capture_output=True,text=True)
        if result.returncode==0:
            st.code(result.stdout); st.success("Evaluation complete. Results saved under outputs/evaluation.json.")
        else:
            st.error(result.stderr or result.stdout or "Evaluation could not run. Confirm test pairs and checkpoint are present.")
    report=ROOT/"outputs"/"evaluation.json"
    if report.exists(): st.download_button("Download latest evaluation JSON",report.read_bytes(),"evaluation.json","application/json")
    st.caption("Formal scores require a genuinely held-out scene split. Desktop latency is not a phone latency result. LPIPS/ΔE2000 can be added once the evaluation environment and target dataset are fixed.")

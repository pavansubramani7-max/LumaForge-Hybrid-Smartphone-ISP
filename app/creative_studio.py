"""UI helpers for LumaForge's creative editing surface."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from app.creative_presets import LOOKS, apply_look

NO_LOOK = "No creative look"
DEPARTMENTS = ("WORLD", "ERA", "CINEMA", "LIGHT")


def _select_active_image(index: int) -> None:
    st.session_state["lf_active_upload"] = index


def _select_look(name: str) -> None:
    st.session_state["lf_preset"] = name
    era_names = list(LOOKS["ERA"])
    if name in era_names:
        st.session_state["lf_era_timeline"] = era_names.index(name)


def _sync_era_timeline() -> None:
    era_names = list(LOOKS["ERA"])
    index = int(st.session_state.get("lf_era_timeline", 0))
    st.session_state["lf_preset"] = era_names[index]


def _current_history_state() -> dict[str, Any]:
    adjustment_prefixes = ("lf_grain_", "lf_scratches_", "lf_vignette_", "lf_fade_", "lf_halation_", "lf_softness_")
    adjustments = {key: st.session_state[key] for key in st.session_state
                   if key.startswith(adjustment_prefixes)}
    return {"department": st.session_state.get("lf_department", "WORLD"),
            "preset": st.session_state.get("lf_preset", NO_LOOK),
            "intensity": st.session_state.get("lf_intensity", 70),
            "adjustments": adjustments}


def _restore_history_state(value: dict[str, Any]) -> None:
    st.session_state["lf_department"] = value["department"]
    st.session_state["lf_preset"] = value["preset"]
    st.session_state["lf_intensity"] = value["intensity"]
    era_names = list(LOOKS["ERA"])
    if value["preset"] in era_names:
        st.session_state["lf_era_timeline"] = era_names.index(value["preset"])
    saved = value.get("adjustments", {})
    adjustment_prefixes = ("lf_grain_", "lf_scratches_", "lf_vignette_", "lf_fade_", "lf_halation_", "lf_softness_")
    for key in list(st.session_state):
        if key.startswith(adjustment_prefixes) and key not in saved:
            st.session_state.pop(key, None)
    for key, value in saved.items():
        st.session_state[key] = value
    st.session_state["lf_history_last"] = dict(value)


def _undo() -> None:
    history = st.session_state.setdefault("lf_undo", [])
    if history:
        st.session_state.setdefault("lf_redo", []).append(_current_history_state())
        _restore_history_state(history.pop())


def _redo() -> None:
    history = st.session_state.setdefault("lf_redo", [])
    if history:
        st.session_state.setdefault("lf_undo", []).append(_current_history_state())
        _restore_history_state(history.pop())


def _reset_look() -> None:
    current = _current_history_state()
    if (current["department"] != "WORLD" or current["preset"] != NO_LOOK
            or current["intensity"] != 70 or current.get("adjustments")):
        st.session_state.setdefault("lf_undo", []).append(current)
    st.session_state.setdefault("lf_redo", []).clear()
    _restore_history_state({"department": "WORLD", "preset": NO_LOOK, "intensity": 70})
    st.session_state["lf_history_last"] = None


def _decode_thumbnail(uploaded: Any) -> np.ndarray:
    """Decode a small source preview without modifying or replacing the upload."""
    data = uploaded.getvalue()
    suffix = Path(uploaded.name).suffix.lower()
    if suffix == ".npy":
        raw = np.load(io.BytesIO(data), allow_pickle=False)
        stride = max(1, int(np.ceil(max(raw.shape[:2]) / 256)))
        raw = raw[::stride, ::stride, ...] if raw.ndim >= 2 else raw
        if raw.ndim == 3:
            raw = raw[..., 0] if raw.shape[-1] in (1, 4) else raw[..., :3]
        if raw.ndim == 2:
            low, high = np.percentile(raw.astype(np.float32), (1, 99.5))
            image = np.clip((raw.astype(np.float32) - low) / max(float(high - low), 1e-6), 0, 1)
            image = np.repeat(image[..., None], 3, axis=-1)
        else:
            image = raw.astype(np.float32)
            if image.max(initial=0) > 1.5:
                image /= float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else max(float(image.max()), 1.0)
            image = image[..., :3]
    elif suffix == ".dng":
        try:
            import rawpy
            with rawpy.imread(io.BytesIO(data)) as dng:
                raw = dng.raw_image_visible.copy().astype(np.float32)
            stride = max(1, int(np.ceil(max(raw.shape[:2]) / 256)))
            raw = raw[::stride, ::stride]
            low, high = np.percentile(raw, (1, 99.5))
            scale = max(float(high - low), 1e-6)
            image = np.repeat(np.clip((raw-low)/scale, 0, 1)[..., None], 3, axis=-1)
        except Exception:
            image = np.full((160, 160, 3), .12, dtype=np.float32)
    else:
        preview = Image.open(io.BytesIO(data)).convert("RGB")
        preview.thumbnail((256, 256), Image.Resampling.BILINEAR)
        image = np.asarray(preview, dtype=np.float32) / 255.0
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    return np.uint8(np.clip(np.nan_to_num(image[..., :3]), 0, 1) * 255 + .5)


def render_upload_gallery(uploads: list[Any], active_index: int) -> None:
    """Show selectable photo cards; the original upload bytes remain untouched."""
    if len(uploads) < 2:
        return
    st.markdown("#### Your photos · select an active image")
    st.caption("Choose a thumbnail to make that exact upload the source for processing and editing.")
    for start in range(0, len(uploads), 4):
        columns = st.columns(4, gap="small")
        for offset, column in enumerate(columns):
            index = start + offset
            if index >= len(uploads):
                break
            with column:
                try:
                    preview = _decode_thumbnail(uploads[index])
                    st.image(preview, caption=uploads[index].name, use_container_width=True)
                except Exception as exc:
                    st.warning(f"Preview unavailable: {exc}")
                label = "✓ Active photo" if index == active_index else "Select photo"
                st.button(label, key=f"lf_choose_upload_{index}",
                          on_click=_select_active_image, args=(index,),
                          use_container_width=True)


def _crop_view(image: np.ndarray, zoom: float, center_x: float, center_y: float) -> np.ndarray:
    image = np.asarray(image)
    h, w = image.shape[:2]
    crop_w, crop_h = max(1, int(w / zoom)), max(1, int(h / zoom))
    x0 = int(np.clip(center_x * w - crop_w / 2, 0, w - crop_w))
    y0 = int(np.clip(center_y * h - crop_h / 2, 0, h - crop_h))
    return image[y0:y0 + crop_h, x0:x0 + crop_w]


def _jpeg_data(image: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(image, 0, 1) * 255 + .5)).save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _render_wipe(before: np.ndarray, after: np.ndarray) -> None:
    h, w = before.shape[:2]
    before_data, after_data = _jpeg_data(before), _jpeg_data(after)
    markup = f"""<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#0c1113;font-family:Arial,sans-serif;color:#f4f5ef}}
.frame{{position:relative;width:100%;aspect-ratio:{w}/{h};overflow:hidden;border:1px solid #344143;border-radius:14px;background:#080b0c}}
.frame img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}}
.after{{clip-path:inset(0 50% 0 0)}}.labels{{position:absolute;inset:12px 12px auto;display:flex;justify-content:space-between;pointer-events:none}}
.label{{padding:6px 10px;border-radius:99px;background:#101718d9;border:1px solid #66736d;font-size:12px}}
.divider{{position:absolute;top:0;bottom:0;left:50%;width:2px;background:#c3f27e;box-shadow:0 0 15px #c3f27e;pointer-events:none}}
input{{position:absolute;left:8%;width:84%;bottom:12px;accent-color:#c3f27e;cursor:ew-resize}}
</style></head><body><div class="frame" id="frame">
<img src="data:image/jpeg;base64,{before_data}" alt="Original photo">
<img class="after" id="after" src="data:image/jpeg;base64,{after_data}" alt="Processed photo">
<div class="labels"><span class="label">ORIGINAL</span><span class="label">LUMAFORGE EDIT</span></div><div class="divider" id="divider"></div>
<input id="slider" type="range" min="0" max="100" value="50" aria-label="Before and after comparison">
</div><script>const s=document.getElementById('slider'),a=document.getElementById('after'),d=document.getElementById('divider');
s.addEventListener('input',()=>{{const v=Number(s.value);a.style.clipPath=`inset(0 ${{100-v}}% 0 0)`;d.style.left=v+'%'}});</script></body></html>"""
    target_height = int(np.clip(1000 * h / max(w, 1), 360, 760))
    components.html(markup, height=target_height, scrolling=False)


def _render_pipeline(input_kind: str, elapsed_ms: float) -> None:
    if input_kind == "rggb":
        names = ("Black level", "Demosaic", "White balance", "Colour matrix", "Denoise", "Tone + gamma", "Sharpen")
    else:
        names = ("RGB decode", "White balance", "Denoise", "Exposure + tone", "Sharpen")
    nodes = []
    for index, name in enumerate(names):
        nodes.append(f"<span class='lf-stage done'><i class='lf-dot'></i>{name}</span>")
        if index < len(names) - 1:
            nodes.append("<span class='lf-arrow' aria-hidden='true'>›</span>")
    st.markdown(
        f"<div class='lf-card'><div class='lf-eyebrow'>ISP pipeline · completed in {elapsed_ms:.1f} ms</div>"
        f"<div class='lf-pipeline'>{''.join(nodes)}</div></div>",
        unsafe_allow_html=True,
    )


def render_creative_studio(source_display: np.ndarray, isp_output: np.ndarray,
                           source_name: str, input_kind: str, elapsed_ms: float) -> np.ndarray:
    """Render the editor; all looks start from the current ISP output, never another look."""
    for key, value in (("lf_department", "WORLD"), ("lf_preset", NO_LOOK), ("lf_intensity", 70)):
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.setdefault("lf_undo", [])
    st.session_state.setdefault("lf_redo", [])
    st.session_state.setdefault("lf_history_last", None)

    st.markdown("<div class='lf-eyebrow'>LUMAFORGE · PHOTO STUDIO</div>", unsafe_allow_html=True)
    title, undo_col, redo_col, reset_col = st.columns([5, 1, 1, 1])
    with title:
        st.subheader(Path(source_name).stem)
        st.caption("Original preserved · edits are rendered from the current ISP output")
    with undo_col:
        st.button("Undo", key="lf_undo_button", on_click=_undo, disabled=not st.session_state["lf_undo"], use_container_width=True)
    with redo_col:
        st.button("Redo", key="lf_redo_button", on_click=_redo, disabled=not st.session_state["lf_redo"], use_container_width=True)
    with reset_col:
        st.button("Reset", key="lf_reset_button", on_click=_reset_look, use_container_width=True)

    _render_pipeline(input_kind, elapsed_ms)
    department = st.radio("Creative departments", DEPARTMENTS, horizontal=True, key="lf_department",
                          label_visibility="collapsed")
    names = list(LOOKS[department])
    if st.session_state.get("lf_preset") not in (*names, NO_LOOK):
        st.session_state["lf_preset"] = names[0]
        if department == "ERA":
            st.session_state["lf_era_timeline"] = 0
    selected_name = st.session_state["lf_preset"]
    if department == "ERA":
        timeline_index = int(st.session_state.get("lf_era_timeline", 0))
        timeline_index = int(np.clip(timeline_index, 0, len(names) - 1))
        st.select_slider("Drag through photographic eras", options=list(range(len(names))),
                         value=timeline_index, format_func=lambda index: names[index],
                         key="lf_era_timeline", on_change=_sync_era_timeline)
        selected_name = st.session_state["lf_preset"]
    intensity = st.slider("Look intensity", 0, 100, key="lf_intensity", help="Applied to the selected photo from its ISP output.") / 100.0

    effect_controls: dict[str, float] = {}
    active_cfg = LOOKS[department].get(selected_name, {})
    effect_specs = (("grain", "Film grain", .20), ("scratches", "Scratches", .20),
                    ("vignette", "Vignette", .40), ("fade", "Fade", .35),
                    ("halation", "Halation", .20), ("softness", "Softness", .25))
    with st.expander("Fine-tune film texture and light", expanded=False):
        for start in range(0, len(effect_specs), 3):
            cols = st.columns(3)
            for col, (field, label, maximum) in zip(cols, effect_specs[start:start + 3]):
                key = f"lf_{field}_{department}_{selected_name}".replace(" ", "_").replace("-", "_")
                with col:
                    effect_controls[field] = st.slider(label, 0.0, maximum,
                        float(active_cfg.get(field, 0.0)), .01, key=key, format="%.2f")

    presets = LOOKS[department]
    st.markdown(f"#### {department.title()} looks")
    st.caption("Select a live thumbnail. Presets are creative color treatments; they do not recreate a real location, era, film, or lighting condition.")
    thumb_source = Image.fromarray(np.uint8(np.clip(isp_output, 0, 1) * 255))
    thumb_base = ImageOps.fit(thumb_source, (88, 88), method=Image.Resampling.BILINEAR)
    thumb_float = np.asarray(thumb_base, dtype=np.float32) / 255.0
    for start in range(0, len(names), 4):
        cols = st.columns(4, gap="small")
        for column, name in zip(cols, names[start:start + 4]):
            cfg = presets[name]
            with column:
                thumb = apply_look(thumb_float, cfg, intensity)
                st.image(np.uint8(thumb * 255 + .5), use_container_width=True)
                st.markdown(f"**{name}**" + (" · active" if selected_name == name else ""))
                st.caption(cfg["description"])
                st.button("Selected" if selected_name == name else "Apply look",
                          key=f"lf_apply_{department}_{name}".replace(" ", "_"),
                          on_click=_select_look, args=(name,), use_container_width=True)
    st.button("Remove creative look", key="lf_clear_look", on_click=_select_look,
              args=(NO_LOOK,), use_container_width=True)

    current = _current_history_state()
    previous = st.session_state.get("lf_history_last")
    if previous is not None and current != previous:
        st.session_state["lf_undo"].append(dict(previous))
        st.session_state["lf_redo"].clear()
    st.session_state["lf_history_last"] = dict(current)

    preset = presets.get(selected_name)
    edited = apply_look(isp_output, preset, intensity, effect_controls)
    st.markdown("#### Original / processed")
    mode_col, zoom_col = st.columns([2, 2])
    with mode_col:
        comparison = st.radio("Comparison view", ("Before / after slider", "Side by side"),
                              horizontal=True, key="lf_compare_mode", label_visibility="collapsed")
    with zoom_col:
        zoom = st.slider("Zoom", 1.0, 4.0, 1.0, .25, key="lf_zoom")
    center_x, center_y = .5, .5
    if zoom > 1:
        pan_x, pan_y = st.columns(2)
        center_x = pan_x.slider("Pan horizontal", 0.0, 1.0, .5, .01, key="lf_pan_x")
        center_y = pan_y.slider("Pan vertical", 0.0, 1.0, .5, .01, key="lf_pan_y")
    before_view = _crop_view(source_display, zoom, center_x, center_y)
    after_view = _crop_view(edited, zoom, center_x, center_y)
    if comparison == "Before / after slider":
        _render_wipe(before_view, after_view)
    else:
        left, right = st.columns(2, gap="small")
        left.markdown("**Original**")
        left.image(np.uint8(np.clip(before_view, 0, 1) * 255 + .5), use_container_width=True)
        right.markdown("**LumaForge edit**")
        right.image(np.uint8(np.clip(after_view, 0, 1) * 255 + .5), use_container_width=True)
    st.caption("Click either side-by-side preview to open Streamlit's full-screen image viewer. Zoom and pan are synchronized.")
    if input_kind == "rggb":
        st.caption("The original side is a colorized Bayer mosaic preview; the edited side is the reconstructed RGB output.")

    export = io.BytesIO()
    Image.fromarray(np.uint8(np.clip(edited, 0, 1) * 255 + .5)).save(export, format="PNG")
    st.download_button("Export edited photo", export.getvalue(),
                       f"{Path(source_name).stem}_lumaforge.png", "image/png", type="primary")
    return edited

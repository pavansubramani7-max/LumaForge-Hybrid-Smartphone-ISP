"""Shared Streamlit state, file decoding and cached ISP model loading."""
from __future__ import annotations
import io,sys
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def configure_sidebar():
    st.sidebar.header("Capture & runtime")
    upload=st.sidebar.file_uploader("RAW / RGB input",type=["npy","png","tif","tiff","jpg","jpeg","dng"],key="hybridisp_upload")
    if upload is not None:
        st.session_state["input_bytes"]=upload.getvalue(); st.session_state["input_name"]=upload.name
    elif st.session_state.get("input_name"):
        st.session_state.pop("input_bytes",None); st.session_state.pop("input_name",None)
    st.session_state["input_kind"]=st.sidebar.selectbox("Input type",["Auto-detect","Bayer RAW","RGB image"],key="global_input_kind")
    st.session_state["bayer_pattern"]=st.sidebar.selectbox("Bayer pattern",["RGGB","BGGR","GRBG","GBRG"],index=0,key="global_bayer_pattern")
    st.session_state["black_level"]=st.sidebar.number_input("Black level",min_value=0.,value=0.,step=1.,key="global_black")
    st.session_state["white_level"]=st.sidebar.number_input("White level",min_value=1.,value=65535.,step=1.,key="global_white")
    st.session_state["adaptive"]=st.sidebar.checkbox("Adaptive noise / illuminant control",value=True,key="global_adaptive")
    st.session_state["device"]=st.sidebar.selectbox("Device",["auto","cpu","cuda"],key="global_device")
    st.session_state["precision"]=st.sidebar.selectbox("Runtime precision",["fp32","fp16","bfloat16"],key="global_precision")
    st.session_state["tile_size"]=st.sidebar.select_slider("Tile size",options=[128,256,512,768],value=512,key="global_tile")
    st.sidebar.caption("FP16/BF16 are enabled on compatible accelerators. INT8 is a calibrated export path, not a live runtime toggle.")

def get_input():
    data=st.session_state.get("input_bytes")
    if not data: return None,None
    name=st.session_state.get("input_name",""); suffix=Path(name).suffix.lower()
    if suffix==".npy": image=np.load(io.BytesIO(data),allow_pickle=False)
    elif suffix==".dng":
        try: import rawpy
        except ImportError as exc: raise RuntimeError("DNG upload requires rawpy. Install the optional dependency from requirements.txt.") from exc
        path=ROOT/"work"/"streamlit_upload.dng"; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        with rawpy.imread(str(path)) as raw: image=raw.raw_image_visible.copy()
    else: image=np.asarray(Image.open(io.BytesIO(data)))
    selected=st.session_state.get("input_kind","Auto-detect")
    kind="raw" if selected=="Bayer RAW" else "rgb" if selected=="RGB image" else ("raw" if image.ndim==2 else "rgb")
    if kind=="raw" and image.ndim!=2: raise ValueError(f"Selected Bayer input requires a 2D mosaic; found {image.shape}")
    if kind=="rgb" and image.ndim not in (2,3): raise ValueError(f"Unsupported image shape {image.shape}")
    return image,kind

@st.cache_resource(show_spinner="Loading ISP models…")
def load_models(device_name="auto",precision="fp32",mobile_signature=None,hybrid_signature=None):
    import torch
    from isp.learned.models import MODEL_REGISTRY
    device=torch.device("cuda" if device_name=="auto" and torch.cuda.is_available() else ("cpu" if device_name=="auto" else device_name))
    models={}; meta={}
    for name,path in (("mobile_unet",ROOT/"models"/"mobile_unet.pt"),("hybrid_refiner",ROOT/"models"/"hybrid_refiner.pt")):
        if not path.exists(): continue
        checkpoint=torch.load(path,map_location=device,weights_only=True)
        arch=checkpoint.get("architecture",name)
        if arch not in MODEL_REGISTRY: continue
        model=MODEL_REGISTRY[arch](base=int(checkpoint.get("base",16))).to(device)
        model.load_state_dict(checkpoint["model"]); model.eval()
        if precision=="fp16" and device.type=="cuda": model.half()
        elif precision=="bfloat16" and device.type=="cuda": model.to(dtype=torch.bfloat16)
        models[name]=model; meta[name]={"checkpoint":str(path),"bytes":path.stat().st_size,"base":checkpoint.get("base"),"epoch":checkpoint.get("epoch"),"synthetic_data":checkpoint.get("synthetic_data",False)}
    return models,meta,str(device)

def active_models():
    mobile_path=ROOT/"models"/"mobile_unet.pt"; hybrid_path=ROOT/"models"/"hybrid_refiner.pt"
    mobile_sig=(mobile_path.stat().st_mtime_ns,mobile_path.stat().st_size) if mobile_path.exists() else None
    hybrid_sig=(hybrid_path.stat().st_mtime_ns,hybrid_path.stat().st_size) if hybrid_path.exists() else None
    return load_models(st.session_state.get("device","auto"),st.session_state.get("precision","fp32"),mobile_sig,hybrid_sig)

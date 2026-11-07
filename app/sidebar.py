"""Shared per-page capture and runtime controls for the multipage app."""
from __future__ import annotations
import streamlit as st
from app.theme import apply_theme

def configure_sidebar():
    apply_theme()
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
    st.sidebar.caption("FP16/BF16 need a compatible accelerator. INT8 is a calibrated export path, not a live runtime toggle.")

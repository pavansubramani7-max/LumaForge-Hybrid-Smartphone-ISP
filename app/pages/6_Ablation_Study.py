from pathlib import Path
import sys
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.components import get_image,upload_reference,score_images
from app.visuals import render_html_table,render_svg_bars
from app.sidebar import configure_sidebar
configure_sidebar()
from isp.classical.pipeline import ClassicalISP
from isp.raw_utils import normalize_raw

st.header("Ablation Study")
image,kind=get_image(); target=upload_reference("ablation_reference")
if image is None or kind!="raw": st.info("Upload a Bayer RAW and aligned RGB target to measure stage ablations.")
elif target is None: st.info("Add an aligned RGB reference to score the stage removals.")
elif st.button("Run classical-stage ablation",type="primary"):
    if target.shape[:2]!=image.shape[:2]: st.error("Reference dimensions must match the RAW mosaic output size.")
    else:
        raw=normalize_raw(image,st.session_state["black_level"],st.session_state["white_level"])
        pipeline=ClassicalISP(); stages=["white_balance","ccm","denoise","tone","sharpen"]
        rows=[]
        full=pipeline.run(raw,"raw",st.session_state["bayer_pattern"],0,1.).image
        rows.append({"variant":"full pipeline",**score_images(full,target)})
        for stage in stages:
            enabled={k:True for k in ["black_level","dead_pixel","white_balance","ccm","denoise","tone","sharpen"]}; enabled[stage]=False
            output=pipeline.run(raw,"raw",st.session_state["bayer_pattern"],0,1.,enabled=enabled).image
            rows.append({"variant":f"without {stage}",**score_images(output,target)})
        st.markdown(render_html_table(rows,"Measured ablation metrics"),unsafe_allow_html=True)
        metric=st.selectbox("Metric change",["PSNR dB","SSIM","CIEDE2000"])
        st.markdown(render_svg_bars({row["variant"]:row[metric] for row in rows},metric,"Measured value"),unsafe_allow_html=True)

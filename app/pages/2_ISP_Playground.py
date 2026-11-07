from pathlib import Path
import sys
import numpy as np
from PIL import Image
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.components import get_image,process_modes,visualized_raw,to_image,upload_reference,score_images
from app.sidebar import configure_sidebar
configure_sidebar()

st.header("ISP Playground")
image,kind=get_image()
if image is None:
    st.info("Upload an image in the global sidebar. Use a 2D Bayer mosaic for RAW reconstruction; RGB inputs skip demosaicing.")
else:
    st.subheader("Processing controls")
    controls=st.columns(4)
    mode=controls[0].selectbox("WB mode",["Adaptive gray-world","Manual gains"],index=0 if st.session_state.get("adaptive",True) else 1)
    demosaic=controls[1].selectbox("Demosaic",["bilinear","malvar_he_cutler","edge_directed"])
    denoise=controls[2].selectbox("Denoise",["bilateral","nlm","guided","wavelet","gaussian"])
    tone=controls[3].selectbox("Tone map",["aces","filmic","reinhard","srgb"])
    wbcols=st.columns(3)
    default=(1.5,1.,1.8) if kind=="raw" else (1.,1.,1.)
    gains=tuple(wbcols[i].slider(("R","G","B")[i]+" gain",.5,3.5,default[i],.05,key=f"play_wb_{i}") for i in range(3))
    c2=st.columns(3); exposure=c2[0].slider("Exposure",.4,2.5,1.,.05); sharp=c2[1].slider("Sharpen",0.,.8,.2,.05); blend=c2[2].slider("Hybrid blend",0.,1.,.85,.05)
    switches=st.columns(5)
    enabled={"black_level":switches[0].checkbox("Black level",True),"white_balance":switches[1].checkbox("WB",True),"ccm":switches[2].checkbox("CCM",True),"denoise":switches[3].checkbox("Denoise",True),"tone":switches[4].checkbox("Tone",True),"sharpen":True}
    st.session_state["adaptive"]=mode=="Adaptive gray-world"; st.session_state["wb_gains"]=gains
    st.session_state["demosaic_method"]=demosaic; st.session_state["denoise_method"]=denoise
    st.session_state["tone_method"]=tone; st.session_state["exposure"]=exposure; st.session_state["sharpen_amount"]=sharp
    st.session_state["fusion_alpha"]=blend
    st.session_state["enabled_stages"]=enabled
    if st.button("Process image",type="primary"):
        try:
            result=process_modes(image,kind)
            st.session_state["last_result"]=result
            if not st.session_state.get("adaptive",True): result["policy"]={}
        except Exception as exc: st.error(f"ISP processing failed: {exc}")
    result=st.session_state.get("last_result")
    if result:
        cols=st.columns(3 if kind=="raw" else 2)
        with cols[0]:
            st.markdown("**Visualized RAW**" if kind=="raw" else "**Input RGB**")
            st.image(visualized_raw(image,st.session_state["bayer_pattern"],st.session_state["black_level"],st.session_state["white_level"]) if kind=="raw" else image,use_container_width=True)
        with cols[1]: st.markdown("**Traditional ISP**"); st.image(to_image(result["traditional"]),use_container_width=True)
        if kind=="raw":
            with cols[2]: st.markdown("**Hybrid ISP**"); st.image(to_image(result["hybrid"]),use_container_width=True)
        st.write("**Adaptive policy**",result.get("policy") or "Manual parameters")
        st.write("**Per-stage timing (ms)**",result.get("timings",{}))
        choose=st.radio("Download",["Traditional","Hybrid"],horizontal=True) if kind=="raw" else "Traditional"
        output=result["hybrid"] if choose=="Hybrid" else result["traditional"]
        buf=__import__("io").BytesIO(); Image.fromarray(to_image(output)).save(buf,format="PNG")
        st.download_button("Save RGB result",buf.getvalue(),"hybridisp_output.png","image/png")
        target=upload_reference("playground_reference")
        if target is not None:
            if target.shape!=output.shape: st.error("Ground-truth dimensions must match the reconstructed image; align before scoring.")
            else: st.json(score_images(output,target))

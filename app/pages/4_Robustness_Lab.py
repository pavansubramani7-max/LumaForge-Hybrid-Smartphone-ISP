from pathlib import Path
import sys
import numpy as np
import streamlit as st

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.components import get_image,process_modes,to_image,upload_reference,score_images
from app.visuals import render_svg_chart
from app.sidebar import configure_sidebar
configure_sidebar()
from isp.raw_utils import normalize_raw,pack_bayer,unpack_bayer
from isp.learned.degradation import degrade_packed

st.header("Robustness Lab")
st.write("Apply controlled sensor noise, exposure, colour-temperature, white-balance, and vignetting changes. The adaptive controller remains enabled during recovery runs. Metrics require an aligned RGB reference.")
image,kind=get_image()
if image is None or kind!="raw": st.info("Upload a 2D Bayer RAW image to run robustness sweeps.")
else:
    c=st.columns(5)
    sigma=c[0].slider("Noise sigma",0.,.12,.03,.005)
    exposure=c[1].slider("Exposure gain",.2,1.6,.8,.05)
    cct=c[2].slider("Colour temperature (K)",2500,9000,5500,250)
    wb_error=c[3].slider("WB error",0.,.4,.12,.02)
    vignette=c[4].slider("Vignetting",0.,.6,.2,.02)
    target=upload_reference("robustness_reference")
    if st.button("Run stress test",type="primary"):
        pattern=st.session_state["bayer_pattern"]
        normalized=normalize_raw(image,st.session_state["black_level"],st.session_state["white_level"])
        packed=pack_bayer(normalized,pattern)
        stressed=degrade_packed(packed,sigma,(exposure,exposure),wb_error,vignette,seed=42,cct_kelvin=cct)
        stressed_raw=unpack_bayer(stressed,pattern)
        result=process_modes(stressed_raw,"raw",pattern,0,1.,True)
        st.session_state["robust_result"]={"clean":process_modes(normalized,"raw",pattern,0,1.,True),
            "stressed":result,"stressed_raw":stressed_raw}
    result=st.session_state.get("robust_result")
    if result:
        cols=st.columns(3)
        cols[0].image(to_image(result["clean"]["traditional"]),caption="Traditional · clean sensor",use_container_width=True)
        cols[1].image(to_image(result["stressed"]["traditional"]),caption="Traditional · stressed",use_container_width=True)
        cols[2].image(to_image(result["stressed"]["hybrid"]),caption="Hybrid · adaptive recovery",use_container_width=True)
        st.write("**Adaptive policy for stressed input**")
        st.json(result["stressed"]["policy"])
        if target is not None:
            if target.shape!=result["stressed"]["hybrid"].shape:
                st.error("Reference dimensions must match the output.")
            else:
                sweep=st.selectbox("Sweep variable",["Noise sigma","Exposure gain","Colour temperature","WB error"])
                if sweep=="Noise sigma": values=np.linspace(0,.1,6); xkey="value"; kwargs=lambda v:{"sigma":float(v)}
                elif sweep=="Exposure gain": values=np.linspace(.25,1.5,6); xkey="value"; kwargs=lambda v:{"exposure":float(v)}
                elif sweep=="Colour temperature": values=np.linspace(2500,9000,6,dtype=int); xkey="value"; kwargs=lambda v:{"cct_kelvin":int(v)}
                else: values=np.linspace(0,.4,6); xkey="value"; kwargs=lambda v:{"wb_error":float(v)}
                rows=[]
                clean_raw=normalize_raw(image,st.session_state["black_level"],st.session_state["white_level"])
                packed_clean=pack_bayer(clean_raw,st.session_state["bayer_pattern"])
                for value in values:
                    params={"sigma":sigma,"exposure":exposure,"cct_kelvin":cct,"wb_error":wb_error}
                    params.update(kwargs(value))
                    degraded=degrade_packed(packed_clean,params["sigma"],(params["exposure"],params["exposure"]),
                        params["wb_error"],vignette,seed=42,cct_kelvin=params["cct_kelvin"])
                    raw_case=unpack_bayer(degraded,st.session_state["bayer_pattern"])
                    outputs=process_modes(raw_case,"raw",st.session_state["bayer_pattern"],0,1.,True)
                    methods=["traditional","hybrid"]
                    if outputs.get("learned") is not None: methods.append("learned")
                    for method in methods:
                        score=score_images(outputs[method],target)
                        rows.append({xkey:float(value),"pipeline":method,"PSNR":score["PSNR dB"],
                            "SSIM":score["SSIM"],"CIEDE2000":score["CIEDE2000"]})
                st.subheader(f"Measured {sweep.lower()} recovery")
                metric=st.selectbox("Curve metric",["PSNR","SSIM","CIEDE2000"],key="robust_metric")
                series={method:[row[metric] for row in rows if row["pipeline"]==method]
                        for method in dict.fromkeys(row["pipeline"] for row in rows)}
                st.markdown(render_svg_chart(series,f"{metric} across {sweep.lower()}",values,sweep,metric),unsafe_allow_html=True)

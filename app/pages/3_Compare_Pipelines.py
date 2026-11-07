from pathlib import Path
import sys
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.components import get_image,process_modes,visualized_raw,to_image,upload_reference,score_images
from app.visuals import render_html_table
from app.sidebar import configure_sidebar
configure_sidebar()

st.header("Compare Pipelines")
image,kind=get_image()
if image is None: st.info("Upload a Bayer RAW or RGB image in the global sidebar.")
else:
    target=upload_reference("compare_reference")
    if st.button("Run Traditional / Learned / Hybrid",type="primary"):
        try: st.session_state["compare_result"]=process_modes(image,kind)
        except Exception as exc: st.error(str(exc))
    result=st.session_state.get("compare_result")
    if result:
        panels=[]
        if kind=="raw": panels.append(("Visualized RAW",visualized_raw(image,st.session_state["bayer_pattern"],st.session_state["black_level"],st.session_state["white_level"])))
        panels.append(("Traditional ISP",result["traditional"]))
        panels.append(("Learned ISP",result["learned"]))
        panels.append(("Hybrid ISP",result["hybrid"]))
        if target is not None: panels.append(("Ground Truth",target))
        columns=st.columns(len(panels))
        for col,(label,array) in zip(columns,panels):
            with col:
                st.markdown(f"**{label}**")
                if array is None: st.warning("Pure learned checkpoint not trained yet")
                else: st.image(to_image(array),use_container_width=True)
        if target is not None:
            if target.shape!=result["traditional"].shape: st.error("Ground truth and outputs must be spatially aligned.")
            else:
                rows=[{"pipeline":name,**score_images(result[name],target)} for name in ("traditional","hybrid")]
                if result["learned"] is not None: rows.append({"pipeline":"learned",**score_images(result["learned"],target)})
                st.subheader("Measured image-quality metrics"); st.markdown(render_html_table(rows),unsafe_allow_html=True)
                import numpy as np
                error=np.mean(np.abs(result["hybrid"]-target),axis=-1)
                st.markdown("**Hybrid absolute-error heatmap**"); st.image(error,use_container_width=True,clamp=True)
        st.caption("Scores are computed from this image/reference pair. No result is inferred from an unaligned reference.")

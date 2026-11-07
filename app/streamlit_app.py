from pathlib import Path
import sys
import streamlit as st

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.sidebar import configure_sidebar
from app.auto_review import render_auto_review

st.set_page_config(page_title="HybridISP Studio",page_icon="📷",layout="wide")
configure_sidebar()
render_auto_review()
st.caption("Problem P2 · Adaptive smartphone RAW-to-RGB reconstruction")
with st.expander("Project and model provenance"):
    st.info("The supplied archive was inspected: it contains payment transaction CSVs, not image data. The current checkpoint was trained on a local synthetic demo set; do not treat its scores as institute-dataset performance.")
    st.write("Use the navigation pages for robustness sweeps, efficiency benchmarks, ablations, training and dataset reports.")

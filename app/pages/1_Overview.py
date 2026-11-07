from pathlib import Path
import json,subprocess,sys
import streamlit as st
from app.state import ROOT,active_models
from app.sidebar import configure_sidebar
configure_sidebar()

st.header("Overview")
st.write("A physics-guided front-end handles sensor calibration and colour sampling. A compact learned model refines restoration; an adaptive controller adjusts denoising and fusion from each frame.")
st.graphviz_chart('''digraph ISP {rankdir=LR; raw[label="Bayer RAW"]; adaptive[label="Noise + illuminant\ncontroller"]; classical[label="WB · demosaic · CCM\nlight denoise"]; learned[label="Mobile U-Net /\nHybrid refiner"]; fusion[label="Confidence fusion"]; tone[label="Tone + sharpen"]; rgb[label="sRGB output"]; raw->adaptive->classical->learned->fusion->tone->rgb;}''')
models,meta,device=active_models(); c1,c2,c3=st.columns(3)
c1.metric("Runtime",device); c2.metric("Available learned models",len(models)); c3.metric("Institute RAW/RGB pairs",0 if not (ROOT/"data"/"pairs").exists() else "Inspecting")
for name,details in meta.items():
    st.success(f"{name}: {details['bytes']/1024**2:.3f} MiB · base={details['base']} · trained epoch={details['epoch']} · synthetic={details['synthetic_data']}")
if not models: st.warning("No compatible model checkpoint found. Run the quick synthetic demo trainer below or train on the institute RAW/RGB pairs after they are available.")
report=ROOT/"outputs"/"synthetic_benchmark"/"benchmark.json"
if report.exists():
    payload=json.loads(report.read_text(encoding="utf-8")); st.subheader("Latest synthetic-only benchmark")
    st.caption(f"{payload['test_images']} held-out generated scenes · {payload['device']} · synthetic_data={payload['synthetic_dataset']}")
    st.json(payload["methods"])
st.divider(); st.subheader("Train a fast local demo")
st.write("Creates only synthetic Bayer/RGB pairs locally—no external dataset download—and trains both pure-learned and hybrid models. Use this to exercise the app, not to report real-camera gains.")
if st.button("Generate synthetic data and quick-train both models",type="primary"):
    with st.status("Generating and training the synthetic demo",expanded=True) as status:
        gen=subprocess.run([sys.executable,"scripts/make_synthetic_dataset.py","--out","data/synthetic_pairs"],cwd=ROOT,capture_output=True,text=True)
        st.code(gen.stdout or gen.stderr)
        if gen.returncode==0:
            train=subprocess.run([sys.executable,"-m","isp.learned.train","--data_root","data/synthetic_pairs","--models","mobile_unet","hybrid_refiner","--quick"],cwd=ROOT,capture_output=True,text=True)
            st.code(train.stdout or train.stderr)
            status.update(label="Training finished" if train.returncode==0 else "Training failed",state="complete" if train.returncode==0 else "error")
            if train.returncode==0: st.success("Models are saved in models/. Reload the app to use them.")
        else: status.update(label="Synthetic data generation failed",state="error")

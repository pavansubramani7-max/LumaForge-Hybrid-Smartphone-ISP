from pathlib import Path
import sys,time,csv,io
import numpy as np
import streamlit as st
import torch
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from app.components import make_pipeline
from app.visuals import render_html_table,render_svg_chart,process_memory_mb
from isp.metrics.efficiency import model_stats,benchmark_callable
from isp.learned.infer import infer_pure_bayer
from app.sidebar import configure_sidebar
configure_sidebar()

st.header("Efficiency Benchmark")
st.write("Measures actual local inference. Desktop numbers are not mobile latency estimates. Use a representative phone and ONNX Runtime single-thread for the final mobile claim.")
pipeline,models,meta,device=make_pipeline()
if not models: st.warning("No learned model checkpoint is available. Train the synthetic demo or upload image pairs first.")
else:
    sizes=st.multiselect("Input resolution",[256,512,1080],default=[256])
    warm=st.number_input("Warm-up runs",1,20,2); runs=st.number_input("Timed runs",1,50,5)
    if st.button("Benchmark selected models",type="primary"):
        rows=[]
        for label in sizes:
            height,width={256:(256,256),512:(512,512),1080:(1080,1920)}[label]
            mosaic=np.full((height,width),.35,np.float32); packed=np.full((height//2,width//2,4),.35,np.float32)
            for name,model in models.items():
                if name=="mobile_unet": fn=lambda m=model,p=packed:infer_pure_bayer(m,p,device,tile_size=256,overlap=32)
                else:
                    from isp.classical.demosaic import bilinear_demosaic
                    base=bilinear_demosaic(mosaic,"RGGB"); raw=packed
                    dtype=next(model.parameters()).dtype
                    def fn(m=model,b=base,r=raw,h=height,w=width,dtype=dtype):
                        base_tensor=torch.from_numpy(b.transpose(2,0,1)[None]).to(device=device,dtype=dtype)
                        raw_tensor=torch.from_numpy(r.transpose(2,0,1)[None]).to(device=device,dtype=dtype)
                        noise=torch.full((1,1,h,w),.02,device=device,dtype=dtype)
                        with torch.inference_mode(): return m(base_tensor,raw_tensor,noise)
                if device=="cuda": torch.cuda.reset_peak_memory_stats()
                timing=benchmark_callable(fn,int(warm),int(runs),device=device)
                stats=model_stats(model,meta[name]["checkpoint"])
                row={"model":name,"resolution":f"{width}x{height}",**timing,**stats,**process_memory_mb()}
                row["peak_cuda_mb"]=float(torch.cuda.max_memory_allocated()/2**20) if device=="cuda" else None
                rows.append(row)
        st.session_state["efficiency_rows"]=rows
    rows=st.session_state.get("efficiency_rows",[])
    if rows:
        st.markdown(render_html_table(rows,"Measured inference efficiency"),unsafe_allow_html=True)
        models_in_rows=list(dict.fromkeys(row["model"] for row in rows))
        selected_resolutions=[{256:"256x256",512:"512x512",1080:"1920x1080"}[v] for v in sizes]
        series={name:[next((row["p50_ms"] for row in rows if row["model"]==name and row["resolution"]==resolution),None)
                      for resolution in selected_resolutions] for name in models_in_rows}
        st.markdown(render_svg_chart(series,"Median inference latency",sizes,"Resolution label (1080 = 1080p)","Latency (ms)"),unsafe_allow_html=True)
        buffer=io.StringIO(); columns=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(buffer,fieldnames=columns); writer.writeheader(); writer.writerows(rows)
        st.download_button("Download efficiency CSV",buffer.getvalue(),"efficiency_benchmark.csv","text/csv")
st.caption("This page measures the FP32/FP16/BF16 checkpoints loaded by the app. INT8, pruned, and distilled models must be exported and loaded as separate artifacts before comparing them. RSS is process-wide; CUDA peak is device allocation. Missing measurements are reported as unavailable.")

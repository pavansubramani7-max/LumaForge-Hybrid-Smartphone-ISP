from pathlib import Path
import sys,json,csv,io
import streamlit as st
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from isp.evaluation.report import build_html_report,build_pdf_report
from app.sidebar import configure_sidebar
configure_sidebar()

st.header("Report Export")
benchmark=ROOT/"results"/"benchmark.json"
if not benchmark.exists(): benchmark=ROOT/"outputs"/"synthetic_benchmark"/"benchmark.json"
payload=json.loads(benchmark.read_text(encoding="utf-8")) if benchmark.exists() else {"methods":{},"test_images":0,"synthetic_dataset":None}
history=ROOT/"outputs"/"training_history.csv"
st.write("Exports use saved run artifacts; this page does not insert expected or placeholder performance values.")
st.json({k:v for k,v in payload.items() if k!="per_image"})
if benchmark.exists():
    st.download_button("Download benchmark JSON",benchmark.read_bytes(),"benchmark.json","application/json")
    rows=[]
    for name,metrics in payload.get("methods",{}).items():
        if "delta_vs_traditional" in metrics: continue
        rows.append({"method":name,**{k:v.get("mean") for k,v in metrics.items() if isinstance(v,dict) and "mean" in v},**{f"{k}_std":v.get("std") for k,v in metrics.items() if isinstance(v,dict) and "std" in v}})
    if rows:
        buf=io.StringIO(); writer=csv.DictWriter(buf,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        st.download_button("Download comparison CSV",buf.getvalue(),"benchmark_summary.csv","text/csv")
    st.download_button("Download HTML report",build_html_report(payload),"hybridisp_report.html","text/html")
    lines=["HybridISP Studio - experiment report",f"Dataset root: {payload.get('data_root')}",f"Test images: {payload.get('test_images')}",f"Synthetic dataset: {payload.get('synthetic_dataset')}",""]
    for method,metrics in payload.get("methods",{}).items():
        for metric,value in metrics.items():
            if isinstance(value,dict) and "mean" in value: lines.append(f"{method} {metric}: {value['mean']:.5f} +/- {value['std']:.5f} (n={value['n']})")
    st.download_button("Download PDF report",build_pdf_report(lines),"hybridisp_report.pdf","application/pdf")
if history.exists(): st.download_button("Download training history",history.read_bytes(),"training_history.csv","text/csv")
else: st.info("No training history available yet. Train using your paired image dataset or run the synthetic quick demo.")

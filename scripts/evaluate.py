"""Scene-level validation report for classical vs learned-refined hybrid ISP."""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
try:
    import psutil
except ImportError:
    psutil=None
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.isp.classical import run_classical
from src.isp.metrics import psnr, ssim, delta_e_76
from isp.metrics.image_metrics import delta_e_2000, ms_ssim, lpips_alex
from src.isp.model import TinyISPUNet, parameter_count, infer_image

def load(path):
    return np.load(path) if path.suffix.lower()==".npy" else np.asarray(Image.open(path))

def resolve_test_root(data_arg=None):
    """Resolve a dataset root or split directory to an existing paired split."""
    project_root = Path(__file__).resolve().parents[1]
    if data_arg:
        root = Path(data_arg).expanduser()
        if not root.is_absolute():
            root = project_root / root
        candidates = [root, root / "test"]
    else:
        candidates = [project_root / "data" / "synthetic_pairs" / "test",
                      project_root / "data" / "pairs" / "test"]
    for candidate in candidates:
        if (candidate / "raw").is_dir() and (candidate / "rgb").is_dir():
            return candidate.resolve()
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise SystemExit(
        "Evaluation data is unavailable or has no raw/ and rgb/ folders. "
        "Pass --data with the test split folder (or its dataset root). Tried:\n" + tried +
        "\nExpected matching RAW/RGB filenames, for example "
        "data/synthetic_pairs/test/raw and data/synthetic_pairs/test/rgb."
    )

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data", "--data_root", dest="data", default=None, help="test split folder or dataset root; defaults to the bundled synthetic test split"); p.add_argument("--checkpoint",default="models/isp_tiny.pt"); p.add_argument("--black",type=float,default=None); p.add_argument("--white",type=float,default=None); p.add_argument("--out",default="outputs/evaluation.json"); a=p.parse_args()
    project_root=Path(__file__).resolve().parents[1]
    checkpoint_path=Path(a.checkpoint).expanduser()
    if not checkpoint_path.is_absolute(): checkpoint_path=project_root/checkpoint_path
    output_path=Path(a.out).expanduser()
    if not output_path.is_absolute(): output_path=project_root/output_path
    device="cuda" if torch.cuda.is_available() else "cpu"; root=resolve_test_root(a.data); rawdir=root/"raw"; rgbdir=root/"rgb"
    model=None
    config={"black":0,"white":None,"wb":(1.5,1.,1.8),"exposure":1.25,"denoise":.08,"sharpen":.25,"noise_feature":.08}
    if checkpoint_path.exists():
        ck=torch.load(checkpoint_path,map_location=device,weights_only=True); config.update(ck.get("isp_config",{})); model=TinyISPUNet().to(device); model.load_state_dict(ck.get("model",ck))
        if ck.get("precision")=="fp16" and device=="cuda": model.half()
        model.eval()
        if device=="cuda": torch.cuda.reset_peak_memory_stats()
    if a.black is not None: config["black"]=a.black
    if a.white is not None: config["white"]=a.white
    synthetic_root=(project_root/"data"/"synthetic_pairs").resolve()
    is_synthetic=root.parent.resolve()==synthetic_root
    report={"data_root":str(root),"synthetic_dataset":is_synthetic,"device":device,"scenes":0,"classical":{},"hybrid":{},"model":None}
    process=psutil.Process() if psutil else None
    report["process_rss_before_mb"]=process.memory_info().rss/2**20 if process else None
    if model:
        report["model"]={"parameters":parameter_count(model),"checkpoint_bytes":checkpoint_path.stat().st_size,
                         "checkpoint":str(checkpoint_path),"synthetic_training":bool(ck.get("synthetic_data",False)),
                         "precision":ck.get("precision","fp32") if "ck" in locals() else "fp32"}
    def quality(image, target):
        return {"psnr_db":psnr(image,target),"ssim":ssim(image,target),
                "ms_ssim":ms_ssim(image,target),"delta_e76":delta_e_76(image,target),
                "delta_e2000":delta_e_2000(image,target),"lpips":lpips_alex(image,target)}
    records={"classical":[],"hybrid":[]}; scene_names=[]; times={"classical":[],"learned_stage":[],"hybrid_total":[]}
    raw_files=sorted(rp for rp in rawdir.iterdir() if rp.suffix.lower() in {".npy",".png",".tif",".tiff"})
    if not raw_files:
        raise SystemExit(f"No supported RAW files found in {rawdir}.")
    for rp in raw_files:
        target_path=next((rgbdir/(rp.stem+e) for e in (".png",".tif",".tiff",".jpg",".jpeg") if (rgbdir/(rp.stem+e)).exists()),None)
        if target_path is None: continue
        raw=load(rp); target=np.asarray(Image.open(target_path).convert("RGB"),np.float32)/255
        white=config["white"] or (float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype,np.integer) else 1.)
        if not times["classical"]:
            _=run_classical(raw,"rggb",config["black"],white,wb=config["wb"],exposure=config["exposure"],denoise_strength=config["denoise"],sharpen_amount=config["sharpen"])
        t=time.perf_counter(); base=run_classical(raw,"rggb",config["black"],white,wb=config["wb"],exposure=config["exposure"],denoise_strength=config["denoise"],sharpen_amount=config["sharpen"]); times["classical"].append((time.perf_counter()-t)*1000)
        if base.shape[:2]!=target.shape[:2]: raise ValueError(f"Dimensions differ for {rp.name}; no silent resizing in scored data")
        records["classical"].append(quality(base,target))
        scene_names.append(rp.name)
        if model:
            feat=np.concatenate([base,np.full((*base.shape[:2],1),config["noise_feature"],np.float32)],axis=-1)
            if device=="cuda": torch.cuda.synchronize()
            if not times["learned_stage"]:
                _=infer_image(model,feat,device)
                if device=="cuda": torch.cuda.synchronize()
            t=time.perf_counter()
            out=infer_image(model,feat,device)
            if device=="cuda": torch.cuda.synchronize()
            learned_ms=(time.perf_counter()-t)*1000
            times["learned_stage"].append(learned_ms)
            times["hybrid_total"].append(times["classical"][-1]+learned_ms)
            records["hybrid"].append(quality(out,target))
    def summarize(rows):
        result={}
        for key in (rows[0] if rows else []):
            values=[r[key] for r in rows if r.get(key) is not None]
            result[key]=float(np.nanmean(values)) if values else None
        return result
    if not records["classical"]:
        raise ValueError(f"No valid paired test images found under {root}. Expected raw/ and rgb/ with matching filename stems.")
    report["scenes"]=len(records["classical"]); report["classical"]=summarize(records["classical"])
    report["quality_std"]={method:{key:(float(np.std([row[key] for row in rows if row.get(key) is not None]))
        if any(row.get(key) is not None for row in rows) else None)
        for key in (rows[0] if rows else [])} for method,rows in records.items() if rows}
    report["per_scene"]=[{"image":name,"classical":records["classical"][i],
        **({"hybrid":records["hybrid"][i]} if i<len(records["hybrid"]) else {})}
        for i,name in enumerate(scene_names)]
    report["latency_ms"]={k:{"median":float(statistics.median(v)),"p95":float(np.percentile(v,95))} for k,v in times.items() if v}
    if model:
        report["hybrid"]=summarize(records["hybrid"])
        report["hybrid_delta_vs_classical"]={}
        for key,value in report["hybrid"].items():
            baseline=report["classical"].get(key)
            if value is not None and baseline is not None:
                report["hybrid_delta_vs_classical"][key]={
                    "absolute":float(value-baseline),
                    "percent":float((value-baseline)/max(abs(baseline),1e-12)*100),
                    "direction":("improved" if (value>baseline if key in {"psnr_db","ssim","ms_ssim"} else value<baseline) else "worse")}
        report["peak_cuda_memory_mb"]=float(torch.cuda.max_memory_allocated()/2**20) if device=="cuda" else None
    report["process_rss_after_mb"]=process.memory_info().rss/2**20 if process else None
    output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
if __name__=="__main__": main()

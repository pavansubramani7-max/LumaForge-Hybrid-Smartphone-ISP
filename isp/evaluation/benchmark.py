"""Actual held-out comparisons for traditional, pure learned and hybrid ISP."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from data.dataset import discover_pairs,_read_raw,make_splits
from isp.raw_utils import normalize_raw
from isp.classical.pipeline import ClassicalISP
from isp.classical.demosaic import bilinear_demosaic
from isp.classical.white_balance import apply_white_balance
from isp.classical.ccm import apply_ccm
from isp.classical.tone import tone_map
from isp.classical.sharpen import unsharp_mask
from isp.learned.models import MODEL_REGISTRY
from isp.learned.infer import infer_pure_bayer
from isp.hybrid.hybrid_pipeline import HybridISPPipeline
from isp.metrics.image_metrics import delta_e_76,delta_e_2000,ms_ssim,lpips_alex
from src.isp.metrics import psnr,ssim

def _load_model(path,device):
    if not Path(path).exists(): return None
    ck=torch.load(path,map_location=device,weights_only=True); name=ck.get("architecture",Path(path).stem)
    if name not in MODEL_REGISTRY: return None
    model=MODEL_REGISTRY[name](base=int(ck.get("base",16))).to(device); model.load_state_dict(ck["model"]); model.eval()
    return model,ck

def _score(pred,target):
    values={"psnr_db":psnr(pred,target),"ssim":ssim(pred,target),"ms_ssim":ms_ssim(pred,target),
            "delta_e76":delta_e_76(pred,target),"delta_e2000":delta_e_2000(pred,target),"lpips":lpips_alex(pred,target)}
    return {k:(float(v) if v is not None and np.isfinite(v) else None) for k,v in values.items()}

def run_dataset_benchmark(data_root,split_file=None,pattern="auto",black=0,white=None,device="cpu",models_dir="models",out_dir="results",fusion_alpha_scale=1.0):
    root=Path(data_root).resolve(); records=discover_pairs(root); manifests=[]
    if split_file:
        explicit=Path(split_file).expanduser().resolve()
        if not explicit.exists(): raise FileNotFoundError(f"Split manifest not found: {explicit}")
        manifests=[explicit]
    elif (root/"splits.json").exists():
        manifests=[root/"splits.json"]
    else:
        manifests=sorted(root.rglob("splits.json"))
        # A missing split is created deterministically instead of scoring the train set by accident.
        if not manifests:
            make_splits(root,root/"splits.json",seed=42)
            manifests=[root/"splits.json"]
    selected=[]; split_roots=[]
    for manifest_path in manifests:
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_root=Path(manifest.get("data_root",manifest_path.parent)).expanduser().resolve()
        if not dataset_root.exists(): dataset_root=manifest_path.parent.resolve()
        split_roots.append(dataset_root)
        wanted=set(manifest.get("test",[]))
        selected.extend(r for r in records if r.key in wanted and r.raw_path.is_relative_to(dataset_root))
    records=selected
    if not records: raise ValueError(f"No test pairs found under {root}; run inspector and create splits.json first")
    device=torch.device(device); models_dir=Path(models_dir)
    mobile_load=_load_model(models_dir/"mobile_unet.pt",device); hybrid_load=_load_model(models_dir/"hybrid_refiner.pt",device)
    mobile=mobile_load[0] if mobile_load else None; hybrid_model=hybrid_load[0] if hybrid_load else None
    config={"tile_size":512,"tile_overlap":64,"wb_gains":(1.5,1.,1.8),
            "fusion_alpha_scale":float(fusion_alpha_scale)}
    pipeline=HybridISPPipeline(ClassicalISP(),hybrid_model,mobile,device,config)
    all_rows=[]
    for record in records:
        raw,meta=_read_raw(record.raw_path)
        if raw.ndim!=2: continue
        target=np.asarray(Image.open(record.rgb_path).convert("RGB"),np.float32)/255
        metadata_path=next((parent/"dataset_metadata.json" for parent in record.raw_path.parents
                            if parent==root or root in parent.parents or parent==root.parent
                            if (parent/"dataset_metadata.json").exists()),None)
        dataset_metadata=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else {}
        raw_pattern=meta.get("bayer_pattern",{}).get("pattern") or meta.get("bayer_pattern_guess",{}).get("pattern","unknown")
        if pattern=="auto":
            raw_pattern=dataset_metadata.get("bayer_pattern",raw_pattern)
            pattern_use=raw_pattern if raw_pattern!="unknown" else "RGGB"
        else: pattern_use=pattern
        if white is None: white_level=float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype,np.integer) else 1.
        else: white_level=white
        pipeline_result=pipeline.process_raw(raw,pattern_use,black,white_level,adaptive=True)
        traditional=pipeline_result.traditional; hybrid=pipeline_result.hybrid
        learned=None
        if mobile is not None:
            normalized=normalize_raw(raw,black,white_level)
            from isp.raw_utils import pack_bayer
            packed=pack_bayer(normalized,pattern_use)
            linear=infer_pure_bayer(mobile,packed,device)
            learned=unsharp_mask(tone_map(linear),.2)
        candidates={"traditional":traditional,"hybrid":hybrid}
        if learned is not None: candidates["learned"]=learned
        for name,image in candidates.items():
            if image.shape!=target.shape: raise ValueError(f"Image dimensions mismatch for {record.key}: {image.shape} vs {target.shape}")
        row={"key":record.key,"traditional":_score(traditional,target),"hybrid":_score(hybrid,target)}
        if learned is not None: row["learned"]=_score(learned,target)
        all_rows.append(row)
    if not all_rows: raise ValueError("No supported 2D Bayer RAW test pairs were found")
    methods=sorted({m for row in all_rows for m in row if m!="key"})
    summary={}
    metric_names=list(all_rows[0]["traditional"].keys())
    for method in methods:
        summary[method]={}
        for metric in metric_names:
            vals=[row[method][metric] for row in all_rows if method in row and row[method][metric] is not None]
            if vals: summary[method][metric]={"mean":float(np.mean(vals)),"std":float(np.std(vals)),"n":len(vals)}
    if "traditional" in summary:
        base=summary["traditional"]
        for method in methods:
            if method=="traditional": continue
            summary[method]["delta_vs_traditional"]={}
            for metric in metric_names:
                if metric in base and metric in summary[method]:
                    b=base[metric]["mean"]; v=summary[method][metric]["mean"]
                    summary[method]["delta_vs_traditional"][metric]=float(v-b)
    synthetic_flags=[]
    for record in records:
        metadata_path=next((parent/"dataset_metadata.json" for parent in record.raw_path.parents
                            if parent==root or root in parent.parents
                            if (parent/"dataset_metadata.json").exists()),None)
        metadata=json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path else {}
        synthetic_flags.append(bool(metadata.get("synthetic",False)))
    result={"data_root":str(root),"test_images":len(all_rows),"device":str(device),
        "fusion_alpha_scale":float(fusion_alpha_scale),
        "synthetic_dataset":bool(synthetic_flags and all(synthetic_flags)),"split_roots":[str(p) for p in split_roots],
        "methods":summary,"per_image":all_rows}
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); (out/"benchmark.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    import csv
    with (out/"benchmark.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["key","method",*metric_names]); writer.writeheader()
        for row in all_rows:
            for method in methods: writer.writerow({"key":row["key"],"method":method,**row.get(method,{})})
    return result

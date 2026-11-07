"""Inspect local RAW/RGB data, report pair candidates and save a sample grid."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.dataset import discover_pairs, _read_raw

RAW_EXTS_FOR_REPORT={".npy", ".png", ".tif", ".tiff", ".dng", ".raw", ".jpg", ".jpeg"}


def bit_depth(array):
    if np.issubdtype(array.dtype, np.integer): return int(np.iinfo(array.dtype).bits)
    return f"float ({array.dtype})"


def raw_preview(raw):
    x = np.asarray(raw, np.float32)
    if x.ndim == 3 and x.shape[-1] == 4:
        x = np.stack([x[...,0], (x[...,1]+x[...,2])/2, x[...,3]], axis=-1)
    elif x.ndim == 3 and x.shape[0] == 4:
        x = np.moveaxis(x,0,-1)
        x = np.stack([x[...,0], (x[...,1]+x[...,2])/2, x[...,3]], axis=-1)
    elif x.ndim == 2:
        # False-colour Bayer preview for inspection only; it is not a demosaiced result.
        h,w=x.shape; h-=h%2; w-=w%2; m=x[:h,:w]
        x=np.zeros((h,w,3),np.float32); x[0::2,0::2,0]=m[0::2,0::2]; x[0::2,1::2,1]=m[0::2,1::2]; x[1::2,0::2,1]=m[1::2,0::2]; x[1::2,1::2,2]=m[1::2,1::2]
    lo,hi=np.percentile(x,[1,99.5]); x=np.clip((x-lo)/max(hi-lo,1e-8),0,1)
    return Image.fromarray(np.uint8(x*255))


def make_grid(pairs, output, count=6, thumb=(420,280)):
    selected=pairs[:count]
    if not selected: return None
    label_h=28; gap=10; rows=[]
    canvas=Image.new("RGB",(thumb[0]*2+gap*3,(thumb[1]+label_h)*len(selected)+gap*(len(selected)+1)),"#202a33")
    draw=ImageDraw.Draw(canvas)
    for i,pair in enumerate(selected):
        y=gap+i*(thumb[1]+label_h+gap)
        try:
            raw,_=_read_raw(pair.raw_path); raw_img=raw_preview(raw)
            rgb_img=Image.open(pair.rgb_path).convert("RGB")
            raw_img.thumbnail(thumb); rgb_img.thumbnail(thumb)
            canvas.paste(raw_img,(gap,y+label_h)); canvas.paste(rgb_img,(gap*2+thumb[0],y+label_h))
            draw.text((gap,y+5),f"RAW preview · {pair.key}",fill="white")
            draw.text((gap*2+thumb[0],y+5),"RGB target",fill="white")
        except Exception as exc:
            draw.text((gap,y+5),f"{pair.key}: {exc}",fill="#ff8e8e")
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); canvas.save(output)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root",default="./data")
    parser.add_argument("--output",default="outputs/dataset_sample_grid.png")
    parser.add_argument("--json_output",default="outputs/dataset_inspection.json")
    parser.add_argument("--samples",type=int,default=6)
    args=parser.parse_args(); root=Path(args.data_root).expanduser().resolve()
    if not root.exists():
        print(json.dumps({"data_root":str(root),"error":"Dataset folder does not exist yet."},indent=2)); return 2
    pairs=discover_pairs(root)
    files=[p for p in root.rglob("*") if p.is_file()]
    extensions=Counter(p.suffix.lower() or "[no extension]" for p in files)
    summary={"data_root":str(root),"files_scanned":len(files),"extension_counts":dict(sorted(extensions.items())),
             "candidate_pairs":len(pairs),"pairs":[]}
    dataset_metadata={}
    for metadata_name in ("dataset_metadata.json","metadata.json"):
        metadata_path=root/metadata_name
        if metadata_path.exists():
            try: dataset_metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError,json.JSONDecodeError): dataset_metadata={}
            break
    summary["non_image_file_examples"]=[str(p.relative_to(root)) for p in files if p.suffix.lower() not in RAW_EXTS_FOR_REPORT][:25]
    for pair in pairs:
        try:
            raw,meta=_read_raw(pair.raw_path)
            target=np.asarray(Image.open(pair.rgb_path))
            guessed=meta.get("bayer_pattern_guess",{})
            authoritative=meta.get("bayer_pattern",{})
            known_pattern=authoritative.get("pattern") or dataset_metadata.get("bayer_pattern")
            bayer=({"pattern":known_pattern,"confidence":1.0,"method":"camera/dataset metadata"}
                   if known_pattern else guessed or {"pattern":"unknown"})
            summary["pairs"].append({"key":pair.key,"raw":str(pair.raw_path),"rgb":str(pair.rgb_path),
                "raw_shape":list(raw.shape),"raw_dtype":str(raw.dtype),"raw_bit_depth":bit_depth(raw),
                "bayer":bayer,"bayer_heuristic":guessed,
                "rgb_shape":list(target.shape),"rgb_dtype":str(target.dtype),"rgb_bit_depth":bit_depth(target)})
        except Exception as exc:
            summary["pairs"].append({"key":pair.key,"raw":str(pair.raw_path),"rgb":str(pair.rgb_path),"error":str(exc)})
    grid=make_grid(pairs,args.output,args.samples)
    summary["sample_grid"]=str(grid) if grid else None
    report_path=Path(args.json_output); report_path.parent.mkdir(parents=True,exist_ok=True)
    report_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    summary["inspection_json"]=str(report_path)
    print(json.dumps(summary,indent=2))
    return 0 if pairs else 1

if __name__=="__main__":
    raise SystemExit(main())

"""Create a tiny, deterministic Bayer/RGB demo set without downloading imagery."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


def make_scene(seed: int, size: int) -> np.ndarray:
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[0:size,0:size]
    t=x/max(size-1,1)
    top=np.array(rng.uniform(.12,.62,3)); bottom=np.array(rng.uniform(.28,.9,3))
    linear=top[None,None,:]*(1-t[...,None])+bottom[None,None,:]*t[...,None]
    # Smooth illumination variation, fine texture, and distinct colour regions.
    radial=((x-size*.52)**2+(y-size*.48)**2)/(size*size)
    linear*=np.clip(1.08-.55*radial,.55,1.1)[...,None]
    image=Image.fromarray(np.uint8(np.clip(linear,0,1)*255))
    draw=ImageDraw.Draw(image)
    for _ in range(18):
        x0=int(rng.integers(0,size-8)); y0=int(rng.integers(0,size-8))
        x1=min(size-1,x0+int(rng.integers(6,max(7,size//3))))
        y1=min(size-1,y0+int(rng.integers(6,max(7,size//3))))
        color=tuple(int(c) for c in rng.integers(25,256,3))
        if rng.random()<.5: draw.rectangle((x0,y0,x1,y1),fill=color,outline=(245,245,245),width=1)
        else: draw.ellipse((x0,y0,x1,y1),fill=color,outline=(245,245,245),width=1)
    # Repeating fine-detail bars make sharpening and demosaic errors measurable.
    for row in range(size//4,size,size//12 or 1):
        draw.line((0,row,size-1,row),fill=(230,230,230),width=1)
    return np.asarray(image,dtype=np.uint8)


def srgb_to_linear(x):
    x=np.asarray(x,np.float32)/255
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def make_raw(rgb, rng, low_light=False):
    linear=srgb_to_linear(rgb)
    scale=float(rng.uniform(.18,.9) if low_light else rng.uniform(.55,1.0))
    linear=np.clip(linear*scale,0,1)
    h,w=linear.shape[:2]
    mosaic=np.empty((h,w),np.float32)
    mosaic[0::2,0::2]=linear[0::2,0::2,0]
    mosaic[0::2,1::2]=linear[0::2,1::2,1]
    mosaic[1::2,0::2]=linear[1::2,0::2,1]
    mosaic[1::2,1::2]=linear[1::2,1::2,2]
    photons=float(rng.uniform(250,1600))
    noisy=rng.poisson(np.maximum(mosaic,0)*photons)/photons
    noisy+=rng.normal(0,float(rng.uniform(.0015,.012)),mosaic.shape)
    return np.uint16(np.clip(noisy,0,1)*65535+.5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",default="data/synthetic_pairs"); p.add_argument("--size",type=int,default=128)
    p.add_argument("--train",type=int,default=24); p.add_argument("--val",type=int,default=3); p.add_argument("--test",type=int,default=3); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(); root=Path(a.out); rng=np.random.default_rng(a.seed); scene_id=0
    split_keys={"train":[],"val":[],"test":[]}
    for split,count in (("train",a.train),("val",a.val),("test",a.test)):
        for kind in ("raw","rgb"):(root/split/kind).mkdir(parents=True,exist_ok=True)
        for _ in range(count):
            rgb=make_scene(a.seed+scene_id,a.size); raw=make_raw(rgb,rng,low_light=(scene_id%3==0))
            stem=f"scene_{scene_id:04d}"
            np.save(root/split/"raw"/f"{stem}.npy",raw)
            Image.fromarray(rgb,"RGB").save(root/split/"rgb"/f"{stem}.png")
            split_keys[split].append(stem.replace("_", ""))
            scene_id+=1
    split_path=root/"splits.json"
    split_path.write_text(__import__("json").dumps({"seed":a.seed,"data_root":str(root.resolve()),
        "fractions":{"train":a.train/scene_id,"val":a.val/scene_id,"test":a.test/scene_id},
        "counts":{"total":scene_id,"train":a.train,"val":a.val,"test":a.test},**split_keys},indent=2),encoding="utf-8")
    (root/"dataset_metadata.json").write_text(__import__("json").dumps({"bayer_pattern":"RGGB","black_level":0,"white_level":65535,
        "synthetic":True,"seed":a.seed,"bit_depth":16,"target_color_space":"sRGB"},indent=2),encoding="utf-8")
    print(f"Generated {scene_id} synthetic RAW/RGB pairs at {root.resolve()} (RGGB, uint16 mosaic, uint8 sRGB target).")
    print(f"Saved deterministic split manifest to {split_path.resolve()}")
    print("Synthetic-only data: use it to exercise the pipeline, not to claim real-camera performance.")

if __name__=="__main__": main()

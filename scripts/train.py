"""Supervised training on aligned RAW/RGB pairs.

Expected directories: data/pairs/{train,val}/raw and .../rgb; matching file stems.
RAW files are 2-D RGGB mosaics (PNG/TIFF/NPY); RGB targets are aligned images.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import sys
import random
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.isp.model import TinyISPUNet, parameter_count
from src.isp.classical import run_classical


def load(path):
    return np.load(path) if path.suffix.lower() == ".npy" else np.asarray(Image.open(path))

def packed_planes(raw):
    x = raw.astype(np.float32)
    if np.issubdtype(raw.dtype, np.integer): x /= np.iinfo(raw.dtype).max
    # Separate RGGB samples are resized to the original spatial resolution.
    h, w = x.shape[:2]
    planes = [x[0::2,0::2], x[0::2,1::2], x[1::2,0::2], x[1::2,1::2]]
    return np.stack([np.asarray(Image.fromarray(p).resize((w,h), Image.Resampling.BILINEAR)) for p in planes])

def classical_features(raw, config):
    white=config["white"]
    if white is None: white=float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype,np.integer) else 1.0
    rgb=run_classical(raw,"rggb",black=config["black"],white=white,wb=config["wb"],exposure=config["exposure"],denoise_strength=config["denoise"],sharpen_amount=config["sharpen"])
    noise=np.full((*rgb.shape[:2],1),config["noise_feature"],np.float32)
    return np.concatenate([rgb,noise],axis=-1)

class PairDataset(Dataset):
    def __init__(self, root, split, patch=256, augment=False, config=None):
        self.root, self.split, self.patch, self.augment = Path(root), split, patch, augment
        self.config=config or {"black":0,"white":None,"wb":(1.5,1.0,1.8),"exposure":1.25,"denoise":.08,"sharpen":.25,"noise_feature":.08}
        self.raw_dir = self.root / split / "raw"; self.rgb_dir = self.root / split / "rgb"
        self.files = sorted(p for p in self.raw_dir.iterdir() if p.suffix.lower() in {".png", ".tif", ".tiff", ".npy"})
        if not self.files: raise FileNotFoundError(f"No RAW mosaics in {self.raw_dir}")
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        rp = self.files[i]; candidates = [self.rgb_dir/(rp.stem+e) for e in (".png", ".tif", ".tiff", ".jpg", ".jpeg")]
        tp = next((p for p in candidates if p.exists()), None)
        if tp is None: raise FileNotFoundError(f"Missing RGB target for {rp.name}")
        raw, target = load(rp), np.asarray(Image.open(tp).convert("RGB"))
        x, y = classical_features(raw,self.config), target.astype(np.float32)/255
        if x.shape[:2] != y.shape[:2]:
            raise ValueError(f"RAW/RGB dimensions differ for {rp.name}: {x.shape[:2]} vs {y.shape[:2]}")
        h, w = y.shape[:2]; size = min(self.patch, h, w)
        if self.augment:
            top = np.random.randint(0, h-size+1); left = np.random.randint(0, w-size+1)
        else:
            top = (h-size)//2; left = (w-size)//2
        x=x[top:top+size,left:left+size]; y=y[top:top+size,left:left+size]
        if self.augment:
            if np.random.rand()<.5: x=x[:,::-1].copy(); y=y[:,::-1].copy()
            if np.random.rand()<.5: x=x[::-1].copy(); y=y[::-1].copy()
            # Sensor/illumination robustness: perturb the conventional ISP feature,
            # keeping the aligned clean sRGB target fixed as the restoration target.
            gain=np.random.uniform(.72,1.28)
            wb=np.random.uniform(.88,1.12,size=(1,1,3)).astype(np.float32)
            observed=np.clip(x[...,:3]*gain*wb,0,1)
            read_sigma=np.random.uniform(.002,.025)
            shot_sigma=np.random.uniform(.0,.025)
            noise=np.random.normal(0,read_sigma,observed.shape).astype(np.float32)
            noise+=np.random.normal(0,1,observed.shape).astype(np.float32)*shot_sigma*np.sqrt(np.maximum(observed,0))
            x=np.concatenate([np.clip(observed+noise,0,1),np.full((*observed.shape[:2],1),read_sigma+shot_sigma,np.float32)],axis=-1)
        return torch.from_numpy(x.transpose(2,0,1).copy()), torch.from_numpy(y.transpose(2,0,1).copy())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--data",default="data/pairs"); ap.add_argument("--epochs",type=int,default=30); ap.add_argument("--batch",type=int,default=4); ap.add_argument("--patch",type=int,default=256); ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--out",default="models/isp_tiny.pt"); ap.add_argument("--amp",action="store_true")
    ap.add_argument("--black",type=float,default=0); ap.add_argument("--white",type=float,default=None)
    ap.add_argument("--red-gain",type=float,default=1.5); ap.add_argument("--green-gain",type=float,default=1.0); ap.add_argument("--blue-gain",type=float,default=1.8)
    ap.add_argument("--exposure",type=float,default=1.25); ap.add_argument("--denoise",type=float,default=.08); ap.add_argument("--sharpen",type=float,default=.25); ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--history",default="outputs/training_history.csv",help="CSV file for per-epoch train and validation losses")
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config={"black":args.black,"white":args.white,"wb":(args.red_gain,args.green_gain,args.blue_gain),"exposure":args.exposure,"denoise":args.denoise,"sharpen":args.sharpen,"noise_feature":args.denoise}
    train=DataLoader(PairDataset(args.data,"train",args.patch,True,config),batch_size=args.batch,shuffle=True,num_workers=0)
    val=DataLoader(PairDataset(args.data,"val",args.patch,False,config),batch_size=1,shuffle=False,num_workers=0)
    model=TinyISPUNet().to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=.5,patience=3,min_lr=1e-6); loss_fn=nn.L1Loss(); scaler=torch.amp.GradScaler("cuda",enabled=args.amp and device.type=="cuda")
    best=float("inf"); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    metadata_path=Path(args.data)/"dataset_metadata.json"
    dataset_metadata={}
    if metadata_path.exists():
        import json
        dataset_metadata=json.loads(metadata_path.read_text(encoding="utf-8"))
    history=[]
    print(f"device={device} params={parameter_count(model):,} model_mb={parameter_count(model)*4/1e6:.2f} fp32")
    for epoch in range(1,args.epochs+1):
        model.train(); total=0
        for x,y in train:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=args.amp and device.type=="cuda"):
                pred=model(x); loss=loss_fn(pred,y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()
        model.eval(); v=0
        with torch.no_grad():
            for x,y in val: v+=loss_fn(model(x.to(device)),y.to(device)).item()
        v/=max(1,len(val)); train_l1=total/max(1,len(train)); scheduler.step(v)
        history.append({"epoch":epoch,"train_l1":train_l1,"val_l1":v,"learning_rate":opt.param_groups[0]["lr"]})
        print(f"epoch={epoch:03d} train_l1={train_l1:.5f} val_l1={v:.5f} lr={opt.param_groups[0]['lr']:.2e}")
        if v<best:
            best=v; torch.save({"model":model.cpu().state_dict(),"base":24,"epoch":epoch,"run_epochs":args.epochs,
                                "best_val_l1":best,"seed":args.seed,"isp_config":config,
                                "synthetic_data":bool(dataset_metadata.get("synthetic",False)),
                                "data_root":str(Path(args.data).resolve()),
                                "dataset_counts":{"train":len(train.dataset),"validation":len(val.dataset)}},out); model.to(device)
    history_path=Path(args.history); history_path.parent.mkdir(parents=True,exist_ok=True)
    with history_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["epoch","train_l1","val_l1","learning_rate"]); writer.writeheader(); writer.writerows(history)
    print(f"history={history_path}")
    print(f"saved={out} best_val_l1={best:.6f}")
if __name__=="__main__": main()

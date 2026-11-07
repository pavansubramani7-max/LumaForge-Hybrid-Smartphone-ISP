"""Config-driven CPU/GPU training for pure MobileUNet and hybrid refiner."""
from __future__ import annotations
import argparse,csv,json,random,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data.dataset import PairedRawRGBDataset,make_splits
from .models import MODEL_REGISTRY,count_parameters
from .losses import CompositeISPLoss

class ModelEMA:
    def __init__(self,model,decay=.995):
        import copy
        self.module=copy.deepcopy(model).eval(); self.decay=decay
        for p in self.module.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self,model):
        source=model.state_dict(); target=self.module.state_dict()
        for key,value in target.items():
            other=source[key].detach()
            if value.is_floating_point(): value.mul_(self.decay).add_(other,alpha=1-self.decay)
            else: value.copy_(other)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",default="configs/default.yaml"); ap.add_argument("--data_root",default=None)
    ap.add_argument("--models",nargs="+",choices=["mobile_unet","pynet_lite","hybrid_refiner"],default=["mobile_unet","hybrid_refiner"])
    ap.add_argument("--epochs",type=int,default=None); ap.add_argument("--batch_size",type=int,default=None)
    ap.add_argument("--patch_size",type=int,default=None); ap.add_argument("--base_channels",type=int,default=None)
    ap.add_argument("--quick",action="store_true",help="Use a short low-channel CPU demo run")
    ap.add_argument("--resume",default=None); ap.add_argument("--seed",type=int,default=None)
    ap.add_argument("--output_dir",default="models",help="Directory for model checkpoints")
    ap.add_argument("--history_path",default="outputs/training_history.csv",help="CSV path for epoch losses")
    args=ap.parse_args(argv)
    config=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root=Path(args.data_root or config["data"]["data_root"]).resolve()
    seed=args.seed if args.seed is not None else config["project"].get("seed",42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    train_cfg=config["training"]; data_cfg=config["data"]; model_cfg=config["model"]
    epochs=args.epochs or (2 if args.quick else train_cfg["epochs"])
    batch=args.batch_size or (2 if args.quick else train_cfg["batch_size"])
    patch=args.patch_size or (64 if args.quick else data_cfg["patch_size"])
    base=args.base_channels or (8 if args.quick else model_cfg["base_channels"])
    split_path=Path(data_cfg.get("splits_file","./splits.json"))
    if not split_path.is_absolute(): split_path=root/split_path.name
    if not split_path.exists():
        manifest=make_splits(root,split_path,seed=seed,train=data_cfg["train_fraction"],val=data_cfg["val_fraction"],test=data_cfg["test_fraction"])
        print(f"Wrote scene split manifest: {split_path} {manifest['counts']}")
    elif json.loads(split_path.read_text(encoding="utf-8")).get("data_root")!=str(root):
        raise ValueError(f"Existing {split_path} belongs to another data root. Move it or set a separate splits_file in config.")
    pattern=config["raw"].get("bayer_pattern","auto"); black=config["raw"].get("black_level",0); white=config["raw"].get("white_level")
    train_ds=PairedRawRGBDataset(root,"train",split_path,patch,pattern,True,black,white,noise_sigma=.015)
    val_ds=PairedRawRGBDataset(root,"val",split_path,patch,pattern,False,black,white,noise_sigma=0)
    generator=torch.Generator().manual_seed(seed)
    train_loader=DataLoader(train_ds,batch_size=batch,shuffle=True,num_workers=data_cfg.get("num_workers",0),generator=generator,drop_last=False)
    val_loader=DataLoader(val_ds,batch_size=1,shuffle=False,num_workers=0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp=bool(train_cfg.get("amp",True) and device.type=="cuda")
    loss_fn=CompositeISPLoss(vgg=0)
    output_dir=Path(args.output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    histories={}
    for model_name in args.models:
        model=MODEL_REGISTRY[model_name](base=base).to(device)
        if args.resume:
            checkpoint=torch.load(args.resume,map_location=device,weights_only=True)
            state=checkpoint.get("model",checkpoint); model.load_state_dict(state,strict=False)
        ema=ModelEMA(model); opt=torch.optim.AdamW(model.parameters(),lr=train_cfg["learning_rate"],weight_decay=train_cfg["weight_decay"])
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(1,epochs))
        scaler=torch.amp.GradScaler("cuda",enabled=amp); best=float("inf"); history=[]
        print(f"model={model_name} device={device} params={count_parameters(model):,} amp={amp}")
        for epoch in range(1,epochs+1):
            model.train(); train_total=0.
            for batch_data in train_loader:
                raw=batch_data["raw"].to(device); baseline=batch_data["baseline"].to(device); target=batch_data["rgb"].to(device); sigma=batch_data["noise_sigma"].to(device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,enabled=amp,dtype=torch.float16 if device.type=="cuda" else torch.bfloat16):
                    if model_name=="hybrid_refiner": pred=model(baseline,raw,sigma)
                    else: pred=model(raw)
                    loss,parts=loss_fn(pred,target)
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt); scaler.update(); ema.update(model); train_total+=float(loss.detach())
            scheduler.step(); ema.module.eval(); val_total=0.
            with torch.inference_mode():
                for batch_data in val_loader:
                    raw=batch_data["raw"].to(device); baseline=batch_data["baseline"].to(device); target=batch_data["rgb"].to(device); sigma=batch_data["noise_sigma"].to(device)
                    pred=ema.module(baseline,raw,sigma) if model_name=="hybrid_refiner" else ema.module(raw)
                    loss,_=loss_fn(pred,target); val_total+=float(loss)
            train_loss=train_total/max(1,len(train_loader)); val_loss=val_total/max(1,len(val_loader))
            history.append({"model":model_name,"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"lr":opt.param_groups[0]["lr"]})
            print(f"epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f} lr={opt.param_groups[0]['lr']:.2e}")
            if val_loss<best:
                best=val_loss; out=output_dir/f"{model_name}.pt"
                torch.save({"model":ema.module.cpu().state_dict(),"architecture":model_name,"base":base,"epoch":epoch,
                            "best_val_loss":best,"seed":seed,"synthetic_data":bool((root/"dataset_metadata.json").exists() and json.loads((root/"dataset_metadata.json").read_text()).get("synthetic")),
                            "config":config},out)
                ema.module.to(device)
        histories[model_name]=history
    history_path=Path(args.history_path); history_path.parent.mkdir(parents=True,exist_ok=True)
    with history_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["model","epoch","train_loss","val_loss","lr"]); writer.writeheader()
        for rows in histories.values(): writer.writerows(rows)
    print(f"saved training history: {history_path.resolve()}")

if __name__=="__main__": main()

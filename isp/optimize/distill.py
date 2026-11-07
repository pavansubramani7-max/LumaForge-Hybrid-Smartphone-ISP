"""Output-distillation trainer: MobileUNet teacher to compact hybrid refiner student."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data.dataset import PairedRawRGBDataset
from isp.learned.models import MobileUNetISP,HybridResidualRefiner
from isp.learned.losses import CompositeISPLoss

def main():
    p=argparse.ArgumentParser(); p.add_argument("--teacher",required=True); p.add_argument("--data_root",default="data"); p.add_argument("--splits",default="splits.json"); p.add_argument("--epochs",type=int,default=5); p.add_argument("--batch_size",type=int,default=2); p.add_argument("--base_channels",type=int,default=8); p.add_argument("--out",default="models/distilled_refiner.pt"); p.add_argument("--kd_weight",type=float,default=.5); a=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck=torch.load(a.teacher,map_location=device,weights_only=True); teacher=MobileUNetISP(base=int(ck.get("base",16))).to(device); teacher.load_state_dict(ck.get("model",ck)); teacher.eval()
    student=HybridResidualRefiner(base=a.base_channels).to(device)
    train=DataLoader(PairedRawRGBDataset(a.data_root,"train",a.splits,128,"auto",True,noise_sigma=.015),batch_size=a.batch_size,shuffle=True)
    loss_fn=CompositeISPLoss(vgg=0); opt=torch.optim.AdamW(student.parameters(),lr=2e-4)
    for epoch in range(a.epochs):
        student.train(); total=0.
        for batch in train:
            raw=batch["raw"].to(device); base=batch["baseline"].to(device); target=batch["rgb"].to(device); noise=batch["noise_sigma"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.inference_mode(): teacher_out=teacher(raw)
            student_out=student(base,raw,noise)
            supervised,_=loss_fn(student_out,target); distill=torch.nn.functional.l1_loss(student_out,teacher_out)
            loss=supervised+a.kd_weight*distill; loss.backward(); torch.nn.utils.clip_grad_norm_(student.parameters(),1.); opt.step(); total+=float(loss.detach())
        print(f"epoch={epoch+1} loss={total/max(1,len(train)):.5f}")
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":student.cpu().state_dict(),"architecture":"hybrid_refiner","base":a.base_channels,"distilled_from":str(a.teacher),"kd_weight":a.kd_weight},out)
    print(f"saved={out} bytes={out.stat().st_size}")
if __name__=="__main__": main()

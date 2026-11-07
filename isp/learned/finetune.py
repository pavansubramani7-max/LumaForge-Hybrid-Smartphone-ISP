"""Fine-tune an existing HybridISP checkpoint on a configured local dataset."""
import argparse,subprocess,sys

def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--config",default="configs/default.yaml"); p.add_argument("--data_root",default=None); p.add_argument("--epochs",type=int,default=5); p.add_argument("--models",nargs="+",default=["hybrid_refiner"]); a=p.parse_args()
    cmd=[sys.executable,"-m","isp.learned.train","--config",a.config,"--epochs",str(a.epochs),"--models",*a.models,"--resume",a.checkpoint]
    if a.data_root: cmd.extend(["--data_root",a.data_root])
    raise SystemExit(subprocess.call(cmd))

if __name__=="__main__": main()

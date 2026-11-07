import tempfile
import unittest
from pathlib import Path
import torch
from torch import nn
from isp.optimize.prune import prune_model,zero_fraction
from isp.optimize.quantize import quantize_dynamic,quantize_static_fx
from isp.optimize.precision import save_precision_checkpoint
from isp.learned.models import MobileUNetISP

class OptimizationTests(unittest.TestCase):
    def test_pruning_creates_requested_sparsity(self):
        model=nn.Sequential(nn.Conv2d(3,8,3),nn.ReLU(),nn.Conv2d(8,3,1))
        prune_model(model,.3)
        self.assertGreaterEqual(zero_fraction(model),.29)

    def test_dynamic_int8_runs(self):
        model=nn.Sequential(nn.Linear(8,16),nn.ReLU(),nn.Linear(16,2)).eval()
        quantized=quantize_dynamic(model)
        output=quantized(torch.rand(4,8))
        self.assertEqual(tuple(output.shape),(4,2))

    def test_static_int8_runs_on_conv_model(self):
        model=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.ReLU(),nn.Conv2d(4,3,1),nn.Sigmoid()).eval()
        calibration=[torch.rand(1,3,16,16) for _ in range(2)]; example=calibration[0]
        quantized=quantize_static_fx(model,calibration,example,lambda x:(x,))
        output=quantized(example)
        self.assertEqual(tuple(output.shape),(1,3,16,16))

    def test_static_int8_runs_on_pure_learned_isp(self):
        model=MobileUNetISP(base=4).eval(); example=torch.rand(1,4,32,32)
        quantized=quantize_static_fx(model,[example],example,lambda x:(x,))
        output=quantized(example)
        self.assertEqual(tuple(output.shape),(1,3,64,64))

    def test_fp16_checkpoint_export(self):
        workspace=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=workspace/"work") as folder:
            source=Path(folder)/"model.pt"; target=Path(folder)/"model_fp16.pt"
            torch.save({"model":nn.Linear(2,2).state_dict()},source)
            size=save_precision_checkpoint(source,target,"fp16")
            saved=torch.load(target,map_location="cpu",weights_only=True)
            self.assertGreater(size,0); self.assertEqual(saved["model"]["weight"].dtype,torch.float16)

if __name__=="__main__": unittest.main()

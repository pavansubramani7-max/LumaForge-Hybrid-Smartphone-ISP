import unittest
import numpy as np
import torch
from isp.learned.models import MobileUNetISP,PyNETLite,HybridResidualRefiner,count_parameters
from isp.learned.infer import infer_hybrid_refiner,infer_pure_bayer

class LearnedModelTests(unittest.TestCase):
    def test_pure_models_reconstruct_full_resolution(self):
        raw=torch.rand(1,4,32,32)
        for model in (MobileUNetISP(base=8),PyNETLite(base=8)):
            model.eval()
            with torch.inference_mode(): output=model(raw)
            self.assertEqual(tuple(output.shape),(1,3,64,64))
            self.assertTrue(torch.isfinite(output).all())

    def test_hybrid_refiner_identity_initialization_and_budget(self):
        model=HybridResidualRefiner(base=8).eval()
        classical=torch.rand(1,3,64,64); packed=torch.rand(1,4,32,32); noise=torch.full((1,1,64,64),.02)
        with torch.inference_mode(): output=model(classical,packed,noise)
        self.assertEqual(tuple(output.shape),tuple(classical.shape))
        self.assertTrue(torch.allclose(output,classical,atol=1e-6))
        self.assertLess(count_parameters(model),500_000)

    def test_tiled_pure_and_hybrid_dimensions(self):
        pure=MobileUNetISP(base=8).eval(); raw=torch.rand(1,4,48,56)
        with torch.inference_mode(): out=infer_pure_bayer(pure,raw[0].permute(1,2,0).numpy(),tile_size=32,overlap=8)
        self.assertEqual(out.shape,(96,112,3))
        hybrid=HybridResidualRefiner(base=8).eval(); base=np.random.default_rng(1).random((96,112,3),dtype=np.float32)
        packed=np.random.default_rng(2).random((48,56,4),dtype=np.float32)
        out=infer_hybrid_refiner(hybrid,base,packed,.02,tile_size=64,overlap=16)
        self.assertEqual(out.shape,base.shape)

if __name__=="__main__": unittest.main()

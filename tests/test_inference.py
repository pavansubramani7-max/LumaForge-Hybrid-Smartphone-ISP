import unittest
import numpy as np
import torch
from src.isp.model import infer_image


class InferenceTests(unittest.TestCase):
    def test_tiled_inference_arbitrary_frame_size(self):
        model=torch.nn.Conv2d(4,3,1).eval()
        image=np.random.default_rng(4).random((137,173,4),dtype=np.float32)
        result=infer_image(model,image,device="cpu",tile_size=64,overlap=16)
        self.assertEqual(result.shape,(137,173,3))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_untiled_inference_preserves_dimensions(self):
        model=torch.nn.Conv2d(4,3,1).eval()
        image=np.zeros((31,29,4),np.float32)
        result=infer_image(model,image,device="cpu",tile_size=64)
        self.assertEqual(result.shape,(31,29,3))


if __name__=="__main__": unittest.main()

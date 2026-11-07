import unittest
import numpy as np
from src.isp.metrics import psnr, ssim, delta_e_76


class MetricTests(unittest.TestCase):
    def test_identical_images(self):
        image=np.random.default_rng(1).random((24,20,3),dtype=np.float32)
        self.assertEqual(psnr(image,image),float("inf"))
        self.assertAlmostEqual(ssim(image,image),1.0,places=5)
        self.assertAlmostEqual(delta_e_76(image,image),0.0,places=6)

    def test_metrics_ordering(self):
        target=np.full((16,16,3),.5,np.float32)
        close=np.clip(target+.02,0,1); far=np.clip(target+.2,0,1)
        self.assertGreater(psnr(close,target),psnr(far,target))
        self.assertLess(ssim(far,target),ssim(close,target))
        self.assertLess(delta_e_76(close,target),delta_e_76(far,target))


if __name__=="__main__": unittest.main()

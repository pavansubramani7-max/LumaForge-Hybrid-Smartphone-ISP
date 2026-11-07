import unittest
import numpy as np
from isp.metrics.image_metrics import delta_e_76,delta_e_2000,ms_ssim

class ColorMetricTests(unittest.TestCase):
    def test_delta_e_identity_and_order(self):
        target=np.random.default_rng(8).random((32,35,3),dtype=np.float32)
        near=np.clip(target+.01,0,1); far=np.clip(target+.2,0,1)
        self.assertAlmostEqual(delta_e_2000(target,target),0,places=6)
        self.assertAlmostEqual(delta_e_76(target,target),0,places=6)
        self.assertLess(delta_e_2000(near,target),delta_e_2000(far,target))

    def test_multiscale_handles_odd_dimensions(self):
        image=np.random.default_rng(9).random((35,31,3),dtype=np.float32)
        self.assertAlmostEqual(ms_ssim(image,image),1,places=5)

if __name__=="__main__": unittest.main()

import unittest
import numpy as np
from isp.classical.pipeline import ClassicalISP
from isp.classical.ccm import interpolate_illuminant_ccm
from isp.hybrid.adaptive_controller import estimate_policy
from isp.hybrid.hybrid_pipeline import HybridISPPipeline

class HybridPipelineTests(unittest.TestCase):
    def test_controller_returns_bounded_policy(self):
        mosaic=np.random.default_rng(7).random((32,32),dtype=np.float32)
        policy=estimate_policy(mosaic,"RGGB",white_level=1.)
        self.assertGreaterEqual(policy.noise_sigma,0)
        self.assertTrue(2500<=policy.cct_kelvin<=9000)
        self.assertTrue(0<=policy.refiner_alpha<=1)

    def test_traditional_and_fallback_modes(self):
        mosaic=np.random.default_rng(9).random((32,32),dtype=np.float32)
        pipeline=HybridISPPipeline(config={"white_level":1.})
        traditional=pipeline.process_raw(mosaic,pattern="RGGB",white_level=1.,adaptive=False,mode="traditional")
        hybrid=pipeline.process_raw(mosaic,pattern="RGGB",white_level=1.,adaptive=False,mode="hybrid")
        self.assertEqual(traditional.traditional.shape,(32,32,3))
        self.assertEqual(hybrid.hybrid.shape,(32,32,3))
        self.assertTrue(np.all(np.isfinite(hybrid.hybrid)))

    def test_adaptive_illuminant_ccm_is_applied(self):
        class CaptureISP(ClassicalISP):
            def run(self,*args,**kwargs):
                self.used_ccm=kwargs.get("ccm")
                return super().run(*args,**kwargs)
        classical=CaptureISP(); mosaic=np.random.default_rng(21).random((32,32),dtype=np.float32)
        result=HybridISPPipeline(classical=classical).process_raw(mosaic,"RGGB",white_level=1.,adaptive=True,mode="traditional")
        expected=interpolate_illuminant_ccm(HybridISPPipeline().config["ccm_presets"],result.policy["cct_kelvin"])
        self.assertIsNotNone(classical.used_ccm)
        np.testing.assert_allclose(classical.used_ccm,expected,atol=1e-6)

if __name__=="__main__": unittest.main()

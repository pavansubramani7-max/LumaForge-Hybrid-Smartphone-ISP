import unittest
import numpy as np
from isp.raw_utils import pack_bayer, unpack_bayer, normalize_raw, pattern_from_indices
from isp.classical.demosaic import bilinear_demosaic, edge_directed_demosaic, malvar_he_cutler_demosaic
from isp.classical.ccm import apply_ccm, fit_ccm, interpolate_illuminant_ccm
from isp.classical.pipeline import ClassicalISP


class ModularISPTests(unittest.TestCase):
    def test_pack_unpack_roundtrip_all_patterns(self):
        mosaic=np.arange(64,dtype=np.float32).reshape(8,8)
        for pattern in ("RGGB","BGGR","GRBG","GBRG"):
            np.testing.assert_array_equal(unpack_bayer(pack_bayer(mosaic,pattern),pattern),mosaic)

    def test_pattern_metadata(self):
        self.assertEqual(pattern_from_indices([[0,1],[1,2]]),"RGGB")

    def test_demosaicers_constant_scene(self):
        mosaic=np.full((16,18),.4,np.float32)
        for fn in (bilinear_demosaic,edge_directed_demosaic,malvar_he_cutler_demosaic):
            rgb=fn(mosaic,"RGGB")
            self.assertEqual(rgb.shape,(16,18,3))
            self.assertTrue(np.all(np.isfinite(rgb)))

    def test_ccm_fit_recovers_known_matrix(self):
        rng=np.random.default_rng(3); camera=rng.random((30,3)); matrix=np.array([[1.1,-.05,0],[0,1.02,-.02],[.02,-.04,1.08]])
        reference=camera@matrix.T
        fit=fit_ccm(camera,reference,regularization=1e-10)
        np.testing.assert_allclose(fit,matrix,atol=1e-5)

    def test_named_ccm_interpolation_uses_illuminant_anchors(self):
        presets={"A":np.zeros((3,3)),"D65":np.ones((3,3))}
        cold=interpolate_illuminant_ccm(presets,2856)
        warm=interpolate_illuminant_ccm(presets,6504)
        np.testing.assert_array_equal(cold,np.zeros((3,3)))
        np.testing.assert_array_equal(warm,np.ones((3,3)))

    def test_mhc_preserves_measured_samples_for_each_pattern(self):
        mosaic=np.random.default_rng(17).random((20,22),dtype=np.float32)
        from isp.raw_utils import BAYER_PATTERNS
        for pattern,grid in BAYER_PATTERNS.items():
            result=malvar_he_cutler_demosaic(mosaic,pattern)
            for y in range(2):
                for x in range(2):
                    channel="RGB".index(grid[y][x])
                    np.testing.assert_allclose(result[y::2,x::2,channel],mosaic[y::2,x::2],atol=1e-6)

    def test_classical_pipeline_stages_and_rgb(self):
        raw=np.full((32,32),.25,np.float32)
        result=ClassicalISP().run(raw,input_kind="raw",pattern="RGGB",white_level=1.,
                                  denoise_method="bilateral",sharpen_amount=0)
        self.assertEqual(result.image.shape,(32,32,3))
        self.assertIn("demosaic",result.timings_ms)
        rgb=np.full((20,24,3),128,np.uint8)
        result=ClassicalISP().run(rgb,input_kind="rgb",wb_gains=(1,1,1))
        self.assertEqual(result.image.shape,(20,24,3))
        self.assertTrue(result.metadata["demosaic_skipped"])


if __name__=="__main__": unittest.main()

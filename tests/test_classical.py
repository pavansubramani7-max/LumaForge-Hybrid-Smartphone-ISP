import unittest
import numpy as np
from src.isp.classical import normalize_raw, demosaic_rggb, run_classical, color_matrix, tone_map, unpack_bayer_4ch


class ClassicalTests(unittest.TestCase):
    def test_black_white_normalization(self):
        raw=np.array([[100,300]],dtype=np.uint16)
        got=normalize_raw(raw,black=100,white=300)
        np.testing.assert_allclose(got,[[0,1]])

    def test_rggb_demosaic_shape_and_known_samples(self):
        mosaic=np.zeros((8,8),np.float32)
        mosaic[0::2,0::2]=.8; mosaic[0::2,1::2]=.4
        mosaic[1::2,0::2]=.4; mosaic[1::2,1::2]=.2
        rgb=demosaic_rggb(mosaic)
        self.assertEqual(rgb.shape,(8,8,3))
        np.testing.assert_allclose(rgb[2,2],[.8,.4,.2],atol=1e-6)
        np.testing.assert_allclose(rgb[2,3],[.8,.4,.2],atol=1e-6)
        np.testing.assert_allclose(rgb[3,2],[.8,.4,.2],atol=1e-6)
        np.testing.assert_allclose(rgb[3,3],[.8,.4,.2],atol=1e-6)

    def test_bilinear_demosaic_all_cfa_patterns(self):
        tiles={"RGGB":((0,1),(1,2)),"BGGR":((2,1),(1,0)),"GRBG":((1,0),(2,1)),"GBRG":((1,2),(0,1))}
        expected=np.array([.8,.4,.2],np.float32)
        for name,tile in tiles.items():
            mosaic=np.empty((12,14),np.float32)
            for y in range(2):
                for x in range(2): mosaic[y::2,x::2]=expected[tile[y][x]]
            got=demosaic_rggb(mosaic,name)
            np.testing.assert_allclose(got,np.broadcast_to(expected,got.shape),atol=1e-6,err_msg=name)

    def test_color_matrix_and_tone_ranges(self):
        image=np.ones((4,4,3),np.float32)*.5
        np.testing.assert_allclose(color_matrix(image,np.eye(3)),image)
        tone=tone_map(image,exposure=1.0)
        self.assertTrue(np.all(np.isfinite(tone)))
        self.assertTrue(np.all((tone>=0)&(tone<=1)))

    def test_rgb_pipeline_preserves_geometry(self):
        image=np.full((16,20,3),128,np.uint8)
        result=run_classical(image,"rgb",wb=(1,1,1))
        self.assertEqual(result.shape,(16,20,3))
        self.assertTrue(np.all((result>=0)&(result<=1)))
        # Rendered sRGB must be decoded before linear-light tone mapping; the
        # old double-gamma path pushed mid-gray close to white.
        result=run_classical(image,"rgb",wb=(1,1,1),exposure=1.0,denoise_strength=0,sharpen_amount=0)
        self.assertTrue(.50 < float(result.mean()) < .75)
        float_rgb=run_classical(np.full((16,20,3),128.,np.float32),"rgb",wb=(1,1,1),exposure=1.0,denoise_strength=0,sharpen_amount=0)
        np.testing.assert_allclose(float_rgb,result,atol=1e-6)

    def test_unpack_packed_bayer_row_major(self):
        packed=np.arange(2*3*4,dtype=np.uint16).reshape(2,3,4)
        mosaic=unpack_bayer_4ch(packed)
        self.assertEqual(mosaic.shape,(4,6))
        np.testing.assert_array_equal(mosaic[0::2,0::2],packed[...,0])
        np.testing.assert_array_equal(mosaic[0::2,1::2],packed[...,1])
        np.testing.assert_array_equal(mosaic[1::2,0::2],packed[...,2])
        np.testing.assert_array_equal(mosaic[1::2,1::2],packed[...,3])


if __name__=="__main__": unittest.main()

import unittest
import numpy as np
from isp.learned.degradation import degrade_packed

class DegradationTests(unittest.TestCase):
    def test_temperature_shift_changes_red_blue_balance(self):
        packed=np.full((32,32,4),.4,np.float32)
        warm=degrade_packed(packed,0,(1,1),0,0,seed=3,cct_kelvin=2500)
        cool=degrade_packed(packed,0,(1,1),0,0,seed=3,cct_kelvin=9000)
        warm_ratio=float(warm[...,0].mean()/warm[...,3].mean())
        cool_ratio=float(cool[...,0].mean()/cool[...,3].mean())
        self.assertGreater(warm_ratio,cool_ratio)

if __name__=="__main__": unittest.main()

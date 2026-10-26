"""Data leakage, exposure-unit and rectified-geometry checks for real clips."""
from pathlib import Path
import unittest
import numpy as np
import torch

from experiments.n1b_v2.real_dynamic import DynamicData, initialize


@unittest.skipUnless(torch.cuda.is_available() and Path('dataset/dsec_dynamic/zurich_city_04_c/left/000143.png').exists(),'downloaded DSEC clip and CUDA required')
class DynamicDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=DynamicData('dataset/dsec_dynamic/zurich_city_04_c',120)

    def test_measured_time_units_and_split(self):
        d=self.data
        self.assertEqual(set(d.train_frames)&set(d.test_frames),set())
        self.assertEqual(len(d.train),36);self.assertEqual(len(d.test),12)
        for s,side in enumerate(('left','right')):
            actual=np.loadtxt(d.root/f'exposure_{side}.txt',delimiter=',')[d.frames]
            np.testing.assert_allclose(d.bounds[s]*d.span*1e6+d.origin,actual,rtol=0,atol=1e-5)
        self.assertTrue(np.all(d.bounds[:,:,1]>d.bounds[:,:,0]))

    def test_stereo_geometry(self):
        left,right=self.data.cameras
        pts=torch.tensor([[0.,0.,5.,1.],[1.,1.,20.,1.]],device='cuda')
        a=(left.w2c@pts.T).T;b=(right.w2c@pts.T).T
        yl=left.fy*a[:,1]/a[:,2]+left.cy;yr=right.fy*b[:,1]/b[:,2]+right.cy
        torch.testing.assert_close(yl,yr,rtol=0,atol=1e-4)
        disparity=left.fx*a[:,0]/a[:,2]+left.cx-(right.fx*b[:,0]/b[:,2]+right.cx)
        baseline=torch.linalg.norm(right.position-left.position)
        torch.testing.assert_close(disparity,left.fx*baseline/pts[:,2],rtol=1e-5,atol=1e-4)

    def test_test_pixels_do_not_change_initialization(self):
        d=self.data;a=initialize(d,7,256);saved=[]
        try:
            for s in (0,1):
                for i in d.test_frames:
                    saved.append((s,i,d.cpu[s][i]));d.cpu[s][i]=np.full_like(d.cpu[s][i],255)
            b=initialize(d,7,256)
            for key in a:torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        finally:
            for s,i,image in saved:d.cpu[s][i]=image


if __name__=='__main__':unittest.main()

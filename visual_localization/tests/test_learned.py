"""Learned-feature pixel coordinates must still agree with Polycam depth."""
import importlib.util
import unittest
import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('kornia'),
                     'Install requirements-enhanced.txt for learned matching tests')
class LearnedGeometryTests(unittest.TestCase):
    def test_rotated_reference_pixels_map_back_exactly(self):
        from wifi_localization.learned import original_pixels
        image=np.arange(35).reshape(5,7)
        original=np.array([[0,0],[6,0],[0,4],[6,4],[3,2]])
        for k in range(4):
            rotated=np.rot90(image,k)
            points=[]
            for x,y in original:
                yy,xx=np.where(rotated==image[y,x]);points.append([xx[0],yy[0]])
            np.testing.assert_array_equal(original_pixels(points,7,5,k),original)

    def test_gravity_selects_the_correct_quarter_turn(self):
        from wifi_localization.learned import upright_quarters
        T=np.eye(4)
        T[:3,:3]=[[0,1,0],[-1,0,0],[0,0,1]]
        self.assertEqual(upright_quarters(T),3)
        T[:3,:3]=np.diag([1.,-1.,-1.])
        self.assertEqual(upright_quarters(T),0)

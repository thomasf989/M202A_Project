import unittest
import numpy as np
from wifi_localization.multiset import read_multiset_pose, compare_to_multiset, fit_rigid_alignment
from wifi_localization.localization import pose_dict


class MultiSetIntegrationTests(unittest.TestCase):
    def test_raw_pose_position_and_explicit_orientation_conversion(self):
        raw=dict(poseFound=True,position=dict(x=1,y=2,z=3),rotation=dict(x=0,y=0,z=0,w=1),confidence=.9,mapIds=['m'])
        r=read_multiset_pose(raw)
        self.assertEqual(r['position_m'],[1.,2.,3.])
        self.assertNotIn('T_world_camera_cv',r)  # do not guess SDK camera convention
        wrapped=read_multiset_pose(dict(camera_convention='opengl',multiset=raw))
        np.testing.assert_allclose(np.array(wrapped['T_world_camera_cv'])[:3,3],[1,2,3])
        self.assertAlmostEqual(np.linalg.det(np.array(wrapped['T_world_camera_cv'])[:3,:3]),1.)

    def test_compare_is_baseline_not_ground_truth(self):
        T=np.eye(4);result=dict(success=True,pose=pose_dict(T))
        baseline=dict(poseFound=True,position=dict(x=.1,y=0,z=0),rotation=dict(x=0,y=0,z=0,w=1))
        out=compare_to_multiset(result,baseline)
        self.assertAlmostEqual(out['multiset_position_difference_m'],.1)
        self.assertIn('not treated as ground truth',out['multiset_note'])

    def test_rigid_control_point_alignment(self):
        src=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        R=np.array([[0,-1,0],[1,0,0],[0,0,1]],float);t=np.array([2,3,4])
        dst=src@R.T+t
        T,stats=fit_rigid_alignment(src,dst)
        np.testing.assert_allclose(T[:3,:3],R,atol=1e-8)
        np.testing.assert_allclose(T[:3,3],t,atol=1e-8)
        self.assertLess(stats['rmse_m'],1e-8)

if __name__=='__main__':unittest.main()

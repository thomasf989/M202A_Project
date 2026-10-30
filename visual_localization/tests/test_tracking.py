import unittest
import cv2
import numpy as np
from wifi_localization.tracking import MappedFeatureTracker
from wifi_localization.localization import pose_dict, pose_errors

class TrackingTests(unittest.TestCase):
    def test_mapped_landmarks_track_without_imu(self):
        rng=np.random.default_rng(2)
        K=np.array([[300.,0,160.],[0,300.,120.],[0,0,1.]])
        xyz=rng.uniform([-1,-.7,3.],[1,.7,5.],(60,3))
        def frame(camera_x):
            pc=xyz-np.array([camera_x,0,0])
            uv=pc[:,:2]/pc[:,2:]*[300,300]+[160,120]
            im=np.zeros((240,320,3),np.uint8)
            for x,y in uv:
                if 4<x<316 and 4<y<236:cv2.circle(im,(round(x),round(y)),3,(255,255,255),-1)
            return im,uv
        first,uv0=frame(0);second,_=frame(.04)
        initial=dict(success=True,pose=pose_dict(np.eye(4)),inlier_query_pixels=uv0.tolist(),inlier_world_points=xyz.tolist())
        tracker=MappedFeatureTracker(min_points=15,max_error=30);tracker.initialize(first,initial)
        r=tracker.track(second,K)
        self.assertTrue(r['success'],r)
        truth=np.eye(4);truth[0,3]=.04
        pe,re=pose_errors(np.asarray(r['pose']['T_world_camera_cv']),truth)
        self.assertLess(pe,.05);self.assertLess(re,1.5)

if __name__=='__main__':unittest.main()

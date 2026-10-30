"""Camera-only mapped-landmark tracking between global relocalizations.

After a successful global localization, its inlier phone pixels already have
metric 3D world points. Lucas-Kanade optical flow tracks those pixels into the
next frame and PnP re-estimates the metric camera pose. This avoids an IMU and
avoids global image retrieval on every video frame.
"""
import cv2
import numpy as np
from .localization import Settings, solve_pose, pose_dict


class MappedFeatureTracker:
    def __init__(self, min_points=18, max_error=20.0):
        self.min_points=int(min_points);self.max_error=float(max_error)
        self.gray=None;self.xy=None;self.xyz=None

    def reset(self):
        self.gray=self.xy=self.xyz=None

    def initialize(self, rgb, localization_result):
        if not localization_result.get('success'):
            raise ValueError('Tracker needs a successful global localization')
        xy=np.asarray(localization_result.get('inlier_query_pixels',[]),np.float32)
        xyz=np.asarray(localization_result.get('inlier_world_points',[]),np.float64)
        if len(xy)!=len(xyz) or len(xy)<self.min_points:
            raise ValueError('Not enough mapped inlier landmarks to initialize tracking')
        self.gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        self.xy=np.ascontiguousarray(xy,dtype=np.float32);self.xyz=np.ascontiguousarray(xyz,dtype=np.float64)

    def track(self, rgb, K, settings=None):
        if self.gray is None:return dict(success=False,reason='tracker_not_initialized',needs_global_relocalization=True)
        gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        prev=np.ascontiguousarray(self.xy.reshape(-1,1,2),dtype=np.float32)
        nxt,status,error=cv2.calcOpticalFlowPyrLK(self.gray,gray,prev,None,
            winSize=(31,31),maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,30,.01))
        if nxt is None:
            return dict(success=False,reason='optical_flow_failed',needs_global_relocalization=True)
        nxt=nxt.reshape(-1,2);status=status.reshape(-1).astype(bool);error=error.reshape(-1)
        good=status&np.isfinite(nxt).all(1)&np.isfinite(error)&(error<self.max_error)
        xy=nxt[good];xyz=self.xyz[good]
        if len(xy)<self.min_points:
            return dict(success=False,reason='too_few_tracked_landmarks',tracked_points=len(xy),needs_global_relocalization=True)
        qids=np.arange(len(xy));groups=np.zeros(len(xy),int)
        s=settings or Settings(min_inliers=max(10,min(self.min_points,len(xy)//2)),top_k=1)
        T,stats=solve_pose(xyz,xy,qids,groups,np.asarray(K,float),(rgb.shape[1],rgb.shape[0]),s)
        if T is None:
            return dict(success=False,method='LK optical flow + map PnP',tracked_points=len(xy),
                        needs_global_relocalization=True,**stats)
        idx=np.asarray(stats.pop('inlier_indices'),int)
        self.gray=gray;self.xy=xy[idx].astype(np.float32);self.xyz=xyz[idx]
        return dict(success=True,method='LK optical flow + map PnP',tracked_points=len(xy),
                    needs_global_relocalization=False,pose=pose_dict(T),
                    inlier_query_pixels=self.xy.tolist(),inlier_world_points=self.xyz.tolist(),**stats)

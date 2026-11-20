#!/usr/bin/env python3
"""Evaluate the desktop equivalent of the iPhone CPU method on held-out photos.
This does not measure on-phone runtime, AR drift, or native implementation parity.
"""
from pathlib import Path
import sys, json, time, argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'visual_localization'))
import cv2
cv2.setNumThreads(2)
from wifi_localization.capture import load_capture
from wifi_localization.features import Database
from wifi_localization.localization import localize, pose_errors
from wifi_localization.polycam import rgb_image,sha256
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--test',required=True);p.add_argument('--database',required=True);p.add_argument('--output',required=True)
a=p.parse_args();train=load_capture(a.train);test=load_capture(a.test);db=Database(a.database,train)
assert not set(train.ids)&set(test.ids),'Train/test overlap'
rows=[]
for rid in test.ids:
 f=test.frames[rid];r=localize(train,db,rgb_image(f.image),f.K,query_hash=sha256(f.image))
 row=dict(id=rid,success=r['success'],seconds=r['elapsed_seconds'],inliers=r.get('inliers',0),reason=r.get('reason'))
 if r['success']:
  row['position_error_m'],row['rotation_error_deg']=pose_errors(np.array(r['pose']['T_world_camera_cv']),f.T_world_cv)
 rows.append(row);print(json.dumps(row),flush=True)
report=dict(method='Python CPU baseline, not native-device validation',references=len(train.ids),queries=len(test.ids),successes=sum(r['success'] for r in rows),results=rows)
Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(report,indent=2))

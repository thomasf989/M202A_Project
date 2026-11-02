"""Optional full learned MeshLoc integration test on a prepared room.

Set M202A_E2E_DATA to a prepared room directory and M202A_E2E_DB to its
feature database directory.  The test uses a known posed frame as a query,
excludes that exact frame from retrieval, and runs the real NetVLAD/LoFTR/
mesh-depth/PoseLib pipeline.  Model downloads and real room data make this
inappropriate for the normal offline regression suite, hence the explicit env
switch.
"""
import os
from pathlib import Path
import numpy as np
import pytest


def test_full_meshloc_known_pose_when_fixture_is_supplied(tmp_path):
    data=os.environ.get("M202A_E2E_DATA")
    db_path=os.environ.get("M202A_E2E_DB")
    if not data or not db_path:
        pytest.skip("Set M202A_E2E_DATA and M202A_E2E_DB for learned end-to-end validation")

    from wifi_localization.capture import load_capture
    from wifi_localization.features import Database
    from wifi_localization.pipeline import LocalizationPipeline
    from wifi_localization.localization import Settings, pose_errors
    from wifi_localization.polycam import rgb_image, sha256

    capture=load_capture(Path(data))
    db=Database(Path(db_path),capture)
    query_ids=list(getattr(capture,"query_ids",[])) or list(getattr(capture,"reference_ids",[]))
    assert query_ids, "fixture needs at least one posed frame"
    qid=query_ids[0]
    frame=capture.frames[qid]
    rgb=rgb_image(frame.image)
    pipe=LocalizationPipeline(capture,db,tmp_path/"cache")
    result=pipe.localize(
        rgb,[frame.K],exclude={qid},query_hash=sha256(frame.image),
        settings=Settings(min_inliers=8,min_inlier_ratio=.04,seed=42),
        solver_mode="meshloc")
    assert result.get("success"), result
    position,rotation=pose_errors(np.asarray(result["pose"]["T_world_camera_cv"]),frame.T_world_cv)
    max_position=float(os.environ.get("M202A_E2E_MAX_POSITION_M","0.50"))
    max_rotation=float(os.environ.get("M202A_E2E_MAX_ROTATION_DEG","15.0"))
    assert position <= max_position, (position,result)
    assert rotation <= max_rotation, (rotation,result)

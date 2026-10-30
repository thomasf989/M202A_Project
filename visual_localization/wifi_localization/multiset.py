"""Utilities for using MultiSet as a map/baseline around our own localizer.

MultiSet's exported GLB/PCD are map assets. They do not expose the proprietary
VPS feature database, so this project treats MultiSet in two deliberately
separate ways:

1. exported textured GLB -> reference geometry for OUR visual localizer;
2. VPS pose JSON -> commercial baseline for evaluation.

All transforms in this module are 4x4 column-vector transforms. The localizer's
canonical camera convention is OpenCV (+X right, +Y down, +Z forward).
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

# OpenCV camera -> OpenGL/AR camera (+X right, +Y up, -Z forward)
CV_TO_OPENGL = np.diag([1.0, -1.0, -1.0, 1.0])


def _finite_vector(value, n, label):
    a = np.asarray(value, dtype=float).reshape(-1)
    if len(a) != n or not np.isfinite(a).all():
        raise ValueError(f"{label} must contain {n} finite values")
    return a


def _matrix(value, label):
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{label} must be a finite 4x4 rigid transform")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=2e-4) or not np.isclose(np.linalg.det(R), 1.0, atol=2e-4):
        raise ValueError(f"{label} rotation is not rigid")
    return T


def read_multiset_pose(source):
    """Read a MultiSet VPS result into a conservative canonical record.

    Accepted forms:
      * raw MultiSet result containing poseFound, position{x,y,z}, rotation{x,y,z,w};
      * wrapper with ``multiset`` holding the raw result plus optional metadata;
      * canonical JSON containing T_world_camera_cv.

    A raw MultiSet result is safe for POSITION comparison when it is from the
    same exported map coordinate frame. Orientation is only converted when the
    JSON explicitly declares camera_convention ('opencv' or 'opengl'). This
    avoids silently assuming an SDK/device convention.
    """
    if isinstance(source, (str, Path)):
        data = json.loads(Path(source).read_text(encoding='utf-8'))
    else:
        data = source
    if not isinstance(data, dict):
        raise ValueError('MultiSet pose JSON must be an object')

    wrapper = data
    raw = data.get('multiset', data)
    if not isinstance(raw, dict):
        raise ValueError('multiset field must be an object')
    if raw.get('poseFound') is False:
        return dict(pose_found=False, confidence=raw.get('confidence'), raw=raw)

    if 'T_world_camera_cv' in wrapper:
        T = _matrix(wrapper['T_world_camera_cv'], 'T_world_camera_cv')
        return dict(pose_found=True, position_m=T[:3, 3].tolist(), T_world_camera_cv=T.tolist(),
                    quaternion_xyzw=Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
                    confidence=wrapper.get('confidence', raw.get('confidence')), map_ids=raw.get('mapIds', []), raw=raw)

    position = raw.get('position')
    rotation = raw.get('rotation')
    if isinstance(position, dict):
        p = _finite_vector([position.get('x'), position.get('y'), position.get('z')], 3, 'position')
    else:
        p = _finite_vector(position, 3, 'position')
    result = dict(pose_found=True, position_m=p.tolist(), confidence=raw.get('confidence'),
                  map_ids=raw.get('mapIds', []), raw=raw)

    if rotation is not None:
        if isinstance(rotation, dict):
            q = _finite_vector([rotation.get('x'), rotation.get('y'), rotation.get('z'), rotation.get('w')], 4, 'rotation')
        else:
            q = _finite_vector(rotation, 4, 'rotation')
        q /= np.linalg.norm(q)
        result['quaternion_raw_xyzw'] = q.tolist()
        convention = wrapper.get('camera_convention') or raw.get('camera_convention')
        if convention is not None:
            if convention not in ['opencv', 'opengl']:
                raise ValueError("camera_convention must be 'opencv' or 'opengl'")
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat(q).as_matrix()
            T[:3, 3] = p
            if convention == 'opengl':
                T = T @ CV_TO_OPENGL
            result['T_world_camera_cv'] = T.tolist()
            result['quaternion_xyzw'] = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()
    return result


def compare_to_multiset(our_result, baseline):
    """Attach baseline errors without calling the MultiSet pose ground truth."""
    baseline = read_multiset_pose(baseline)
    out = dict(multiset_baseline={k: v for k, v in baseline.items() if k != 'raw'},
               multiset_note='Commercial VPS baseline; not treated as ground truth.')
    if not baseline.get('pose_found') or not our_result.get('success'):
        return out
    ours = np.asarray(our_result['pose']['position_m'], float)
    theirs = np.asarray(baseline['position_m'], float)
    out['multiset_position_difference_m'] = float(np.linalg.norm(ours - theirs))
    if baseline.get('T_world_camera_cv') is not None:
        A = np.asarray(our_result['pose']['T_world_camera_cv'], float)
        B = np.asarray(baseline['T_world_camera_cv'], float)
        out['multiset_rotation_difference_deg'] = float(np.degrees(
            Rotation.from_matrix(A[:3, :3].T @ B[:3, :3]).magnitude()))
    return out


def fit_rigid_alignment(source_points, target_points):
    """Least-squares rigid T_target_source from >=3 corresponding 3D points."""
    src = np.asarray(source_points, float)
    dst = np.asarray(target_points, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3 or len(src) < 3:
        raise ValueError('Need matching Nx3 source/target points with N >= 3')
    if not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise ValueError('Control points must be finite')
    sc, dc = src.mean(0), dst.mean(0)
    X, Y = src - sc, dst - dc
    U, _, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = dc - R @ sc
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    residuals = np.linalg.norm((src @ R.T + t) - dst, axis=1)
    return T, dict(rmse_m=float(np.sqrt(np.mean(residuals**2))), median_m=float(np.median(residuals)),
                   max_m=float(np.max(residuals)), points=len(src))


def transform_pose(T_target_source, T_source_camera):
    return _matrix(T_target_source, 'T_target_source') @ _matrix(T_source_camera, 'T_source_camera')

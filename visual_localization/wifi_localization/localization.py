"""Geometric localization. Query ground truth is never used by this module."""
from dataclasses import dataclass, asdict
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .features import extract, match


@dataclass
class Settings:
    top_k: int = 12
    ratio: float = .75
    reprojection_px: float = 4.
    min_inliers: int = 12
    min_inlier_ratio: float = .12
    iterations: int = 4000
    seed: int = 42


def pose_dict(T):
    return dict(T_world_camera_cv=T.tolist(), position_m=T[:3, 3].tolist(),
                quaternion_xyzw=Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
                forward_world=T[:3, 2].tolist())


def pose_errors(estimated, truth):
    position = float(np.linalg.norm(estimated[:3, 3]-truth[:3, 3]))
    rotation = float(np.degrees(Rotation.from_matrix(estimated[:3, :3].T @ truth[:3, :3]).magnitude()))
    return position, rotation


def project_errors(xyz, xy, K, rvec, tvec):
    projected = cv2.projectPoints(xyz, rvec, tvec, K, None)[0].reshape(-1, 2)
    R = cv2.Rodrigues(rvec)[0]
    positive = ((xyz @ R.T + tvec.reshape(3))[:, 2] > .05)
    return np.linalg.norm(projected-xy, axis=1), positive


def unique_inliers(qids, errors, valid):
    """One vote per query feature, even if it matched multiple reference frames."""
    chosen = []
    seen = set()
    for i in np.argsort(errors):
        q = int(qids[i])
        if valid[i] and q not in seen:
            seen.add(q)
            chosen.append(i)
    return np.asarray(chosen, dtype=int)


def _quality_score(inliers, ratio, median_px, occupied, supporting_refs, min_inliers):
    """A conservative 0..1 diagnostic confidence, not a calibrated probability."""
    inlier_term = min(1.0, inliers/max(min_inliers*3.0, 1.0))
    ratio_term = np.clip((ratio-.08)/.42, 0., 1.)
    reproj_term = float(np.exp(-max(median_px, 0.)/3.0))
    coverage_term = min(1.0, occupied/8.0)
    reference_term = min(1.0, supporting_refs/3.0)
    score = .28*inlier_term + .24*ratio_term + .22*reproj_term + .16*coverage_term + .10*reference_term
    return float(np.clip(score, 0., 1.))


def solve_pose(xyz, xy, qids, group_ids, K, image_size, settings):
    xyz=np.asarray(xyz,float);xy=np.asarray(xy,float);qids=np.asarray(qids);group_ids=np.asarray(group_ids)
    finite=np.isfinite(xyz).all(1)&np.isfinite(xy).all(1)
    xyz,xy,qids,group_ids=xyz[finite],xy[finite],qids[finite],group_ids[finite]
    if len(xyz) < settings.min_inliers:
        return None, {'reason': 'too_few_2d_3d_correspondences', 'correspondences': len(xyz)}
    hypotheses = []
    # Pooled and per-reference hypotheses avoid one bad reference dominating.
    groups = [np.arange(len(xyz))] + [np.flatnonzero(group_ids == g) for g in np.unique(group_ids)]
    for indices in groups:
        if len(indices) < settings.min_inliers:
            continue
        cv2.setRNGSeed(settings.seed)
        ok, r, t, inl = cv2.solvePnPRansac(np.ascontiguousarray(xyz[indices], dtype=np.float64),
            np.ascontiguousarray(xy[indices], dtype=np.float64), K, None,
            iterationsCount=settings.iterations, reprojectionError=settings.reprojection_px,
            confidence=.999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inl is None or not np.isfinite(r).all() or not np.isfinite(t).all():
            continue
        e, pos = project_errors(xyz, xy, K, r, t)
        good = unique_inliers(qids, e, pos & (e < settings.reprojection_px))
        if len(good):
            hypotheses.append((len(good), -float(np.median(e[good])), r, t, good))
    if not hypotheses:
        return None, {'reason': 'pnp_ransac_failed', 'correspondences': len(xyz)}
    _, _, r, t, inliers = max(hypotheses, key=lambda h: h[:2])
    for _ in range(2):
        if len(inliers) < settings.min_inliers:
            break
        r, t = cv2.solvePnPRefineLM(xyz[inliers], xy[inliers], K, None, r, t)
        e, positive = project_errors(xyz, xy, K, r, t)
        inliers = unique_inliers(qids, e, positive & (e < settings.reprojection_px))
    total_unique = len(np.unique(qids))
    fraction = len(inliers)/max(total_unique, 1)
    width, height = image_size
    if len(inliers):
        cells=np.floor(xy[inliers]/[max(width/4,1),max(height/4,1)]).astype(int)
        cells=np.clip(cells,[0,0],[3,3])
        occupied = len(np.unique(cells,axis=0))
    else:occupied=0
    supporting_refs=len(np.unique(group_ids[inliers])) if len(inliers) else 0
    median=float(np.median(e[inliers])) if len(inliers) else float('inf')
    p90=float(np.percentile(e[inliers],90)) if len(inliers) else float('inf')
    stats = dict(correspondences=len(xyz), unique_query_matches=total_unique, inliers=len(inliers),
                 inlier_ratio=fraction, occupied_grid_cells=occupied, supporting_references=supporting_refs,
                 median_reprojection_px=median, p90_reprojection_px=p90)
    stats['pose_confidence']=_quality_score(len(inliers),fraction,median,occupied,supporting_refs,settings.min_inliers)
    stats['confidence_note']='Heuristic geometric confidence (0..1), not a calibrated probability.'
    if len(inliers) < settings.min_inliers or fraction < settings.min_inlier_ratio or occupied < 4:
        return None, dict(stats, reason='insufficient_geometric_support')
    spread = np.linalg.svd(xyz[inliers]-xyz[inliers].mean(axis=0), compute_uv=False)
    if len(spread)<2 or spread[1] < .03:
        return None, dict(stats, reason='nearly_collinear_3d_points')
    R = cv2.Rodrigues(r)[0]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R.T, -R.T @ t.reshape(3)
    return T, dict(stats, inlier_indices=inliers.tolist())


def consensus_position(capture, ranking, radius=.75):
    chosen = ranking[:8]
    centers = np.array([capture.frames[id].T_world_cv[:3, 3] for id, _ in chosen])
    weights = np.array([max(score, 1e-6)**4 for _, score in chosen])
    dist = np.linalg.norm(centers[:, None]-centers[None, :], axis=2)
    support = dist <= radius
    winner = np.argmax(support @ weights)
    good = support[winner]
    return np.average(centers[good], axis=0, weights=weights[good]).tolist()


def localize(capture, db, rgb, K, settings=None, exclude=(), query_hash=None, candidate_intrinsics=None):
    """CPU baseline retained for ablation/fallback."""
    start = time.perf_counter()
    s = settings or Settings()
    cv2.setRNGSeed(s.seed)
    xyq, dq = extract(rgb, db.meta['nfeatures'])
    excluded = set(exclude)
    if query_hash:
        excluded.update(id for id, h in db.meta['reference_image_hashes'].items() if h == query_hash)
    rank = db.rank(dq, excluded)
    result = dict(success=False, settings=asdict(s), query_features=len(xyq),
                  excluded_reference_count=len(set(db.ids) & excluded), retrieval_method='RootSIFT TF-IDF')
    if not rank:
        return dict(result, reason='no_eligible_references', elapsed_seconds=time.perf_counter()-start)
    result['retrieval'] = [dict(id=id, score=score) for id, score in rank[:s.top_k]]
    result['baseline_nearest'] = pose_dict(capture.frames[rank[0][0]].T_world_cv)
    result['baseline_consensus_position_m'] = consensus_position(capture, rank)
    all_xyz, all_xy, all_q, all_g = [], [], [], []
    matches = []
    for group, (id, score) in enumerate(rank[:s.top_k]):
        ref = db.features(id)
        pairs, distances = match(dq, ref['desc'], s.ratio)
        pairs = pairs[ref['valid'][pairs[:, 1]]] if len(pairs) else pairs
        matches.append(dict(reference=id, depth_matches=len(pairs), score=score))
        if not len(pairs):
            continue
        all_xyz.append(ref['xyz'][pairs[:, 1]])
        all_xy.append(xyq[pairs[:, 0]])
        all_q.append(pairs[:, 0])
        all_g.append(np.full(len(pairs), group, int))
    result['matches'] = matches
    if all_xyz:
        xyz, xy, qids, groups = map(np.concatenate, [all_xyz, all_xy, all_q, all_g])
        solutions = []
        failures=[]
        candidates = candidate_intrinsics if candidate_intrinsics is not None else [K]
        for candidate in candidates:
            T, stats = solve_pose(xyz, xy, qids, groups, candidate, (rgb.shape[1], rgb.shape[0]), s)
            if T is not None:
                solutions.append((T, stats, candidate))
            else:failures.append(stats)
        if solutions:
            T, stats, K = max(solutions, key=lambda x: (x[1]['inliers'], x[1].get('pose_confidence',0), -x[1]['median_reprojection_px']))
        else:
            T = None
            stats=max(failures,key=lambda x:(x.get('inliers',0),x.get('inlier_ratio',0))) if failures else {'reason':'pnp_ransac_failed'}
        result.update(stats)
        if T is not None:
            result.update(success=True, pose=pose_dict(T))
            result['query_intrinsics'] = np.asarray(K).tolist()
            idx = np.asarray(result.pop('inlier_indices'), int)
            result['inlier_query_pixels'] = xy[idx].tolist()
            result['inlier_world_points'] = xyz[idx].tolist()
    else:
        result['reason'] = 'no_valid_depth_matches'
    result['elapsed_seconds'] = time.perf_counter()-start
    return result

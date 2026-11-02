"""ALIKED + LightGlue metric localization with geometric pair verification.

Preferred hierarchy:
    multi-crop DINOv2 retrieval -> spatial neighbour expansion ->
    ALIKED/LightGlue -> fundamental-matrix verification -> reference 3D points
    -> multi-hypothesis PnP/RANSAC.

The query image is never given a ground-truth pose.  MultiSet is not used by
this solver; it is only an optional external baseline.
"""
import hashlib
from pathlib import Path
import os
import time
import cv2
import numpy as np
import torch
from ._vendor.lightglue import ALIKED, LightGlue
from ._vendor.lightglue.utils import numpy_image_to_torch
from .features import match
from .polycam import rgb_image
from .localization import Settings, pose_dict, solve_pose


def upright_quarters(T_world_cv):
    gravity_image = np.asarray(T_world_cv)[1, :2]
    if np.linalg.norm(gravity_image) < .25:
        return 0
    rotations = [np.eye(2), np.array([[0,1],[-1,0]]), -np.eye(2), np.array([[0,-1],[1,0]])]
    return int(np.argmin([(R @ gravity_image)[1] for R in rotations]))


def original_pixels(pixels, width, height, quarters):
    x, y = np.asarray(pixels).T
    if quarters == 0:return np.column_stack([x,y])
    if quarters == 1:return np.column_stack([width-1-y,x])
    if quarters == 2:return np.column_stack([width-1-x,height-1-y])
    if quarters == 3:return np.column_stack([y,height-1-x])
    raise ValueError('Rotation must be 0,1,2 or 3 quarter turns')


def fundamental_filter(query_xy, reference_xy, threshold_px=2.0):
    """Return a mask of pairwise geometrically consistent matches.

    LightGlue is strong but its scores are not a geometric guarantee.  A
    fundamental-matrix RANSAC before lifting reference points to 3D removes a
    large fraction of repeated-texture and synthetic-vs-real false matches.
    """
    q = np.asarray(query_xy, np.float32).reshape(-1, 2)
    r = np.asarray(reference_xy, np.float32).reshape(-1, 2)
    if len(q) < 10:
        return np.ones(len(q), dtype=bool), 'not_enough_for_F'
    method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.FM_RANSAC
    try:
        F, mask = cv2.findFundamentalMat(q, r, method, float(threshold_px), .999, 10000)
    except TypeError:
        F, mask = cv2.findFundamentalMat(q, r, method, float(threshold_px), .999)
    if F is None or mask is None:
        return np.ones(len(q), dtype=bool), 'F_failed'
    mask = mask.reshape(-1).astype(bool)
    # If a model was found but it explains almost nothing, treating all raw
    # matches as valid is worse than rejecting this reference pair.
    if mask.sum() < 8:
        return np.zeros(len(q), dtype=bool), 'F_low_support'
    return mask, 'F_ransac'


class LearnedLocalizer:
    def __init__(self, capture, reference_ids, cache_root, progress=None, max_keypoints=4096, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        # Avoid the old hard-coded 2-thread bottleneck on CPU.
        if device == 'cpu':
            torch.set_num_threads(max(2, min(8, (os.cpu_count() or 4)//2)))
        self.capture = capture
        self.ids = list(reference_ids)
        self.progress = progress or (lambda message, fraction: None)
        self.progress(f'Loading ALIKED + LightGlue on {device}…', .35)
        self.extractor = ALIKED(max_num_keypoints=int(max_keypoints)).eval().to(device)
        self.matcher = LightGlue(features='aliked').eval().to(device)
        key = hashlib.sha256((capture.fingerprint()+','.join(self.ids)+f'aliked-{max_keypoints}-1280-geom-v3').encode()).hexdigest()[:20]
        self.directory = Path(cache_root)/key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.references = {}

    @torch.inference_mode()
    def extract(self, rgb):
        image = numpy_image_to_torch(np.ascontiguousarray(rgb)).to(self.device)
        data = self.extractor.extract(image, resize=1280)
        return dict(xy=data['keypoints'][0].detach().cpu().numpy(),
            desc=data['descriptors'][0].detach().cpu().numpy(), size=data['image_size'][0].detach().cpu().numpy())

    def tensor_features(self, data):
        return dict(keypoints=torch.from_numpy(data['xy'].astype(np.float32))[None].to(self.device),
            descriptors=torch.from_numpy(data['desc'].astype(np.float32))[None].to(self.device),
            image_size=torch.from_numpy(data['size'].astype(np.float32))[None].to(self.device))

    def reference(self, id):
        if id in self.references:return self.references[id]
        file = self.directory/f'{id}.npz'
        if file.exists():
            with np.load(file, allow_pickle=False) as z:
                data = {k:z[k] for k in z.files}
            data['desc'] = data['desc'].astype(np.float32)
        else:
            f = self.capture.frames[id]
            rgb = rgb_image(f.image)
            quarter = upright_quarters(f.T_world_cv)
            rotated = np.rot90(rgb, quarter).copy()
            data = self.extract(rotated)
            pixels = original_pixels(data['xy'], f.width, f.height, quarter)
            xyz, valid = self.capture.world_points(f, pixels)
            data.update(original_xy=pixels, xyz=xyz, valid=valid,
                        rotated_width=np.array(rotated.shape[1]), rotated_height=np.array(rotated.shape[0]))
            np.savez_compressed(file, **data)
        self.references[id] = data
        return data

    def _slow_rank(self, query, eligible, top_k):
        rankings = []
        for i,id in enumerate(eligible):
            self.progress(f'Learned reference pre-ranking: {i+1}/{len(eligible)}', .38+.16*(i+1)/max(len(eligible),1))
            ref = self.reference(id)
            pairs,_ = match(query['desc'], ref['desc'], ratio=.85)
            score = int(ref['valid'][pairs[:,1]].sum()) if len(pairs) else 0
            rankings.append((id,float(score)))
        rankings.sort(key=lambda row:-row[1])
        return rankings[:top_k], 'ALIKED tentative-match retrieval'

    @torch.inference_mode()
    def localize(self, rgb, candidate_intrinsics, exclude=(), query_hash=None, reference_hashes=None,
                 candidate_ranking=None, top_k=40, retrieval_method=None, settings=None):
        started = time.perf_counter()
        query = self.extract(rgb)
        result = dict(success=False, method='DINOv2 + ALIKED/LightGlue + F-RANSAC + 2D-3D PnP',
                      query_features=len(query['xy']))
        excluded = set(exclude)
        if query_hash:
            excluded.update(id for id,h in (reference_hashes or {}).items() if h == query_hash)
        eligible = [id for id in self.ids if id not in excluded]
        eligible_set = set(eligible)
        result['excluded_reference_count'] = len(set(self.ids)&excluded)
        if len(query['xy']) < 12 or not eligible:
            return dict(result, reason='too_few_query_features_or_references', elapsed_seconds=time.perf_counter()-started)

        if candidate_ranking is None:
            candidates, used_method = self._slow_rank(query, eligible, top_k)
        else:
            candidate_ranking = [(id,float(score)) for id,score in candidate_ranking if id in eligible_set]
            candidates = candidate_ranking[:top_k]
            used_method = retrieval_method or 'external global retrieval'
        result['retrieval_method'] = used_method
        result['retrieval'] = [dict(id=id, score=float(score)) for id,score in candidates]
        if not candidates:
            return dict(result, reason='no_eligible_references', elapsed_seconds=time.perf_counter()-started)

        fq = self.tensor_features(query)
        all_xyz, all_xy, all_qids, all_groups = [],[],[],[]
        matches = []
        for group,(id,score) in enumerate(candidates):
            self.progress(f'LightGlue + geometry: {group+1}/{len(candidates)}', .52+.28*(group+1)/max(len(candidates),1))
            ref = self.reference(id)
            if len(ref['xy']) < 12:
                matches.append(dict(reference=id, raw_matches=0, geometric_matches=0, depth_matches=0,
                                    retrieval_score=float(score)))
                continue
            data = self.matcher({'image0':fq,'image1':self.tensor_features(ref)})
            pairs = data['matches'][0].detach().cpu().numpy()
            scores = data.get('scores')
            scores = scores[0].detach().cpu().numpy() if scores is not None and len(scores) else np.ones(len(pairs))
            raw_count = len(pairs)
            if raw_count:
                fmask, fmethod = fundamental_filter(query['xy'][pairs[:,0]], ref['xy'][pairs[:,1]])
                pairs = pairs[fmask]
                scores = scores[fmask]
            else:
                fmethod = 'no_matches'
            geometric_count = len(pairs)
            if geometric_count:
                depth_mask = ref['valid'][pairs[:,1]]
                pairs = pairs[depth_mask]
                scores = scores[depth_mask]
            matches.append(dict(reference=id, raw_matches=raw_count, geometric_matches=geometric_count,
                                depth_matches=len(pairs), retrieval_score=float(score), geometry_filter=fmethod,
                                median_match_score=float(np.median(scores)) if len(scores) else 0.))
            if len(pairs):
                all_xyz.append(ref['xyz'][pairs[:,1]])
                all_xy.append(query['xy'][pairs[:,0]])
                all_qids.append(pairs[:,0])
                all_groups.append(np.full(len(pairs),group,int))
        result['matches'] = matches
        if not all_xyz:
            return dict(result, reason='no_geometrically_valid_depth_matches', elapsed_seconds=time.perf_counter()-started)

        xyz,xy,qids,groups = map(np.concatenate,[all_xyz,all_xy,all_qids,all_groups])
        solutions, failures = [], []
        self.progress('Validating metric camera pose…', .83)
        s = settings or Settings(top_k=top_k)
        for K in candidate_intrinsics:
            T,stats = solve_pose(xyz,xy,qids,groups,K,(rgb.shape[1],rgb.shape[0]),s)
            if T is not None:solutions.append((T,stats,K))
            else:failures.append(stats)
        if solutions:
            T,stats,K = max(solutions,key=lambda row:(row[1]['inliers'],row[1].get('pose_confidence',0),-row[1]['median_reprojection_px']))
            idx = np.asarray(stats.pop('inlier_indices'),int)
            result.update(stats,success=True,pose=pose_dict(T),query_intrinsics=np.asarray(K).tolist(),
                inlier_query_pixels=xy[idx].tolist(),inlier_world_points=xyz[idx].tolist())
        else:
            if failures:
                result.update(max(failures,key=lambda row:(row.get('inliers',0),row.get('inlier_ratio',0),-row.get('median_reprojection_px',1e9))))
            else:
                result['reason']='pnp_ransac_failed'
        result['elapsed_seconds'] = time.perf_counter()-started
        return result

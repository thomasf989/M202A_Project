"""ALIKED + LightGlue fallback with gravity-uprighted reference images.

Every detected reference point is mapped back to the ORIGINAL reference RGB
pixel grid before depth lifting. Camera/world transforms are never rotated.
Queries remain in their EXIF-oriented pixel grid. No query joins the map.
"""
import hashlib
from pathlib import Path
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


class LearnedLocalizer:
    def __init__(self, capture, reference_ids, cache_root, progress=None):
        torch.set_num_threads(2)
        self.capture = capture
        self.ids = list(reference_ids)
        self.progress = progress or (lambda message, fraction: None)
        self.progress('正在加载增强匹配模型，首次使用需要联网下载…', .35)
        self.extractor = ALIKED(max_num_keypoints=2048).eval().cpu()
        self.matcher = LightGlue(features='aliked').eval().cpu()
        key = hashlib.sha256((capture.fingerprint()+','.join(self.ids)+'aliked-2048-upright-v1').encode()).hexdigest()[:20]
        self.directory = Path(cache_root)/key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.references = {}

    @torch.inference_mode()
    def extract(self, rgb):
        data = self.extractor.extract(numpy_image_to_torch(np.ascontiguousarray(rgb)), resize=1024)
        return dict(xy=data['keypoints'][0].cpu().numpy(),
            desc=data['descriptors'][0].cpu().numpy(), size=data['image_size'][0].cpu().numpy())

    def tensor_features(self, data):
        return dict(keypoints=torch.from_numpy(data['xy'].astype(np.float32))[None],
            descriptors=torch.from_numpy(data['desc'].astype(np.float32))[None],
            image_size=torch.from_numpy(data['size'].astype(np.float32))[None])

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
            data = self.extract(np.rot90(rgb, quarter).copy())
            pixels = original_pixels(data['xy'], f.width, f.height, quarter)
            xyz, valid = self.capture.world_points(f, pixels)
            data.update(original_xy=pixels, xyz=xyz, valid=valid)
            # Preserve full precision for stable warm/cold-cache correspondences.
            np.savez_compressed(file, **data)
        self.references[id] = data
        return data

    @torch.inference_mode()
    def localize(self, rgb, candidate_intrinsics, exclude=(), query_hash=None, reference_hashes=None):
        started = time.perf_counter()
        query = self.extract(rgb)
        result = dict(success=False, method='ALIKED + LightGlue + depth PnP', query_features=len(query['xy']))
        excluded = set(exclude)
        if query_hash:
            excluded.update(id for id,h in (reference_hashes or {}).items() if h == query_hash)
        eligible = [id for id in self.ids if id not in excluded]
        result['excluded_reference_count'] = len(set(self.ids)&excluded)
        if len(query['xy']) < 12 or not eligible:
            return dict(result, reason='too_few_2d_3d_correspondences', elapsed_seconds=time.perf_counter()-started)
        rankings = []
        for i,id in enumerate(eligible):
            self.progress(f'增强参考特征：{i+1}/{len(eligible)}', .38+.18*(i+1)/len(eligible))
            ref = self.reference(id)
            pairs,_ = match(query['desc'], ref['desc'], ratio=.85)
            score = int(ref['valid'][pairs[:,1]].sum()) if len(pairs) else 0
            rankings.append((id,score))
        rankings.sort(key=lambda row:-row[1])
        candidates = rankings[:20]
        result['retrieval'] = [dict(id=id, learned_tentative_matches=count) for id,count in candidates]
        fq = self.tensor_features(query)
        all_xyz, all_xy, all_qids, all_groups = [],[],[],[]
        matches = []
        for group,(id,_) in enumerate(candidates):
            self.progress(f'增强特征匹配：{group+1}/{len(candidates)}', .57+.20*(group+1)/len(candidates))
            ref = self.reference(id)
            if len(ref['xy']) < 12:continue
            data = self.matcher({'image0':fq,'image1':self.tensor_features(ref)})
            pairs = data['matches'][0].cpu().numpy()
            pairs = pairs[ref['valid'][pairs[:,1]]] if len(pairs) else pairs
            matches.append(dict(reference=id, depth_matches=len(pairs)))
            if len(pairs):
                all_xyz.append(ref['xyz'][pairs[:,1]])
                all_xy.append(query['xy'][pairs[:,0]])
                all_qids.append(pairs[:,0])
                all_groups.append(np.full(len(pairs),group,int))
        result['matches'] = matches
        if not all_xyz:
            return dict(result, reason='no_valid_depth_matches', elapsed_seconds=time.perf_counter()-started)
        xyz,xy,qids,groups = map(np.concatenate,[all_xyz,all_xy,all_qids,all_groups])
        solutions, failures = [], []
        self.progress('正在验证增强匹配的三维位置…', .79)
        settings = Settings()
        for K in candidate_intrinsics:
            T,stats = solve_pose(xyz,xy,qids,groups,K,(rgb.shape[1],rgb.shape[0]),settings)
            if T is not None:solutions.append((T,stats,K))
            else:failures.append(stats)
        if solutions:
            T,stats,K = max(solutions,key=lambda row:(row[1]['inliers'],-row[1]['median_reprojection_px']))
            idx = np.asarray(stats.pop('inlier_indices'),int)
            result.update(stats,success=True,pose=pose_dict(T),query_intrinsics=np.asarray(K).tolist(),
                inlier_query_pixels=xy[idx].tolist(),inlier_world_points=xyz[idx].tolist())
        else:
            result.update(max(failures,key=lambda row:row.get('inliers',0)))
        result['elapsed_seconds'] = time.perf_counter()-started
        return result

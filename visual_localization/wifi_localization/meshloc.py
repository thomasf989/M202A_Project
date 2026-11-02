"""MeshLoc-style mesh-based visual localization.

This module implements the main ideas from:
  Panek, Kukelova, Sattler, "MeshLoc: Mesh-Based Visual Localization", ECCV 2022.

Primary path (paper-style indoor configuration):
    NetVLAD top-K retrieval -> LoFTR indoor matching -> mesh-depth lifting ->
    individual 2D-3D correspondences -> P3P LO-RANSAC (PoseLib when available)
    -> nonlinear refinement -> MeshLoc position averaging.

The existing dense Blender synthetic reference generation is intentionally left
unchanged.  MeshLoc also evaluated merging of repeated 2D-3D matches and an
on-the-fly covisibility approximation; both are implemented as optional
ablation switches, while the default follows the paper's strong simple variant:
individual matches + position averaging, without covisibility filtering.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import math
import time
from collections import defaultdict, deque

import cv2
import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .polycam import rgb_image


# HLoc's public NetVLAD checkpoint exported from the original MATLAB model.
NETVLAD_URL = "https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat"


@dataclass
class MeshLocSettings:
    top_k: int = 50
    retrieval_mode: str = "hybrid"  # hybrid for synthetic maps, netvlad for paper baseline
    retrieval_pool: int = 250
    diversify_dense_references: bool = True
    diversity_translation_m: float = 0.30
    diversity_rotation_deg: float = 15.0
    matcher: str = "loftr_indoor_new"
    use_merge: bool = False
    use_covisibility: bool = False
    position_averaging: bool = True
    reprojection_px: float = 20.0
    min_inliers: int = 10
    min_inlier_ratio: float = 0.06
    min_ransac_iterations: int = 10_000
    max_ransac_iterations: int = 100_000
    query_cluster_px: float = 4.0
    merge_reprojection_px: float = 4.0
    position_volume_half_m: float = 0.25
    position_step_m: float = 0.05
    seed: int = 42


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _safe_unit(x, axis=-1):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


class NetVLADRetriever:
    """NetVLAD global retrieval using the Pitts30K checkpoint used by HLoc.

    The model is cached by torch.hub and reference descriptors are cached per
    room.  The first run downloads the public checkpoint once.
    """

    def __init__(self, capture, reference_ids, cache_root, progress=None, device=None):
        self.capture = capture
        self.ids = list(reference_ids)
        self.progress = progress or (lambda message, fraction: None)
        self.device = device or _device()
        key = hashlib.sha256((capture.fingerprint() + ",".join(self.ids) + "netvlad-pitts30k-v1").encode()).hexdigest()[:20]
        self.directory = Path(cache_root) / key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.directory / "netvlad_reference_embeddings.npz"
        self.model = None
        self.reference_embeddings = None

    def _load_model(self):
        if self.model is not None:
            return
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torchvision import models

        class NetVLADLayer(nn.Module):
            def __init__(self, input_dim=512, clusters=64):
                super().__init__()
                self.score_proj = nn.Conv1d(input_dim, clusters, 1, bias=False)
                self.centers = nn.Parameter(torch.empty(input_dim, clusters))
                self.output_dim = input_dim * clusters

            def forward(self, x):
                b = x.shape[0]
                scores = F.softmax(self.score_proj(x), dim=1)
                diff = x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1)
                desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
                desc = F.normalize(desc, dim=1)
                desc = F.normalize(desc.reshape(b, -1), dim=1)
                return desc

        class NetVLADModel(nn.Module):
            def __init__(self, checkpoint):
                super().__init__()
                mat = loadmat(checkpoint, struct_as_record=False, squeeze_me=True)
                backbone = list(models.vgg16(weights=None).children())[0]
                self.backbone = nn.Sequential(*list(backbone.children())[:-2])
                self.vlad = NetVLADLayer()
                self.whiten = nn.Linear(self.vlad.output_dim, 4096)

                conv_layers = [l for l in self.backbone.children() if isinstance(l, nn.Conv2d)]
                matlab_conv = [l for l in np.atleast_1d(mat["net"].layers) if hasattr(l, "weights") and len(np.atleast_1d(l.weights)) >= 2]
                # First 13 weighted layers are VGG conv layers in the published checkpoint.
                for layer, mlayer in zip(conv_layers, matlab_conv[:len(conv_layers)]):
                    w = torch.tensor(np.asarray(mlayer.weights[0])).float().permute(3, 2, 0, 1)
                    b = torch.tensor(np.asarray(mlayer.weights[1]).squeeze()).float()
                    layer.weight = nn.Parameter(w)
                    layer.bias = nn.Parameter(b)

                # The NetVLAD and whitening layers use fixed indices in the public HLoc checkpoint.
                vl = mat["net"].layers[30]
                score_w = torch.tensor(np.asarray(vl.weights[0])).float().permute(1, 0).unsqueeze(-1)
                center_w = -torch.tensor(np.asarray(vl.weights[1])).float()
                self.vlad.score_proj.weight = nn.Parameter(score_w)
                self.vlad.centers = nn.Parameter(center_w)

                wl = mat["net"].layers[33]
                ww = torch.tensor(np.asarray(wl.weights[0])).float().squeeze().permute(1, 0)
                wb = torch.tensor(np.asarray(wl.weights[1]).squeeze()).float()
                self.whiten.weight = nn.Parameter(ww)
                self.whiten.bias = nn.Parameter(wb)
                self.register_buffer("mean", torch.tensor(np.asarray(mat["net"].meta.normalization.averageImage[0, 0])).float().reshape(1, 3, 1, 1))

            def forward(self, image):
                image = image * 255.0 - self.mean
                x = self.backbone(image)
                x = F.normalize(x.reshape(x.shape[0], x.shape[1], -1), dim=1)
                x = self.vlad(x)
                return F.normalize(self.whiten(x), dim=1)

        checkpoint = Path(torch.hub.get_dir()) / "netvlad" / "VGG16-NetVLAD-Pitts30K.mat"
        if not checkpoint.exists():
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self.progress("Downloading NetVLAD Pitts30K checkpoint (first run only)…", 0.20)
            torch.hub.download_url_to_file(NETVLAD_URL, str(checkpoint), progress=True)
        self.progress(f"Loading NetVLAD on {self.device}…", 0.24)
        self.model = NetVLADModel(checkpoint).eval().to(self.device)

    def _tensor(self, rgb):
        import torch
        import torch.nn.functional as F
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()[None] / 255.0
        h, w = image.shape[-2:]
        scale = min(1.0, 1024.0 / max(h, w))
        nh, nw = max(32, int(round(h * scale))), max(32, int(round(w * scale)))
        if (nh, nw) != (h, w):
            image = F.interpolate(image, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
        return image.to(self.device)

    def embed(self, rgb):
        import torch
        self._load_model()
        with torch.inference_mode():
            descriptor = self.model(self._tensor(rgb))[0].float().cpu().numpy()
        return _safe_unit(descriptor.reshape(1, -1))[0]

    def _ensure_references(self):
        if self.reference_embeddings is not None:
            return
        if self.cache_file.exists():
            with np.load(self.cache_file, allow_pickle=False) as z:
                ids = z["ids"].astype(str).tolist()
                if ids == self.ids:
                    self.reference_embeddings = z["embeddings"].astype(np.float32)
                    return
        vectors = []
        for i, rid in enumerate(self.ids):
            self.progress(f"NetVLAD reference descriptors: {i+1}/{len(self.ids)}", 0.25 + 0.20 * (i + 1) / max(1, len(self.ids)))
            vectors.append(self.embed(rgb_image(self.capture.frames[rid].image)))
        self.reference_embeddings = np.stack(vectors).astype(np.float32)
        np.savez_compressed(self.cache_file, ids=np.asarray(self.ids), embeddings=self.reference_embeddings)

    def rank(self, rgb, exclude=()):
        self._ensure_references()
        q = self.embed(rgb)
        scores = self.reference_embeddings @ q
        excluded = set(exclude)
        return [(self.ids[i], float(scores[i])) for i in np.argsort(-scores) if self.ids[i] not in excluded]


def reciprocal_rank_fusion(rankings, k=60.0):
    """Fuse retrieval systems without assuming comparable score scales.

    This is used only by the synthetic-map robustness mode.  The strict paper
    baseline remains NetVLAD-only.  Reciprocal-rank fusion is intentionally
    scale-free, so NetVLAD and DINOv2 can contribute without arbitrary score
    calibration.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, (rid, _score) in enumerate(ranking):
            fused[str(rid)] += 1.0 / (float(k) + rank + 1.0)
    return sorted(fused.items(), key=lambda row: (-row[1], row[0]))


def pose_diverse_ranking(capture, ranking, top_k=50, pool=250, translation_m=.30, rotation_deg=15.):
    """Suppress near-duplicate synthetic poses while preserving NetVLAD order.

    Dense Blender databases can return dozens of almost identical views from one
    location. MeshLoc's top-50 retrieval budget is more useful when it covers a
    neighborhood of distinct camera centers/orientations. We first inspect a
    larger NetVLAD pool, keep pose-diverse entries, then back-fill with skipped
    entries so this adaptation can never reduce the requested top-k count.
    """
    ranking = list(ranking)[:max(int(pool), int(top_k))]
    if not ranking or top_k <= 0:
        return []
    cos_threshold = math.cos(math.radians(float(rotation_deg)))
    kept, skipped = [], []
    pose_cache = {}

    def pose(rid):
        if rid not in pose_cache:
            frame = capture.frames[rid]
            center = np.asarray(frame.T_world_cv[:3, 3], float)
            forward = _safe_unit(np.asarray(frame.T_world_cv[:3, 2], float).reshape(1, 3))[0]
            pose_cache[rid] = (center, forward)
        return pose_cache[rid]

    for row in ranking:
        rid, score = row
        center, forward = pose(rid)
        duplicate = False
        for krid, _ in kept:
            kcenter, kforward = pose(krid)
            if np.linalg.norm(center - kcenter) < translation_m and float(np.dot(forward, kforward)) > cos_threshold:
                duplicate = True
                break
        (skipped if duplicate else kept).append((rid, score))
        if len(kept) >= top_k:
            break
    if len(kept) < top_k:
        used = {rid for rid, _ in kept}
        for row in skipped + ranking:
            if row[0] not in used:
                kept.append(row); used.add(row[0])
                if len(kept) >= top_k:
                    break
    return kept[:top_k]


def _resize_loftr(rgb, max_side=640, max_area=640 * 480):
    """Resize for indoor LoFTR while returning original-coordinate scale."""
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w), math.sqrt(max_area / max(h * w, 1)))
    nh = max(32, int(round(h * scale / 8)) * 8)
    nw = max(32, int(round(w * scale / 8)) * 8)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    return resized, np.array([w / nw, h / nh], np.float32)


class LoFTRMatcher:
    """Indoor LoFTR matcher, one of the local matchers evaluated by MeshLoc."""

    def __init__(self, device=None):
        import torch
        from kornia.feature import LoFTR
        self.torch = torch
        self.device = device or _device()
        self.model = LoFTR(pretrained="indoor_new").eval().to(self.device)

    def _gray_tensor(self, rgb):
        torch = self.torch
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return torch.from_numpy(gray).float()[None, None].to(self.device) / 255.0

    @property
    def inference_mode(self):
        return self.torch.inference_mode

    def match(self, query_rgb, reference_rgb):
        q, qscale = _resize_loftr(query_rgb)
        r, rscale = _resize_loftr(reference_rgb)
        with self.torch.inference_mode():
            out = self.model({"image0": self._gray_tensor(q), "image1": self._gray_tensor(r)})
        qxy = out["keypoints0"].detach().cpu().numpy().astype(np.float32) * qscale
        rxy = out["keypoints1"].detach().cpu().numpy().astype(np.float32) * rscale
        conf = out["confidence"].detach().cpu().numpy().astype(np.float32)
        return qxy, rxy, conf


def cluster_query_positions(query_xy, radius_px=4.0):
    """Assign repeatable query IDs to dense pairwise matches by pixel clustering."""
    pts = np.asarray(query_xy, np.float32).reshape(-1, 2)
    if not len(pts):
        return np.empty(0, np.int32)
    # Grid hashing is deterministic and fast; adjacent cells are merged greedily.
    cell = max(float(radius_px), 1.0)
    buckets = defaultdict(list)
    centers = []
    ids = np.full(len(pts), -1, np.int32)
    for i, p in enumerate(pts):
        gx, gy = np.floor(p / cell).astype(int)
        best = None
        best_d = float("inf")
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for cid in buckets.get((gx + dx, gy + dy), []):
                    d = float(np.linalg.norm(p - centers[cid]))
                    if d <= radius_px and d < best_d:
                        best, best_d = cid, d
        if best is None:
            best = len(centers)
            centers.append(p.copy())
            buckets[(gx, gy)].append(best)
        else:
            centers[best] = 0.7 * centers[best] + 0.3 * p
        ids[i] = best
    return ids


def _project_world(frame, xyz):
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    Rwc = frame.T_world_cv[:3, :3]
    twc = frame.T_world_cv[:3, 3]
    cam = (xyz - twc) @ Rwc
    z = cam[:, 2]
    uv = np.column_stack([frame.K[0, 0] * cam[:, 0] / z + frame.K[0, 2], frame.K[1, 1] * cam[:, 1] / z + frame.K[1, 2]])
    return uv, z


def merge_correspondences(capture, query_xy, xyz, qids, ref_ids, ref_xy, threshold_px=4.0):
    """MeshLoc-style optional merging of multiple 3D matches per query feature.

    For each query cluster, choose the candidate 3D point with the strongest
    reprojection consensus across its matched database observations, then refine
    that point by robust multi-view reprojection optimization. If no candidate
    has at least two supporting observations, keep the individual matches.
    """
    query_xy = np.asarray(query_xy, float)
    xyz = np.asarray(xyz, float)
    qids = np.asarray(qids, int)
    ref_xy = np.asarray(ref_xy, float)
    ref_ids = np.asarray(ref_ids, object)
    out_q, out_xyz, out_ids, out_refs = [], [], [], []

    for qid in np.unique(qids):
        idx = np.flatnonzero(qids == qid)
        if len(idx) == 1:
            i = idx[0]
            out_q.append(query_xy[i]); out_xyz.append(xyz[i]); out_ids.append(qid); out_refs.append(ref_ids[i])
            continue
        observations = [(capture.frames[str(ref_ids[i])], ref_xy[i]) for i in idx]
        best = None
        for i in idx:
            candidate = xyz[i]
            errors = []
            for frame, obs in observations:
                uv, z = _project_world(frame, candidate[None])
                errors.append(float(np.linalg.norm(uv[0] - obs)) if z[0] > 0.05 else float("inf"))
            errors = np.asarray(errors)
            mask = errors <= threshold_px
            score = int(mask.sum())
            cost = float(np.minimum(errors**2, threshold_px**2).sum())
            key = (score, -cost)
            if best is None or key > best[0]:
                best = (key, candidate.copy(), mask)
        _, seed, mask = best
        if int(mask.sum()) < 2:
            for i in idx:
                out_q.append(query_xy[i]); out_xyz.append(xyz[i]); out_ids.append(qid); out_refs.append(ref_ids[i])
            continue
        inlier_obs = [observations[j] for j in np.flatnonzero(mask)]

        def residual(point):
            rows = []
            for frame, obs in inlier_obs:
                uv, z = _project_world(frame, np.asarray(point)[None])
                if z[0] <= 0.05:
                    rows.extend([threshold_px * 3, threshold_px * 3])
                else:
                    rows.extend((uv[0] - obs).tolist())
            return np.asarray(rows)

        refined = least_squares(residual, seed, loss="huber", f_scale=threshold_px, max_nfev=50).x
        out_q.append(query_xy[idx[0]])
        out_xyz.append(refined)
        out_ids.append(qid)
        out_refs.append(ref_ids[idx[0]])

    return (np.asarray(out_q, np.float32), np.asarray(out_xyz, np.float64),
            np.asarray(out_ids, np.int32), np.asarray(out_refs, object))


def covisibility_components(qids, ref_ids):
    """On-the-fly MeshLoc covisibility approximation from shared query matches."""
    qids = np.asarray(qids, int)
    ref_ids = np.asarray(ref_ids, object)
    refs = sorted({str(x) for x in ref_ids})
    graph = {r: set() for r in refs}
    by_q = defaultdict(set)
    for q, r in zip(qids, ref_ids):
        by_q[int(q)].add(str(r))
    for rs in by_q.values():
        rs = list(rs)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                graph[rs[i]].add(rs[j]); graph[rs[j]].add(rs[i])
    components = []
    seen = set()
    for root in refs:
        if root in seen:
            continue
        todo = deque([root]); seen.add(root); comp = []
        while todo:
            r = todo.popleft(); comp.append(r)
            for n in graph[r]:
                if n not in seen:
                    seen.add(n); todo.append(n)
        components.append(comp)
    return components


def _camera_dict(K, image_size):
    w, h = image_size
    K = np.asarray(K, float)
    return {"model": "PINHOLE", "width": int(w), "height": int(h),
            "params": [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]}


def _reprojection_errors(T_world_cv, xyz, xy, K):
    Rwc = T_world_cv[:3, :3]
    twc = T_world_cv[:3, 3]
    Rcw = Rwc.T
    tcw = -Rcw @ twc
    cam = np.asarray(xyz) @ Rcw.T + tcw
    z = cam[:, 2]
    uv = np.column_stack([K[0, 0] * cam[:, 0] / z + K[0, 2], K[1, 1] * cam[:, 1] / z + K[1, 2]])
    return np.linalg.norm(uv - np.asarray(xy), axis=1), z > 0.05


def _opencv_lo_ransac(xy, xyz, K, settings):
    """Portable fallback: P3P RANSAC + iterative local optimization + Cauchy refinement."""
    cv2.setRNGSeed(settings.seed)
    flag = getattr(cv2, "SOLVEPNP_P3P", cv2.SOLVEPNP_AP3P)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(xyz, np.float64), np.asarray(xy, np.float64), np.asarray(K, np.float64), None,
        iterationsCount=settings.max_ransac_iterations, reprojectionError=settings.reprojection_px,
        confidence=0.9999, flags=flag)
    if not ok or inliers is None:
        return None, {"reason": "p3p_ransac_failed", "backend": "opencv"}

    def residual(params):
        r, t = params[:3], params[3:]
        projected = cv2.projectPoints(np.asarray(xyz, np.float64), r, t, K, None)[0].reshape(-1, 2)
        return (projected - xy).reshape(-1)

    params = np.r_[rvec.reshape(3), tvec.reshape(3)]
    # Local optimization / robust nonlinear refinement analogous to MeshLoc's
    # PoseLib LO-RANSAC post-refinement with a Cauchy loss.
    for _ in range(3):
        opt = least_squares(residual, params, loss="cauchy", f_scale=max(1.0, settings.reprojection_px / 2), max_nfev=100)
        params = opt.x
        R = cv2.Rodrigues(params[:3])[0]
        T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ params[3:]
        err, pos = _reprojection_errors(T, xyz, xy, K)
        mask = pos & (err <= settings.reprojection_px)
        if mask.sum() < 4:
            break
        # Refine only current inliers, then reclassify.
        def rin(p):
            projected = cv2.projectPoints(np.asarray(xyz[mask], np.float64), p[:3], p[3:], K, None)[0].reshape(-1, 2)
            return (projected - xy[mask]).reshape(-1)
        params = least_squares(rin, params, loss="cauchy", f_scale=max(1.0, settings.reprojection_px / 2), max_nfev=80).x

    R = cv2.Rodrigues(params[:3])[0]
    T = np.eye(4); T[:3, :3] = R.T; T[:3, 3] = -R.T @ params[3:]
    err, pos = _reprojection_errors(T, xyz, xy, K)
    mask = pos & (err <= settings.reprojection_px)
    return T, {"inlier_mask": mask, "backend": "OpenCV P3P + local Cauchy refinement"}


def estimate_pose_meshloc(xy, xyz, K, image_size, settings):
    """P3P LO-RANSAC + nonlinear refinement, using PoseLib when installed."""
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    finite = np.isfinite(xy).all(1) & np.isfinite(xyz).all(1)
    xy, xyz = xy[finite], xyz[finite]
    if len(xy) < max(4, settings.min_inliers):
        return None, {"reason": "too_few_2d_3d_correspondences", "correspondences": int(len(xy))}
    try:
        import poselib
        camera = _camera_dict(K, image_size)
        ransac = {"max_reproj_error": float(settings.reprojection_px),
                  "min_iterations": int(settings.min_ransac_iterations),
                  "max_iterations": int(settings.max_ransac_iterations),
                  "success_prob": 0.9999, "seed": int(settings.seed)}
        try:
            pose, info = poselib.estimate_absolute_pose(xy, xyz, camera, ransac, {})
        except Exception:
            # Compatibility with PoseLib versions that call the threshold max_error.
            ransac["max_error"] = ransac.pop("max_reproj_error")
            pose, info = poselib.estimate_absolute_pose(xy, xyz, camera, ransac, {})
        Rcw = np.asarray(pose.R, float)
        tcw = np.asarray(pose.t, float).reshape(3)
        T = np.eye(4); T[:3, :3] = Rcw.T; T[:3, 3] = -Rcw.T @ tcw
        raw_mask = np.asarray(info.get("inliers", []), bool)
        if raw_mask.shape != (len(xy),):
            err, pos = _reprojection_errors(T, xyz, xy, K)
            raw_mask = pos & (err <= settings.reprojection_px)
        backend = "PoseLib P3P LO-RANSAC + nonlinear refinement"
    except Exception as exc:
        T, fb = _opencv_lo_ransac(xy, xyz, K, settings)
        if T is None:
            return None, dict(fb, poselib_error=str(exc), correspondences=int(len(xy)))
        raw_mask = np.asarray(fb["inlier_mask"], bool)
        backend = fb["backend"]

    err, pos = _reprojection_errors(T, xyz, xy, K)
    mask = raw_mask & pos & (err <= settings.reprojection_px)
    inliers = int(mask.sum())
    ratio = inliers / max(len(xy), 1)
    stats = {
        "correspondences": int(len(xy)), "inliers": inliers, "inlier_ratio": float(ratio),
        "median_reprojection_px": float(np.median(err[mask])) if inliers else float("inf"),
        "p90_reprojection_px": float(np.percentile(err[mask], 90)) if inliers else float("inf"),
        "pose_backend": backend, "inlier_mask": mask,
    }
    if inliers < settings.min_inliers or ratio < settings.min_inlier_ratio:
        return None, dict(stats, reason="insufficient_geometric_support")
    return T, stats


def position_average(T_world_cv, xyz, xy, K, settings):
    """MeshLoc +REF position refinement with an MSAC-style weight.

    Rotation is held fixed while nearby camera centers are sampled.  Instead
    of weighting candidates only by raw inlier count, each sample receives a
    truncated residual score: low-error inliers contribute more than matches
    sitting just under the RANSAC threshold.  This is much closer to robust
    MSAC scoring and avoids preferring a noisier pose merely because it has one
    or two additional borderline inliers.
    """
    if not settings.position_averaging:
        return T_world_cv, {"position_averaging": False}
    T = np.asarray(T_world_cv, float).copy()
    center = T[:3, 3].copy()
    half = float(settings.position_volume_half_m)
    step = float(settings.position_step_m)
    if half <= 0 or step <= 0:
        return T, {"position_averaging": False}
    threshold = float(settings.reprojection_px)
    threshold2 = threshold * threshold
    offsets = np.arange(-half, half + step * 0.5, step)
    candidates, weights, supports = [], [], []
    best_score = 0.0
    for dx in offsets:
        for dy in offsets:
            for dz in offsets:
                c = center + [dx, dy, dz]
                Tc = T.copy(); Tc[:3, 3] = c
                err, pos = _reprojection_errors(Tc, xyz, xy, K)
                finite = np.isfinite(err) & pos
                residual2 = np.full(len(err), threshold2, float)
                residual2[finite] = np.minimum(err[finite] ** 2, threshold2)
                # Equivalent to threshold^2*N - truncated squared-error cost.
                msac_support = float(threshold2 * len(err) - residual2.sum())
                count = int(np.sum(finite & (err <= threshold)))
                if count and msac_support > 0:
                    candidates.append(c); weights.append(msac_support); supports.append(count)
                    best_score = max(best_score, msac_support)
    if not weights:
        return T, {"position_averaging": True, "position_samples_with_support": 0}
    weights = np.asarray(weights, float)
    refined = np.average(np.asarray(candidates), axis=0, weights=weights)
    T[:3, 3] = refined
    return T, {
        "position_averaging": True,
        "position_samples_with_support": int(len(weights)),
        "position_shift_m": float(np.linalg.norm(refined - center)),
        "position_refinement_score": "MSAC-style truncated reprojection residual",
        "position_refinement_best_score": float(best_score),
        "position_refinement_best_support": int(max(supports)),
    }


class MeshLocLocalizer:
    def __init__(self, capture, reference_ids, cache_root, progress=None, device=None):
        self.capture = capture
        self.ids = list(reference_ids)
        self.cache_root = Path(cache_root)
        self.progress = progress or (lambda message, fraction: None)
        self.device = device or _device()
        self.retriever = NetVLADRetriever(capture, reference_ids, self.cache_root / "netvlad", self.progress, self.device)
        self.dino_retriever = None
        self.matcher = None

    def _dino(self):
        if self.dino_retriever is None:
            from .retrieval import DinoV2Retriever
            self.dino_retriever = DinoV2Retriever(
                self.capture, self.ids, self.cache_root / "dino",
                device=self.device, progress=self.progress)
        else:
            self.dino_retriever.progress = self.progress
        return self.dino_retriever

    def _matcher(self):
        if self.matcher is None:
            self.progress(f"Loading LoFTR indoor matcher on {self.device}…", 0.46)
            self.matcher = LoFTRMatcher(self.device)
        return self.matcher

    def localize(self, rgb, candidate_intrinsics, exclude=(), query_hash=None, reference_hashes=None,
                 settings=None):
        s = settings or MeshLocSettings()
        started = time.perf_counter()
        excluded = set(exclude)
        if query_hash:
            excluded.update(rid for rid, h in (reference_hashes or {}).items() if h == query_hash)

        netvlad_ranking = self.retriever.rank(rgb, excluded)
        ranking = netvlad_ranking
        retrieval_components = {"netvlad": netvlad_ranking[:10]}
        retrieval_warning = None
        if str(s.retrieval_mode).lower() == "hybrid":
            try:
                dino_ranking = self._dino().rank(rgb, excluded)
                retrieval_components["dinov2"] = dino_ranking[:10]
                ranking = reciprocal_rank_fusion([netvlad_ranking, dino_ranking])
            except Exception as exc:
                # Keep the MeshLoc solver active even if DINO is unavailable.
                retrieval_warning = f"DINOv2 retrieval unavailable; using NetVLAD only: {exc}"
        elif str(s.retrieval_mode).lower() != "netvlad":
            raise ValueError(f"Unknown MeshLoc retrieval_mode: {s.retrieval_mode}")
        if s.diversify_dense_references:
            candidates = pose_diverse_ranking(
                self.capture, ranking, top_k=min(s.top_k, len(ranking)),
                pool=min(s.retrieval_pool, len(ranking)),
                translation_m=s.diversity_translation_m,
                rotation_deg=s.diversity_rotation_deg)
        else:
            candidates = ranking[:min(s.top_k, len(ranking))]
        result = {
            "success": False,
            "method": "MeshLoc-style: retrieval + LoFTR + bilinear mesh-depth lifting + P3P LO-RANSAC + robust position refinement",
            "settings": asdict(s),
            "retrieval_method": (("Hybrid NetVLAD + multi-crop DINOv2 (RRF)" if str(s.retrieval_mode).lower()=="hybrid" else "NetVLAD Pitts30K")
                                 + (" + pose-diverse suppression" if s.diversify_dense_references else "")),
            "retrieval_components_top10": {name: [{"id": r, "score": float(v)} for r, v in rows]
                                            for name, rows in retrieval_components.items()},
            "retrieval_pool_top10": [{"id": r, "score": float(v)} for r, v in ranking[:10]],
            "retrieval": [{"id": r, "score": float(v)} for r, v in candidates],
            "excluded_reference_count": len(set(self.ids) & excluded),
            "correspondence_identity_note": (
                "LoFTR is a dense pairwise matcher; cross-reference query-feature clustering is approximate. "
                "MeshLoc correspondence merging/covisibility are therefore disabled by default."),
        }
        if retrieval_warning:
            result["retrieval_warning"] = retrieval_warning
        if not candidates:
            return dict(result, reason="no_eligible_references", elapsed_seconds=time.perf_counter() - started)

        matcher = self._matcher()
        q_all, xyz_all, refxy_all, refid_all, confidence_all = [], [], [], [], []
        pair_stats = []
        for j, (rid, score) in enumerate(candidates):
            self.progress(f"MeshLoc LoFTR matching: {j+1}/{len(candidates)}", 0.48 + 0.30 * (j + 1) / max(1, len(candidates)))
            frame = self.capture.frames[rid]
            qxy, rxy, conf = matcher.match(rgb, rgb_image(frame.image))
            raw_count = int(len(qxy))
            if len(qxy):
                world, valid = self.capture.world_points(frame, rxy)
                qxy, rxy, conf, world = qxy[valid], rxy[valid], conf[valid], world[valid]
            else:
                world = np.empty((0, 3), float)
            pair_stats.append({"reference": rid, "retrieval_score": float(score), "raw_matches": raw_count,
                               "depth_matches": int(len(qxy)), "median_match_confidence": float(np.median(conf)) if len(conf) else 0.0})
            if len(qxy):
                q_all.append(qxy); xyz_all.append(world); refxy_all.append(rxy)
                refid_all.extend([rid] * len(qxy)); confidence_all.append(conf)

        result["matches"] = pair_stats
        if not q_all:
            return dict(result, reason="no_valid_mesh_depth_matches", elapsed_seconds=time.perf_counter() - started)

        qxy = np.concatenate(q_all).astype(np.float32)
        xyz = np.concatenate(xyz_all).astype(np.float64)
        refxy = np.concatenate(refxy_all).astype(np.float32)
        refids = np.asarray(refid_all, object)
        conf = np.concatenate(confidence_all).astype(np.float32)
        qids = cluster_query_positions(qxy, s.query_cluster_px)
        result["raw_2d3d_correspondences"] = int(len(xyz))
        result["query_match_clusters"] = int(len(np.unique(qids)))

        if s.use_merge:
            result["merge_warning"] = (
                "Experimental with LoFTR: query identities are formed by pixel clustering, "
                "not native stable keypoint IDs. Keep disabled for paper-style LoFTR results.")
            qxy, xyz, qids, refids = merge_correspondences(
                self.capture, qxy, xyz, qids, refids, refxy, s.merge_reprojection_px)
            result["merged_2d3d_correspondences"] = int(len(xyz))

        # Optional paper ablation: approximate covisibility filtering. The paper's
        # final simple variant disables this because it could over-segment matches.
        components = [sorted(set(map(str, refids)))]
        if s.use_covisibility:
            components = covisibility_components(qids, refids)
        result["covisibility_components"] = [len(c) for c in components]

        solutions = []
        failures = []
        self.progress("MeshLoc P3P LO-RANSAC pose estimation…", 0.82)
        for K in candidate_intrinsics:
            for comp in components:
                mask = np.isin(refids.astype(str), comp)
                if mask.sum() < s.min_inliers:
                    continue
                T, stats = estimate_pose_meshloc(qxy[mask], xyz[mask], K, (rgb.shape[1], rgb.shape[0]), s)
                if T is None:
                    failures.append(stats); continue
                inmask = np.asarray(stats.pop("inlier_mask"), bool)
                in_xyz = xyz[mask][inmask]
                in_xy = qxy[mask][inmask]
                Tpa, pastats = position_average(T, in_xyz, in_xy, K, s)
                err, pos = _reprojection_errors(Tpa, xyz[mask], qxy[mask], K)
                final_mask = pos & (err <= s.reprojection_px)
                final_inliers = int(final_mask.sum())
                score = (final_inliers, -float(np.median(err[final_mask])) if final_inliers else -1e9)
                solutions.append((score, Tpa, K, mask, final_mask, dict(stats, **pastats,
                                   inliers=final_inliers,
                                   median_reprojection_px=float(np.median(err[final_mask])) if final_inliers else float("inf"))))

        if not solutions:
            if failures:
                result.update(max(failures, key=lambda x: (x.get("inliers", 0), x.get("inlier_ratio", 0))))
            result.setdefault("reason", "meshloc_pose_estimation_failed")
            result["elapsed_seconds"] = time.perf_counter() - started
            return result

        _, T, K, comp_mask, final_mask, stats = max(solutions, key=lambda x: x[0])
        result.update(stats)
        result["success"] = True
        result["pose"] = {
            "T_world_camera_cv": T.tolist(),
            "position_m": T[:3, 3].tolist(),
            "quaternion_xyzw": Rotation.from_matrix(T[:3, :3]).as_quat().tolist(),
            "forward_world": T[:3, 2].tolist(),
        }
        result["query_intrinsics"] = np.asarray(K).tolist()
        local_xyz = xyz[comp_mask]
        local_xy = qxy[comp_mask]
        result["inlier_query_pixels"] = local_xy[final_mask].tolist()
        result["inlier_world_points"] = local_xyz[final_mask].tolist()
        result["elapsed_seconds"] = time.perf_counter() - started
        return result

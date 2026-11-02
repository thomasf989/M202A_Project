"""Learned global retrieval utilities for difficult indoor viewpoints.

The metric pose is never taken from DINOv2.  DINOv2 only proposes candidate
reference images.  Pose still comes from local correspondences + reference
geometry + PnP/RANSAC.

v3 adds two robustness ideas that matter for phone photos of a synthetic room:
  * multi-crop retrieval, so portrait/landscape/FOV differences do not destroy
    the global descriptor;
  * pose-neighbour expansion, so once a likely location is retrieved we also
    try nearby synthetic cameras with different yaw/pitch/height.
"""
from pathlib import Path
import hashlib
import math
import numpy as np


def cosine_rank(query_embedding, reference_embeddings, ids, exclude=()):
    q = np.asarray(query_embedding, np.float32).reshape(-1)
    refs = np.asarray(reference_embeddings, np.float32)
    if refs.ndim != 2 or refs.shape[1] != len(q) or len(refs) != len(ids):
        raise ValueError('Embedding matrix shape does not match IDs/query')
    q /= max(float(np.linalg.norm(q)), 1e-8)
    refs = refs / np.maximum(np.linalg.norm(refs, axis=1, keepdims=True), 1e-8)
    scores = refs @ q
    excluded = set(exclude)
    return [(ids[i], float(scores[i])) for i in np.argsort(-scores) if ids[i] not in excluded]


def _crop(rgb, x0, y0, x1, y1):
    h, w = rgb.shape[:2]
    xa, xb = int(round(x0*w)), int(round(x1*w))
    ya, yb = int(round(y0*h)), int(round(y1*h))
    xa, ya = max(0, xa), max(0, ya)
    xb, yb = min(w, xb), min(h, yb)
    if xb-xa < 32 or yb-ya < 32:
        return rgb
    return np.ascontiguousarray(rgb[ya:yb, xa:xb])


def retrieval_views(rgb, reference=False):
    """Return a small set of crops for FOV/aspect-ratio robustness.

    Synthetic references are generally square/landscape while phone queries
    are often portrait.  A single whole-image descriptor is brittle in that
    situation.  We therefore cache only three views per reference (reasonable
    build cost), but use several cheap query crops at run time.
    """
    rgb = np.ascontiguousarray(rgb)
    views = [rgb]
    # Keep reference indexing to one descriptor per rendered view so a room
    # with ~1-3k synthetic views remains practical on CPU. Query-time crops
    # are cheap and provide most of the portrait/FOV robustness.
    if reference:
        return views
    # Center portrait-like and landscape-like query crops are useful because a
    # phone image may be a narrow crop of a much wider synthetic render.
    views.append(_crop(rgb, .18, 0., .82, 1.))
    views.append(_crop(rgb, 0., .18, 1., .82))
    if not reference:
        h, w = rgb.shape[:2]
        if h >= w:  # portrait query: top/center/bottom contextual crops
            views += [_crop(rgb, 0., 0., 1., .76), _crop(rgb, 0., .12, 1., .88), _crop(rgb, 0., .24, 1., 1.)]
        else:       # landscape query: left/center/right contextual crops
            views += [_crop(rgb, 0., 0., .76, 1.), _crop(rgb, .12, 0., .88, 1.), _crop(rgb, .24, 0., 1., 1.)]
    return views[:6]



def expand_pose_ranking(capture, ranking, max_candidates=40, seed_count=10, neighbors_per_seed=6,
                        neighbor_radius_m=1.75):
    """Expand visual retrieval with spatially nearby reference cameras.

    Synthetic databases contain many views at the same/nearby camera center
    but different yaw/pitch.  DINO may retrieve the correct location but not
    the exact orientation.  Trying pose neighbours is much cheaper and more
    reliable than hoping every neighbouring yaw appears independently in the
    global Top-K.
    """
    ranking = [(rid, float(score)) for rid, score in ranking if rid in capture.frames]
    if not ranking:
        return []
    ids = [rid for rid, _ in ranking]
    all_ids = list(getattr(capture, 'reference_ids', capture.ids))
    centers = np.array([capture.frames[rid].T_world_cv[:3, 3] for rid in all_ids], float)
    index = {rid: i for i, rid in enumerate(all_ids)}

    output = []
    used = set()
    def add(rid, score):
        if rid not in used and len(output) < max_candidates:
            used.add(rid); output.append((rid, float(score)))

    # Keep the strongest visual candidates first.
    for rid, score in ranking[:min(len(ranking), max(12, seed_count))]:
        add(rid, score)

    for rid, score in ranking[:seed_count]:
        if len(output) >= max_candidates:
            break
        c = capture.frames[rid].T_world_cv[:3, 3]
        dist = np.linalg.norm(centers-c, axis=1)
        order = np.argsort(dist)
        added = 0
        for j in order:
            d = float(dist[j])
            if d > neighbor_radius_m and added >= 2:
                break
            nid = all_ids[int(j)]
            if nid in used:
                continue
            # Position-neighbour score is only for ordering/display.  Metric
            # pose never trusts this value.
            nscore = score * math.exp(-d/max(neighbor_radius_m, 1e-6)) * .985
            add(nid, nscore)
            added += 1
            if added >= neighbors_per_seed or len(output) >= max_candidates:
                break
    return output


class DinoV2Retriever:
    """DINOv2 indoor retrieval with disk-cached multi-crop reference embeddings."""
    def __init__(self, capture, reference_ids, cache_root, model_name='dinov2_vits14', device=None, progress=None):
        self.capture = capture
        self.ids = list(reference_ids)
        self.model_name = model_name
        if device is None:
            try:
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except Exception:
                device = 'cpu'
        self.device = device
        self.progress = progress or (lambda message, fraction: None)
        key = hashlib.sha256((capture.fingerprint()+','.join(self.ids)+model_name+'-global-multicrop-v3').encode()).hexdigest()[:20]
        self.directory = Path(cache_root) / key
        self.directory.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.directory / 'reference_embeddings.npz'
        self.model = None
        self.embeddings = None

    def _load_model(self):
        if self.model is not None:
            return
        try:
            import os
            import torch
            repo = os.environ.get('DINOV2_REPO')
            self.progress(f'Loading DINOv2 global retrieval model on {self.device}…', .30)
            if repo:
                self.model = torch.hub.load(repo, self.model_name, source='local')
            else:
                self.model = torch.hub.load('facebookresearch/dinov2', self.model_name, trust_repo=True)
            self.model.eval().to(self.device)
        except Exception as e:
            raise RuntimeError(f'DINOv2 unavailable: {e}') from e

    def _tensor(self, rgb):
        import torch
        import torch.nn.functional as F
        image = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()[None] / 255.0
        h, w = image.shape[-2:]
        # 784 keeps more indoor detail than the old 700-pixel cap while still
        # being practical on CPU.  DINOv2 patch size is 14.
        scale = min(1.0, 784.0 / max(h, w))
        nh = max(14, int(round(h * scale / 14)) * 14)
        nw = max(14, int(round(w * scale / 14)) * 14)
        image = F.interpolate(image, size=(nh, nw), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=image.dtype)[None, :, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=image.dtype)[None, :, None, None]
        return ((image - mean) / std).to(self.device)

    def embed(self, rgb):
        import torch
        self._load_model()
        with torch.inference_mode():
            x = self._tensor(rgb)
            if hasattr(self.model, 'forward_features'):
                out = self.model.forward_features(x)
                if isinstance(out, dict):
                    cls = out.get('x_norm_clstoken')
                    patches = out.get('x_norm_patchtokens')
                    emb = cls
                    if cls is not None and patches is not None and patches.ndim == 3:
                        # Blend scene-level CLS with average local context.  Both
                        # are in the same DINO embedding space.
                        emb = .78*cls + .22*patches.mean(dim=1)
                    if emb is None:
                        emb = next(v for v in out.values() if hasattr(v, 'shape'))
                else:
                    emb = out
            else:
                emb = self.model(x)
            emb = emb.reshape(emb.shape[0], -1)[0].float().cpu().numpy()
        emb /= max(float(np.linalg.norm(emb)), 1e-8)
        return emb.astype(np.float32)

    def embed_views(self, rgb, reference=False):
        return np.stack([self.embed(v) for v in retrieval_views(rgb, reference=reference)]).astype(np.float32)

    def _build(self):
        from .polycam import rgb_image
        vectors = []
        for i, rid in enumerate(self.ids):
            self.progress(f'DINOv2 reference embeddings: {i+1}/{len(self.ids)}', .32 + .18*(i+1)/max(len(self.ids),1))
            vectors.append(self.embed_views(rgb_image(self.capture.frames[rid].image), reference=True))
        self.embeddings = np.stack(vectors)  # [N, reference_views, D]
        np.savez_compressed(self.cache_file, ids=np.asarray(self.ids), embeddings=self.embeddings)

    def _ensure(self):
        if self.embeddings is not None:
            return
        if self.cache_file.exists():
            with np.load(self.cache_file, allow_pickle=False) as z:
                ids = z['ids'].astype(str).tolist()
                emb = z['embeddings'].astype(np.float32)
                if ids == self.ids and emb.ndim == 3:
                    self.embeddings = emb
                    return
        self._build()

    def rank(self, rgb, exclude=()):
        self._ensure()
        q = self.embed_views(rgb, reference=False)  # [Q,D]
        # Max similarity across query/reference crops makes retrieval tolerant
        # to portrait photos, digital zoom, and narrower phone FOVs.
        scores = np.einsum('nvd,qd->nvq', self.embeddings, q).max(axis=(1, 2))
        excluded = set(exclude)
        return [(self.ids[i], float(scores[i])) for i in np.argsort(-scores) if self.ids[i] not in excluded]

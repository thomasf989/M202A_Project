"""Optional learned global retrieval for large viewpoint changes.

DINOv2 is used only to rank candidate reference views. It is not trusted to
produce metric pose. Metric pose still comes from local feature matches,
reference geometry, and PnP/RANSAC.
"""
from pathlib import Path
import hashlib
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


class DinoV2Retriever:
    """DINOv2 CLS-token retrieval with disk-cached reference embeddings.

    First use may download the official DINOv2 repository/weights through
    torch.hub. Set DINOV2_REPO to a local clone to make first use offline.
    """
    def __init__(self, capture, reference_ids, cache_root, model_name='dinov2_vits14', device='cpu', progress=None):
        self.capture = capture
        self.ids = list(reference_ids)
        self.model_name = model_name
        self.device = device
        self.progress = progress or (lambda message, fraction: None)
        key = hashlib.sha256((capture.fingerprint()+','.join(self.ids)+model_name+'-global-v2').encode()).hexdigest()[:20]
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
            self.progress('Loading DINOv2 global retrieval model…', .32)
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
        scale = min(1.0, 700.0 / max(h, w))
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
                emb = out.get('x_norm_clstoken') if isinstance(out, dict) else out
            else:
                emb = self.model(x)
            emb = emb.reshape(emb.shape[0], -1)[0].float().cpu().numpy()
        emb /= max(float(np.linalg.norm(emb)), 1e-8)
        return emb.astype(np.float32)

    def _build(self):
        from .polycam import rgb_image
        vectors = []
        for i, id in enumerate(self.ids):
            self.progress(f'DINOv2 reference embeddings: {i+1}/{len(self.ids)}', .34 + .16*(i+1)/max(len(self.ids),1))
            vectors.append(self.embed(rgb_image(self.capture.frames[id].image)))
        self.embeddings = np.stack(vectors)
        np.savez_compressed(self.cache_file, ids=np.asarray(self.ids), embeddings=self.embeddings)

    def _ensure(self):
        if self.embeddings is not None:
            return
        if self.cache_file.exists():
            with np.load(self.cache_file, allow_pickle=False) as z:
                ids = z['ids'].astype(str).tolist()
                if ids == self.ids:
                    self.embeddings = z['embeddings'].astype(np.float32)
                    return
        self._build()

    def rank(self, rgb, exclude=()):
        self._ensure()
        q = self.embed(rgb)
        return cosine_rank(q, self.embeddings, self.ids, exclude)

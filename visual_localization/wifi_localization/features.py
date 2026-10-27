"""RootSIFT, reference-only visual vocabulary retrieval, and local matching."""
from pathlib import Path
import cv2
import numpy as np
from .polycam import rgb_image, sha256, read_json, write_json


def extract(rgb, nfeatures=3000):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Suppress the artificial border in Polycam undistorted images.
    mask = np.uint8(gray > 8) * 255
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))
    kps, desc = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=.025).detectAndCompute(gray, mask)
    if desc is None:
        return np.empty((0, 2), np.float32), np.empty((0, 128), np.float32)
    desc /= np.maximum(desc.sum(axis=1, keepdims=True), 1e-8)
    return np.array([k.pt for k in kps], np.float32), np.sqrt(desc).astype(np.float32)


def assign_words(desc, centers):
    if not len(desc):
        return np.empty(0, int)
    return np.array([m.trainIdx for m in cv2.BFMatcher(cv2.NORM_L2).match(desc, centers)], dtype=int)


def histogram(desc, centers):
    h = np.bincount(assign_words(desc, centers), minlength=len(centers)).astype(np.float32)
    return h / max(h.sum(), 1)


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def build_database(capture, out, query_every=5, nfeatures=3000, words=128, seed=42, progress=None):
    out = Path(out)
    if (out / 'database.json').exists():
        raise ValueError(f"Database exists: {out}. Use a new output directory to avoid stale results.")
    if query_every is not None and query_every < 3:
        raise ValueError("query_every must be at least 3")
    official_split = hasattr(capture, 'reference_ids')
    queries = list(capture.query_ids) if official_split else (capture.ids[query_every//2::query_every] if query_every is not None else [])
    references = list(capture.reference_ids) if official_split else [id for id in capture.ids if id not in queries]
    if (query_every is not None and len(queries) < 1) or len(references) < 4:
        raise ValueError("Need at least one query and four reference images")
    cv2.setRNGSeed(seed)
    rng = np.random.default_rng(seed)
    (out / 'features').mkdir(parents=True, exist_ok=True)
    samples = []
    ref_desc = []
    hashes = {}
    for j, id in enumerate(references):
        f = capture.frames[id]
        xy, desc = extract(rgb_image(f.image), nfeatures)
        xyz, valid = capture.world_points(f, xy)
        np.savez_compressed(out / 'features' / f'{id}.npz', xy=xy, desc=desc, xyz=xyz, valid=valid)
        ref_desc.append(desc)
        if len(desc):
            samples.append(desc[rng.choice(len(desc), min(200, len(desc)), replace=False)])
        hashes[id] = sha256(f.image)
        if progress:
            progress(j+1, len(references))
        if j % 10 == 0 or j == len(references)-1:
            print(f"Reference features {j+1}/{len(references)}: {len(xy)} keypoints, {valid.sum()} valid depth", flush=True)
    if not samples or sum(map(len, samples)) < 16:
        raise ValueError("Too few reference features to create a vocabulary")
    training = np.concatenate(samples)
    words = min(int(words), len(training))
    _, _, centers = cv2.kmeans(training, words, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4), 1, cv2.KMEANS_PP_CENTERS)
    hist = np.stack([histogram(d, centers) for d in ref_desc])
    idf = (np.log((len(references)+1)/(np.count_nonzero(hist, axis=0)+1)) + 1).astype(np.float32)
    vectors = normalize(np.sqrt(hist) * idf)
    np.savez_compressed(out / 'retrieval.npz', centers=centers, idf=idf, vectors=vectors)
    metadata = dict(version=1, method='RootSIFT + visual-word TF-IDF + depth PnP',
        capture_fingerprint=capture.fingerprint(), reference_ids=references, query_ids=queries,
        reference_image_hashes=hashes, nfeatures=nfeatures, words=words, seed=seed,
        alignment=capture.alignment, depth_scale=capture.depth_scale,
        dataset_format=getattr(capture, 'dataset_format', 'polycam'),
        split=(getattr(capture, 'split_description', 'Official 7-Scenes train/test SEQUENCES; only sampled train frames enter the database') if official_split else
               'Every nth corrected frame held out before vocabulary construction' if query_every is not None
               else 'All corrected frames as references for interactive localization'), query_every=query_every)
    write_json(out / 'database.json', metadata)
    return metadata


class Database:
    def __init__(self, path, capture):
        self.path = Path(path)
        self.meta = read_json(self.path / 'database.json')
        if self.meta.get('version') != 1:
            raise ValueError("Unsupported database version")
        if self.meta['capture_fingerprint'] != capture.fingerprint():
            raise ValueError("Database input/configuration changed. Rebuild the database in a new directory.")
        self.ids = self.meta['reference_ids']
        if hasattr(capture, 'reference_ids'):
            if not set(self.ids) <= set(capture.reference_ids) or not set(self.meta['query_ids']) <= set(capture.query_ids):
                raise ValueError('7-Scenes database violates the official sequence split. Rebuild it.')
        with np.load(self.path / 'retrieval.npz', allow_pickle=False) as d:
            self.centers, self.idf, self.vectors = d['centers'], d['idf'], d['vectors']
        self.cache = {}

    def features(self, id):
        if id not in self.cache:
            with np.load(self.path / 'features' / f'{id}.npz', allow_pickle=False) as d:
                self.cache[id] = {k: d[k] for k in d.files}
        return self.cache[id]

    def rank(self, desc, exclude=()):
        vector = normalize(np.sqrt(histogram(desc, self.centers)) * self.idf)
        scores = self.vectors @ vector
        excluded = set(exclude)
        return [(self.ids[i], float(scores[i])) for i in np.argsort(-scores) if self.ids[i] not in excluded]


def match(desc_query, desc_reference, ratio=.75):
    if min(len(desc_query), len(desc_reference)) < 2:
        return np.empty((0, 2), int), np.empty(0)
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=64))
    forward = matcher.knnMatch(desc_query, desc_reference, k=2)
    backward = matcher.knnMatch(desc_reference, desc_query, k=1)
    reverse = {m[0].queryIdx: m[0].trainIdx for m in backward if m}
    kept = [(a.queryIdx, a.trainIdx, a.distance) for pair in forward if len(pair) == 2
            for a, b in [pair] if a.distance < ratio*b.distance and reverse.get(a.trainIdx) == a.queryIdx]
    if not kept:
        return np.empty((0, 2), int), np.empty(0)
    rows = np.asarray(kept)
    return rows[:, :2].astype(int), rows[:, 2]

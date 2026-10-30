"""Polycam input and explicit coordinate conventions.

Column vectors. T_world_camera maps camera coordinates to world coordinates.
Polycam camera: +X right, +Y up, -Z forward. OpenCV: +X right, +Y down,
+Z forward. World is the exported GLB frame, in meters. No axis reordering.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageOps

CV_TO_POLYCAM = np.diag([1., -1., -1., 1.])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rigid_matrix(T, label):
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError(f"{label}: expected a finite 4x4 transform")
    R = T[:3, :3]
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError(f"{label}: invalid homogeneous last row")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(R), 1., atol=1e-4):
        raise ValueError(f"{label}: transform is not a proper rigid rotation")
    return T


def camera_matrix(c):
    K = np.array([[c['fx'], 0, c['cx']], [0, c['fy'], c['cy']], [0, 0, 1]], dtype=float)
    if not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    return K


def rgb_image(path, orient=False):
    with Image.open(path) as im:
        if orient:
            im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB")).copy()


def prepare_capture(archive, glb, output):
    """Extract only capture data; reject traversal and avoid overwriting files."""
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be empty: {output}")
    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if not n.endswith('/') and
                 not any(s.startswith('.') or s == '__MACOSX' for s in Path(n).parts)]
        roots = [n[:-len('optimized_roomplan.json')] for n in names if n.endswith('optimized_roomplan.json')]
        if len(roots) != 1:
            raise ValueError("ZIP must contain exactly one optimized_roomplan.json")
        root = roots[0]
        selected = []
        for n in names:
            if not n.startswith(root):
                continue
            rel = Path(n[len(root):])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError(f"Unsafe ZIP path: {n}")
            if rel.parts and (rel.parts[0] == 'keyframes' or rel.name in ['optimized_roomplan.json', 'roomplan.json']):
                selected.append((n, output / rel))
        output.mkdir(parents=True, exist_ok=True)
        for n, dest in selected:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(n) as src, dest.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    shutil.copy2(glb, output / 'room.glb')
    return output


@dataclass
class Frame:
    id: str
    image: Path
    camera: Path
    depth: Path
    confidence: Path
    K: np.ndarray
    T_world_cv: np.ndarray
    width: int
    height: int


class Capture:
    def __init__(self, root, alignment='forward', depth_scale=.001):
        self.root = Path(root).resolve()
        self.depth_scale = float(depth_scale)
        if not np.isfinite(self.depth_scale) or self.depth_scale <= 0:
            raise ValueError("depth_scale must be positive")
        room = read_json(self.root / 'optimized_roomplan.json')
        a = room['alignment_transform']
        A = rigid_matrix([[a[f't{i}{j}'] for j in range(4)] for i in range(4)], 'alignment_transform')
        self.alignment = alignment
        self.A = {'forward': A, 'inverse': np.linalg.inv(A), 'identity': np.eye(4)}[alignment]
        self.raw_alignment = A
        self.frames = {}
        images = self.root / 'keyframes/corrected_images'
        cameras = sorted((self.root / 'keyframes/corrected_cameras').glob('*.json'))
        if not cameras:
            raise ValueError("No corrected_cameras JSON files. Export Polycam raw data with corrected frames.")
        for path in cameras:
            c = read_json(path)
            source = np.eye(4)
            source[:3, :] = [[c[f't_{i}{j}'] for j in range(4)] for i in range(3)]
            rigid_matrix(source, path.name)
            image = images / f'{path.stem}.jpg'
            depth = self.root / 'keyframes/depth' / f'{path.stem}.png'
            conf = self.root / 'keyframes/confidence' / f'{path.stem}.png'
            for p in [image, depth, conf]:
                if not p.is_file():
                    raise ValueError(f"Missing matching input: {p}")
            with Image.open(image) as im:
                if im.size != (int(c['width']), int(c['height'])):
                    raise ValueError(f"Image/intrinsic size mismatch: {image}")
            self.frames[path.stem] = Frame(path.stem, image, path, depth, conf,
                camera_matrix(c), self.A @ source @ CV_TO_POLYCAM, int(c['width']), int(c['height']))
        self.ids = sorted(self.frames, key=int)
        self.glb = self.root / 'room.glb'
        self._mesh = None
        self._ray = None

    def fingerprint(self):
        """Hash data content, not modification times, so stale caches are rejected."""
        h = hashlib.sha256()
        h.update(self.A.tobytes())
        h.update(str(self.depth_scale).encode())
        h.update(b'polycam-depth-plus-sparse-mesh-consistency-v1')
        for id in self.ids:
            f = self.frames[id]
            for p in [f.image, f.camera, f.depth, f.confidence]:
                h.update(sha256(p).encode())
        return h.hexdigest()

    def _mesh_z_at_pixels(self, frame, uv):
        """Sparse ray-cast mesh depth in the reference RGB grid.

        This is used only as a consistency gate for sensor depth. If ray
        casting is unavailable or the mesh has no hit for a pixel, the sensor
        depth is preserved rather than rejected.
        """
        if not hasattr(self, 'glb') or not Path(self.glb).is_file() or not len(uv):
            return np.full(len(uv), np.nan)
        try:
            if getattr(self, '_ray', None) is None:
                from .visualization import load_mesh
                from trimesh.ray.ray_triangle import RayMeshIntersector
                self._mesh = load_mesh(self.glb)
                self._ray = RayMeshIntersector(self._mesh)
            x=(uv[:,0]-frame.K[0,2])/frame.K[0,0]
            y=(uv[:,1]-frame.K[1,2])/frame.K[1,1]
            directions_camera=np.column_stack([x,y,np.ones(len(uv))])
            directions_world=directions_camera @ frame.T_world_cv[:3,:3].T
            directions_world/=np.maximum(np.linalg.norm(directions_world,axis=1,keepdims=True),1e-12)
            origins=np.repeat(frame.T_world_cv[None,:3,3],len(uv),axis=0)
            locations,index_ray,_=self._ray.intersects_location(origins,directions_world,multiple_hits=False)
            mesh_z=np.full(len(uv),np.nan,float)
            if len(index_ray):
                camera=(locations-frame.T_world_cv[:3,3]) @ frame.T_world_cv[:3,:3]
                mesh_z[index_ray]=camera[:,2]
            return mesh_z
        except Exception:
            return np.full(len(uv), np.nan)

    def world_points(self, frame, pixels, min_confidence=127, edge_threshold=.15, mesh_tolerance=.25):
        """Lift RGB pixels with depth, confidence, edge and mesh checks.

        Polycam sensor depth is associated to corrected RGB by scaled pixel
        position and is therefore approximate. To remove large bad lifts near
        depth discontinuities/misalignment, valid sensor samples are compared
        against a sparse ray-cast of the exported room mesh. A finite mesh hit
        that disagrees by more than ``mesh_tolerance`` meters is rejected.
        """
        uv = np.asarray(pixels, dtype=float).reshape(-1, 2)
        d = np.asarray(Image.open(frame.depth), dtype=np.float64) * self.depth_scale
        conf = np.asarray(Image.open(frame.confidence))
        if d.shape != conf.shape or d.ndim != 2:
            raise ValueError(f"Depth/confidence shape mismatch: {frame.id}")
        h, w = d.shape
        dp = (uv + .5) * [w / frame.width, h / frame.height] - .5
        ij = np.rint(dp).astype(int)
        valid = (ij[:, 0] >= 1) & (ij[:, 0] < w - 1) & (ij[:, 1] >= 1) & (ij[:, 1] < h - 1)
        ij = np.clip(ij, [1, 1], [w-2, h-2])
        x, y = ij.T
        patches = np.stack([d[y+dy, x+dx] for dy in [-1, 0, 1] for dx in [-1, 0, 1]], axis=1)
        z = d[y, x]
        valid &= (z > .15) & (z < 5.0) & (conf[y, x] >= min_confidence)
        valid &= (patches.min(axis=1) > 0) & (np.ptp(patches, axis=1) < edge_threshold)
        if mesh_tolerance is not None and mesh_tolerance > 0 and valid.any():
            mesh_z=self._mesh_z_at_pixels(frame,uv)
            comparable=np.isfinite(mesh_z)&(mesh_z>.05)
            valid &= ~comparable | (np.abs(mesh_z-z) <= float(mesh_tolerance))
        pc = np.column_stack([(uv[:, 0]-frame.K[0, 2])/frame.K[0, 0]*z,
                              (uv[:, 1]-frame.K[1, 2])/frame.K[1, 1]*z, z])
        return pc @ frame.T_world_cv[:3, :3].T + frame.T_world_cv[:3, 3], valid


def query_calibration(path, size):
    """JSON uses fx,fy,cx,cy,width,height, optional OpenCV distortion array."""
    c = read_json(path)
    if not {'width', 'height'} <= c.keys():
        raise ValueError("Query calibration must include width and height")
    if (int(c['width']), int(c['height'])) != tuple(size):
        raise ValueError(f"Calibration size {(c['width'], c['height'])} differs from oriented query {size}. "
                         "Calibrate this resolution/orientation, or update the intrinsics explicitly.")
    dist = np.array(c.get('distortion', [0, 0, 0, 0, 0]), dtype=float)
    if dist.size not in [4, 5, 8, 12, 14] or not np.isfinite(dist).all():
        raise ValueError("distortion must contain 4, 5, 8, 12 or 14 finite OpenCV coefficients")
    return camera_matrix(c), dist

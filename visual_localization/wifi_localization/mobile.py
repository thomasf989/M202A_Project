"""Export reference-only maps for the offline iPhone localizer.

The package contains precomputed features and a display mesh, never query
photos, query poses, original image files, or learned-model weights.
"""
from pathlib import Path
import json
import os
import struct
import tempfile

import cv2
import numpy as np
import trimesh

from .features import Database
from .visualization import load_mesh

MAGIC = b'RLOC0001'
MESH_MAGIC = b'RMSH0001'
MAX_MAP_BYTES = 256 * 1024 * 1024
MAX_MESH_BYTES = 64 * 1024 * 1024


def _display_mesh(path):
    mesh = load_mesh(path)
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError('The room model has no triangles')
    colors = (mesh.visual.to_color() if hasattr(mesh.visual, 'to_color') else mesh.visual).vertex_colors
    # Only the display mesh is simplified. Exported localization points retain
    # the original GLB coordinates and geometry used by the desktop database.
    voxel = .02
    for _ in range(12):
        _, inverse = np.unique(np.floor(mesh.vertices / voxel).astype(np.int64), axis=0, return_inverse=True)
        count = np.bincount(inverse)
        vertices = np.stack([np.bincount(inverse, weights=mesh.vertices[:, d]) / count for d in range(3)], axis=1)
        faces = inverse[mesh.faces]
        faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
        _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        faces = faces[np.sort(first)]
        if len(vertices) <= 200_000 and len(faces) <= 300_000:
            break
        voxel *= 1.5
    if not len(faces) or len(vertices) > 200_000 or len(faces) > 300_000:
        raise ValueError('Display mesh cannot be reduced to the mobile limits')
    rgba = np.stack([np.bincount(inverse, weights=colors[:, d]) / count for d in range(4)], axis=1) / 255.
    normals = trimesh.Trimesh(vertices, faces, process=False).vertex_normals
    packed = np.asarray(np.column_stack([vertices, normals, rgba]), dtype='<f4')
    if not np.isfinite(packed).all():
        raise ValueError('Display mesh contains nonfinite vertices or colors')
    return (MESH_MAGIC + struct.pack('<II', len(vertices), len(faces)) +
            packed.tobytes() + np.asarray(faces, dtype='<u4').tobytes())


def export_mobile(capture, database, output, name='Room', progress=None):
    """Write one atomic .roomloc file, preserving the database reference split."""
    db = database if isinstance(database, Database) else Database(database, capture)
    if db.meta['capture_fingerprint'] != capture.fingerprint():
        raise ValueError('Room data changed. Rebuild the reference database before exporting.')
    if set(db.ids) & set(db.meta.get('query_ids', [])):
        raise ValueError('Query frames must never enter a mobile reference map')
    if not 4 <= len(db.ids) <= 1000:
        raise ValueError('Mobile maps require 4 to 1000 reference frames')
    if getattr(capture, 'reference_ids', None) is not None and not set(db.ids) <= set(capture.reference_ids):
        raise ValueError('Test frames must never enter a mobile reference map')
    output = Path(output)
    if output.suffix.lower() != '.roomloc':
        raise ValueError('Choose an output filename ending in .roomloc')
    output.parent.mkdir(parents=True, exist_ok=True)
    name = str(name).strip()[:80] or 'Room'
    selected = [(i, frame_id, db.features(frame_id)) for i, frame_id in enumerate(db.ids)
                if len(db.features(frame_id)['desc']) >= 2]
    if len(selected) < 4:
        raise ValueError('Mobile maps require at least four textured reference frames')
    with tempfile.TemporaryDirectory(prefix='roomloc-export-') as temp:
        map_path = Path(temp) / 'map.yml'
        storage = cv2.FileStorage(str(map_path), cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_BASE64)
        try:
            storage.write('format', 'roomloc-map-v1')
            storage.write('name', name)
            storage.write('units', 'meters')
            storage.write('camera_convention', 'opencv-camera-to-world')
            storage.write('nfeatures', int(db.meta['nfeatures']))
            storage.write('dataset_format', getattr(capture, 'dataset_format', 'polycam'))
            storage.write('centers', np.asarray(db.centers, np.float32))
            storage.write('idf', np.asarray(db.idf, np.float32).reshape(1, -1))
            storage.write('vectors', np.asarray(db.vectors[[i for i, _, _ in selected]], np.float32))
            profile = getattr(capture, 'profile', None)
            storage.write('dataset_camera_json', json.dumps(profile or {}, allow_nan=False))
            storage.startWriteStruct('references', cv2.FileNode_SEQ)
            for index, (_, frame_id, ref) in enumerate(selected):
                desc = np.asarray(ref['desc'], np.float32)
                xyz = np.asarray(ref['xyz'], np.float32).copy()
                valid = np.asarray(ref['valid'], bool) & np.isfinite(xyz).all(axis=1)
                if desc.ndim != 2 or desc.shape[1] != 128 or not 2 <= len(desc) <= 10_000:
                    raise ValueError(f'Invalid mobile descriptors: {frame_id}')
                if xyz.shape != (len(desc), 3) or not np.isfinite(desc).all():
                    raise ValueError(f'Invalid mobile features: {frame_id}')
                xyz[~valid] = 0
                storage.startWriteStruct('', cv2.FileNode_MAP)
                storage.write('id', frame_id)
                storage.write('image_sha256', db.meta.get('reference_image_hashes', {}).get(frame_id, ''))
                storage.write('descriptors', desc)
                storage.write('world_points', xyz)
                storage.write('valid', valid.astype(np.uint8).reshape(-1, 1))
                storage.endWriteStruct()
                if progress:
                    progress(index + 1, len(selected))
            storage.endWriteStruct()
        finally:
            storage.release()
        size = map_path.stat().st_size
        if size > MAX_MAP_BYTES:
            raise ValueError('Mobile reference map exceeds 256 MiB. Rebuild with fewer reference frames or features.')
        if progress:
            progress(len(selected), len(selected))
        mesh = _display_mesh(capture.glb)
        if len(mesh) > MAX_MESH_BYTES:
            raise ValueError('Display mesh exceeds 64 MiB')
        fd, temporary = tempfile.mkstemp(prefix='.' + output.name + '-', dir=output.parent)
        try:
            with os.fdopen(fd, 'wb') as stream, map_path.open('rb') as source:
                stream.write(struct.pack('<8sQQ', MAGIC, size, len(mesh)))
                while chunk := source.read(1024 * 1024):
                    stream.write(chunk)
                stream.write(mesh)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    return dict(path=str(output), name=name, references=len(selected), bytes=output.stat().st_size,
                method='RootSIFT + TF-IDF + PnP/RANSAC', format='roomloc-v1')

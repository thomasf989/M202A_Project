"""Build an offline iPhone room package from reference-only data.
No neural-model conversion: exports the existing RootSIFT/TF-IDF/PnP baseline.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .capture import load_capture
from .features import Database, build_database
from .polycam import sha256, write_json
from .visualization import load_mesh


def export(data, database, output, name="Scanned room"):
    import cv2
    cv2.setNumThreads(2)
    capture = load_capture(data)
    if hasattr(capture, '_mesh_z_at_pixels'):
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        capture._mesh = load_mesh(capture.glb)
        capture._ray = RayMeshIntersector(capture._mesh)
    if not (Path(database) / 'database.json').exists():
        build_database(capture, database, query_every=None)
    db = Database(database, capture)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    map_id = hashlib.sha256((sha256(capture.glb) + capture.fingerprint()).encode()).hexdigest()
    # OpenCV FileStorage JSON matrices, portable between Python and C++.
    import cv2
    fs = cv2.FileStorage(str(output / 'features.json'), cv2.FILE_STORAGE_WRITE | cv2.FILE_STORAGE_FORMAT_JSON | cv2.FILE_STORAGE_BASE64)
    fs.write('version', 1); fs.write('nfeatures', db.meta['nfeatures'])
    fs.write('centers', db.centers); fs.write('idf', db.idf.reshape(1, -1)); fs.write('vectors', db.vectors)
    fs.write('count', len(db.ids))
    for i, rid in enumerate(db.ids):
        f = db.features(rid)
        fs.write(f'desc_{i}', f['desc'].astype(np.float32))
        fs.write(f'xyz_{i}', f['xyz'].astype(np.float32))
        fs.write(f'valid_{i}', f['valid'].astype(np.uint8).reshape(-1, 1))
    fs.release()
    mesh = load_mesh(capture.glb)
    write_json(output / 'room.json', dict(mapID=map_id, name=name,
        vertices=np.asarray(mesh.vertices).round(5).tolist(), triangles=mesh.faces.tolist()))
    write_json(output / 'summary.json', dict(mapID=map_id, name=name))
    write_json(output / 'wifi-empty.json', dict(mapID=map_id, network='Not surveyed', radius=1.5, samples=[]))
    write_json(output / 'manifest.json', dict(version=1, mapID=map_id, reference_count=len(db.ids),
        reference_ids=db.ids, coordinate_frame='GLB meters; row-major camera-to-world CV poses',
        method='Offline RootSIFT + TF-IDF + PnP; not the MeshLoc neural pipeline',
        files={p.name: sha256(p) for p in output.iterdir() if p.name != 'manifest.json'}))
    # Single-file, checksummed package. No archive paths can escape on import.
    payloads = [output / 'room.json', output / 'features.json']
    header = json.dumps(dict(version=1, mapID=map_id, name=name,
        files=[dict(name=p.name, size=p.stat().st_size, sha256=sha256(p)) for p in payloads])).encode()
    import struct, shutil
    with output.with_suffix('.wifiroom').open('wb') as f:
        f.write(b'WIFIROOM'); f.write(struct.pack('<I', len(header))); f.write(header)
        for p in payloads:
            with p.open('rb') as source: shutil.copyfileobj(source, f)
    return map_id


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True); p.add_argument('--database', required=True); p.add_argument('--output', required=True)
    p.add_argument('--name', default='Scanned room')
    a=p.parse_args(); print(export(a.data, a.database, a.output, a.name))

if __name__ == '__main__': main()

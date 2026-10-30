"""Desktop export -> the same native C++ solver shipped in the iPhone app."""
import hashlib
import json
import os
import re
from pathlib import Path
import struct
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from urllib.request import Request, urlopen
import threading

import cv2
import numpy as np
from PIL import Image
import trimesh

from wifi_localization.features import build_database, Database
from wifi_localization.mobile import export_mobile, MAGIC, MESH_MAGIC


class SyntheticCapture:
    """Four reference views and one unseen view of a textured metric plane."""
    def __init__(self, root):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_format = 'posed-images'; self.alignment = 'forward'; self.depth_scale = .001
        self.reference_ids = ['ref0', 'ref1', 'ref2', 'ref3']; self.query_ids = ['unseen_query']
        self.ids = self.reference_ids + self.query_ids
        self.split_description = 'Four reference camera positions and a distinct held-out camera position'
        self.profile = dict(width=640, height=480, fx=500., fy=500., cx=320., cy=240., distortion=[0.] * 5)
        self.K = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
        self.frames = {}; rng = np.random.default_rng(32)
        texture = rng.integers(25, 240, (480, 640, 3), dtype=np.uint8)
        for y in range(35, 460, 70):
            for x in range(35, 620, 70):
                cv2.circle(texture, (x, y), 14, (250, 250, 250), 2)
                cv2.putText(texture, str(x + y), (x - 15, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (10, 10, 10))
        centers = [np.array([x, 0., 0.]) for x in [-.12, -.04, .04, .12]] + [np.array([.07, .06, .04])]
        for id, center in zip(self.ids, centers):
            plane = np.eye(3); plane[:, 2] -= center / 4.
            transform = self.K @ plane @ np.linalg.inv(self.K)
            rgb = cv2.warpPerspective(texture, transform, (640, 480))
            image = self.root / f'{id}.png'; Image.fromarray(rgb).save(image)
            T = np.eye(4); T[:3, 3] = center
            self.frames[id] = SimpleNamespace(image=image, K=self.K.copy(), T_world_cv=T, width=640, height=480)
        self.glb = self.root / 'room.glb'
        mesh = trimesh.Trimesh([[-2.56, -1.92, 4], [2.56, -1.92, 4], [2.56, 1.92, 4], [-2.56, 1.92, 4]], [[0, 1, 2], [0, 2, 3]], process=False)
        mesh.visual.vertex_colors = [[100, 150, 180, 255], [150, 180, 100, 255], [180, 100, 150, 255], [160, 160, 160, 255]]
        mesh.export(self.glb)

    def world_points(self, frame, xy):
        center = frame.T_world_cv[:3, 3]
        rays = np.column_stack([xy, np.ones(len(xy))]) @ np.linalg.inv(self.K).T
        points = rays * (4 - center[2]) + center
        return points, np.ones(len(xy), bool)

    def fingerprint(self):
        return hashlib.sha256(b''.join(self.frames[id].image.read_bytes() for id in self.ids) + self.glb.read_bytes()).hexdigest()


class MobileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(2)
        cls.temp = tempfile.TemporaryDirectory(); cls.root = Path(cls.temp.name)
        cls.capture = SyntheticCapture(cls.root / 'capture')
        cls.db_path = cls.root / 'database'
        build_database(cls.capture, cls.db_path, query_every=None, nfeatures=800, words=32)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def export(self, folder):
        path = folder / 'room.roomloc'
        result = export_mobile(self.capture, self.db_path, path, 'Synthetic room')
        with path.open('rb') as stream:
            magic, mapsize, meshsize = struct.unpack('<8sQQ', stream.read(24))
            self.assertEqual(magic, MAGIC)
            self.assertEqual(path.stat().st_size, 24 + mapsize + meshsize)
            map_path = folder / 'map.yml'; map_path.write_bytes(stream.read(mapsize))
            mesh = stream.read(meshsize); self.assertEqual(mesh[:8], MESH_MAGIC)
            nv, nf = struct.unpack('<II', mesh[8:16]); self.assertEqual(len(mesh), 16 + nv * 40 + nf * 12)
        return result, map_path

    def test_reference_split_geometry_and_bounded_package(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); result, path = self.export(folder)
            self.assertEqual(result['references'], 4)
            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            try:
                refs = fs.getNode('references')
                self.assertEqual([refs.at(i).getNode('id').string() for i in range(refs.size())], self.capture.reference_ids)
                self.assertEqual(fs.getNode('vectors').mat().shape, (4, 32))
                first = refs.at(0).getNode('world_points').mat()
                np.testing.assert_allclose(first[:, 2], 4., atol=1e-6)
            finally:
                fs.release()
            self.assertNotIn('unseen_query', path.read_text())

    def test_invalid_features_do_not_replace_previous_export(self):
        with tempfile.TemporaryDirectory() as d:
            output = Path(d) / 'keep.roomloc'; output.write_bytes(b'keep existing package')
            db = Database(self.db_path, self.capture)
            db.features('ref0')['desc'][0, 0] = np.nan
            with self.assertRaises(ValueError): export_mobile(self.capture, db, output)
            self.assertEqual(output.read_bytes(), b'keep existing package')
            db = Database(self.db_path, self.capture); db.meta['query_ids'].append('ref0')
            with self.assertRaisesRegex(ValueError, 'Query frames'): export_mobile(self.capture, db, output)

    def test_browser_export_and_download_are_local_and_reference_only(self):
        from wifi_localization.app import LocalServer, LocalApp
        with tempfile.TemporaryDirectory() as d:
            app = LocalApp.__new__(LocalApp)
            from concurrent.futures import ThreadPoolExecutor
            app.capture = self.capture; app.db = Database(self.db_path, self.capture)
            app.output = Path(d); app.jobs = {}; app.lock = threading.RLock()
            app.state = {'ready': True}; app.worker = ThreadPoolExecutor(max_workers=1)
            server = LocalServer(('127.0.0.1', 0), app)
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(Request(base + '/api/mobile-export', data=b'{}', headers={'Content-Type': 'application/json'}), timeout=20) as response:
                    job = json.load(response)
                app.worker.submit(lambda: None).result(timeout=30)
                with urlopen(base + '/api/jobs/' + job['id'], timeout=20) as response: result = json.load(response)
                self.assertEqual(result['status'], 'done', result)
                with urlopen(base + result['result']['url'], timeout=20) as response:
                    self.assertEqual(response.read(8), MAGIC)
                self.assertNotIn('path', result['result'])
            finally:
                server.shutdown(); server.server_close(); app.close()

    @unittest.skipUnless(os.environ.get('ROOMLOC_NATIVE_CLI'), 'Set ROOMLOC_NATIVE_CLI to the compiled C++ test tool')
    def test_export_to_native_solver_on_unseen_camera_and_failure(self):
        binary = os.environ['ROOMLOC_NATIVE_CLI']
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); _, path = self.export(folder)
            metadata = subprocess.run([binary, '--inspect', str(path)], check=True, text=True, capture_output=True)
            self.assertEqual(json.loads(metadata.stdout)['references'], 4)
            frame = self.capture.frames['unseen_query']
            image = cv2.imread(str(frame.image))
            raw = folder / 'query.bgr'; raw.write_bytes(struct.pack('<II', 640, 480) + image.tobytes())
            request = folder / 'camera.json'
            request.write_text(json.dumps(dict(mode='saved', original_width=640, original_height=480, calibration=self.capture.profile)))
            run = subprocess.run([binary, str(path), str(raw), str(request)], check=True, capture_output=True, text=True)
            result = json.loads(run.stdout); self.assertTrue(result['success'], result)
            self.assertFalse(result['camera_estimated'])
            position = result['pose']['position_m']
            self.assertLess(np.linalg.norm(position - frame.T_world_cv[:3, 3]), .04)
            rotation = np.array(result['pose']['T_world_camera_cv'])[:3, :3]
            self.assertLess(np.linalg.norm(rotation - np.eye(3)), .04)
            # The result agrees with the existing Python geometric pipeline.
            from wifi_localization.localization import localize
            rgb = np.asarray(Image.open(frame.image).convert('RGB'))
            baseline = localize(self.capture, Database(self.db_path, self.capture), rgb, self.capture.K)
            self.assertTrue(baseline['success'], baseline.get('reason'))
            self.assertLess(np.linalg.norm(np.asarray(position) - baseline['pose']['position_m']), .04)
            # Byte-identical reference photos exclude themselves before retrieval.
            reference = self.capture.frames['ref0']
            raw.write_bytes(struct.pack('<II', 640, 480) + cv2.imread(str(reference.image)).tobytes())
            request.write_text(json.dumps(dict(mode='saved', original_width=640, original_height=480,
                calibration=self.capture.profile, source_sha256=hashlib.sha256(reference.image.read_bytes()).hexdigest())))
            excluded = json.loads(subprocess.run([binary, str(path), str(raw), str(request)], check=True, capture_output=True, text=True).stdout)
            self.assertEqual(excluded['excluded_reference_count'], 1)
            request.write_text(json.dumps(dict(mode='saved', original_width=640, original_height=480, calibration=self.capture.profile)))
            raw.write_bytes(struct.pack('<II', 640, 480) + np.full_like(image, 127).tobytes())
            failed = json.loads(subprocess.run([binary, str(path), str(raw), str(request)], check=True, capture_output=True, text=True).stdout)
            self.assertFalse(failed['success']); self.assertNotIn('pose', failed)
            request.write_text(json.dumps(dict(mode='dataset', original_width=640, original_height=480)))
            invalid = subprocess.run([binary, str(path), str(raw), str(request)], capture_output=True, text=True)
            self.assertNotEqual(invalid.returncode, 0)
            # Corrupt matrix dimensions are rejected before a huge allocation.
            corrupt = folder / 'corrupt.yml'
            text, count = re.subn(r'(centers:.*?rows:\s*)\d+', r'\g<1>99999999', path.read_text(), count=1, flags=re.S)
            self.assertEqual(count, 1); corrupt.write_text(text)
            badmap = subprocess.run([binary, '--inspect', str(corrupt)], capture_output=True, text=True)
            self.assertNotEqual(badmap.returncode, 0)
            # The separate dataset mode uses the package camera and rejects a
            # phone-sized image instead of silently applying dataset intrinsics.
            original_format = self.capture.dataset_format
            try:
                self.capture.dataset_format = '7-scenes'
                _, dataset_path = self.export(folder)
            finally:
                self.capture.dataset_format = original_format
            raw.write_bytes(struct.pack('<II', 640, 480) + image.tobytes())
            request.write_text(json.dumps(dict(mode='dataset', original_width=640, original_height=480)))
            dataset_result = json.loads(subprocess.run([binary, str(dataset_path), str(raw), str(request)], check=True, capture_output=True, text=True).stdout)
            self.assertTrue(dataset_result['success'], dataset_result)
            self.assertTrue(dataset_result['camera_estimated'])
            request.write_text(json.dumps(dict(mode='dataset', original_width=1280, original_height=960)))
            wrong_size = subprocess.run([binary, str(dataset_path), str(raw), str(request)], capture_output=True, text=True)
            self.assertNotEqual(wrong_size.returncode, 0)


if __name__ == '__main__': unittest.main()

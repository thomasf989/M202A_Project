"""Exercise the same portable room builder linked into the iPhone app."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from test_mobile import SyntheticCapture


@unittest.skipUnless(os.environ.get('ROOMLOC_NATIVE_CLI'), 'Set ROOMLOC_NATIVE_CLI to the compiled C++ tool')
class PhoneBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.capture = SyntheticCapture(cls.root / 'source')
        cls.binary = os.environ['ROOMLOC_NATIVE_CLI']

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def manifest(self, folder, convention='opencv'):
        cap = self.capture
        flip = np.diag([1., -1., -1., 1.])
        frames = []
        for id in cap.ids:
            f = cap.frames[id]
            frames.append(dict(id=id, image=id + '.png', split='test' if id in cap.query_ids else 'reference',
                               camera_to_world=(f.T_world_cv @ (flip if convention == 'opengl' else np.eye(4))).tolist()))
            # The held-out image and pose don't have to be present to build.
            if id in cap.reference_ids: shutil.copy2(f.image, folder / (id + '.png'))
        meta = dict(format='posed-images-v1', camera_convention=convention, intrinsics=dict(cap.profile), frames=frames)
        (folder / 'references.json').write_text(json.dumps(meta))
        return meta

    def build(self, source, output, maximum=120, cancel=False, glb=None, timeout=90):
        return subprocess.run([self.binary, '--cancel-build' if cancel else '--build-room',
                               str(glb or self.capture.glb), str(source), str(output), 'Phone room', str(maximum)],
                              capture_output=True, text=True, timeout=timeout)

    def raw_frames(self, folder, ids=None):
        folder.mkdir(parents=True, exist_ok=True)
        flip = np.diag([1., -1., -1., 1.])
        for i, id in enumerate(ids or self.capture.reference_ids):
            stem = f'frame-{i:06d}'
            shutil.copy2(self.capture.frames[id].image, folder / (stem + '.color.png'))
            np.savetxt(folder / (stem + '.pose.txt'), flip @ self.capture.frames[id].T_world_cv)

    def inspect(self, path):
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            refs = storage.getNode('references')
            return [(refs.at(i).getNode('id').string(), refs.at(i).getNode('image_sha256').string(),
                     refs.at(i).getNode('world_points').mat(), refs.at(i).getNode('valid').mat()) for i in range(refs.size())]
        finally:
            storage.release()

    def test_phone_build_recovers_held_out_pose_and_keeps_reference_hashes(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; source.mkdir(); output = Path(d) / 'room'
            self.manifest(source)
            result = self.build(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['references'], 4)
            refs = self.inspect(output / 'map.yml')
            self.assertEqual([ref[0] for ref in refs], self.capture.reference_ids)
            for id, hash_, world, valid in refs:
                self.assertEqual(hash_, hashlib.sha256((source / (id + '.png')).read_bytes()).hexdigest())
                np.testing.assert_allclose(world[valid.ravel().astype(bool), 2], 4, atol=1e-5)
            frame = self.capture.frames['unseen_query']
            raw = Path(d) / 'query.bgr'
            raw.write_bytes(struct.pack('<II', 640, 480) + cv2.imread(str(frame.image)).tobytes())
            camera = Path(d) / 'camera.json'
            camera.write_text(json.dumps(dict(mode='saved', original_width=640, original_height=480, calibration=self.capture.profile)))
            localized = subprocess.run([self.binary, str(output / 'map.yml'), str(raw), str(camera)], capture_output=True, text=True, check=True)
            pose = json.loads(localized.stdout)
            self.assertTrue(pose['success'], pose)
            self.assertLess(np.linalg.norm(pose['pose']['position_m'] - frame.T_world_cv[:3, 3]), .04)
            self.assertLess(np.linalg.norm(np.array(pose['pose']['T_world_camera_cv'])[:3, :3] - np.eye(3)), .04)
            # Display geometry keeps the same metric coordinates and colors.
            data = (output / 'mesh.bin').read_bytes()
            magic, nv, nf = struct.unpack('<8sII', data[:16])
            self.assertEqual(magic, b'RMSH0001')
            self.assertEqual(len(data), 16 + nv * 40 + nf * 12)
            packed = np.frombuffer(data[16:16 + nv * 40], dtype='<f4').reshape(nv, 10)
            np.testing.assert_allclose(packed[:, 2], 4, atol=1e-6)
            self.assertGreater(float(packed[:, 6:9].std()), .01)

    def test_opengl_camera_convention_matches_opencv(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; source.mkdir(); output = Path(d) / 'room'
            self.manifest(source, 'opengl')
            result = self.build(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            for _, _, world, valid in self.inspect(output / 'map.yml'):
                np.testing.assert_allclose(world[valid.ravel().astype(bool), 2], 4, atol=1e-5)

    def test_prepared_seven_scenes_ignores_missing_test_images_and_raw_depth(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; source.mkdir(); output = Path(d) / 'room'
            frames = []
            flip = np.diag([1., -1., -1., 1.])
            for i, id in enumerate(self.capture.reference_ids):
                seq, stem = 'seq-01', f'frame-{i:06d}'
                (source / seq).mkdir(exist_ok=True)
                shutil.copy2(self.capture.frames[id].image, source / seq / (stem + '.color.png'))
                np.savetxt(source / seq / (stem + '.pose.txt'), flip @ self.capture.frames[id].T_world_cv)
                frames.append(dict(id=seq + '_' + stem, sequence=seq, stem=stem, split='train'))
            frames.append(dict(id='seq-03_frame-000000', sequence='seq-03', stem='frame-000000', split='test'))
            meta = dict(format='7-scenes-v1', depth_source='mesh_rendered_rgb', world_to_glb=flip.tolist(),
                        rgb_intrinsics=self.capture.profile, train_sequences=['seq-01'], test_sequences=['seq-03'], frames=frames)
            (source / 'seven_scenes.json').write_text(json.dumps(meta))
            result = self.build(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['dataset_format'], '7-scenes')
            self.assertTrue(all(row[0].startswith('seq-01_') for row in self.inspect(output / 'map.yml')))
            meta['train_sequences'].append('seq-03')
            (source / 'seven_scenes.json').write_text(json.dumps(meta))
            bad = self.build(source, Path(d) / 'overlap')
            self.assertNotEqual(bad.returncode, 0)
            self.assertFalse((Path(d) / 'overlap/map.yml').exists())

    def test_original_seven_scenes_builds_without_test_sequence(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; (source / 'seq-01').mkdir(parents=True); output = Path(d) / 'room'
            (source / 'TrainSplit.txt').write_text('sequence1\n')
            (source / 'TestSplit.txt').write_text('sequence3\n')
            flip = np.diag([1., -1., -1., 1.])
            for i, id in enumerate(self.capture.reference_ids):
                stem = f'frame-{i:06d}'
                shutil.copy2(self.capture.frames[id].image, source / 'seq-01' / (stem + '.color.png'))
                np.savetxt(source / 'seq-01' / (stem + '.pose.txt'), flip @ self.capture.frames[id].T_world_cv)
            result = self.build(source, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['references'], 4)
            self.assertTrue(all('seq-03' not in row[0] for row in self.inspect(output / 'map.yml')))

    def test_sampling_is_bounded_and_uniform(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; source.mkdir(); output = Path(d) / 'room'
            meta = self.manifest(source)
            refs = meta['frames'][:4]
            meta['frames'] = [dict(refs[i % 4], id=f'view-{i}') for i in range(9)]
            (source / 'references.json').write_text(json.dumps(meta))
            result = self.build(source, output, maximum=4)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual([r[0] for r in self.inspect(output / 'map.yml')], ['view-0', 'view-2', 'view-4', 'view-6'])

    def test_raw_flat_sequence_builds_without_any_split_file(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; self.raw_frames(source)
            result = self.build(source, Path(d) / 'room')
            self.assertEqual(result.returncode, 0, result.stderr)
            refs = self.inspect(Path(d) / 'room/map.yml')
            self.assertEqual(len(refs), 4)
            for i, (_, hash_, world, valid) in enumerate(refs):
                image = source / f'frame-{i:06d}.color.png'
                self.assertEqual(hash_, hashlib.sha256(image.read_bytes()).hexdigest())
                np.testing.assert_allclose(world[valid.ravel().astype(bool), 2], 4, atol=1e-5)

    def test_no_train_split_uses_all_sequences_even_with_test_split(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'
            self.raw_frames(source / 'Heads/seq-01', self.capture.reference_ids[:2])
            self.raw_frames(source / 'Heads/seq-03', self.capture.reference_ids[2:])
            # With no TrainSplit, this old test label must not exclude uploaded RGBs.
            (source / 'Heads/TestSplit.txt').write_text('sequence3\n')
            result = self.build(source, Path(d) / 'room')
            self.assertEqual(result.returncode, 0, result.stderr)
            refs = self.inspect(Path(d) / 'room/map.yml')
            self.assertEqual(len({row[0] for row in refs}), 4)
            expected = {hashlib.sha256(self.capture.frames[id].image.read_bytes()).hexdigest()
                        for id in self.capture.reference_ids}
            self.assertEqual({row[1] for row in refs}, expected)

    def test_no_split_still_requires_matching_poses(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; self.raw_frames(source)
            (source / 'frame-000002.pose.txt').unlink()
            result = self.build(source, Path(d) / 'room')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('frame-000002.pose.txt', result.stderr)
            self.assertFalse((Path(d) / 'room/map.yml').exists())

    def test_explicit_train_split_does_not_require_test_split(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; self.raw_frames(source / 'seq-01')
            self.raw_frames(source / 'seq-03')
            (source / 'TrainSplit.txt').write_text('sequence1\n')
            result = self.build(source, Path(d) / 'room')
            self.assertEqual(result.returncode, 0, result.stderr)
            refs = self.inspect(Path(d) / 'room/map.yml')
            self.assertEqual(len(refs), 4)
            self.assertTrue(all(row[0].startswith('seq-01_') for row in refs))

    def test_thousand_reference_map_builds_and_reopens_with_bounded_features(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; source.mkdir(); output = Path(d) / 'room'
            meta = self.manifest(source)
            # Reuse four textured fixtures to stress the 1,000-row index and
            # serialized feature budget, without storing 1,000 source images.
            refs = meta['frames'][:4]
            meta['frames'] = [dict(refs[i % 4], id=f'view-{i}') for i in range(1000)]
            (source / 'references.json').write_text(json.dumps(meta))
            rejected = self.build(source, Path(d) / 'too-many', maximum=1001)
            self.assertNotEqual(rejected.returncode, 0)
            result = self.build(source, output, maximum=1000, timeout=300)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['references'], 1000)
            self.assertLessEqual((output / 'map.yml').stat().st_size, 256 * 1024 * 1024)
            storage = cv2.FileStorage(str(output / 'map.yml'), cv2.FILE_STORAGE_READ)
            try:
                self.assertEqual(int(storage.getNode('reference_feature_limit').real()), 300)
                refs = storage.getNode('references')
                sizes = [refs.at(i).getNode('descriptors').mat().shape[0] for i in range(refs.size())]
                self.assertEqual(refs.size(), 1000)
                self.assertTrue(all(12 <= n <= 300 for n in sizes))
                self.assertLessEqual(sum(sizes), 300000)
            finally:
                storage.release()
            reopened = subprocess.run([self.binary, '--inspect', str(output / 'map.yml')],
                                      capture_output=True, text=True, timeout=90)
            self.assertEqual(reopened.returncode, 0, reopened.stderr)
            self.assertEqual(json.loads(reopened.stdout)['references'], 1000)

    def test_cancellation_and_invalid_input_preserve_existing_rooms(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'input'; source.mkdir(); self.manifest(source)
            old = root / 'old'; old.mkdir(); (old / 'map.yml').write_text('keep existing map')
            for cancel in [False, True]:
                output = old if not cancel else root / 'cancelled'
                result = self.build(source, output, cancel=cancel)
                self.assertNotEqual(result.returncode, 0)
            self.assertEqual((old / 'map.yml').read_text(), 'keep existing map')
            self.assertFalse(list((root / 'cancelled').iterdir()))
            meta = json.loads((source / 'references.json').read_text())
            meta['frames'][0]['image'] = '../outside.png'
            (source / 'references.json').write_text(json.dumps(meta))
            result = self.build(source, root / 'bad')
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'bad/map.yml').exists())

    def test_glb_node_transform_is_applied_and_malformed_buffer_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'input'; source.mkdir(); self.manifest(source)
            data = self.capture.glb.read_bytes()
            size, kind = struct.unpack('<II', data[12:20]); self.assertEqual(kind, 0x4e4f534a)
            gltf = json.loads(data[20:20 + size]); binary = data[20 + size:]
            # Move the scene away from all references, so no depth support exists.
            gltf['nodes'][0]['translation'] = [0, 0, 30]
            def save(value):
                blob = json.dumps(value).encode(); blob += b' ' * (-len(blob) % 4)
                path = root / 'modified.glb'
                path.write_bytes(struct.pack('<4sII', b'glTF', 2, 20 + len(blob) + len(binary)) + struct.pack('<II', len(blob), kind) + blob + binary)
                return path
            result = self.build(source, root / 'moved', glb=save(gltf))
            self.assertNotEqual(result.returncode, 0)
            gltf['buffers'][0]['byteLength'] = 2 ** 32 - 1
            result = self.build(source, root / 'bad-buffer', glb=save(gltf))
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / 'bad-buffer/map.yml').exists())

    def test_polycam_corrected_cameras_and_literal_room_name(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; source.mkdir(); output = Path(d) / 'room'
            images = source / 'keyframes/corrected_images'; images.mkdir(parents=True)
            cameras = source / 'keyframes/corrected_cameras'; cameras.mkdir(parents=True)
            flip = np.diag([1., -1., -1., 1.]); alignment = np.eye(4); alignment[0, 3] = .3
            (source / 'optimized_roomplan.json').write_text(json.dumps(dict(alignment_transform={f't{i}{j}': alignment[i, j] for i in range(4) for j in range(4)})))
            for i, id in enumerate(self.capture.reference_ids):
                f = self.capture.frames[id]
                Image.open(f.image).save(images / f'{i}.jpg', quality=97)
                camera = dict(self.capture.profile)
                pose = np.linalg.inv(alignment) @ f.T_world_cv @ flip
                camera.update({f't_{i}{j}': pose[i, j] for i in range(3) for j in range(4)})
                (cameras / f'{i}.json').write_text(json.dumps(camera))
            name = '{手机房间}'
            run = subprocess.run([self.binary, '--build-room', str(self.capture.glb), str(source), str(output), name, '120'], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            metadata = json.loads(run.stdout)
            self.assertEqual(metadata['name'], name)
            self.assertEqual(metadata['dataset_format'], 'polycam-mesh')
            for _, _, world, valid in self.inspect(output / 'map.yml'):
                np.testing.assert_allclose(world[valid.ravel().astype(bool), 2], 4, atol=1e-5)

    def test_rotated_reference_and_invalid_intrinsics_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d) / 'input'; source.mkdir(); meta = self.manifest(source)
            image = Image.open(source / 'ref0.png')
            exif = Image.Exif(); exif[274] = 6
            image.save(source / 'rotated.jpg', exif=exif)
            meta['frames'][0]['image'] = 'rotated.jpg'
            (source / 'references.json').write_text(json.dumps(meta))
            result = self.build(source, Path(d) / 'rotated')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('EXIF', result.stderr)
            meta['frames'][0]['image'] = 'ref0.png'; meta['intrinsics']['fx'] = 1e308
            (source / 'references.json').write_text(json.dumps(meta))
            result = self.build(source, Path(d) / 'bad-camera')
            self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()

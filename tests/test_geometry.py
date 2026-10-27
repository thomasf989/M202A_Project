"""Numerical contracts, including recovery from deliberately wrong matches."""
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from wifi_localization.polycam import Frame, Capture, CV_TO_POLYCAM, rigid_matrix, write_json, query_calibration
from wifi_localization.localization import solve_pose, pose_errors, Settings
from wifi_localization.cli import aggregate


class GeometryTests(unittest.TestCase):
    def test_pose_with_30_percent_outliers(self):
        rng = np.random.default_rng(31)
        K = np.array([[700., 0, 512], [0, 700, 384], [0, 0, 1]])
        pc = rng.uniform([-2., -1.5, 3.], [2., 1.5, 6.], (200, 3))
        truth = np.eye(4)
        truth[:3, :3] = Rotation.from_euler('xyz', [20, -40, 10], degrees=True).as_matrix()
        truth[:3, 3] = [1.2, .7, -2.]
        xyz = pc @ truth[:3, :3].T + truth[:3, 3]
        xy = pc[:, :2]/pc[:, 2:] * [700, 700] + [512, 384]
        xy += rng.normal(0, .3, xy.shape)
        xy[:60] = rng.uniform([0, 0], [1024, 768], (60, 2))
        estimated, stats = solve_pose(xyz, xy, np.arange(200), np.zeros(200, int), K, (1024, 768), Settings())
        self.assertIsNotNone(estimated, stats)
        p, r = pose_errors(estimated, truth)
        self.assertLess(p, .02)
        self.assertLess(r, .3)
        self.assertGreater(stats['inliers'], 130)

    def test_cv_axis_and_metric_depth(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            depth = root/'depth.png';confidence = root/'confidence.png'
            Image.fromarray(np.full((8, 8), 2000, np.uint16)).save(depth)
            Image.fromarray(np.full((8, 8), 255, np.uint8)).save(confidence)
            f = Frame('1', root/'image.jpg', root/'camera.json', depth, confidence,
                np.array([[10., 0, 4], [0, 10, 4], [0, 0, 1]]), CV_TO_POLYCAM, 8, 8)
            capture = Capture.__new__(Capture);capture.depth_scale=.001
            xyz, valid = capture.world_points(f, [[4., 4.], [5., 5.]])
            np.testing.assert_allclose(xyz, [[0, 0, -2], [.2, -.2, -2]], atol=1e-8)
            self.assertTrue(valid.all())
            Image.fromarray(np.zeros((8, 8), np.uint8)).save(confidence)
            _, valid = capture.world_points(f, [[4., 4.]])
            self.assertFalse(valid[0])

    def test_failed_queries_stay_in_denominator(self):
        results = [dict(success=True, position_error_m=.1, rotation_error_deg=2., elapsed_seconds=1.),
                   dict(success=False, elapsed_seconds=1.)]
        s = aggregate(results)
        threshold=next(t for t in s['accuracy_thresholds_all_queries'] if t['position_m']==.25)
        self.assertEqual(threshold['rate'], .5)

    def test_bad_transform_and_query_intrinsics_rejected(self):
        with self.assertRaises(ValueError):
            rigid_matrix(np.diag([1., 1., -1., 1.]), 'reflection')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'intrinsics.json'
            write_json(p, dict(fx=100, fy=100, cx=50, cy=50, width=100, height=100))
            with self.assertRaises(ValueError):
                query_calibration(p, (200, 100))


if __name__ == '__main__':
    unittest.main()

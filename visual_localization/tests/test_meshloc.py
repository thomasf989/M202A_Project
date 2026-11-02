import numpy as np
from types import SimpleNamespace

from wifi_localization.meshloc import cluster_query_positions, covisibility_components, position_average, MeshLocSettings


def test_query_position_clustering_groups_nearby_points():
    xy = np.array([[10., 10.], [11., 11.], [100., 100.], [102., 101.]], np.float32)
    ids = cluster_query_positions(xy, radius_px=4.)
    assert ids[0] == ids[1]
    assert ids[2] == ids[3]
    assert ids[0] != ids[2]


def test_covisibility_components_from_shared_query_features():
    qids = np.array([0, 0, 1, 2, 2, 3])
    refs = np.array(['a', 'b', 'b', 'c', 'd', 'e'], object)
    comps = [set(c) for c in covisibility_components(qids, refs)]
    assert {'a', 'b'} in comps
    assert {'c', 'd'} in comps
    assert {'e'} in comps


def test_position_averaging_preserves_rotation_and_can_shift_center():
    K = np.array([[500., 0., 320.], [0., 500., 240.], [0., 0., 1.]])
    truth = np.eye(4)
    truth[:3, 3] = [0., 0., 0.]
    xyz = np.array([[-1., -1., 5.], [1., -1., 5.], [-1., 1., 5.], [1., 1., 5.], [0., 0., 4.]])
    xy = np.column_stack([500 * xyz[:, 0] / xyz[:, 2] + 320, 500 * xyz[:, 1] / xyz[:, 2] + 240])
    initial = truth.copy(); initial[:3, 3] = [0.10, 0., 0.]
    s = MeshLocSettings(position_volume_half_m=.15, position_step_m=.05, reprojection_px=8.)
    refined, stats = position_average(initial, xyz, xy, K, s)
    assert np.allclose(refined[:3, :3], initial[:3, :3])
    assert abs(refined[0, 3]) < abs(initial[0, 3])
    assert stats['position_averaging'] is True


def test_p3p_fallback_recovers_synthetic_pose():
    from wifi_localization.meshloc import estimate_pose_meshloc
    K = np.array([[700., 0., 320.], [0., 700., 240.], [0., 0., 1.]])
    # Non-coplanar points in front of identity camera.
    xyz = np.array([
        [-1.0, -0.5, 4.0], [1.0, -0.5, 4.5], [-0.8, 0.8, 5.0], [0.9, 0.7, 5.5],
        [0.0, 0.0, 3.5], [-1.2, 0.3, 6.0], [1.3, 0.2, 6.2], [0.2, -1.0, 5.8],
        [-0.3, 1.2, 4.8], [0.5, -0.2, 7.0], [1.0, 1.0, 7.5], [-1.0, 1.0, 6.8],
    ])
    xy = np.column_stack([700 * xyz[:, 0] / xyz[:, 2] + 320, 700 * xyz[:, 1] / xyz[:, 2] + 240])
    s = MeshLocSettings(min_inliers=6, min_inlier_ratio=.3, reprojection_px=2., max_ransac_iterations=2000,
                        min_ransac_iterations=100)
    T, stats = estimate_pose_meshloc(xy, xyz, K, (640, 480), s)
    assert T is not None
    assert np.linalg.norm(T[:3, 3]) < 0.05
    assert stats['inliers'] >= 10


def test_pose_diverse_ranking_avoids_near_duplicate_top_refs():
    from wifi_localization.meshloc import pose_diverse_ranking

    def frame(center, yaw_deg):
        yaw = np.deg2rad(yaw_deg)
        T = np.eye(4)
        T[:3, 3] = np.asarray(center, float)
        # CV camera +Z is forward.
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        right = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        down = np.cross(forward, right)
        T[:3, :3] = np.column_stack([right, down, forward])
        return SimpleNamespace(T_world_cv=T)

    capture = SimpleNamespace(frames={
        'a': frame([0.0, 0.0, 1.5], 0),
        'b': frame([0.05, 0.0, 1.5], 3),   # near duplicate of a
        'c': frame([1.0, 0.0, 1.5], 0),    # distinct position
        'd': frame([0.0, 0.0, 1.5], 90),   # distinct orientation
    })
    ranking = [('a', 1.0), ('b', .99), ('c', .98), ('d', .97)]
    out = pose_diverse_ranking(capture, ranking, top_k=3, pool=4,
                               translation_m=.30, rotation_deg=15.)
    ids = [rid for rid, _ in out]
    assert ids[0] == 'a'
    assert 'c' in ids and 'd' in ids
    assert 'b' not in ids


def test_meshloc_only_mode_never_silently_falls_back(monkeypatch, tmp_path):
    from wifi_localization.pipeline import LocalizationPipeline
    db = SimpleNamespace(ids=[], meta={"reference_image_hashes": {}, "nfeatures": 100})
    capture = SimpleNamespace()
    pipe = LocalizationPipeline(capture, db, tmp_path)

    def fail_meshloc(*args, **kwargs):
        return {"success": False, "reason": "expected_meshloc_failure", "method": "MeshLoc"}

    def forbidden_legacy():
        raise AssertionError("legacy fallback must not run in MeshLoc-only mode")

    monkeypatch.setattr(pipe, "_run_meshloc", fail_meshloc)
    monkeypatch.setattr(pipe, "_legacy_models", forbidden_legacy)
    K = np.array([[500., 0., 32.], [0., 500., 32.], [0., 0., 1.]])
    result = pipe.localize(np.zeros((64, 64, 3), np.uint8), [K], solver_mode="meshloc")
    assert result["success"] is False
    assert result["reason"] == "expected_meshloc_failure"
    assert "fallback" not in result.get("method", "").lower()


def test_paper_settings_use_meshloc_release_thresholds(monkeypatch, tmp_path):
    from wifi_localization.pipeline import LocalizationPipeline
    db = SimpleNamespace(ids=[], meta={"reference_image_hashes": {}})
    capture = SimpleNamespace()
    pipe = LocalizationPipeline(capture, db, tmp_path)
    captured = {}

    class Dummy:
        def localize(self, *args, **kwargs):
            captured["settings"] = args[-1]
            return {"success": False, "reason": "dummy"}

    monkeypatch.setattr(pipe, "_meshloc_model", lambda: Dummy())
    base = SimpleNamespace(min_inliers=10, min_inlier_ratio=.08, seed=42)
    K = np.eye(3)
    result = pipe._run_meshloc(np.zeros((8, 8, 3), np.uint8), [K], (), None, base, paper=True)
    settings = captured["settings"]
    assert settings.top_k == 20
    assert settings.reprojection_px == 20.0
    assert settings.min_ransac_iterations == 10_000
    assert settings.max_ransac_iterations == 100_000
    assert settings.retrieval_mode == "netvlad"
    assert result["solver_mode"] == "meshloc-paper"

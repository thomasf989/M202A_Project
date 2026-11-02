"""Reusable localization pipeline with explicit solver modes.

Solver modes
------------
meshloc
    Recommended for the MultiSet/Blender synthetic database.  Keeps the
    MeshLoc geometric pipeline but mitigates the real-vs-synthetic retrieval
    gap with scale-free fusion of NetVLAD and multi-crop DINOv2, followed by
    pose-diverse suppression.

meshloc-paper
    Paper-style ablation baseline: NetVLAD-only retrieval, LoFTR, mesh-depth
    lifting, PoseLib P3P/LO-RANSAC, 20 px threshold, 10k-100k iterations, and
    +REF position refinement.  MeshLoc's official 12-Scenes example used a
    DenseVLAD top-20 retrieval file; this project uses NetVLAD because our
    synthetic reference database has no supplied DenseVLAD vocabulary/file.

legacy
    Previous DINOv2/ALIKED-LightGlue/RootSIFT pipeline only.

auto
    Robust MeshLoc first, then the legacy learned and CPU fallbacks.  This is
    convenient for demos but should not be used when reporting MeshLoc-only
    experimental success rates.
"""
from pathlib import Path
from .features import extract
from .localization import localize as cpu_localize, Settings
from .retrieval import expand_pose_ranking


VALID_SOLVER_MODES = {"meshloc", "meshloc-paper", "legacy", "auto"}


class LocalizationPipeline:
    def __init__(self, capture, db, cache_root, progress=None):
        self.capture = capture
        self.db = db
        self.cache_root = Path(cache_root)
        self.progress = progress or (lambda message, fraction: None)
        self.meshloc = None
        self.learned = None
        self.retriever = None

    def _meshloc_model(self):
        if self.meshloc is None:
            from .meshloc import MeshLocLocalizer
            self.meshloc = MeshLocLocalizer(
                self.capture, self.db.ids, self.cache_root / "meshloc", self.progress)
        else:
            self.meshloc.progress = self.progress
            self.meshloc.retriever.progress = self.progress
            if self.meshloc.dino_retriever is not None:
                self.meshloc.dino_retriever.progress = self.progress
        return self.meshloc

    def _legacy_models(self):
        if self.learned is None:
            from .learned import LearnedLocalizer
            self.learned = LearnedLocalizer(
                self.capture, self.db.ids, self.cache_root / "local", self.progress)
        else:
            self.learned.progress = self.progress

    def _run_meshloc(self, rgb, candidate_intrinsics, exclude, query_hash, settings, paper=False):
        from .meshloc import MeshLocSettings
        if paper:
            # Closest practical reproduction of the official 12-Scenes command
            # given this project's synthetic database. The official release uses
            # DenseVLAD top-20 retrieval; here NetVLAD is the available global
            # descriptor, while the rest of the pose stack follows that setup.
            ms = MeshLocSettings(
                top_k=20,
                retrieval_mode="netvlad",
                retrieval_pool=20,
                diversify_dense_references=False,
                reprojection_px=20.0,
                min_inliers=max(8, int(settings.min_inliers)),
                min_inlier_ratio=min(float(settings.min_inlier_ratio), 0.06),
                min_ransac_iterations=10_000,
                max_ransac_iterations=100_000,
                seed=int(settings.seed),
                use_merge=False,
                use_covisibility=False,
                position_averaging=True,
            )
        else:
            ms = MeshLocSettings(
                top_k=50,
                retrieval_mode="hybrid",
                retrieval_pool=250,
                diversify_dense_references=True,
                diversity_translation_m=0.30,
                diversity_rotation_deg=15.0,
                reprojection_px=20.0,
                min_inliers=max(8, int(settings.min_inliers)),
                min_inlier_ratio=min(float(settings.min_inlier_ratio), 0.06),
                min_ransac_iterations=10_000,
                max_ransac_iterations=100_000,
                seed=int(settings.seed),
                use_merge=False,
                use_covisibility=False,
                position_averaging=True,
            )
        result = self._meshloc_model().localize(
            rgb, candidate_intrinsics, exclude, query_hash,
            self.db.meta.get("reference_image_hashes", {}), ms)
        result["solver_mode"] = "meshloc-paper" if paper else "meshloc"
        if paper:
            result["paper_baseline_note"] = (
                "MeshLoc 12-Scenes used DenseVLAD top-20 retrieval. This project uses "
                "NetVLAD top-20 because the synthetic database does not include the "
                "paper's precomputed DenseVLAD retrieval file/vocabulary.")
        return result

    def localize(self, rgb, candidate_intrinsics, exclude=(), query_hash=None,
                 settings=None, enhanced=True, solver_mode="auto"):
        s = settings or Settings()
        solver_mode = str(solver_mode or "auto").lower()
        if solver_mode not in VALID_SOLVER_MODES:
            raise ValueError(f"Unknown localization solver mode: {solver_mode}")

        result = None
        warnings = []

        # MeshLoc-only modes never silently fall through to a different solver.
        if solver_mode in {"meshloc", "meshloc-paper", "auto"} and enhanced:
            try:
                result = self._run_meshloc(
                    rgb, candidate_intrinsics, exclude, query_hash, s,
                    paper=(solver_mode == "meshloc-paper"))
            except Exception as e:
                warnings.append(f"MeshLoc pipeline: {e}")
                result = {
                    "success": False,
                    "reason": "meshloc_runtime_error",
                    "method": "MeshLoc",
                    "solver_mode": solver_mode,
                    "runtime_error": str(e),
                }
            if solver_mode in {"meshloc", "meshloc-paper"}:
                if warnings:
                    result["enhancement_warnings"] = warnings
                return result

        # Legacy learned path. In `legacy` it is the requested primary solver;
        # in `auto` it is explicitly labeled as a fallback.
        expanded = None
        global_rank = None
        if solver_mode in {"legacy", "auto"} and (result is None or not result.get("success")):
            previous = result
            try:
                self._legacy_models(); method = None
                try:
                    if self.retriever is None:
                        from .retrieval import DinoV2Retriever
                        self.retriever = DinoV2Retriever(
                            self.capture, self.db.ids, self.cache_root / "global", progress=self.progress)
                    else:
                        self.retriever.progress = self.progress
                    global_rank = self.retriever.rank(rgb, exclude)
                    expanded = expand_pose_ranking(
                        self.capture, global_rank, max_candidates=40, seed_count=10,
                        neighbors_per_seed=6, neighbor_radius_m=1.75)
                    method = "Multi-crop DINOv2 + pose-neighbour expansion"
                except Exception as e:
                    warnings.append(f"DINOv2 legacy retrieval: {e}")
                    _, desc = extract(rgb, self.db.meta["nfeatures"])
                    global_rank = self.db.rank(desc, exclude)
                    expanded = expand_pose_ranking(
                        self.capture, global_rank, max_candidates=32, seed_count=8,
                        neighbors_per_seed=5, neighbor_radius_m=1.5)
                    method = "RootSIFT TF-IDF + pose-neighbour expansion"
                legacy = self.learned.localize(
                    rgb, candidate_intrinsics, exclude, query_hash,
                    self.db.meta["reference_image_hashes"], candidate_ranking=expanded,
                    top_k=min(40, len(expanded) if expanded else 40),
                    retrieval_method=method, settings=s)
                prefix = "Legacy solver: " if solver_mode == "legacy" else "Legacy fallback: "
                legacy["method"] = prefix + legacy.get("method", "learned localizer")
                legacy["solver_mode"] = solver_mode
                if previous is not None:
                    legacy["meshloc_attempt"] = {k: previous.get(k) for k in
                        ["method", "reason", "correspondences", "inliers",
                         "retrieval_method", "retrieval", "runtime_error"]}
                if legacy.get("success") or result is None or solver_mode == "legacy":
                    result = legacy
            except Exception as e:
                warnings.append(f"Legacy learned solver: {e}")

        # CPU RootSIFT fallback is used in auto/legacy only, never MeshLoc-only.
        if solver_mode in {"legacy", "auto"} and (result is None or not result.get("success")):
            previous = result
            base = cpu_localize(
                self.capture, self.db, rgb, candidate_intrinsics[0], s, exclude, query_hash,
                candidate_intrinsics=candidate_intrinsics)
            base["method"] = ("Legacy solver: " if solver_mode == "legacy" else "Last-resort fallback: ") + \
                             "RootSIFT + TF-IDF + mesh/depth PnP"
            base["solver_mode"] = solver_mode
            if previous is not None:
                base["previous_attempt"] = {k: previous.get(k) for k in
                    ["method", "reason", "query_features", "correspondences",
                     "inliers", "retrieval_method", "retrieval"]}
            if base.get("success") or result is None:
                result = base

        if result is None:
            result = {"success": False, "reason": "no_solver_result", "solver_mode": solver_mode}
        if global_rank is not None:
            result["fallback_global_retrieval_top10"] = [
                {"id": rid, "score": float(score)} for rid, score in global_rank[:10]]
        if expanded is not None:
            result["fallback_expanded_candidate_count"] = len(expanded)
        if warnings:
            result["enhancement_warnings"] = warnings
        return result

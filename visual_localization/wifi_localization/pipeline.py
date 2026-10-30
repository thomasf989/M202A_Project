"""Reusable hierarchical localization pipeline for CLI/evaluation code."""
from pathlib import Path
from .features import extract
from .localization import localize as cpu_localize, Settings


class LocalizationPipeline:
    def __init__(self, capture, db, cache_root, progress=None):
        self.capture=capture;self.db=db;self.cache_root=Path(cache_root)
        self.progress=progress or (lambda message,fraction:None)
        self.learned=None;self.retriever=None

    def _models(self):
        if self.learned is None:
            from .learned import LearnedLocalizer
            self.learned=LearnedLocalizer(self.capture,self.db.ids,self.cache_root/'local',self.progress)
        else:self.learned.progress=self.progress

    def localize(self, rgb, candidate_intrinsics, exclude=(), query_hash=None, settings=None, enhanced=True):
        s=settings or Settings();result=None;warnings=[]
        if enhanced:
            try:
                self._models();ranking=None;method=None
                try:
                    if self.retriever is None:
                        from .retrieval import DinoV2Retriever
                        self.retriever=DinoV2Retriever(self.capture,self.db.ids,self.cache_root/'global',progress=self.progress)
                    else:self.retriever.progress=self.progress
                    ranking=self.retriever.rank(rgb,exclude);method='DINOv2 global retrieval'
                except Exception as e:
                    warnings.append(f'DINOv2 retrieval: {e}')
                    _,desc=extract(rgb,self.db.meta['nfeatures'])
                    ranking=self.db.rank(desc,exclude);method='RootSIFT TF-IDF retrieval (learned local matching)'
                result=self.learned.localize(rgb,candidate_intrinsics,exclude,query_hash,
                    self.db.meta['reference_image_hashes'],candidate_ranking=ranking,
                    top_k=max(12,s.top_k),retrieval_method=method,settings=s)
            except Exception as e:
                warnings.append(f'Learned matcher: {e}')
        if result is None or not result.get('success'):
            previous=result
            base=cpu_localize(self.capture,self.db,rgb,candidate_intrinsics[0],s,exclude,query_hash,
                              candidate_intrinsics=candidate_intrinsics)
            base['method']='RootSIFT + visual-word TF-IDF + depth PnP (CPU baseline)'
            if previous is not None:
                base['learned_attempt']={k:previous.get(k) for k in
                    ['method','reason','query_features','correspondences','inliers','pose_confidence','retrieval_method','retrieval']}
            if base.get('success') or result is None:result=base
        if warnings:
            result['enhancement_warnings']=warnings
        return result

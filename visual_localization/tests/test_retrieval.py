import unittest
import numpy as np
from wifi_localization.retrieval import cosine_rank, retrieval_views, expand_pose_ranking


class _Frame:
    def __init__(self, xyz):
        self.T_world_cv=np.eye(4); self.T_world_cv[:3,3]=xyz


class _Capture:
    def __init__(self):
        self.reference_ids=['a','b','c','d']
        self.ids=list(self.reference_ids)
        self.frames={
            'a':_Frame([0,0,1]),
            'b':_Frame([0,0,1]),  # same center, different synthetic orientation
            'c':_Frame([.4,0,1]),
            'd':_Frame([5,0,1]),
        }


class RetrievalTests(unittest.TestCase):
    def test_cosine_rank_and_exclusion(self):
        rows=cosine_rank([1,0],[[0,1],[1,0],[.8,.2]],['a','b','c'],exclude={'b'})
        self.assertEqual([r[0] for r in rows],['c','a'])
        self.assertGreater(rows[0][1],rows[1][1])

    def test_phone_query_uses_multiple_crops_but_reference_index_is_compact(self):
        portrait=np.zeros((1200,800,3),np.uint8)
        self.assertEqual(len(retrieval_views(portrait,reference=True)),1)
        self.assertGreaterEqual(len(retrieval_views(portrait,reference=False)),4)

    def test_pose_neighbour_expansion_adds_same_location_views(self):
        rows=expand_pose_ranking(_Capture(), [('a',.9),('d',.7)], max_candidates=4,
                                 seed_count=1,neighbors_per_seed=2,neighbor_radius_m=1.)
        ids=[r[0] for r in rows]
        self.assertIn('a',ids)
        self.assertIn('b',ids)
        self.assertIn('c',ids)


if __name__=='__main__':unittest.main()

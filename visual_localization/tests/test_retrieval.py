import unittest
from wifi_localization.retrieval import cosine_rank

class RetrievalTests(unittest.TestCase):
    def test_cosine_rank_and_exclusion(self):
        rows=cosine_rank([1,0],[[0,1],[1,0],[.8,.2]],['a','b','c'],exclude={'b'})
        self.assertEqual([r[0] for r in rows],['c','a'])
        self.assertGreater(rows[0][1],rows[1][1])

if __name__=='__main__':unittest.main()

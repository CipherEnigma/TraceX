from tracex_ir import metrics as M
from tracex_ir.baselines import GlobalPopularity, frequency_baseline, retrieval_baseline
from tracex_ir.retrieval import IDF, bm25, jaccard, lcs_length, retrieve, retrieve_demonstrations, tfidf_cosine


def T(tid, pois):
    return {"traj_id": tid, "checkins": [{"poi_id": p} for p in pois]}


HIST = [T(1, [1, 2, 3]), T(2, [4, 5, 6]), T(3, [1, 2, 9])]
IDF_ = IDF(HIST + [T(4, [7, 8])])


def test_metrics():
    assert M.precision_at_k([1, 2, 3], {1, 3}, 2) == 0.5
    assert M.recall_at_k([1, 2, 3], {1, 3}, 1) == 0.5
    assert M.reciprocal_rank([5, 2], {2}) == 0.5
    assert abs(M.average_precision([1, 2, 3], {1, 3}) - (1 + 2 / 3) / 2) < 1e-9
    assert M.ndcg_at_k([1], {1}, 5) == 1.0 and M.accuracy_at_k([1, 2], 2, 1) == 0.0


def test_similarities():
    assert jaccard([1, 2], [2, 3]) == 1 / 3
    assert lcs_length([1, 2, 3, 4], [2, 4, 5]) == 2
    assert lcs_length([], [1]) == 0
    assert tfidf_cosine([1, 2], [1, 2], IDF_) > 0.999
    assert tfidf_cosine([1], [4], IDF_) == 0
    assert bm25([1], [4, 5], IDF_) == 0 and bm25([1], [1, 2], IDF_) > 0


def test_retrieve_orders_and_ties():
    for m in ["jaccard", "lcs", "tfidf", "bm25", "hybrid"]:
        top = retrieve(m, [1, 2], HIST, IDF_)
        assert top[0][0]["traj_id"] in (1, 3)
        assert top[-1][0]["traj_id"] == 2
    # all-zero scores: most recent trajectory first
    assert retrieve("recent", [1], HIST, IDF_)[0][0]["traj_id"] == 3


def test_baselines_and_demo_do_not_use_target():
    pop = GlobalPopularity(HIST)
    inst = {"prefix": [{"poi_id": 1}, {"poi_id": 2}], "target": {"poi_id": 3}}
    r = frequency_baseline(inst, HIST, pop, n=5)
    assert r[0] in (1, 2) and len(set(r)) == len(r)
    r2 = retrieval_baseline("lcs", inst, HIST, IDF_, pop, top_k=1, n=5)
    assert len(r2) == 5
    d = retrieve_demonstrations(inst, HIST, IDF_, method="lcs", top_k=2)
    assert len(d) == 2 and {"context", "next", "score", "traj_id"} <= set(d[0])

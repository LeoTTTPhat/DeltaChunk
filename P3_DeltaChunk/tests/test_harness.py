"""embed cache, index policies, LLM cache round-trip."""
import numpy as np
import pytest

from deltachunk.index import HnswPolicy, exact_search, fresh_reference, recall_at_k


def test_vector_cache_roundtrip(tmp_path, monkeypatch):
    from deltachunk import embed
    c = embed.VectorCache("m", path=str(tmp_path / "c.sqlite"))
    k = [embed.vkey("m", "a"), embed.vkey("m", "b")]
    c.put_many([(k[0], np.ones(4)), (k[1], np.zeros(4))])
    g = c.get_many(k + [embed.vkey("m", "zz")])
    assert len(g) == 2 and g[k[0]].dtype == np.float16 and float(g[k[0]].sum()) == 4.0
    assert embed.vkey("m", "a") != embed.vkey("m", "a", prefix="q: ")


@pytest.mark.parametrize("policy", ["TOMB", "REUSE", "REBUILD"])
def test_index_policies_exact_on_tiny(policy):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 16)).astype(np.float32); X /= np.linalg.norm(X, axis=1, keepdims=True)
    ix = HnswPolicy(16, policy, tau=0.2, efS=300, capacity=64)
    ix.insert(list(range(200)), X[:200])
    ix.delete(list(range(0, 120, 2)))
    ix.insert(list(range(200, 300)), X[200:])
    ids = sorted(ix.live); q = X[rng.choice(300, 20)] + 0.01
    truth = exact_search(ids, np.stack([ix.live[i] for i in ids]), q, 10)
    assert recall_at_k(ix.search(q, 10), truth) == 1.0          # tiny case: HNSW with large ef is exact
    assert recall_at_k(fresh_reference(ix, q, 10), truth) == 1.0
    assert all(i not in ix.live for i in range(0, 120, 2)) and len(ix.live) == 240
    found = {i for row in ix.search(X[:120:2], 10) for i in row}
    assert not found & set(range(0, 120, 2))                    # deleted ids never returned
    if policy == "REBUILD":
        assert ix.rebuilds >= 1
    if policy == "REUSE":
        assert ix.slots == 240 + ix.tombs


def test_llm_cache_roundtrip(tmp_path, monkeypatch):
    from deltachunk.qa import LLM
    calls = []
    llm = LLM(cache_path=str(tmp_path / "llm.sqlite"))
    monkeypatch.setattr(llm, "_post", lambda payload: calls.append(payload) or
                        {"choices": [{"message": {"content": "<think></think> Paris"}}], "usage": {"prompt_tokens": 5}})
    assert llm.chat("Capital of France?") == "Paris"
    assert llm.chat("Capital of France?") == "Paris"
    assert len(calls) == 1 and calls[0]["messages"][0]["content"].endswith("/no_think") and calls[0]["temperature"] == 0.0
    assert llm.stats["hits"] == 1


def test_exact_topk_equals_faiss_and_bruteforce(tmp_path):
    """FAISS is run in a fresh subprocess: its OpenMP runtime aborts once torch has run in the same process (macOS)."""
    import subprocess, sys
    from deltachunk.index import exact_topk
    rng = np.random.default_rng(1)
    X = rng.normal(size=(1000, 32)).astype(np.float32); Q = rng.normal(size=(37, 32)).astype(np.float32)
    np.save(tmp_path / "X.npy", X); np.save(tmp_path / "Q.npy", Q)
    code = ("import numpy as np, faiss, sys; X=np.load(sys.argv[1]); Q=np.load(sys.argv[2]); ix=faiss.IndexFlatIP(32); ix.add(X); "
            "np.save(sys.argv[3], ix.search(Q, 10)[1])")
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "X.npy"), str(tmp_path / "Q.npy"), str(tmp_path / "I.npy")], check=True)
    I = np.load(tmp_path / "I.npy")
    ours = exact_topk(X, Q, 10)
    assert (ours == I).all()
    assert (ours == np.argsort(-(Q @ X.T), axis=1, kind="stable")[:, :10]).all()

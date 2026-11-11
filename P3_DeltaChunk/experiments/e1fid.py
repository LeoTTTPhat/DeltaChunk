"""E1-fid (B3.5): LangChain indexing API fidelity. Replays WIKI + K8S documents version by version (global timestamp
order) through langchain_core.indexing.index(cleanup="incremental", source_id_key="source") with an
InMemoryRecordManager and an InMemoryVectorStore over a counting fake embedding, using our REC chunks.
Setting 1: metadata = {"source"} only.  Setting 2: + start_index (RecursiveCharacterTextSplitter add_start_index=True).
Compared per version against our counts: S1 (per-document multiset), S1-distinct, and S2 (global live set).
usage: python experiments/e1fid.py [--docs 50] [--tag _smoke]
"""
import argparse, json, os, time
from collections import Counter

from common import RESULTS, calibration, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.delta import ScopeState, apply_edit


class CountingEmb:
    def __init__(self):
        from langchain_core.embeddings import DeterministicFakeEmbedding
        self.inner = DeterministicFakeEmbedding(size=8); self.n = 0

    def embed_documents(self, texts):
        self.n += len(texts); return self.inner.embed_documents(texts)

    def embed_query(self, t):
        return self.inner.embed_query(t)


def run(stream_docs, rec, start_index: bool, batch_size: int = 100):
    from langchain_core.documents import Document
    from langchain_core.indexing import InMemoryRecordManager, index
    from langchain_core.vectorstores import InMemoryVectorStore
    emb = CountingEmb(); vs = InMemoryVectorStore(emb)
    rm = InMemoryRecordManager(namespace="fid"); rm.create_schema()
    st = ScopeState(); rows = []
    events = sorted(((v["ts"], d, i, v) for d, vers in stream_docs for i, v in enumerate(vers)), key=lambda x: (x[0], str(x[1]), x[2]))
    prev = {}
    for ts, d, i, v in events:
        V = Version(v["text"]); cs = rec.chunk(V)
        md = (lambda c: {"source": str(d), "start_index": c.start}) if start_index else (lambda c: {"source": str(d)})
        docs = [Document(page_content=c.text, metadata=md(c)) for c in cs]
        r = index(docs, rm, vs, cleanup="incremental", source_id_key="source", key_encoder="sha256", batch_size=batch_size)
        keys = [c.key for c in cs]
        ours = apply_edit(st, str(d), keys, {k: 1 for k in keys})
        old = prev.get(d, Counter()); new = Counter(keys)
        s1d = len(set(new) - set(old))
        prev[d] = new
        rows.append(dict(doc=str(d), first=i == 0, lc_added=r["num_added"], lc_deleted=r["num_deleted"], S1=ours["S1_ins"],
                         S1_distinct=s1d, S2=ours["S2_ins"], S1_del=ours["S1_del"], S2_del=ours["S2_del"],
                         S1_distinct_del=len(set(old) - set(new))))
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", type=int, default=50); ap.add_argument("--tag", default="")
    ap.add_argument("--streams", default="wiki,k8s"); a = ap.parse_args(); t0 = time.time()
    out = {}
    for s in a.streams.split(","):
        rec = make("REC", calibration(s).get("200", {}).get("REC"))
        try:
            groups = doc_groups(load_stream(s, a.docs))
        except Exception as e:
            out[s] = {"error": repr(e)}; continue
        for setting, si, bs in (("source_only", False, 100), ("source_only_batch10000", False, 10000), ("start_index", True, 100)):
            rows = [r for r in run(groups, rec, si, bs) if not r["first"]]
            n = len(rows)
            res = dict(edits=n, lc_added_total=sum(r["lc_added"] for r in rows))
            for k in ("S1", "S1_distinct", "S2"):
                res[f"match_{k}"] = sum(r["lc_added"] == r[k] for r in rows) / max(1, n)
                res[f"{k}_total"] = sum(r[k] for r in rows)
            res["match_del_S1_distinct"] = sum(r["lc_deleted"] == r["S1_distinct_del"] for r in rows) / max(1, n)
            res["mismatch_examples"] = [r for r in rows if r["lc_added"] != r["S1_distinct"]][:5]
            out[f"{s}/{setting}"] = res
            print(s, setting, {k: v for k, v in res.items() if k != "mismatch_examples"}, flush=True)
    out["explanation"] = ("LangChain hashes page_content + metadata; with the source in metadata identical chunk texts in "
                          "different documents get different record keys, so index() deduplicates per document and per "
                          "batch (distinct keys): it matches S1-distinct, not the global S2 scope. With start_index, any "
                          "length change before a chunk changes its metadata hash and forces a re-embedding. With the default batch_size=100, "
                          "incremental cleanup runs after every batch and deletes the source's records not yet seen in this run, so "
                          "a document with more than 100 chunks re-embeds its later batches on every update; with batch_size=10000 the "
                          "counts equal S1-distinct.")
    out["meta"] = run_meta(seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(RESULTS, f"e1_fid{a.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

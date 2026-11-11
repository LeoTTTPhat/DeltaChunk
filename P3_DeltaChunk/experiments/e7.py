"""E7: maintenance timing and unit costs (measured under contention; labeled as such).

  timing : WIKI-20 per-revision replay in timestamp order for FIX, REC, STRUCT-M, SentCDC, KARA, DOC under the online
           regime (one encode call per edit) and the micro-batched regime (calls of >= 64 chunks across edits); cold
           (uncached) encoder; hnswlib index (TOMB) maintained alongside; 3 repetitions with the chunker order rotated
           and the regime order alternated; the token-count cache is cleared before every replay and a warm-up encode
           precedes every replay. Initial indexing (each document's first version) is done and timed BEFORE the clock
           for edits starts and is excluded from per-edit costs. In the batched regime a delete of a chunk that is still
           pending is cancelled in the batch (counted), so index state and delete counts equal the online regime's
           (asserted). (B5 fixes of review findings B4-E7a/B4-m.)
  micro  : unit costs alpha0 (per-call overhead), alpha (per token), beta (insert / vector), gamma (delete / vector).
           Every timed call encodes exactly `size` distinct texts drawn without replacement from a pool of >= 5x256
           texts (asserted), sizes in random order, >= 10 repetitions, each call's own token count recorded; alpha and
           alpha0 by Theil-Sen robust regression over all calls (OLS reported too); alpha0 must not exceed the median
           1-text call (B5 fix of review finding B4-C3: the build timed truncated/empty batches).
  rho    : HNSW (re)build time rho(N) for N in {100k, 300k, 1M} (vectors: real cached chunk vectors, padded with
           perturbed copies, labeled).
  llm    : E7c per-chunk LLM extraction (entity/relation triples, qwen3:8b) on 200 chunks: seconds/call, tokens/call.
usage: python experiments/e7.py timing|micro|rho|llm [--docs 20] [--reps 3] [--tag _smoke]
"""
import argparse, json, os, time
from collections import Counter

import numpy as np
import pandas as pd
from common import RESULTS, calibration, doc_groups, load_snapshot, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.embed import Embedder
from deltachunk.index import HnswPolicy
from deltachunk.text import ntok_many

CHUNKERS = ["FIX", "REC", "STRUCT-M", "SentCDC", "KARA", "DOC"]


def replay(name, cal, events, emb, regime, batch=64, max_wait=16):
    """`batched`: a call is made when >= `batch` chunks are pending OR the oldest pending chunk has waited `max_wait`
    edits (bounded staleness; without the bound, pending chunks of frequently edited documents are cancelled before
    they are ever embedded, which conflates micro-batching with coalescing). `encoded` counts chunks actually embedded."""
    """Replay `events` (ts, doc, i, text). Phase 1 (untimed for per-edit costs, reported as init_s): index every
    document's first version. Phase 2 (timed): the edits (i > 0)."""
    from deltachunk import text as _text
    _text._TOK_CACHE.clear()                       # cold token-count cache for every replay
    emb._encode(["warm-up encode"])                # warm-up call (device / graph initialisation)
    ch = make(name, cal.get(name)); state, prev = {}, {}
    ix = HnswPolicy(384, "TOMB", capacity=4096); ids = {}; nid = 0
    t_chunk = t_embed = t_index = 0.0; calls = 0; pending = []; n_ins = n_del = toks = cancelled = 0

    encoded = 0

    def flush(items):
        nonlocal t_embed, t_index, calls, nid, encoded
        if not items:
            return
        t = time.perf_counter(); X = emb._encode([x[1] for x in items]); t_embed += time.perf_counter() - t; calls += 1
        encoded += len(items)
        t = time.perf_counter()
        new_ids = list(range(nid, nid + len(items))); nid += len(items)
        ix.insert(new_ids, X)
        for (key, _), i in zip(items, new_ids):
            ids.setdefault(key, []).append(i)
        t_index += time.perf_counter() - t

    def chunk(d, text):
        v = Version(text)
        if ch.stateful:
            cs, state[d] = ch.update(v, state.get(d))
        else:
            cs = ch.chunk(v)
        new = Counter((c.key, c.text) for c in cs); old = prev.get(d, Counter())
        add, rem = new - old, old - new; prev[d] = new
        return add, rem

    # phase 1: initial indexing of every first version (one batched pass), timed separately
    t0 = time.perf_counter(); init_items = []
    for ts, d, i, text in events:
        if i == 0:
            add, _ = chunk(d, text); init_items += list(add.elements())
    for k in range(0, len(init_items), 256):
        flush(init_items[k:k + 256])
    init_s = time.perf_counter() - t0
    t_chunk = t_embed = t_index = 0.0; calls = 0; encoded = 0
    # phase 2: edits
    t_wall = time.perf_counter()
    n_edits_seen = 0; first_wait = 0
    for ts, d, i, text in events:
        if i == 0:
            continue
        n_edits_seen += 1
        t = time.perf_counter()
        add, rem = chunk(d, text)
        t_chunk += time.perf_counter() - t
        t = time.perf_counter()
        rm = []
        for (k, txt), c in rem.items():
            for _ in range(c):
                if ids.get(k):
                    rm.append(ids[k].pop())
                else:   # still pending in the micro-batch: cancel it instead of inserting a dead vector
                    j = next(j for j, x in enumerate(pending) if x[0] == k)
                    pending.pop(j); cancelled += 1
        ix.delete(rm); t_index += time.perf_counter() - t
        items = list(add.elements()); n_ins += len(items); n_del += len(rm)
        toks += sum(ntok_many([x[1] for x in items]))
        if regime == "online":
            flush(items)
        else:
            if items and not pending:
                first_wait = n_edits_seen
            pending += items
            if pending and (len(pending) >= batch or n_edits_seen - first_wait >= max_wait):
                flush(pending); pending = []
    flush(pending)
    wall = time.perf_counter() - t_wall
    live = sum(len(v) for v in ids.values())
    return dict(t_chunk=t_chunk, t_embed=t_embed, t_index=t_index, calls=calls, ins=n_ins, dele=n_del, toks=toks,
                cancelled_pending=cancelled, encoded=encoded, live_vectors=live, batch=batch, max_wait=max_wait, init_s=init_s, init_chunks=len(init_items), edit_wall_s=wall)


def timing(a):
    cal = calibration("wiki").get("200", {})
    groups = doc_groups(load_stream("wiki", a.docs))
    events = sorted(((v["ts"], str(d), i, v["text"]) for d, vers in groups for i, v in enumerate(vers)), key=lambda x: x[:3])
    n_edits = sum(1 for e in events if e[2] > 0)
    emb = Embedder(cache=False)
    rows = []
    for rep in range(a.reps):
        order = CHUNKERS[rep % len(CHUNKERS):] + CHUNKERS[:rep % len(CHUNKERS)]
        for ci, name in enumerate(order):
            live = {}
            regimes = ("online", "batched") if (rep + ci) % 2 == 0 else ("batched", "online")   # alternate regime order
            for regime in regimes:
                r = replay(name, cal, events, emb, regime)
                live[regime] = (r["live_vectors"], r["dele"] + r["cancelled_pending"])
                rows.append(dict(rep=rep, chunker=name, regime=regime, edits=n_edits, wall_s=r["edit_wall_s"], **r,
                                 per_1k_edits_s=r["edit_wall_s"] / n_edits * 1000, load=json.dumps(load_snapshot())))
                print(rep, name, regime, round(r["edit_wall_s"], 1), "s", flush=True)
            assert live["online"] == live["batched"], f"index state differs between regimes for {name}: {live}"
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, f"e7_timing{a.tag}.csv"), index=False)
    json.dump(run_meta(device=emb.device, docs=a.docs, edits=n_edits), open(os.path.join(RESULTS, f"e7_timing{a.tag}.meta.json"), "w"), indent=1)


def micro(a):
    from scipy.stats import theilslopes
    emb = Embedder(cache=False)
    rec = make("REC", calibration("wiki").get("200", {}).get("REC"))
    texts, ndocs = [], 0
    for _, vers in doc_groups(load_stream("wiki")):      # enough documents for a pool of >= 5 x 256 distinct texts
        texts += [c.text for c in rec.chunk(Version(vers[-1]["text"]))]; ndocs += 1
        if len(set(texts)) >= 6 * 256:
            break
    texts = list(dict.fromkeys(texts)); assert len(texts) >= 5 * 256, len(texts)
    toks = np.array(ntok_many(texts))
    rng = np.random.default_rng(7)
    emb._encode(texts[:64])                                   # warm-up
    sizes = (1, 4, 16, 64, 256); calls = []
    plan = [s for s in sizes for _ in range(a.micro_reps)]; rng.shuffle(plan)
    for size in plan:
        idx = rng.choice(len(texts), size=size, replace=False); batch = [texts[i] for i in idx]
        assert len(batch) == size
        t = time.perf_counter(); emb._encode(batch); sec = time.perf_counter() - t
        calls.append(dict(size=int(size), tokens=int(toks[idx].sum()), seconds=sec))
    C = pd.DataFrame(calls)
    res = {f"call_{s}": dict(seconds=float(g.seconds.median()), tokens=float(g.tokens.median()), n=len(g)) for s, g in C.groupby("size")}
    alpha, alpha0, lo, hi = theilslopes(C.seconds.values, C.tokens.values)
    ols_alpha, ols_alpha0 = np.polyfit(C.tokens.values, C.seconds.values, 1)
    one = float(C[C["size"] == 1].seconds.median())
    if alpha0 > one:
        print(f"WARNING: alpha0 {alpha0:.3f} s > median single-text call {one:.3f} s", flush=True)
    X = emb._encode(texts[:2000])
    ix = HnswPolicy(384, "TOMB", capacity=4096)
    t = time.perf_counter(); ix.insert(list(range(len(X))), X, threads=1); beta = (time.perf_counter() - t) / len(X)
    t = time.perf_counter(); ix.delete(list(range(0, len(X), 2))); gamma = (time.perf_counter() - t) / (len(X) // 2)
    out = dict(alpha0=float(alpha0), alpha=float(alpha), alpha_ci95=[float(lo), float(hi)], alpha_ols=float(ols_alpha),
               alpha0_ols=float(ols_alpha0), single_call_s=one, alpha0_le_single_call=bool(alpha0 <= one),
               beta=float(beta), gamma=float(gamma), calls=res, pool_texts=len(texts), pool_docs=ndocs,
               reps_per_size=a.micro_reps, fit="Theil-Sen over all timed calls (seconds ~ alpha0 + alpha * tokens)",
               meta=run_meta(device=emb.device, note="measured under contention"))
    C.to_csv(os.path.join(RESULTS, f"e7_unit_calls{a.tag}.csv"), index=False)
    json.dump(out, open(os.path.join(RESULTS, f"e7_unit_costs{a.tag}.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k not in ("meta", "calls")})


def rho(a):
    from deltachunk.embed import VectorCache
    import sqlite3
    c = VectorCache("BAAI/bge-small-en-v1.5-mps")
    real = np.stack([np.frombuffer(x, np.float16).astype(np.float32) for (x,) in c.db.execute("SELECT x FROM v LIMIT 1000000")]) \
        if len(c) else np.random.default_rng(0).normal(size=(10000, 384)).astype(np.float32)
    rng = np.random.default_rng(0); out = dict(real_vectors=len(real))
    for N in [int(x) for x in a.sizes.split(",")]:
        reps = -(-N // len(real))
        X = np.vstack([real + (0 if r == 0 else rng.normal(scale=0.02, size=real.shape).astype(np.float32)) for r in range(reps)])[:N]
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        ix = HnswPolicy(384, "TOMB", capacity=N + 16, build_threads=8)
        t = time.perf_counter(); ix.insert(list(range(N)), X, threads=8); out[f"rho_{N}"] = time.perf_counter() - t
        print(N, out[f"rho_{N}"], flush=True); del ix, X
    out["meta"] = run_meta(build_threads=8, note="vectors beyond the real cached ones are perturbed copies (labeled)")
    json.dump(out, open(os.path.join(RESULTS, f"e7_rho{a.tag}.json"), "w"), indent=1)


EXTRACT = """Extract the entities and relations stated in the text as JSON: {{"entities": [...], "relations": [[head, relation, tail], ...]}}.
Text: {t}"""


def llm(a):
    from deltachunk.qa import LLM
    groups = doc_groups(load_stream("wiki", 40))
    texts = list(dict.fromkeys(c.text for _, vers in groups for c in make("REC", calibration("wiki").get("200", {}).get("REC")).chunk(Version(vers[-1]["text"]))))[:a.n]
    # fresh cache per timing run: a shared cache would time cache hits on reruns (review finding B4-m E7c)
    L = LLM(cache_path=os.path.join(RESULTS, "..", "data", f"llm_cache_e7c_{time.strftime('%Y%m%d%H%M%S')}.sqlite"))
    t = time.perf_counter(); outs = L.chat_many([EXTRACT.format(t=x) for x in texts], max_tokens=256); wall = time.perf_counter() - t
    out = dict(chunks=len(texts), wall_s=wall, calls=L.stats["calls"], hits=L.stats["hits"], sec_per_call_wall=wall / max(1, L.stats["calls"]),
               mean_latency_s=L.stats["seconds"] / max(1, L.stats["calls"]), prompt_tokens=L.stats["prompt_tokens"],
               completion_tokens=L.stats["completion_tokens"], parallel=L.parallel,
               meta=run_meta(llm_model=L.model, note="GPU shared with other projects; measured under contention"))
    json.dump(out, open(os.path.join(RESULTS, f"e7c_llm{a.tag}.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k != "meta"})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["timing", "micro", "rho", "llm"])
    ap.add_argument("--docs", type=int, default=20); ap.add_argument("--reps", type=int, default=3); ap.add_argument("--tag", default="")
    ap.add_argument("--sizes", default="100000,300000,1000000"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--micro-reps", dest="micro_reps", type=int, default=12)
    a = ap.parse_args(); {"timing": timing, "micro": micro, "rho": rho, "llm": llm}[a.cmd](a)


if __name__ == "__main__":
    main()

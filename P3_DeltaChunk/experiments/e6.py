"""E6 (reduced index study): replay WIKI document histories in global timestamp order through an HNSW index under
TOMB / REUSE / REBUILD(tau) for chunkers REC, SentCDC, DIFF[REC], on top of a background of distractor vectors.
Index entries are per (document, chunk instance) (S1 semantics). Vectors come from the shared content-addressed cache.
Every `probe` edits: recall@10 of held-out queries (QA questions + texts of recently deleted chunks) against exact
search over the live vectors and against a freshly built HNSW; tombstone share, slots, latency. Final efSearch sweep.
B5 fixes (review finding B4-E6): REBUILD's tombstone share is measured over the churned partition (background slots
excluded) and swept over --taus; question queries are restricted to the replayed documents (WIKI-100), topped up with
units of their final versions if too few; recall is reported per query type (questions / deleted-chunk texts); every
cell is repeated for each --seeds value (HNSW random_seed).
Runner session 2 (Nov 9 ~20:45): one cell takes ~10 min (45 cells ~ 8 h) and nothing was checkpointed, so one memwatch
stop lost every finished cell. Each (chunker, policy, seed) cell now writes results/raw/e6<tag>/<config hash>/<cell>.pkl
(hash over src/ and the cell-independent arguments) and is skipped on resume. Cells are claimed with a lock file
(<cell>.lock holding the PID), so extra processes can share the cell list (`--no-final`: compute cells only); the
process that writes the CSV waits for cells claimed by live processes and computes cells whose owner died.
Default taus reduced to 0.3 (the F9 design point; env E6_TAUS): with the 50k static background every rebuild re-inserts
~52k vectors (several seconds), and on the churned partition tau=0.05 fires ~0.3 times per edit (smoke: 323 rebuilds in
1,000 edits), i.e. ~2,400 rebuilds and hours per cell over the 7,955 WIKI-100 edits; three tau=0.05 cells had run
45 min without finishing. tau in {0.05, 0.1} is Tier B (seed 0 only, if time allows).
usage: python experiments/e6.py [--docs 100] [--edits 1000000] [--bg 200000] [--probe 2000] [--tag _smoke]
"""
import argparse, hashlib, json, os, pickle, re, time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from common import P3, RAW, RESULTS, calibration, code_hash, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.embed import Embedder
from deltachunk.index import HnswPolicy, exact_search, fresh_reference, recall_at_k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", type=int, default=100); ap.add_argument("--edits", type=int, default=10 ** 9)
    ap.add_argument("--bg", type=int, default=200000); ap.add_argument("--probe", type=int, default=2000); ap.add_argument("--tag", default="")
    ap.add_argument("--chunkers", default="REC,SentCDC,DIFF[REC]"); ap.add_argument("--policies", default="TOMB,REUSE,REBUILD")
    ap.add_argument("--taus", default=os.environ.get("E6_TAUS", "0.3")); ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--fresh_every", type=int, default=5)
    ap.add_argument("--qtag", default="")
    ap.add_argument("--no-final", dest="no_final", action="store_true", help="compute (claim) cells only; no CSV/meta")
    a = ap.parse_args(); t0 = time.time()
    cal = calibration("wiki").get("200", {})
    emb = Embedder()
    groups = doc_groups(load_stream("wiki", a.docs))
    events = sorted(((v["ts"], str(d), i, v["text"]) for d, vers in groups for i, v in enumerate(vers)), key=lambda x: x[:3])
    # held-out queries: QA questions if present, else first sentences of final versions
    qf = os.path.join(RESULTS, "qa", f"questions_wiki{a.qtag}.jsonl")
    docset = {str(d) for d, _ in groups}
    qtexts = [q["question"] for q in map(json.loads, open(qf)) if str(q["doc_id"]) in docset][:a.nq] if os.path.exists(qf) else []
    n_questions = len(qtexts)
    if len(qtexts) < a.nq:     # top up with prose units of the replayed documents' final versions (labeled)
        import hashlib
        extra = [u.text for d, vers in groups for u in Version(vers[-1]["text"]).units if u.kind == "prose" and u.tok >= 10]
        extra.sort(key=lambda t: hashlib.sha256(t.encode()).hexdigest())
        qtexts += extra[:a.nq - len(qtexts)]
    Qbase = emb.encode_queries(qtexts)
    bg = np.zeros((0, 384), np.float32)
    bgf = os.path.join(P3, "data", "streams", "wiki_bg.parquet")
    if a.bg and os.path.exists(bgf):
        rec = make("REC", cal.get("REC")); texts = []
        for t in pd.read_parquet(bgf).text:
            texts += [c.text for c in rec.chunk(Version(t))]
            if len(texts) >= a.bg:
                break
        bg = emb.encode(texts[:a.bg])
    rows = []
    pols = [(p, None) for p in a.policies.split(",") if p != "REBUILD"]
    if "REBUILD" in a.policies.split(","):
        pols += [("REBUILD", float(t)) for t in a.taus.split(",")]
    cells = [(name, pol, tau, int(seed)) for name in a.chunkers.split(",") for pol, tau in pols for seed in a.seeds.split(",")]
    qh = hashlib.sha256("\n".join(qtexts).encode()).hexdigest()[:12]
    ck = os.path.join(RAW, f"e6{a.tag}", code_hash("e6", a.docs, a.edits, a.bg, len(bg), a.probe, a.nq, a.fresh_every, qh, cal))
    os.makedirs(ck, exist_ok=True)

    def cfile(c, ext):
        return os.path.join(ck, re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{c[0]}__{c[1]}__{c[2]}__s{c[3]}") + ext)

    def owner_alive(c):
        try:
            pid = int(open(cfile(c, ".lock")).read().strip() or 0)
            os.kill(pid, 0); return pid != os.getpid()
        except (OSError, ValueError):
            return False

    def claim(c):
        lk = cfile(c, ".lock")
        for _ in range(2):
            try:
                fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
                return True
            except FileExistsError:
                if owner_alive(c):
                    return False
                try:
                    os.remove(lk)       # stale lock of a dead process
                except FileNotFoundError:
                    pass
        return False

    def todo():
        """Next cell to compute: unfinished and claimable; None when every unfinished cell is owned by a live process."""
        for c in cells:
            if not os.path.exists(cfile(c, ".pkl")) and claim(c):
                if os.path.exists(cfile(c, ".pkl")):     # finished between the check and the claim
                    os.remove(cfile(c, ".lock")); continue
                return c
        return None

    resumed = [c for c in cells if os.path.exists(cfile(c, ".pkl"))]
    print(f"e6: {len(cells)} cells, {len(resumed)} already done in {os.path.relpath(ck, P3)}", flush=True)
    while True:
        cell = todo()
        if cell is None:
            if all(os.path.exists(cfile(x, ".pkl")) for x in cells) or a.no_final:
                break
            time.sleep(30); continue     # cells owned by other live processes: wait for them (or for their death)
        name, pol, tau, seed = cell
        n_rows0 = len(rows)
        if True:
            ch = make(name, cal.get(name))
            ix = HnswPolicy(384, pol, tau=tau or 0.3, capacity=len(bg) + 1024, seed=seed, static=len(bg))
            polname = pol if tau is None else f"REBUILD({tau})"
            t = time.perf_counter()
            if len(bg):
                ix.insert(list(range(len(bg))), bg, threads=8)
            build_s = time.perf_counter() - t
            nid = len(bg); inst = defaultdict(lambda: defaultdict(list)); state = {}; prev = {}
            deleted_texts = []; n_edits = 0; ins_s = del_s = 0.0
            for ts, d, i, text in events:
                v = Version(text)
                if ch.stateful:
                    cs, state[d] = ch.update(v, state.get(d))
                else:
                    cs = ch.chunk(v)
                new = Counter(c.text for c in cs); old = prev.get(d, Counter())
                add, rem = new - old, old - new
                rm_ids = []
                for txt, c in rem.items():
                    for _ in range(c):
                        rm_ids.append(inst[d][txt].pop()); deleted_texts.append(txt)
                t = time.perf_counter(); ix.delete(rm_ids); del_s += time.perf_counter() - t
                add_texts = list(add.elements())
                if add_texts:
                    X = emb.encode(add_texts); ids = list(range(nid, nid + len(add_texts))); nid += len(add_texts)
                    for txt, j in zip(add_texts, ids):
                        inst[d][txt].append(j)
                    t = time.perf_counter(); ix.insert(ids, X, threads=1); ins_s += time.perf_counter() - t
                prev[d] = new
                if i > 0:
                    n_edits += 1
                    if n_edits % a.probe == 0 or n_edits == a.edits:
                        rows.append(dict(probe(ix, emb, Qbase, deleted_texts[-a.nq:], name, polname, n_edits, ins_s, del_s, build_s,
                                               fresh=(n_edits // a.probe) % a.fresh_every == 0, nqq=n_questions), seed=seed))
                if n_edits >= a.edits:
                    break
            r = probe(ix, emb, Qbase, deleted_texts[-a.nq:], name, polname, n_edits, ins_s, del_s, build_s, sweep=True, nqq=n_questions)
            r["final"] = True; r["seed"] = seed; rows.append(r)
            print(name, pol, {k: r[k] for k in ("edits", "recall_exact", "recall_fresh", "tomb_share", "rebuilds")}, flush=True)
        tmp = cfile(cell, ".pkl.tmp"); pickle.dump(rows[n_rows0:], open(tmp, "wb")); os.replace(tmp, cfile(cell, ".pkl"))
        try:
            os.remove(cfile(cell, ".lock"))
        except FileNotFoundError:
            pass
    if a.no_final:
        return
    rows = [r for c in cells for r in pickle.load(open(cfile(c, ".pkl"), "rb"))]
    df = pd.DataFrame(rows); df.to_csv(os.path.join(RESULTS, f"e6_index{a.tag}.csv"), index=False)
    json.dump(run_meta(seconds=round(time.time() - t0, 1), background=len(bg), docs=a.docs, embed=emb.stats, hnsw=dict(M=16, efC=200, efS=100),
                       question_queries=n_questions, unit_queries=len(qtexts) - n_questions, seeds=a.seeds, taus=a.taus,
                       cells=len(cells), cells_resumed=len(resumed), checkpoint=os.path.relpath(ck, P3),
                       tau_basis="tombstones / (slots - background slots)"),
              open(os.path.join(RESULTS, f"e6_index{a.tag}.meta.json"), "w"), indent=1)


def probe(ix, emb, Qbase, deleted, name, pol, n_edits, ins_s, del_s, build_s, sweep=False, fresh=True, nqq=0):
    Q = np.vstack([Qbase, emb.encode(deleted)]) if deleted else Qbase
    ids = list(ix.live); X = np.stack([ix.live[i] for i in ids])
    truth = exact_search(ids, X, Q, 10)
    t = time.perf_counter(); found = ix.search(Q, 10); lat = (time.perf_counter() - t) / len(Q) * 1000
    nb = len(Qbase); fr = fresh_reference(ix, Q, 10) if fresh else None
    r = dict(chunker=name, policy=pol, edits=n_edits, live=len(ids), slots=ix.slots, tomb_share=ix.tombs / max(1, ix.slots),
             tomb_share_churned=ix.tombs / max(1, ix.slots - ix.static),
             recall_exact=recall_at_k(found, truth), recall_fresh=(recall_at_k(fr, truth) if fresh else float("nan")),
             recall_exact_questions=recall_at_k(found[:nqq], truth[:nqq]), recall_exact_units=recall_at_k(found[nqq:nb], truth[nqq:nb]),
             recall_exact_deleted=recall_at_k(found[nb:], truth[nb:]),
             lat_ms=lat, rebuilds=ix.rebuilds, rebuild_s=ix.rebuild_seconds, insert_s=ins_s, delete_s=del_s, bg_build_s=build_s)
    if sweep:
        for ef in (50, 100, 200):
            r[f"recall_ef{ef}"] = recall_at_k(ix.search(Q, 10, ef=ef), truth)
    return r


if __name__ == "__main__":
    main()

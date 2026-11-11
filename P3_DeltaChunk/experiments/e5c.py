"""E5c: CTX (contextual retrieval) cost per chunk with and without llama-server prompt caching.
For WIKI documents of <= 3,000 tokens (slot context is 32768 / 8 = 4096), every STRUCT-M chunk gets one LLM call whose
prompt holds the whole document followed by the chunk (Anthropic's contextual-retrieval prompt shape). Calls of one
document run back to back on one client thread, so with cache_prompt=True the shared document prefix is reused.
B5 (review finding B4-E5c): calls go straight to the server (no response cache, so every call executes); the two
variants are interleaved per document in ABBA order over --reps repetitions; before every no-cache call the slot's
prompt cache is invalidated by a distinct short prompt, and llama-server's `timings` (prompt_n = prompt tokens actually
processed, prompt_ms, predicted_ms) are logged per call to show prefix reuse; costs are reported per document-length
stratum (<=1k, 1-2k, 2-3k tokens). Documents above 3,000 tokens do not fit a slot (32768 / 8); stated.
E-phase budget fix (Nov 9, runner session 1): the full design (200 chunks x 3 reps x 2 variants + 600 eviction calls, one
client thread, ~10 s per call under contention) needs ~4 h of LLM time, far above the 15-min E5c budget in experiments.md,
and it blocked E9 in the sequential GPU chain. The driver now (a) checkpoints every finished (rep, document) block to
results/raw/e5c<tag>/calls.csv and resumes from it, and (b) stops starting new blocks after --max-minutes (default 45,
env E5C_MAX_MINUTES) of wall time. A block holds both variants of one document, so the ABBA balance is kept; the JSON
records the completed calls, documents and repetitions, and whether the time cap ended the run.
usage: python experiments/e5c.py [--chunks 200] [--reps 3] [--max-minutes 45] [--tag _smoke]
"""
import argparse, json, os, time

import pandas as pd
import requests

from common import P3, RESULTS, calibration, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.qa import DEFAULT_MODEL, DEFAULT_URL

PROMPT = """<document>
{doc}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{chunk}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--chunks", type=int, default=200); ap.add_argument("--tag", default="")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-minutes", dest="max_minutes", type=float, default=float(os.environ.get("E5C_MAX_MINUTES", 45)))
    a = ap.parse_args(); t0 = time.time()
    ch = make("STRUCT-M", calibration("wiki").get("200", {}).get("STRUCT-M"))
    jobs = []
    for d, vers in doc_groups(load_stream("wiki")):
        v = Version(vers[-1]["text"])
        if sum(u.tok for u in v.units) > 3000:
            continue
        dtok = sum(u.tok for u in v.units)
        jobs += [(str(d), v.text, c.text, dtok) for c in ch.chunk(v)]
        if len(jobs) >= a.chunks:
            break
    jobs = jobs[:a.chunks]
    def call(prompt, cache_prompt):
        msgs = [{"role": "user", "content": prompt + " /no_think"}]
        for attempt in range(12):   # memguard may restart the server (memwatch stop + cooldown): wait up to ~10 min
            t = time.perf_counter()
            try:
                js = requests.post(DEFAULT_URL, json=dict(model=DEFAULT_MODEL, messages=msgs, max_tokens=80, temperature=0.0,
                                                          cache_prompt=cache_prompt), timeout=600).json()
                break
            except Exception:
                if attempt == 11:
                    raise
                time.sleep(min(60, 2 ** attempt))
        tm = js.get("timings", {}); u = js.get("usage", {})
        return dict(wall_s=time.perf_counter() - t, prompt_tokens=u.get("prompt_tokens"), prompt_n=tm.get("prompt_n"),
                    prompt_ms=tm.get("prompt_ms"), predicted_ms=tm.get("predicted_ms"))
    by_doc = {}
    for d, doc, chunk, dtok in jobs:
        by_doc.setdefault(d, []).append((doc, chunk, dtok))
    ck_dir = os.path.join(RESULTS, "raw", f"e5c{a.tag}"); os.makedirs(ck_dir, exist_ok=True)
    ck = os.path.join(ck_dir, "calls.csv")
    rows = pd.read_csv(ck, dtype={"doc": str}).to_dict("records") if os.path.exists(ck) else []
    done_blocks = {(int(r["rep"]), str(r["doc"])) for r in rows}
    t_start = time.time(); capped = False
    for rep in range(a.reps):
        for di, (d, items) in enumerate(by_doc.items()):
            if (rep, d) in done_blocks:
                continue
            if (time.time() - t_start) / 60 > a.max_minutes:
                capped = True; break
            order = ("no_prompt_cache", "prompt_cache") if (di + rep) % 2 == 0 else ("prompt_cache", "no_prompt_cache")
            block = []
            for variant in order:
                for k, (doc, chunk, dtok) in enumerate(items):
                    if variant == "no_prompt_cache":
                        call(f"Reply OK. {rep}-{d}-{k}", False)        # evict the document prefix from the slot cache
                    r = call(PROMPT.format(doc=doc, chunk=chunk), variant == "prompt_cache")
                    block.append(dict(rep=rep, doc=d, chunk_i=k, variant=variant, doc_tok=dtok, **r))
            rows += block
            pd.DataFrame(rows).to_csv(ck + ".tmp", index=False); os.replace(ck + ".tmp", ck)
        if capped:
            print(f"time cap {a.max_minutes} min reached in rep {rep}", flush=True); break
        print("rep", rep, "done", flush=True)
    D = pd.DataFrame(rows); D.to_csv(os.path.join(RESULTS, f"e5c_ctx_calls{a.tag}.csv"), index=False)
    D["stratum"] = pd.cut(D.doc_tok, [0, 1000, 2000, 3000], labels=["<=1k", "1-2k", "2-3k"])
    out = {v: dict(calls=int(len(g)), sec_per_call=float(g.wall_s.mean()), sec_per_call_median=float(g.wall_s.median()),
                   prompt_tokens_mean=float(g.prompt_tokens.mean()), prompt_n_mean=float(pd.to_numeric(g.prompt_n).mean()),
                   per_stratum={str(k): float(h.wall_s.mean()) for k, h in g.groupby("stratum", observed=True)})
           for v, g in D.groupby("variant")}
    out["completed"] = dict(calls=int(len(D)), docs=int(D.doc.nunique()), reps_started=int(D.rep.nunique()),
                            full_reps=int(sum(D[D.rep == r].doc.nunique() == len(by_doc) for r in D.rep.unique())),
                            time_capped=capped, max_minutes=a.max_minutes, planned_docs=len(by_doc), planned_reps=a.reps)
    out["meta"] = run_meta(seconds=round(time.time() - t0, 1), reps=a.reps, docs=len(by_doc),
                           note="ABBA per document, one client thread; GPU shared (measured under contention)")
    json.dump(out, open(os.path.join(RESULTS, f"e5c_ctx_cost{a.tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1))


if __name__ == "__main__":
    main()

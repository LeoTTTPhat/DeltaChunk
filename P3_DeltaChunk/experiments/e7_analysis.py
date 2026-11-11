"""E7 analysis (all costs DERIVED from measured counts x measured unit costs; labeled as such in the paper).

1. Derived maintenance cost per 1k retained revisions (whatever the sync interval) per (stream, chunker, scope, interval, regime):
   online   : calls = edits with INS > 0, cost = calls*alpha0 + TOKS*alpha + INS*beta + DEL*gamma + chunk time
   batched  : calls = ceil(sum INS / 64)
   llm      : per-chunk LLM extraction, INS * seconds per call (E7c) + the batched embedding cost
2. Variance decomposition of log10(cost) on BALANCED cell sets only (B5 fix of review finding B4-VD: S2/S3 exist only
   at per-revision sync, so the full factorial is unbalanced and marginal SS are not additive): design A = chunker x
   scope x regime at interval=revision; design B = chunker x interval x regime at scope S1; one observation per cell,
   chunkers kept only if present in every cell, so main-effect SS are orthogonal and additive (remainder =
   interactions). Each design is reported with and without the LLM-extraction regime, and within each regime the
   chunker share and the max/min chunker cost ratio.
3. Judge-3 table: tombstone-rebuild threshold tau* = sqrt(2 rho(N) DEL_rate / (N q c_q)) chosen with the chunker's
   own DEL rate (chunker-aware) vs the stream-average DEL rate (index-only); c_q = latency increase per unit tombstone
   share (E6), q = queries per edit (parameter, default 10). Reports whether the chosen tau bucket differs; `differs` is
   NaN (not False) when tau* is undefined (rho_N missing or c_q <= 0), and the table is flagged invalid.
4. Staleness simulation: edits replayed at their real timestamps (per stream); ONE absolute embedding budget grid per
   stream shared by all chunkers (multiples of REC's mean load in re-embedded tokens/hour; B5 fix of review finding
   B4-ST: per-chunker budgets made every chunker run at the same utilisation); work = re-embedded tokens; FIFO queue;
   staleness = time from an edit to the moment all its chunks are embedded; an edit with zero work has zero staleness.
usage: python experiments/e7_analysis.py [--tag ""] [--utag ""]
"""
import argparse, json, math, os

import numpy as np
import pandas as pd
from common import RESULTS, run_meta

TAU_BUCKETS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def load(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def balanced_ss(d: pd.DataFrame, facs: list[str]) -> dict:
    """Main-effect SS shares of log10(cost) on the balanced subset of `d` (one cell per factor combination; chunkers
    missing any cell dropped). Returns shares per factor, the interaction remainder and the cell counts."""
    if d.empty:
        return {}
    others = [f for f in facs if f != "chunker"]
    need = d.groupby(others).ngroups if others else 1
    keep = [c for c, h in d.groupby("chunker") if h.groupby(others).ngroups == need and len(h) == need] if others else list(d.chunker.unique())
    d = d[d.chunker.isin(keep)]
    if d.empty or d.groupby(facs).size().max() != 1:
        return {"note": "no balanced cell set"}
    y = np.log10(d.cost_per_1k_edits_s); tot = float(((y - y.mean()) ** 2).sum())
    if tot <= 0:
        return {}
    out = {f: float(((y.groupby(d[f]).transform("mean") - y.mean()) ** 2).sum() / tot) for f in facs}
    out["interactions"] = 1.0 - sum(out.values()); out["cells"] = int(len(d)); out["chunkers"] = len(keep)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--utag", default="")
    ap.add_argument("--q", type=float, default=10.0); ap.add_argument("--N", type=int, default=300000)
    a = ap.parse_args()
    U = load(os.path.join(RESULTS, f"e7_unit_costs{a.utag}.json"), {})
    L = load(os.path.join(RESULTS, f"e7c_llm{a.utag}.json"), {})
    R = load(os.path.join(RESULTS, f"e7_rho{a.utag}.json"), {})
    unit = {k: U.get(k, 0.0) for k in ("alpha0", "alpha", "beta", "gamma")}
    llm_s = L.get("sec_per_call_wall", float("nan"))
    E = pd.read_parquet(os.path.join(RESULTS, f"e1_edits{a.tag}.parquet"))
    E = E[E["T"] == 200]
    rows = []
    nrev = E[E.interval == "revision"].groupby(["stream", "chunker"]).size().to_dict()
    for (s, c, iv), g in E.groupby(["stream", "chunker", "interval"]):
        # normalize by the number of retained revisions (not coalesced pairs) so sync intervals are comparable
        n = nrev.get((s, c), len(g))
        for scope in ("S1", "S2", "S3"):
            ins = g["ins"] if scope == "S1" else g.get(f"{scope}_ins")
            toks = g["toks"] if scope == "S1" else g.get(f"{scope}_toks")
            dele = g["dele"] if scope == "S1" else g.get("S2_del")   # index deletes are identical under S2 and S3
            if ins is None or ins.isna().all():
                continue
            I, T_, D = ins.sum(), toks.sum(), dele.sum(); chunk = g.chunk_total_ms.sum() / 1000
            base = T_ * unit["alpha"] + I * unit["beta"] + D * unit["gamma"] + chunk
            costs = {"online": (ins > 0).sum() * unit["alpha0"] + base, "batched": math.ceil(I / 64) * unit["alpha0"] + base}
            costs["llm"] = costs["batched"] + I * llm_s
            for regime, cst in costs.items():
                rows.append(dict(stream=s, chunker=c, interval=iv, scope=scope, regime=regime, revisions=n, applied_edits=len(g), ins_per_edit=I / n,
                                 del_per_edit=D / n, cost_per_1k_edits_s=cst / n * 1000))
    C = pd.DataFrame(rows); C.to_csv(os.path.join(RESULTS, f"e7_derived_costs{a.tag}.csv"), index=False)
    # 2. variance decomposition on balanced designs, per stream
    vd = {}
    for s, g in C.groupby("stream"):
        g = g[np.isfinite(g.cost_per_1k_edits_s) & (g.cost_per_1k_edits_s > 0)]
        designs = {"A_scope@revision": (g[g.interval == "revision"], ["chunker", "scope", "regime"]),
                   "B_interval@S1": (g[g.scope == "S1"], ["chunker", "interval", "regime"])}
        vd[s] = {}
        for dname, (d, facs) in designs.items():
            for variant, dd in (("all_regimes", d), ("without_llm", d[d.regime != "llm"])):
                vd[s][f"{dname}/{variant}"] = balanced_ss(dd, facs)
            vd[s][f"{dname}/within_regime"] = {
                r: dict(balanced_ss(dr, [f for f in facs if f != "regime"]),
                        chunker_max_min_ratio=float(dr.groupby("chunker").cost_per_1k_edits_s.mean().max() /
                                                    dr.groupby("chunker").cost_per_1k_edits_s.mean().min()))
                for r, dr in d.groupby("regime")}
    # 3. judge-3: tau* chunker-aware vs index-only
    E6 = pd.read_csv(os.path.join(RESULTS, f"e6_index{a.utag}.csv")) if os.path.exists(os.path.join(RESULTS, f"e6_index{a.utag}.csv")) else None
    c_q, c_q_ci = float("nan"), [float("nan"), float("nan")]
    if E6 is not None and len(E6) > 2 and E6.tomb_share.std() > 0:
        from scipy.stats import linregress
        lr = linregress(E6.tomb_share, E6.lat_ms / 1000)
        c_q = float(lr.slope); c_q_ci = [float(lr.slope - 1.96 * lr.stderr), float(lr.slope + 1.96 * lr.stderr)]
    rho_N = R.get(f"rho_{a.N}", R.get("rho_300000", float("nan")))
    j3_valid = bool(np.isfinite(rho_N) and np.isfinite(c_q) and c_q > 0)
    if not j3_valid:
        print(f"WARNING judge-3 undefined: rho_N={rho_N} c_q={c_q} (c_q must be > 0); differs = NaN", flush=True)
    j3 = []
    for (s, iv), g in C[(C.scope == "S1") & (C.regime == "batched")].groupby(["stream", "interval"]):
        avg = g.del_per_edit.mean()
        for r in g.itertuples():
            def tau(d):
                if c_q <= 0 or not np.isfinite(rho_N):
                    return float("nan")
                return math.sqrt(2 * rho_N * d / (a.N * a.q * c_q))
            ta, ti = tau(r.del_per_edit), tau(avg)
            b = lambda t: min(TAU_BUCKETS, key=lambda x: abs(x - t)) if np.isfinite(t) else None
            j3.append(dict(stream=s, interval=iv, chunker=r.chunker, del_per_edit=r.del_per_edit, tau_chunker_aware=ta,
                           tau_index_only=ti, bucket_aware=b(ta), bucket_index_only=b(ti),
                           differs=(float(b(ta) != b(ti)) if (np.isfinite(ta) and np.isfinite(ti)) else float("nan"))))
    J = pd.DataFrame(j3); J.to_csv(os.path.join(RESULTS, f"e7_judge3{a.tag}.csv"), index=False)
    # 4. staleness simulation (per revision, S1): one absolute token budget grid per stream (multiples of REC's load)
    st = []
    for s, g in E[(E.interval == "revision")].groupby("stream"):
        ref = g[g.chunker == "REC"] if (g.chunker == "REC").any() else g
        span_h = max(1.0, (ref.ts.max() - ref.ts.min()) / 3600)
        ref_load = ref["toks"].sum() / span_h                      # REC's mean re-embedded tokens per hour
        for c, h in g.groupby("chunker"):
            h = h.sort_values(["ts", "doc_id", "rev_new"]); ts = h.ts.values.astype(float); work = h["toks"].values.astype(float)
            for mult in (0.5, 1.1, 2.0, 10.0):
                B = max(1e-9, ref_load * mult); free_at = ts.min(); lat = []
                for t, w in zip(ts, work):
                    if w <= 0:
                        lat.append(0.0); continue           # nothing to re-embed: fresh immediately
                    start = max(free_at, t); free_at = start + w / B * 3600; lat.append(free_at - t)
                st.append(dict(stream=s, chunker=c, budget_x_rec_load=mult, budget_tokens_per_h=B,
                               load_tokens_per_h=float(work.sum() / span_h), utilisation=float(work.sum() / span_h / B),
                               staleness_mean_h=float(np.mean(lat)) / 3600, staleness_p95_h=float(np.percentile(lat, 95)) / 3600))
    S = pd.DataFrame(st); S.to_csv(os.path.join(RESULTS, f"e7_staleness{a.tag}.csv"), index=False)
    out = dict(unit_costs=unit, llm_sec_per_call=llm_s, rho_N=rho_N, c_q=c_q, c_q_ci95=c_q_ci, judge3_valid=j3_valid,
               variance_decomposition=vd,
               judge3_differs_share=(float(J.differs.mean()) if (len(J) and j3_valid) else None),
               meta=run_meta(note="derived costs: measured counts x unit costs measured under contention"))
    json.dump(out, open(os.path.join(RESULTS, f"e7_analysis{a.tag}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1, default=str))


if __name__ == "__main__":
    main()

"""Figures from results/ (PDF into results/figures/). Each figure is produced only if its inputs exist.
F1: fraction of the new version re-embedded (TOKS/tok(d')) and index ops per edit, per chunker x stream (S1, per
    revision, T=128 = size-matched; T=200 as f1_cost_T200), mean with document-clustered bootstrap 95% CI.  F6: total re-embedded tokens at each sync interval relative to per-revision sync.
usage: python experiments/plots.py [--tag _smoke] [--paper]
"""
import argparse, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from common import RESULTS

FIG = os.path.join(RESULTS, "figures")
ORDER = ["FIX", "FIXO", "REC", "STRUCT-M", "STRUCT-NM", "SENTWIN", "SentCDC", "GCDC", "BCDC", "STL", "STL+snap", "KARA",
         "DIFF[REC]", "DIFF[STRUCT-M]", "DOC"]


FULL_WIDTH = 7.16
NAVY, TEAL, ORANGE, GRAY = "#24618A", "#248278", "#BB572A", "#606D78"
STREAM_ORDER = ("wiki", "k8s", "cpy")
STREAM_STYLE = dict(zip(STREAM_ORDER, ((NAVY, "o"), (TEAL, "s"), (ORANGE, "^"))))
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.edgecolor": "#7F8991",
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def save(fig, name, tag):
    fig.savefig(os.path.join(FIG, f"{name}{tag}.pdf"))
    plt.close(fig)


def grid(ax, axis="y"):
    ax.grid(axis=axis, color="#E0E5E8", lw=0.6)


def marker(color, shape, label):
    return Line2D([], [], color=color, marker=shape, linestyle="none", label=label, markersize=4)


def aligned_rows(axes, names):
    rows = np.arange(len(names))
    for ax in axes:
        for row in rows[::2]:
            ax.axhspan(row - 0.5, row + 0.5, color="#F3F5F7", zorder=0)
        ax.set_ylim(len(names) - 0.5, -0.5)
        ax.set_yticks(rows)
        ax.tick_params(axis="y", length=0)
        grid(ax, "x")
    axes[0].set_yticklabels(names)
    return rows


def cluster_ci(df, col, B=1000, seed=0):
    """Document-clustered bootstrap CI of the edit-level mean."""
    g = df.groupby("doc_id")[col].agg(["sum", "size"])
    s, n = g["sum"].values, g["size"].values
    rng = np.random.default_rng(seed); k = len(s)
    if k == 0:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, k, size=(B, k))
    boots = s[idx].sum(1) / n[idx].sum(1)
    return s.sum() / n.sum(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def at_T(E, T):
    """Per-revision S1 rows at calibration point T; SENTWIN and STRUCT-NM are size-independent (run at T=200 only) and
    are taken from the T=200 rows (unmatched by design)."""
    d = E[(E["T"] == T) & (E.interval == "revision")]
    extra = E[(E["T"] == 200) & (E.interval == "revision") & E.chunker.isin(["SENTWIN", "STRUCT-NM"]) & ~E.chunker.isin(set(d.chunker))]
    return pd.concat([d, extra])


UNMATCHED = {"SENTWIN", "STRUCT-NM"}          # no size parameter: realized sizes differ from the matched configurations


def f1(E, tag, T=128):
    """Clustered edit-cost estimates, grouped by chunker and stream. Round 2: tick labels carry the achieved mean chunk
    size per stream (WIKI/K8S/CPY, edit-weighted as in the matched-size analysis); configurations without a size
    parameter are marked with a dagger and drawn with open markers; a double dagger marks a configuration whose size
    sweep does not reach the common size on some stream (results/round2/matched_size_table.csv)."""
    d = at_T(E, T)
    streams = [stream for stream in STREAM_ORDER if stream in set(d.stream)]
    names = [c for c in ORDER if c in set(d.chunker)]
    size = d.groupby(["chunker", "stream"]).chunk_tok_mean.mean() if "chunk_tok_mean" in d else None
    unsupported = set()
    mt = os.path.join(RESULTS, "round2", "matched_size_table.csv")
    if os.path.exists(mt):
        m = pd.read_csv(mt)
        unsupported = set(m[(m.size_label == "common") & ~m.supported.astype(bool)].chunker)
    labels = []
    for c in names:
        mark = "\u2020" if c in UNMATCHED else ("\u2021" if c in unsupported else "")
        sz = "/".join(f"{size[(c, s_)]:.0f}" for s_ in streams if size is not None and (c, s_) in size.index)
        labels.append(f"{c}{mark}  {sz}")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 3.25), sharey=True)
    rows = aligned_rows(axes, labels)
    axes[0].tick_params(axis="y", labelsize=6.6)
    offsets = np.linspace(-0.23, 0.23, len(streams)) if len(streams) > 1 else [0]
    groups = d.groupby(["stream", "chunker"])
    for ax, col, label in zip(axes, ("toks_frac", "ops"),
                              ("New-version tokens re-embedded", "Index operations per edit")):
        for stream, offset in zip(streams, offsets):
            color, shape = STREAM_STYLE[stream]
            for row, chunker in zip(rows, names):
                key = (stream, chunker)
                if key not in groups.groups:
                    continue
                m, lo, hi = cluster_ci(groups.get_group(key), col)
                ax.errorbar(m, row + offset, xerr=[[max(0, m - lo)], [max(0, hi - m)]],
                            fmt=shape, color=color, markersize=3, elinewidth=0.8, capsize=1.2,
                            markerfacecolor="none" if chunker in UNMATCHED else color)
        ax.set_xlabel(label)
        ax.set_xlim(left=0)
    axes[0].xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    handles = [marker(*STREAM_STYLE[stream], stream.upper()) for stream in streams]
    fig.legend(handles=handles, loc="upper center", ncol=len(streams), bbox_to_anchor=(0.6, 1.0))
    fig.text(0.005, 0.955, "Configuration  achieved tokens W/K/C", fontsize=6.6, ha="left", va="center", color="#404A52")
    fig.subplots_adjust(left=0.235, right=0.985, top=0.88, bottom=0.16, wspace=0.10)
    name = "f1_cost" if T == 128 else f"f1_cost_T{T}"
    save(fig, name, tag)


def f6(E, tag):
    d = E[E["T"] == 200]
    if d.empty:
        return
    labels = {"revision": "Every edit", "daily": "Daily", "weekly": "Weekly",
              "monthly": "Monthly", "release": "Release dates"}
    streams = [stream for stream in STREAM_ORDER if stream in set(d.stream)]
    styles = {
        "FIX": (ORANGE, "o", "-"), "REC": (NAVY, "s", "-"),
        "STRUCT-M": (NAVY, "^", "--"), "SentCDC": (TEAL, "D", "-"),
        "STL": (ORANGE, "v", "--"), "KARA": (TEAL, "P", "--"),
        "DIFF[REC]": (GRAY, "<", "-."), "DOC": (GRAY, "x", ":"),
    }
    totals = d.groupby(["stream", "chunker", "interval"]).toks.sum()
    fig, axes = plt.subplots(1, len(streams), figsize=(FULL_WIDTH, 2.05), squeeze=False, sharey=True)
    handles = {}
    for ax, stream in zip(axes[0], streams):
        ivs = [iv for iv in labels if iv in set(d.loc[d.stream == stream, "interval"])]
        for name, (color, shape, line) in styles.items():
            key = (stream, name, "revision")
            if key not in totals or totals[key] <= 0:
                continue
            y = [totals.get((stream, name, iv), np.nan) / totals[key] for iv in ivs]
            handles[name], = ax.plot(range(len(ivs)), y, marker=shape, linestyle=line,
                                     color=color, linewidth=0.85, markersize=3, label=name)
        ax.set_xticks(range(len(ivs)), [labels[iv] for iv in ivs], rotation=30, ha="right")
        ax.set_title(stream.upper(), loc="left")
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        grid(ax)
    axes[0][0].set_ylabel("Tokens vs. every-edit sync")
    fig.legend(handles=list(handles.values()), loc="upper center", ncol=8, bbox_to_anchor=(0.53, 1.02),
               handletextpad=0.4, columnspacing=0.85)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.72, bottom=0.26, wspace=0.17)
    save(fig, "f6_sync", tag)


def f2(tag, r3=False):
    """Compare measurements with the bound at one radius per stream."""
    suffix = "_r3" if r3 else ""
    paths = [os.path.join(RESULTS, f"e2_thm1_{s}{suffix}{tag}.parquet") for s in STREAM_ORDER]
    data = [pd.read_parquet(path) for path in paths if os.path.exists(path)]
    if not data:
        return
    D = pd.concat(data)
    if "r" in D.columns:
        assert D.groupby("stream").r.nunique().max() == 1, "F2 would pool several r values"
    D = D[~D.guard]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.16))
    for stream in STREAM_ORDER:
        d = D[D.stream == stream]
        if d.empty:
            continue
        color, shape = STREAM_STYLE[stream]
        g = d.groupby(d.l_new.clip(upper=12))[["ins", "bound_ins"]].mean()
        axes[0].plot(g.index, g.ins, marker=shape, color=color, lw=0.9, ms=3)
        axes[0].plot(g.index, g.bound_ins, marker=shape, color=color, ls="--", lw=0.9, ms=3,
                     markerfacecolor="white")
    m = D.dropna(subset=["mc_mean"])
    if not m.empty:
        g = m.groupby(m.l_new.clip(upper=6))[["ins", "mc_mean"]].mean()
        axes[1].plot(g.index, g.ins, "o-", color=NAVY, lw=0.9, ms=3, label="Measured")
        axes[1].plot(g.index, g.mc_mean, "s--", color=GRAY, lw=0.9, ms=3, label="Monte Carlo")
        axes[1].legend(loc="upper left", handlelength=1.4)
    axes[0].set_title("(a) Covered edits", loc="left")
    axes[1].set_title("(b) Sampled single-region edits", loc="left")
    axes[0].set_xticks([0, 3, 6, 9, 12], ["0", "3", "6", "9", "12+"])
    axes[1].set_xticks([0, 2, 4, 6], ["0", "2", "4", "6+"])
    axes[0].set_ylabel("Chunks inserted per edit")
    for ax in axes:
        ax.set_xlabel("New units in edited span")
        grid(ax)
    handles = [marker(*STREAM_STYLE[s], s.upper()) for s in STREAM_ORDER if s in set(D.stream)]
    handles.extend([Line2D([], [], color=GRAY, lw=0.9, label="Measured"),
                    Line2D([], [], color=GRAY, lw=0.9, ls="--", label="Bound")])
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), bbox_to_anchor=(0.51, 1.02))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.73, bottom=0.22, wspace=0.26)
    save(fig, f"f2_thm1{suffix}", tag)


def f3(tag):
    import glob
    files = sorted(f for f in glob.glob(os.path.join(RESULTS, f"e3_drift_*{tag}.csv")) if tag or "_smoke" not in f)
    if not files:
        return
    D = pd.concat([pd.read_csv(f) for f in files])
    if "cohort" in D.columns:
        D = pd.concat([g[g.cohort == ">=100"] if (g.cohort == ">=100").any() else g[g.cohort == "all"]
                       for _, g in D.groupby("stream")])
    D = D[D.t >= 0]
    methods = {"KARA": (TEAL, "D"), "DIFF[REC]": (NAVY, "o"), "DIFF[STRUCT-M]": (ORANGE, "s")}
    lines = {"wiki": "-", "k8s": "--", "cpy": ":"}
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH, 2.10))
    for (c, stream), g in D.groupby(["chunker", "stream"]):
        color, shape = methods[c]
        g = g.sort_values("t")
        for ax, col in zip(axes, ("ks", "divergence", "out_of_range_m")):
            ax.plot(g.t, g[col], marker=shape, color=color, linestyle=lines[stream], ms=2.5, lw=0.85)
    for ax, label in zip(axes, ("KS distance", "Key divergence", "Outside 50–480 tokens")):
        ax.set_title(label, loc="left")
        ax.set_xlabel("Edit index")
        ax.set_ylim(bottom=0)
        grid(ax)
    axes[2].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    handles = [marker(*style, label) for label, style in methods.items()]
    handles.extend(Line2D([], [], color=GRAY, linestyle=style, label=stream.upper()) for stream, style in lines.items())
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.51, 1.02),
               columnspacing=1.05, handletextpad=0.35)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.73, bottom=0.22, wspace=0.35)
    save(fig, "f3_drift", tag)


def f4(E, tag):
    """Cost vs evidence recall@1024 at nominal T. Round 2: one distinct color per configuration, filled markers for
    the native-distractor run and open markers (joined by a thin line) for the common REC background (CPU) run, the
    realized final-snapshot chunk size next to each point, and an open dagger-marked STRUCT-NM (no size parameter)."""
    f = os.path.join(RESULTS, f"e4_quality{tag}.csv")
    if not os.path.exists(f):
        return
    meta = f.replace(".csv", ".meta.json")
    T, achieved = 128, {}
    if os.path.exists(meta):
        with open(meta) as handle:
            mj = json.load(handle)
        T = int(mj.get("T", 128)); achieved = mj.get("achieved_tok", {})
    Q = pd.read_csv(f)
    shared = None
    for extra in ("_sharedbg", "_r2matched"):
        g = os.path.join(RESULTS, f"e4_quality{extra}{tag}.csv")
        if os.path.exists(g):
            x = pd.read_csv(g)
            x = x[x.chunker.isin(set(Q.chunker))]
            shared = x if shared is None else pd.concat([shared, x[~x.chunker.isin(set(shared.chunker))]])
    costs = at_T(E, T).groupby(["stream", "chunker"]).toks_frac.mean()
    streams = [stream for stream in STREAM_ORDER if stream in set(Q.stream)]
    styles = {   # Okabe-Ito palette, one color per configuration
        "FIX": ("#E69F00", "X"), "REC": ("#0072B2", "o"), "STRUCT-M": ("#56B4E9", "s"), "STRUCT-NM": ("#6E6E6E", "P"),
        "SentCDC": ("#009E73", "^"), "STL": ("#D55E00", "v"), "KARA": ("#CC79A7", "D"),
    }
    path = os.path.join(RESULTS, f"e4_stats{tag}.csv")
    intervals = pd.read_csv(path).set_index(["stream", "config"]) if os.path.exists(path) else None
    fig, axes = plt.subplots(1, len(streams), figsize=(FULL_WIDTH, 2.2), sharex=True, sharey=True, squeeze=False)
    handles = {}
    for stream, ax in zip(streams, axes[0]):
        for chunker, g in Q[Q.stream == stream].groupby("chunker"):
            key = (stream, chunker)
            if key not in costs:
                continue
            x = costs[key]
            y = g["recall@1024"].mean()
            color, shape = styles.get(chunker, (GRAY, "."))
            hollow = chunker in UNMATCHED
            if intervals is not None and key in intervals.index:
                ci = intervals.loc[key]
                if np.isclose(y, ci["mean"]):
                    yerr = [[max(0, y - ci.ci_lo)], [max(0, ci.ci_hi - y)]]
                    ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor=color, alpha=0.5, lw=0.75, capsize=1.4)
            ax.plot(x, y, linestyle="none", marker=shape, color=color, markersize=4.6,
                    markerfacecolor="white" if hollow else color, markeredgewidth=0.9, zorder=3)
            if shared is not None:
                h = shared[(shared.stream == stream) & (shared.chunker == chunker)]
                if len(h):
                    y2 = h["recall@1024"].mean()
                    ax.plot([x, x], [y, y2], color=color, lw=0.6, alpha=0.8, zorder=2)
                    ax.plot(x, y2, linestyle="none", marker=shape, color=color, markersize=3.6, markerfacecolor="white",
                            markeredgewidth=0.8, zorder=3)
            sz = achieved.get(chunker, {}).get(stream)
            if sz is not None:
                ax.annotate(f"{sz:.0f}", (x, y), xytext=(3.5, -1.5), textcoords="offset points", fontsize=5.6,
                            color=color, va="top")
            handles[chunker] = Line2D([], [], color=color, marker=shape, linestyle="none", markersize=4,
                                      markerfacecolor="white" if hollow else color,
                                      label=chunker + ("\u2020" if hollow else ""))
        ax.set_title(stream.upper(), loc="left")
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        grid(ax)
        grid(ax, "x")
    axes[0, 0].set_ylabel("Evidence recall@1024")
    fig.supxlabel("New-version tokens re-embedded per edit", fontsize=8, y=0.04)
    labels = [c for c in styles if c in handles] + [c for c in handles if c not in styles]
    hs = [handles[c] for c in labels]
    if shared is not None:
        hs.append(Line2D([], [], color="#404A52", marker="o", markerfacecolor="white", linestyle="none", markersize=3.6,
                         label="common background"))
    fig.legend(handles=hs, loc="upper center", ncol=len(hs), bbox_to_anchor=(0.51, 1.02), columnspacing=0.8,
               handletextpad=0.3, fontsize=7)
    fig.subplots_adjust(left=0.10, right=0.985, top=0.74, bottom=0.22, wspace=0.17)
    save(fig, "f4_pareto", tag)


def f5(tag):
    f = os.path.join(RESULTS, f"e5a_late{tag}.parquet")
    if not os.path.exists(f):
        return
    D = pd.read_parquet(f)
    g = D.groupby(D.dist.clip(upper=10))
    series = [
        (g.cos_late.mean(), NAVY, "o", "-", "LATE mean"),
        (g.cos_late.quantile(0.05), ORANGE, "s", "--", "LATE drift p95"),
        (g.cos_ctl.mean(), GRAY, "^", ":", "LATE-ctl mean"),
    ]
    fig, ax = plt.subplots(figsize=(3.34, 2.26))
    for values, color, shape, line, label in series:
        ax.plot(values.index, 1e3 * (1 - values), marker=shape, linestyle=line,
                color=color, lw=0.9, ms=3, label=label)
    ax.set_xticks([1, 3, 5, 7, 10], ["1", "3", "5", "7", "10+"])
    ax.set_xlabel("Chunks from nearest change")
    ax.set_ylabel(r"Cosine drift ($10^3(1 - \cos)$)")
    ax.set_title("(a) Locality of vector drift", loc="left")
    ax.legend(loc="upper right", handlelength=1.4)
    grid(ax)
    fig.subplots_adjust(left=0.18, right=0.97, top=0.84, bottom=0.23)
    save(fig, "f5_late", tag)


def f5b(tag):
    """Align forward-token cost and absolute recall by maintenance policy."""
    f = os.path.join(RESULTS, f"e5b_policies{tag}.json")
    if not os.path.exists(f):
        return
    with open(f) as handle:
        j = json.load(handle)
    rec, cost = j.get("recall", {}), j.get("cost", {})
    pols = [p for p in ["EXACT", "STALE", "WINDOW-0", "WINDOW-1", "WINDOW-2", "WINDOW-4"] if p in rec and p in cost]
    if not pols or not cost.get("EXACT", {}).get("forward_tokens"):
        return
    exact = cost["EXACT"]["forward_tokens"]
    sf = os.path.join(RESULTS, f"e5b_stats{tag}.csv")
    ci = pd.read_csv(sf).set_index("policy")[["ci_lo", "ci_hi"]] if os.path.exists(sf) else None
    fig, axes = plt.subplots(1, 2, figsize=(3.78, 2.26), sharey=True)
    rows = aligned_rows(axes, pols)
    for row, policy in zip(rows, pols):
        color, shape = {"EXACT": (NAVY, "s"), "STALE": (ORANGE, "^")}.get(policy, (TEAL, "o"))
        x = cost[policy]["forward_tokens"] / exact
        axes[0].plot(x, row, linestyle="none", marker=shape, ms=4, color=color)
        axes[1].plot(rec[policy], row, linestyle="none", marker=shape, ms=4, color=color)
        if ci is not None and policy in ci.index:
            lo, hi = ci.loc[policy]
            axes[1].hlines(row, lo, hi, color=color, lw=1, zorder=2)
    axes[0].set_xlim(0, 1.07)
    axes[0].set_xticks([0, 0.5, 1])
    for ax in axes:
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes[0].set_xlabel("Tokens / EXACT")
    axes[1].set_xlabel("Recall@1024")
    fig.text(0.20, 0.88, "(b) Edit-proximal retrieval", fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.25, right=0.97, top=0.82, bottom=0.23, wspace=0.33)
    save(fig, "f5b_policies", tag)


def f9(tag):
    f = os.path.join(RESULTS, f"e6_index{tag}.csv")
    if not os.path.exists(f):
        return
    D = pd.read_csv(f); D = D[D.get("final", False) != True] if "final" in D else D
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for (c, p), g in D.groupby(["chunker", "policy"]):
        m = g.groupby("edits").recall_exact.agg(["mean", "min", "max"])   # mean over HNSW seeds, error bar = range
        ax.errorbar(m.index, m["mean"], yerr=[m["mean"] - m["min"], m["max"] - m["mean"]], marker="o", ms=2, lw=0.6,
                    capsize=1, label=f"{c} {p}")
    ax.set_xlabel("edits", fontsize=7); ax.set_ylabel("recall@10 vs exact (mean, seed range)", fontsize=7); ax.legend(fontsize=4); ax.tick_params(labelsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"f9_index{tag}.pdf")); plt.close(fig)


def f7(tag):
    f = os.path.join(RESULTS, f"e7_derived_costs{tag}.csv")
    if not os.path.exists(f):
        return
    C = pd.read_csv(f)
    C = C[(C.scope == "S1") & (C.interval == "revision")]
    names = [c for c in ORDER if c in set(C.chunker)]
    streams = [stream for stream in STREAM_ORDER if stream in set(C.stream)]
    if not streams or not names:
        return
    means = C.groupby(["stream", "chunker", "regime"]).cost_per_1k_edits_s.mean()
    fig, axes = plt.subplots(1, len(streams), figsize=(FULL_WIDTH, 3.05), squeeze=False, sharex=True, sharey=True)
    rows = aligned_rows(axes[0], names)
    styles = {"online": (NAVY, "s"), "batched": (TEAL, "o"), "llm": (ORANGE, "^")}
    for ax, stream in zip(axes[0], streams):
        for (regime, (color, shape)), offset in zip(styles.items(), [-0.20, 0, 0.20]):
            values = [means.get((stream, c, regime), np.nan) for c in names]
            ax.plot(values, rows + offset, marker=shape, color=color, linestyle="none", ms=3.4)
        ax.set_xscale("log")
        ax.set_xticks([10, 1000, 100000], ["10", "1k", "100k"])
        ax.set_title(stream.upper(), loc="left")
    handles = [marker(*style, "Per-chunk LLM" if regime == "llm" else regime.capitalize())
               for regime, style in styles.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.55, 1.01))
    fig.supxlabel("Derived seconds per 1,000 edits (log scale)", fontsize=8, y=0.05)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.82, bottom=0.15, wspace=0.10)
    save(fig, "f7_regimes", tag)


def f8(tag):
    import glob
    fs = [f for f in glob.glob(os.path.join(RESULTS, f"e8_sens_*{tag}.csv")) if tag or "_smoke" not in f]
    if not fs:
        return
    D = pd.concat([pd.read_csv(f).assign(stream=os.path.basename(f).split("_")[2].replace(".csv", "")) for f in fs])
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for fam, mk in zip(["SentCDC", "REC", "STRUCT-M", "GCDC", "STL", "BCDC"], "os^v<>"):
        d = D[D.config.str.startswith(fam + " ")]
        ax.scatter(d.mean_chunk_tok, d.toks_frac, marker=mk, s=10, color="black", label=fam)
    ax.set_xlabel("mean chunk tokens", fontsize=7); ax.set_ylabel("TOKS / tok(d') per edit", fontsize=7); ax.legend(fontsize=5)
    ax.tick_params(labelsize=6); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"f8_sensitivity{tag}.pdf")); plt.close(fig)


def main():
    global FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--paper", action="store_true", help="Generate only the eight manuscript panels")
    a = ap.parse_args()
    if a.paper:
        FIG = os.path.join(os.path.dirname(RESULTS), "paper", "figures")
    os.makedirs(FIG, exist_ok=True)
    f = os.path.join(RESULTS, f"e1_edits{a.tag}.parquet")
    if os.path.exists(f):
        E = pd.read_parquet(f, columns=["stream", "doc_id", "chunker", "T", "interval", "toks_frac", "toks", "ins", "dele",
                                        "chunk_tok_mean"])
        E["ops"] = E.ins + E.dele
        f1(E, a.tag)
        if not a.paper:
            f1(E, a.tag, T=200)
        f6(E, a.tag)
        f4(E, a.tag)
    f2(a.tag)
    f3(a.tag)
    f5(a.tag)
    f5b(a.tag)
    f7(a.tag)
    if not a.paper:
        f2(a.tag, r3=True)
        f9(a.tag)
        f8(a.tag)
    print("figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()

"""B2.2 + B2.4 build checks.
(a) diff minimality spot-check: unit-identity SequenceMatcher(autojunk=False) vs `git diff --no-index --minimal` on the
    same unit-per-line rendering, 200 adjacent pairs (pooled over streams): compare deleted+inserted unit counts.
(b) adapters: Steadlith chunk text vs the source span it reports (whitespace-insensitive); KARA chunk text vs the
    canonical span (exact); 20 documents per stream (final + previous version).
Writes results/b22_diff_minimality.json and results/adapters.json.
"""
import json, os, random, re, subprocess, tempfile
from common import RESULTS, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.diff import regions

NS = lambda s: re.sub(r"\s", "", s)


def git_minimal(a_lines, b_lines):
    with tempfile.TemporaryDirectory() as d:
        fa, fb = os.path.join(d, "a"), os.path.join(d, "b")
        open(fa, "w").write("\n".join(a_lines) + "\n"); open(fb, "w").write("\n".join(b_lines) + "\n")
        out = subprocess.run(["git", "diff", "--no-index", "--minimal", "--numstat", fa, fb], capture_output=True, text=True).stdout
        if not out.strip():
            return 0, 0
        add, dele = out.split()[:2]
        return int(dele), int(add)


def main():
    rng = random.Random(0)
    streams = {}
    for s in ("wiki", "k8s", "cpy"):
        try:
            streams[s] = doc_groups(load_stream(s, 60))
        except Exception as e:
            print("skip", s, e)
    # (a)
    pairs = []
    for s, groups in streams.items():
        for d, vers in groups:
            for i in range(1, len(vers)):
                pairs.append((s, d, vers[i - 1]["text"], vers[i]["text"]))
    sample = rng.sample(pairs, min(200, len(pairs)))
    mism, rows = 0, []
    for s, d, a, b in sample:
        ua, ub = Version(a).units, Version(b).units
        regs = regions(ua, ub)
        ours = (sum(r.l_old for r in regs), sum(r.l_new for r in regs))
        la = [u.ihash.hex() for u in ua]; lb = [u.ihash.hex() for u in ub]
        g = git_minimal(la, lb)
        rows.append(dict(stream=s, ours_del=ours[0], ours_ins=ours[1], git_del=g[0], git_ins=g[1]))
        mism += ours != g
    worse = sum(r["ours_del"] + r["ours_ins"] > r["git_del"] + r["git_ins"] for r in rows)
    excess = sum(max(0, r["ours_del"] + r["ours_ins"] - r["git_del"] - r["git_ins"]) for r in rows)
    json.dump(dict(pairs=len(rows), mismatches=mism, ours_larger=worse, excess_units_total=excess,
                   units_changed_total_ours=sum(r["ours_del"] + r["ours_ins"] for r in rows),
                   units_changed_total_git=sum(r["git_del"] + r["git_ins"] for r in rows), rows=rows, meta=run_meta()),
              open(os.path.join(RESULTS, "b22_diff_minimality.json"), "w"), indent=1)
    print("diff minimality:", len(rows), "pairs, mismatches", mism, "ours larger", worse, "excess units", excess)
    # (b)
    res = {}
    for s, groups in streams.items():
        stl = dict(chunks=0, mismatch=0); stls = dict(chunks=0, mismatch=0); kara = dict(chunks=0, mismatch=0)
        for d, vers in groups[:20]:
            for txt in [v["text"] for v in vers[-2:]]:
                v = Version(txt)
                for name, acc in (("STL", stl), ("STL+snap", stls)):
                    for c in make(name).chunk(v):
                        acc["chunks"] += 1
                        acc["mismatch"] += c.start < 0 or NS(c.text) != NS(v.text[c.start:c.end])
                for c in make("KARA").chunk(v):
                    kara["chunks"] += 1; kara["mismatch"] += v.text[c.start:c.end] != c.text
        res[s] = {"STL": stl, "STL+snap": stls, "KARA": kara}
    res["notes"] = ("KARA runs on our segmentation units (exact source text, no token decode); Steadlith keys are over its "
                    "own normalized chunk text; mismatch = chunk text differs (ignoring whitespace) from the reported source span.")
    res["meta"] = run_meta()
    json.dump(res, open(os.path.join(RESULTS, "adapters.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "meta"}, indent=1))


if __name__ == "__main__":
    main()

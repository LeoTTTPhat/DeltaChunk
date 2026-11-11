"""E5a: invalidation radius of late-chunking embeddings. For sampled WIKI edits, both versions are chunked by STRUCT-M
(calibrated) and embedded by LATE (jina-v2-small, whole-document forward pass, span mean pooling) and LATE-ctl (each
chunk alone). For every chunk whose text is unchanged, record cos(old, new) and its distance (in chunks) to the nearest
changed chunk of the new version. LATE-ctl gives cos = 1 for unchanged text by construction (control).
usage: python experiments/e5a.py [--edits 300] [--tag _smoke] [--stream wiki]
"""
import argparse, hashlib, json, os, time
from collections import Counter

import numpy as np
import pandas as pd
from common import RESULTS, calibration, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.late import LateEncoder


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--edits", type=int, default=300); ap.add_argument("--tag", default="")
    ap.add_argument("--stream", default="wiki"); a = ap.parse_args(); t0 = time.time()
    ch = make("STRUCT-M", calibration(a.stream).get("200", {}).get("STRUCT-M"))
    enc = LateEncoder()
    groups = doc_groups(load_stream(a.stream))
    pairs = []
    for d, vers in groups:
        for i in range(1, len(vers)):
            pairs.append((hashlib.sha256(f"e5a:{d}:{i}".encode()).hexdigest(), d, vers[i - 1], vers[i]))
    pairs.sort(key=lambda x: x[0])
    rows, used = [], 0
    for _, d, vo_, vn_ in pairs:
        if used >= a.edits:
            break
        vo, vn = Version(vo_["text"]), Version(vn_["text"])
        co, cn = ch.chunk(vo), ch.chunk(vn)
        ko = Counter(c.text for c in co)
        changed = [j for j, c in enumerate(cn) if ko[c.text] == 0]
        if not changed or len(changed) == len(cn):
            continue
        used += 1
        Eo = enc.embed_spans(vo.text, [(c.start, c.end) for c in co]); En = enc.embed_spans(vn.text, [(c.start, c.end) for c in cn])
        old_idx = {}
        for j, c in enumerate(co):
            old_idx.setdefault(c.text, j)
        Ao = enc.embed_alone([c.text for c in cn if c.text in old_idx])
        Ac = enc.embed_alone([co[old_idx[c.text]].text for c in cn if c.text in old_idx])
        k = 0
        for j, c in enumerate(cn):
            if c.text not in old_idx:
                continue
            dist = min(abs(j - x) for x in changed)
            rows.append(dict(doc_id=str(d), rev_new=int(vn_["rev_idx"]), chunk=j, n_chunks=len(cn), dist=dist,
                             cos_late=float(En[j] @ Eo[old_idx[c.text]]), cos_ctl=float(Ao[k] @ Ac[k]),
                             doc_tok=sum(u.tok for u in vn.units)))
            k += 1
    df = pd.DataFrame(rows); df.to_parquet(os.path.join(RESULTS, f"e5a_late{a.tag}.parquet"), index=False)
    summ = df.groupby(df.dist.clip(upper=10)).agg(n=("cos_late", "size"), cos_late_mean=("cos_late", "mean"),
                                                  cos_late_p05=("cos_late", lambda x: np.percentile(x, 5)), cos_ctl_min=("cos_ctl", "min"))
    json.dump(dict(edits=used, chunks=len(df), by_distance=summ.reset_index().to_dict("records"),
                   meta=run_meta(seconds=round(time.time() - t0, 1), forward_tokens=enc.forward_tokens, device=enc.device)),
              open(os.path.join(RESULTS, f"e5a_late{a.tag}.json"), "w"), indent=1)
    print(summ.round(4))


if __name__ == "__main__":
    main()

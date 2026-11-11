"""B3.1: MPS embedding throughput (bge-small fp16/fp32, batch 64) on 1,000 real REC chunks, fp16-vs-fp32 cosine,
and a CPU (8 threads) reference. Writes results/embed_bench.json."""
import json, os, time
import numpy as np
from common import RESULTS, doc_groups, load_stream, run_meta

from deltachunk.chunkers import Version, make
from deltachunk.embed import Embedder
from deltachunk.text import ntok_many

texts = []
for s in ("k8s", "cpy", "wiki"):
    try:
        for d, vers in doc_groups(load_stream(s, 40)):
            texts += [c.text for c in make("REC").chunk(Version(vers[-1]["text"]))]
    except Exception:
        pass
texts = list(dict.fromkeys(texts))[:1000]
toks = ntok_many(texts)
out = dict(n=len(texts), mean_tokens=float(np.mean(toks)))
res = {}
for dev, fp16 in (("mps", True), ("mps", False), ("cpu", False)):
    if dev == "cpu":
        import torch; torch.set_num_threads(8)
    e = Embedder(device=dev, fp16=fp16, cache=False)
    e.encode(texts[:64])  # warm-up
    t = time.perf_counter(); x = e.encode(texts); dt = time.perf_counter() - t
    res[f"{dev}_{'fp16' if fp16 else 'fp32'}"] = x
    out[f"{dev}_{'fp16' if fp16 else 'fp32'}_chunks_per_s"] = round(len(texts) / dt, 1)
    # online regime: single-chunk calls
    t = time.perf_counter()
    for tt in texts[:30]:
        e.encode([tt])
    out[f"{dev}_{'fp16' if fp16 else 'fp32'}_single_call_ms"] = round((time.perf_counter() - t) / 30 * 1000, 1)
cos = (res["mps_fp16"] * res["mps_fp32"]).sum(1)
cos2 = (res["mps_fp32"] * res["cpu_fp32"]).sum(1)
out.update(fp16_vs_fp32_cos_min=float(cos.min()), fp16_vs_fp32_cos_mean=float(cos.mean()),
           mps_vs_cpu_cos_min=float(cos2.min()), meta=run_meta(threads_cpu=8))
json.dump(out, open(os.path.join(RESULTS, "embed_bench.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "meta"}, indent=1))

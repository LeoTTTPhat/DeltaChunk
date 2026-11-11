"""B1.3: jina-v2-small late-chunking sanity under the pinned stack.
Mean-pooled token states over a single-chunk document must equal the model's standard embedding (cos > 0.999)."""
import json, os, sys, time, platform
import numpy as np, torch, transformers, sentence_transformers
from sentence_transformers import SentenceTransformer
P3 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
docs = [f"Document {i}. " + " ".join(["The river flows through the valley and the town is old."] * (3 + i)) for i in range(20)]
dev = "mps" if torch.backends.mps.is_available() else "cpu"
t0 = time.time(); out = {"model": "jinaai/jina-embeddings-v2-small-en", "transformers": transformers.__version__,
                        "sentence_transformers": sentence_transformers.__version__, "torch": torch.__version__, "device": dev}
try:
    m = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True, device=dev)
    std = m.encode(docs, normalize_embeddings=True, batch_size=8)
    tf = m[0].auto_model; tok = m.tokenizer
    cos = []
    for d, s in zip(docs, std):
        enc = tok(d, return_tensors="pt").to(dev)
        with torch.no_grad(): h = tf(**enc).last_hidden_state[0]
        mask = enc["attention_mask"][0].unsqueeze(-1).float()
        v = (h * mask).sum(0) / mask.sum(); v = torch.nn.functional.normalize(v, dim=0).cpu().numpy()
        cos.append(float(np.dot(v, s)))
    out.update(path="sentence-transformers 5.7.0 + transformers 4.x, trust_remote_code (no patch)", min_cos=min(cos), mean_cos=float(np.mean(cos)),
               ok=min(cos) > 0.999, pooling=str(m[1]) if len(m) > 1 else None, max_seq_length=m.max_seq_length)
except Exception as e:
    out.update(path="FAILED", error=repr(e)[:500], ok=False)
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(P3, "results", "late_stack.json"), "w"), indent=1); print(json.dumps(out, indent=1))

"""Batched sentence-transformers encoding (fp16 on MPS by default) with a content-addressed on-disk vector cache.

Cache key = blake2b(model name, prefix, text); value = float16 vector. One SQLite file per model under data/vec_cache/.
`MEMGUARD_SHRINK=1` (memguard retry) lowers the batch size.
"""
from __future__ import annotations

import hashlib, os, sqlite3, time

import numpy as np

P3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CACHE_DIR = os.environ.get("P3_VEC_CACHE", os.path.join(P3, "data", "vec_cache"))
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def vkey(model: str, text: str, prefix: str = "") -> bytes:
    return hashlib.blake2b(f"{model}\x00{prefix}\x00{text}".encode(), digest_size=16).digest()


class VectorCache:
    def __init__(self, model: str, path: str | None = None):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = path or os.path.join(CACHE_DIR, model.replace("/", "__") + ".sqlite")
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS v (k BLOB PRIMARY KEY, x BLOB)")

    def get_many(self, keys: list[bytes]) -> dict[bytes, np.ndarray]:
        out = {}
        for i in range(0, len(keys), 500):
            part = keys[i:i + 500]
            q = "SELECT k, x FROM v WHERE k IN (%s)" % ",".join("?" * len(part))
            for k, x in self.db.execute(q, part):
                out[bytes(k)] = np.frombuffer(x, dtype=np.float16)
        return out

    def put_many(self, items: list[tuple[bytes, np.ndarray]]):
        self.db.executemany("INSERT OR REPLACE INTO v VALUES (?, ?)", [(k, v.astype(np.float16).tobytes()) for k, v in items])
        self.db.commit()

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM v").fetchone()[0]


class Embedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str | None = None, fp16: bool = True,
                 batch_size: int = 64, cache: bool = True, max_seq_length: int = 512):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.m = SentenceTransformer(model, device=self.device, trust_remote_code=True)
        self.m.max_seq_length = max_seq_length
        self.fp16 = fp16 and self.device != "cpu"
        if self.fp16:
            self.m.half()
        self.batch_size = batch_size // 4 if os.environ.get("MEMGUARD_SHRINK") else batch_size
        self.cache = VectorCache(model + ("" if self.fp16 else "-fp32") + f"-{self.device}") if cache else None
        self.stats = dict(encoded=0, hits=0, calls=0, seconds=0.0)

    def _encode(self, texts: list[str]) -> np.ndarray:
        t = time.perf_counter()
        x = self.m.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        self.stats["seconds"] += time.perf_counter() - t; self.stats["encoded"] += len(texts); self.stats["calls"] += 1
        return x.astype(np.float32)

    def encode(self, texts: list[str], prefix: str = "") -> np.ndarray:
        """Unit-normalized float32 vectors; cached by content."""
        if not texts:
            return np.zeros((0, self.m.get_sentence_embedding_dimension()), np.float32)
        if self.cache is None:
            return self._encode([prefix + t for t in texts])
        keys = [vkey(self.model_name, t, prefix) for t in texts]
        got = self.cache.get_many(list(set(keys)))
        miss = list(dict.fromkeys(k for k in keys if k not in got))
        self.stats["hits"] += len(texts) - sum(1 for k in keys if k not in got)
        if miss:
            idx = {k: i for i, k in enumerate(keys)}
            mt = [prefix + texts[idx[k]] for k in miss]
            for i in range(0, len(mt), 4096):
                vecs = self._encode(mt[i:i + 4096])
                items = list(zip(miss[i:i + 4096], vecs))
                self.cache.put_many(items)
                got.update({k: v.astype(np.float16) for k, v in items})
        return np.stack([got[k].astype(np.float32) for k in keys])

    def encode_queries(self, qs: list[str]) -> np.ndarray:
        return self.encode(qs, prefix=BGE_QUERY_PREFIX if "bge" in self.model_name else ("query: " if "e5" in self.model_name else ""))

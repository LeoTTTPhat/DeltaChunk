"""ANN index maintenance policies on hnswlib 0.8.0 plus exact (numpy BLAS; equal to FAISS IndexFlatIP) and fresh-rebuild references.

TOMB    : deletes are hnswlib mark_deleted tombstones; inserts get new labels.
REUSE   : allow_replace_deleted=True; inserts reuse tombstoned slots (replace_deleted=True).
REBUILD : TOMB, and a full rebuild from the live vectors when tombstones / slots > tau.
Stream inserts use num_threads=1 (deterministic graph); bulk builds use `build_threads`.
"""
from __future__ import annotations

import time

import numpy as np


class HnswPolicy:
    def __init__(self, dim: int, policy: str = "TOMB", tau: float = 0.3, M: int = 16, efC: int = 200, efS: int = 100,
                 capacity: int = 1024, build_threads: int = 8, seed: int = 0, static: int = 0):
        """`static`: number of never-deleted (background) slots excluded from the REBUILD trigger's tombstone share,
        so tau is measured over the churned partition (B5 fix of review finding B4-E6)."""
        import hnswlib
        self.static = static
        self.dim, self.policy, self.tau, self.M, self.efC, self.efS = dim, policy, tau, M, efC, efS
        self.build_threads, self.seed = build_threads, seed
        self.live: dict[int, np.ndarray] = {}        # external id -> vector
        self.label_of: dict[int, int] = {}           # external id -> hnsw label
        self.id_of: dict[int, int] = {}              # hnsw label -> external id
        self.next_label = 0; self.tombs = 0; self.rebuilds = 0; self.rebuild_seconds = 0.0
        self._new(capacity)

    def _new(self, capacity: int):
        import hnswlib
        self.ix = hnswlib.Index(space="ip", dim=self.dim)
        self.ix.init_index(max_elements=max(16, capacity), ef_construction=self.efC, M=self.M, random_seed=self.seed,
                           allow_replace_deleted=self.policy == "REUSE")
        self.ix.set_ef(self.efS)

    def _ensure(self, extra: int):
        need = self.ix.get_current_count() + extra
        if need > self.ix.get_max_elements():
            self.ix.resize_index(max(need, 2 * self.ix.get_max_elements()))

    def insert(self, ids: list[int], vecs: np.ndarray, threads: int = 1):
        if not len(ids):
            return
        vecs = np.asarray(vecs, np.float32)
        if self.policy == "REUSE" and self.tombs:
            k = min(self.tombs, len(ids))
            # replace_deleted reuses tombstoned slots: labels are fresh, slots are recycled
            labels = np.arange(self.next_label, self.next_label + k)
            self.ix.add_items(vecs[:k], labels, num_threads=threads, replace_deleted=True)
            self.tombs -= k
            self._register(ids[:k], labels, vecs[:k]); ids, vecs = ids[k:], vecs[k:]
            if not len(ids):
                return
        self._ensure(len(ids))
        labels = np.arange(self.next_label, self.next_label + len(ids))
        self.ix.add_items(vecs, labels, num_threads=threads)
        self._register(ids, labels, vecs)

    def _register(self, ids, labels, vecs):
        for i, l, v in zip(ids, labels, vecs):
            self.live[int(i)] = v; self.label_of[int(i)] = int(l); self.id_of[int(l)] = int(i)
        self.next_label = int(labels[-1]) + 1

    def delete(self, ids: list[int]):
        for i in ids:
            l = self.label_of.pop(int(i), None)
            if l is None:
                continue
            self.ix.mark_deleted(l); self.id_of.pop(l, None); self.live.pop(int(i), None); self.tombs += 1
        if self.policy == "REBUILD" and self.slots and self.tombs / max(1, self.slots - self.static) > self.tau:
            self.rebuild()

    @property
    def slots(self) -> int:
        return self.ix.get_current_count()

    def rebuild(self):
        t = time.perf_counter()
        ids = list(self.live); vecs = np.stack([self.live[i] for i in ids]) if ids else np.zeros((0, self.dim), np.float32)
        self.label_of.clear(); self.id_of.clear(); self.next_label = 0; self.tombs = 0
        live = dict(self.live); self.live.clear()
        self._new(max(16, int(len(ids) * 1.3)))
        if ids:
            labels = np.arange(len(ids)); self.ix.add_items(vecs, labels, num_threads=self.build_threads)
            self._register(ids, labels, vecs)
        self.rebuild_seconds += time.perf_counter() - t; self.rebuilds += 1

    def search(self, q: np.ndarray, k: int = 10, ef: int | None = None) -> list[list[int]]:
        if ef:
            self.ix.set_ef(ef)
        k = min(k, len(self.live))
        if k == 0:
            return [[] for _ in range(len(q))]
        labels, _ = self.ix.knn_query(np.asarray(q, np.float32), k=k, num_threads=1)
        if ef:
            self.ix.set_ef(self.efS)
        return [[self.id_of[int(l)] for l in row] for row in labels]


def exact_topk(X: np.ndarray, Q: np.ndarray, k: int = 10, block: int = 256) -> np.ndarray:
    """Exact inner-product top-k (numpy/Accelerate BLAS). FAISS IndexFlatIP gives the same result, but loading FAISS's
    OpenMP runtime after torch has run on MPS aborts the process on macOS ("OMP: System error #22"), so the exact
    reference is computed with numpy."""
    X = np.asarray(X, np.float32); Q = np.asarray(Q, np.float32); k = min(k, len(X))
    out = np.empty((len(Q), k), np.int64)
    for i in range(0, len(Q), block):
        S = Q[i:i + block] @ X.T
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(S, part, 1), axis=1, kind="stable")
        out[i:i + block] = np.take_along_axis(part, order, 1)
    return out


def exact_search(ids: list[int], vecs: np.ndarray, q: np.ndarray, k: int = 10) -> list[list[int]]:
    if not len(ids):
        return [[] for _ in range(len(q))]
    I = exact_topk(vecs, q, k)
    return [[ids[j] for j in row] for row in I]


def fresh_reference(index: HnswPolicy, q: np.ndarray, k: int = 10) -> list[list[int]]:
    """A freshly built HNSW (same parameters) over the same live vectors."""
    ref = HnswPolicy(index.dim, "TOMB", M=index.M, efC=index.efC, efS=index.efS, capacity=len(index.live) + 16,
                     build_threads=index.build_threads, seed=index.seed)
    ids = list(index.live)
    if ids:
        ref.insert(ids, np.stack([index.live[i] for i in ids]), threads=index.build_threads)
    return ref.search(q, k)


def recall_at_k(found: list[list[int]], truth: list[list[int]], k: int = 10) -> float:
    s = [len(set(f[:k]) & set(t[:k])) / max(1, min(k, len(t))) for f, t in zip(found, truth)]
    return float(np.mean(s)) if s else 1.0

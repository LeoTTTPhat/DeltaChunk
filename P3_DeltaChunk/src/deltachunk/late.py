"""Late chunking with jina-embeddings-v2-small-en (B1.3 sanity: mean-pooled token states of a single-chunk document
equal the model's standard embedding, cos > 0.999; results/late_stack.json).

`LateEncoder.embed_spans(text, spans)` runs the transformer over the whole document (macro-windows of 8,192 tokens
with 512 tokens of overlap; each token's state is taken from the window in which it has the most context), then
mean-pools the token states inside each char span. `region=(a, b)` restricts the forward pass to text[a:b]
(WINDOW-D re-encoding). `embed_alone` is the LATE-ctl control (each chunk encoded alone by the same model).
"""
from __future__ import annotations

import os

import numpy as np

MODEL = "jinaai/jina-embeddings-v2-small-en"


class LateEncoder:
    def __init__(self, device: str | None = None, max_window: int = 8192, overlap: int = 512):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.st = SentenceTransformer(MODEL, device=self.device, trust_remote_code=True)
        self.tf = self.st[0].auto_model; self.tok = self.st.tokenizer
        self.max_window, self.overlap = max_window, overlap
        self.forward_tokens = 0

    def token_states(self, text: str):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        n = len(ids); W = self.max_window - 2
        states = [None] * n; best = [-1] * n
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        start = 0
        while True:
            end = min(n, start + W)
            x = self.torch.tensor([[cls] + ids[start:end] + [sep]], device=self.device)
            with self.torch.no_grad():
                h = self.tf(input_ids=x, attention_mask=self.torch.ones_like(x)).last_hidden_state[0, 1:-1].float().cpu().numpy()
            self.forward_tokens += x.shape[1]
            for j in range(end - start):
                ctx = min(j, end - start - 1 - j)
                if ctx > best[start + j]:
                    best[start + j] = ctx; states[start + j] = h[j]
            if end >= n:
                break
            start = end - self.overlap
        return np.stack(states) if n else np.zeros((0, 512), np.float32), offs

    def embed_spans(self, text: str, spans: list[tuple[int, int]], region: tuple[int, int] | None = None) -> np.ndarray:
        base = 0
        if region is not None:
            base = region[0]; text = text[region[0]:region[1]]
        H, offs = self.token_states(text)
        out = []
        for a, b in spans:
            a -= base; b -= base
            sel = [i for i, (x, y) in enumerate(offs) if x >= a and y <= b and y > x]
            v = H[sel].mean(0) if sel else np.zeros(H.shape[1] if len(H) else 512, np.float32)
            out.append(v / (np.linalg.norm(v) + 1e-12))
        return np.stack(out) if out else np.zeros((0, 512), np.float32)

    def embed_alone(self, texts: list[str]) -> np.ndarray:
        return self.st.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

import numpy as np
from deltachunk.late import LateEncoder


def test_late_single_chunk_equals_standard_and_windows():
    enc = LateEncoder()
    docs = ["The river flows through the old town. " * (i + 2) for i in range(5)]
    for d in docs:
        a = enc.embed_spans(d, [(0, len(d))])[0]; b = enc.embed_alone([d])[0]
        assert float(a @ b) > 0.995   # span pooling excludes [CLS]/[SEP]; with them it is > 0.999 (B1.3)
    long = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(200))
    full = enc.embed_spans(long, [(0, 40), (len(long) - 40, len(long))])
    enc.max_window, enc.overlap = 128, 32
    win = enc.embed_spans(long, [(0, 40), (len(long) - 40, len(long))])
    assert full.shape == win.shape and np.all((full * win).sum(1) > 0.8)

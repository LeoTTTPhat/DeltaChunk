"""Chunk / Version types and the stateless chunkers of experiments.md (Tier A).

Every chunker receives a Version (units + canonical text) and returns Chunks with char spans into `version.text`.
Chunk keys are H(model, chunk text [, salt]); DOC/LATE/CTX salt the key with the document hash (Prop. 5).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

from deltachunk.text import Unit, canon, key, ntok, ntok_many, segment, token_offsets, unit_offsets

T_MAX = 480


@dataclass
class Chunk:
    text: str
    start: int            # char span in version.text
    end: int
    u0: int = -1          # unit range [u0, u1) when unit-aligned
    u1: int = -1
    salt: str = ""
    pieces: int = 1       # guard pieces this chunk came from (SentCDC guard accounting)

    @property
    def key(self) -> str:
        return key(self.text, self.salt)


class Version:
    """One document version: units, canonical text, unit char offsets."""

    def __init__(self, raw: str | None = None, units: list[Unit] | None = None):
        self.units = units if units is not None else segment(raw)
        self.text = canon(self.units)
        self.uoff = unit_offsets(self.units)
        self._h = None

    @property
    def doc_hash(self) -> str:
        if self._h is None:
            self._h = hashlib.blake2b(self.text.encode(), digest_size=12).hexdigest()
        return self._h

    def unit_chunk(self, s: int, e: int, **kw) -> Chunk:
        a, b = self.uoff[s][0], self.uoff[e - 1][1]
        return Chunk(self.text[a:b], a, b, s, e, **kw)

    def units_in(self, a: int, b: int) -> tuple[int, int]:
        """Unit index range whose text lies inside char range [a, b)."""
        s = next((i for i, (x, y) in enumerate(self.uoff) if x >= a), len(self.uoff))
        e = s
        while e < len(self.uoff) and self.uoff[e][1] <= b:
            e += 1
        return s, e


def span_chunk(text: str, a: int, b: int, salt: str = "") -> Chunk | None:
    """Chunk for text[a:b] with surrounding whitespace trimmed; None if empty."""
    seg = text[a:b]
    l = len(seg) - len(seg.lstrip()); r = len(seg.rstrip())
    if r <= l:
        return None
    return Chunk(seg[l:r], a + l, a + r, salt=salt)


def locate(text: str, pieces: list[str], base: int = 0, overlap: bool = False) -> list[Chunk]:
    """Map chunk strings that occur in order in `text` to spans. Without overlap the search for piece k starts at the
    END of piece k-1 (B5 fix of review finding B4-LOC: restarting at the previous start + 1 mapped short pieces that
    also occur inside the previous chunk to wrong, overlapping spans). With overlap (REC-O) a piece may start inside
    its predecessor, so the search starts at the predecessor's start + 1 (LangChain's add_start_index rule uses an
    approximate char offset). A piece that cannot be found in order raises (no unanchored fallback)."""
    out, pos = [], 0
    for p in pieces:
        i = text.find(p, pos)
        if i < 0:
            raise ValueError(f"locate: piece not found in order at {pos}: {p[:60]!r}")
        out.append(Chunk(p, base + i, base + i + len(p))); pos = i + 1 if overlap else i + len(p)
    return out


def check_layout(text: str, chunks: list[Chunk], where: str = "") -> None:
    """Invariant for span-defined layouts: ordered, non-overlapping, every chunk's text equals its span."""
    prev = 0
    for c in chunks:
        if c.start < 0:
            continue
        if c.start < prev or text[c.start:c.end] != c.text:
            raise AssertionError(f"layout invariant violated {where}: span ({c.start},{c.end}) prev_end {prev}")
        prev = c.end


def window_split(text: str, a: int, b: int, max_tok: int = T_MAX) -> list[Chunk]:
    """Split text[a:b] into token windows of <= max_tok tokens (by tokenizer offsets)."""
    sub = text[a:b]
    offs = token_offsets(sub)
    out, k = [], 0
    while k < len(offs):
        e = min(k + max_tok, len(offs))
        x = offs[k][0]; y = offs[e][0] if e < len(offs) else len(sub)
        c = span_chunk(text, a + x, a + y)
        if c:
            out.append(c)
        k = e
    return out


def enforce_max(v_text: str, chunks: list[Chunk], max_tok: int = T_MAX) -> list[Chunk]:
    out = []
    toks = ntok_many([c.text for c in chunks])
    for c, t in zip(chunks, toks):
        if t > max_tok and c.start >= 0:
            out.extend(window_split(v_text, c.start, c.end, max_tok))
        else:
            out.append(c)
    return out


class Chunker:
    name = "base"
    stateful = False

    def chunk(self, v: Version) -> list[Chunk]:
        raise NotImplementedError

    def params(self) -> dict:
        return {k: getattr(self, k) for k in vars(self) if not k.startswith("_")}


# ----------------------------------------------------------------------------- offset-defined
class Fixed(Chunker):
    """FIX / FIXO: windows of T tokens (stride T - overlap), cut by tokenizer offsets on the canonical text."""

    def __init__(self, T: int = 200, overlap: int = 0, name: str | None = None):
        self.T, self.overlap = T, overlap
        self.name = name or ("FIX" if overlap == 0 else "FIXO")

    def chunk(self, v: Version) -> list[Chunk]:
        offs = token_offsets(v.text)
        out, k, n = [], 0, len(offs)
        while k < n:
            e = min(k + self.T, n)
            c = span_chunk(v.text, offs[k][0], offs[e][0] if e < n else len(v.text))
            if c:
                out.append(c)
            if e == n:
                break
            k = e - self.overlap
        return out


class SentWin(Chunker):
    """SENTWIN: one unit per vector (context expanded at query time)."""
    name = "SENTWIN"

    def chunk(self, v: Version) -> list[Chunk]:
        return [v.unit_chunk(i, i + 1) for i in range(len(v.units)) if v.units[i].text.strip()]


# ----------------------------------------------------------------------------- greedy packing
class Recursive(Chunker):
    """REC: LangChain RecursiveCharacterTextSplitter (default separators, keep_separator=True, strip_whitespace=True),
    length = bge-small token count (identical to from_huggingface_tokenizer), chunk_overlap as given (REC: 0)."""

    def __init__(self, T: int = 200, overlap: int = 0, name: str | None = None):
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        self.T, self.overlap = T, overlap
        self.name = name or ("REC" if overlap == 0 else "REC-O")
        self._sp = RecursiveCharacterTextSplitter(chunk_size=T, chunk_overlap=overlap, length_function=ntok)

    def split_text(self, text: str) -> list[str]:
        import logging
        logging.getLogger("langchain_text_splitters.base").setLevel(logging.ERROR)
        return self._sp.split_text(text)

    def chunk(self, v: Version) -> list[Chunk]:
        return enforce_max(v.text, locate(v.text, self.split_text(v.text), overlap=self.overlap > 0))

    def chunk_range(self, v: Version, a: int, b: int) -> list[Chunk]:
        return enforce_max(v.text, locate(v.text[a:b], self.split_text(v.text[a:b]), base=a, overlap=self.overlap > 0))


def _greedy_units(v: Version, s: int, e: int, T: int) -> list[tuple[int, int]]:
    out, cur, tok = [], s, 0
    for i in range(s, e):
        t = v.units[i].tok
        if i > cur and tok + t > T:
            out.append((cur, i)); cur, tok = i, 0
        tok += t
    if e > cur:
        out.append((cur, e))
    return out


def _blocks(v: Version, s: int = 0, e: int | None = None) -> list[tuple[int, int]]:
    e = len(v.units) if e is None else e
    out, cur = [], s
    for i in range(s, e):
        if v.units[i].par_end or i == e - 1:
            out.append((cur, i + 1)); cur = i + 1
    return out


class StructMerge(Chunker):
    """STRUCT-M: hard flush at headings; whole paragraphs packed greedily up to T; long paragraphs split at units."""
    name = "STRUCT-M"

    def __init__(self, T: int = 200):
        self.T = T

    def ranges(self, v: Version, s: int = 0, e: int | None = None) -> list[tuple[int, int]]:
        out, cur, tok = [], None, 0
        for bs, be in _blocks(v, s, e):
            bt = sum(u.tok for u in v.units[bs:be])
            head = v.units[bs].is_heading
            if cur is not None and (head or tok + bt > self.T):
                out.append((cur, bs)); cur, tok = None, 0
            if cur is None and bt > self.T and not head:
                parts = _greedy_units(v, bs, be, self.T)
                out.extend(parts[:-1]); cur = parts[-1][0]; tok = sum(u.tok for u in v.units[cur:be]); continue
            if cur is None:
                cur = bs
            tok += bt
        if cur is not None:
            out.append((cur, len(v.units) if e is None else e))
        return out

    def chunk(self, v: Version) -> list[Chunk]:
        return enforce_max(v.text, [v.unit_chunk(a, b) for a, b in self.ranges(v)])

    def chunk_range(self, v: Version, a: int, b: int) -> list[Chunk]:
        s, e = v.units_in(a, b)
        return enforce_max(v.text, [v.unit_chunk(x, y) for x, y in self.ranges(v, s, e)]) if e > s else []


class StructNoMerge(Chunker):
    """STRUCT-NM: one chunk per paragraph/block (heading attached to the next block); blocks > T_max split at units."""
    name = "STRUCT-NM"

    def __init__(self, T_max: int = T_MAX):
        self.T_max = T_max

    def chunk(self, v: Version) -> list[Chunk]:
        out, pending = [], None
        for bs, be in _blocks(v):
            if v.units[bs].is_heading and be - bs == 1 and be < len(v.units):
                pending = bs if pending is None else pending; continue
            s = pending if pending is not None else bs; pending = None
            for a, b in _greedy_units(v, s, be, self.T_max):
                out.append(v.unit_chunk(a, b))
        if pending is not None:
            out.append(v.unit_chunk(pending, len(v.units)))
        return enforce_max(v.text, out)


# ----------------------------------------------------------------------------- local content-defined (MAXP)
def unit_hash(u: Unit) -> float:
    return int.from_bytes(u.ihash, "big") / 2 ** 64


def maxp_cuts(h: list[float], r: int, anchor_after: list[bool] | None = None) -> list[int]:
    """Cut after i iff h[i] < h[j] for every j != i in [i-r, i+r] inside the document, or anchor_after[i].
    Only i < n-1 can be a cut (the document end is always a boundary)."""
    n = len(h); cuts = []
    for i in range(n - 1):
        if anchor_after is not None and anchor_after[i]:
            cuts.append(i); continue
        hi = h[i]; ok = True
        for j in range(max(0, i - r), min(n, i + r + 1)):
            if j != i and not hi < h[j]:
                ok = False; break
        if ok:
            cuts.append(i)
    return cuts


def spans_from_cuts(n: int, cuts: list[int]) -> list[tuple[int, int]]:
    out, s = [], 0
    for c in cuts:
        out.append((s, c + 1)); s = c + 1
    if s < n:
        out.append((s, n))
    return out


class SentCDC(Chunker):
    """MAXP over sentence-unit identity hashes; paragraph bias λ; heading anchors; content-local max split."""
    name = "SentCDC"

    def __init__(self, r: int = 3, lam: float = 0.5, T_max: int = T_MAX):
        self.r, self.lam, self.T_max = r, lam, T_max

    def hashes(self, v: Version) -> list[float]:
        return [unit_hash(u) * (self.lam if u.par_end else 1.0) for u in v.units]

    def chunk(self, v: Version) -> list[Chunk]:
        n = len(v.units)
        if n == 0:
            return []
        h = self.hashes(v)
        anchors = [i + 1 < n and v.units[i + 1].is_heading for i in range(n)]
        out = []
        for s, e in spans_from_cuts(n, maxp_cuts(h, self.r, anchors)):
            parts = self._split_max(v, h, s, e)
            for a, b in parts:
                out.append(v.unit_chunk(a, b, pieces=len(parts)))
        return out

    def _split_max(self, v, h, s, e):
        if e - s <= 1 or sum(u.tok for u in v.units[s:e]) <= self.T_max:
            return [(s, e)]
        k = min(range(s, e - 1), key=lambda i: (h[i], i))
        return self._split_max(v, h, s, k + 1) + self._split_max(v, h, k + 1, e)


class GuardedCDC(Chunker):
    """GCDC: sentence-unit CDC with FastCDC-style guards: no cut before T_min tokens (cut-point skipping), candidate
    cut after unit i iff unit hash < p, forced cut before exceeding T_max; heading anchors."""
    name = "GCDC"

    def __init__(self, T_min: int = 100, p: float = 0.3, T_max: int = T_MAX):
        self.T_min, self.p, self.T_max = T_min, p, T_max

    def chunk(self, v: Version) -> list[Chunk]:
        n = len(v.units); out, s, tok = [], 0, 0
        for i in range(n):
            tok += v.units[i].tok
            last = i == n - 1
            nxt = 0 if last else v.units[i + 1].tok
            cut = last or v.units[i + 1].is_heading or tok + nxt > self.T_max or (tok >= self.T_min and unit_hash(v.units[i]) < self.p)
            if cut:
                out.append(v.unit_chunk(s, i + 1)); s, tok = i + 1, 0
        return out


class ByteCDC(Chunker):
    """BCDC: byte-level FastCDC (fastcdc 1.7.0) over the UTF-8 canonical text; cuts snapped to the next whitespace."""
    name = "BCDC"

    def __init__(self, avg: int = 900, min_size: int | None = None, max_size: int | None = None):
        self.avg = avg; self.min_size = min_size or avg // 4; self.max_size = max_size or avg * 4

    def chunk(self, v: Version) -> list[Chunk]:
        from fastcdc import fastcdc
        data = v.text.encode("utf-8")
        if not data:
            return []
        cuts, last = [], 0
        for c in fastcdc(data, min_size=self.min_size, avg_size=self.avg, max_size=self.max_size):
            e = c.offset + c.length
            while e < len(data) and data[e:e + 1] not in (b" ", b"\n", b"\t"):
                e += 1
            if e > last:
                cuts.append(e); last = e
        if not cuts or cuts[-1] < len(data):
            cuts.append(len(data))
        out, a = [], 0
        for e in cuts:
            seg = data[a:e].decode("utf-8")
            ca = len(data[:a].decode("utf-8"))
            ch = span_chunk(v.text, ca, ca + len(seg))
            if ch:
                out.append(ch)
            a = e
        return enforce_max(v.text, out)


class DocLevel(Chunker):
    """DOC: document-level replace. REC chunks whose keys include the document hash (every chunk re-embedded)."""
    name = "DOC"

    def __init__(self, base: Chunker | None = None):
        self._base = base or Recursive(200)

    def chunk(self, v: Version) -> list[Chunk]:
        cs = self._base.chunk(v)
        for c in cs:
            c.salt = v.doc_hash
        return cs

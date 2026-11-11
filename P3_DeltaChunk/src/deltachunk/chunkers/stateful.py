"""History-dependent chunkers (DIFF[base], KARA) and the Steadlith adapter.

Stateful chunkers expose `update(v_new, state) -> (chunks, state)`; `state=None` means a fresh chunking.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from deltachunk.chunkers.base import Chunk, Chunker, Version, check_layout, enforce_max, span_chunk, T_MAX
from deltachunk.text import ntok, ntok_many, token_offsets


# ----------------------------------------------------------------------------- DIFF[base]
def equal_char_blocks(old: Version, new: Version):
    """Equal unit blocks as char ranges: [(old_a, old_b, shift)], plus the set of old insertion boundaries (chars)."""
    a = [u.ihash for u in old.units]; b = [u.ihash for u in new.units]
    sm = SequenceMatcher(None, a, b, autojunk=False)
    blocks, inserts = [], set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            oa, ob = old.uoff[i1][0], old.uoff[i2 - 1][1]
            blocks.append((oa, ob, new.uoff[j1][0] - oa))
        elif tag == "insert":
            inserts.add(old.uoff[i1][0] if i1 < len(old.uoff) else len(old.text))
    return blocks, inserts


class DiffAnchored(Chunker):
    """DIFF[base]: keep previous chunks that lie inside unchanged text, re-chunk each maximal dirty run with `base`;
    an undersized first/last re-chunked piece (< T_small tokens) is merged once with its clean neighbor (<= T_MAX)."""
    stateful = True

    def __init__(self, base, T_small: int = 50):
        self.base = base; self.T_small = T_small
        self.name = f"DIFF[{base.name}]"

    def chunk(self, v: Version) -> list[Chunk]:
        return self.base.chunk(v)

    def update(self, v: Version, state):
        if state is None:
            cs = self.base.chunk(v)
            return cs, (v, cs)
        old_v, old_cs = state
        if old_v.text == v.text:
            return old_cs, state
        blocks, inserts = equal_char_blocks(old_v, v)
        # classify old chunks
        clean: list[Chunk | None] = []
        bi = 0
        for c in old_cs:
            m = None
            if c.start >= 0 and c.start not in inserts:
                for (oa, ob, sh) in blocks:
                    if oa <= c.start and c.end <= ob:
                        m = Chunk(c.text, c.start + sh, c.end + sh); break
            clean.append(m)
        # walk: clean chunks kept; dirty runs re-chunked over the new text between clean neighbours
        out: list[Chunk] = []
        i, n = 0, len(old_cs)
        prev_end = 0
        while i < n:
            if clean[i] is not None:
                out.append(clean[i]); prev_end = clean[i].end; i += 1; continue
            j = i
            while j < n and clean[j] is None:
                j += 1
            nxt_start = clean[j].start if j < n else len(v.text)
            pieces = self.base.chunk_range(v, prev_end, nxt_start) if nxt_start > prev_end else []
            pieces = [p for p in pieces if p.text]
            # merge undersized boundary pieces with clean neighbours once
            if pieces and out and ntok(pieces[0].text) < self.T_small:
                m = span_chunk(v.text, out[-1].start, pieces[0].end)
                if m and ntok(m.text) <= T_MAX:
                    out[-1] = m; pieces = pieces[1:]
            if pieces and j < n and ntok(pieces[-1].text) < self.T_small:
                m = span_chunk(v.text, pieces[-1].start, clean[j].end)
                if m and ntok(m.text) <= T_MAX:
                    clean[j] = m; pieces = pieces[:-1]
            out.extend(pieces)
            if out:
                prev_end = out[-1].end
            i = j
        # text appended after the last old chunk / when all old chunks were clean but new text exists between them
        out = self._fill_gaps(v, out)
        check_layout(v.text, out, self.name)
        return out, (v, out)

    def _fill_gaps(self, v: Version, cs: list[Chunk]) -> list[Chunk]:
        """Re-chunk any non-whitespace new text not covered by a chunk (e.g. an insertion between two clean chunks)."""
        res, pos = [], 0
        for c in sorted(cs, key=lambda c: c.start):
            if v.text[pos:c.start].strip():
                res.extend(self.base.chunk_range(v, pos, c.start))
            res.append(c); pos = max(pos, c.end)
        if v.text[pos:].strip():
            res.extend(self.base.chunk_range(v, pos, len(v.text)))
        return res


# ----------------------------------------------------------------------------- KARA (kara-toolkit 0.3.2)
class KaraAdapter(Chunker):
    """KARA with sentence units: kara.KARAUpdater over a BaseDocumentChunker whose units are our segmentation units
    (unit text + separator; units above `chunk_size` pre-split into token windows), unit length = bge tokens.
    Chunk text = concatenated units (exact source text, no decode step); key over the stripped text."""
    stateful = True
    name = "KARA"

    def __init__(self, chunk_size: int = 260):
        import kara
        self.chunk_size = chunk_size
        outer = self

        class _UnitChunker(kara.BaseDocumentChunker):
            def __init__(self):
                super().__init__(chunk_size=chunk_size, overlap=0)
                self.units_for = {}

            def _split_to_units(self, text):
                return self.units_for[text]

            def create_chunks(self, text):
                return self._merge_units_greedy(self._split_to_units(text), self.chunk_size)

            def unit_length(self, unit):
                return ntok(unit)

            def serialize_units(self, units):
                # KARA's reuse objective hashes this; use the stripped text = the counted key's input (review B4-m KARA)
                return "".join(units).strip().encode("utf-8")

        self._ch = _UnitChunker()
        self._upd = kara.KARAUpdater(self._ch)

    def _units(self, v: Version) -> list[str]:
        """Unit strings (text + separator) covering the canonical text; any unit above chunk_size tokens is split by
        tokenizer offsets, shrinking the window until every piece (re-tokenized on its own) fits."""
        out = []
        for u in v.units:
            s = u.text + u.sep
            if ntok(s) <= self.chunk_size:
                out.append(s); continue
            out.extend(self._split(s, self.chunk_size))
        return out

    def _split(self, s: str, window: int) -> list[str]:
        offs = token_offsets(s); pieces, k = [], 0
        while k < len(offs):
            e = min(k + window, len(offs))
            a = 0 if k == 0 else offs[k][0]; b = offs[e][0] if e < len(offs) else len(s)
            pieces.append(s[a:b]); k = e
        res = []
        for p in pieces:
            if ntok(p) > self.chunk_size and window > 8:
                res.extend(self._split(p, max(8, int(window * 0.8))))
            else:
                res.append(p)
        return res

    def _to_chunks(self, v: Version, coll) -> list[Chunk]:
        out, pos = [], 0
        for cd in coll.chunks:
            txt = "".join(cd.splits)
            c = span_chunk(v.text, pos, pos + len(txt)); pos += len(txt)
            if c:
                out.append(c)
        assert pos == len(v.text), "KARA units must cover the canonical text"
        return out

    def chunk(self, v: Version) -> list[Chunk]:
        return self.update(v, None)[0]

    def update(self, v: Version, state):
        self._ch.units_for = {v.text: self._units(v)}
        if state is None:
            res = self._upd.create_collection([v.text])
        else:
            res = self._upd.update_collection(state, [v.text])
        coll = res.new_chunked_doc
        return enforce_max(v.text, self._to_chunks(v, coll)), coll


# ----------------------------------------------------------------------------- Steadlith (pinned 74c5b55b)
class SteadlithAdapter(Chunker):
    """STL / STL+snap: steadlith.chunk.chunk_text (`cdc-rabin` / `cdc-rabin+snap`) on the canonical text with the
    bge-small token count per word; keys over Steadlith's chunk text (its own normalized rendering)."""

    def __init__(self, min_tokens: int = 120, max_tokens: int = 320, snap: bool = False, name: str | None = None):
        self.min_tokens, self.max_tokens, self.snap = min_tokens, max_tokens, snap
        self.name = name or ("STL+snap" if snap else "STL")

    fallbacks = 0   # versions whose over-long words had to be split (Steadlith rejects a word above max_tokens)

    def _split_long_words(self, text: str) -> str:
        import re
        lim, piece = self.max_tokens // 2, max(1, self.max_tokens // 4)

        def fix(m):
            w = m.group(0)
            if ntok(w) <= lim:
                return w
            offs = token_offsets(w); out, k = [], 0
            while k < len(offs):
                e = min(k + piece, len(offs)); a = offs[k][0]; b = offs[e][0] if e < len(offs) else len(w)
                out.append(w[a:b]); k = e
            return " ".join(out)
        return re.sub(r"\S+", fix, text)

    def chunk(self, v: Version) -> list[Chunk]:
        from steadlith.chunk import CDCParams, chunk_text
        if not v.text.strip():
            return []
        p = CDCParams(min_tokens=self.min_tokens, max_tokens=self.max_tokens, snap_to_boundaries=self.snap)
        text = v.text
        try:
            res = chunk_text(text, p, token_counter=ntok, tokenizer_id="bge-small-en-v1.5")
        except ValueError:
            # Same operational limit Bian et al. report: a whitespace-delimited word longer than max_tokens is rejected.
            # We split such words into token pieces separated by spaces (counted in `fallbacks`).
            text = self._split_long_words(text); SteadlithAdapter.fallbacks += 1
            res = chunk_text(text, p, token_counter=ntok, tokenizer_id="bge-small-en-v1.5")
        out = []
        for c in res.chunks:
            md = c.metadata
            out.append(Chunk(c.page_content, md.get("start_offset", -1), md.get("end_offset", -1)))
        return enforce_max(text, out)

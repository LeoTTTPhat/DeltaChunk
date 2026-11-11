"""Normalization, unit segmentation with identity flags, canonical rendering, token counting.

A document version is turned into a list of Units. Every chunker works on the units or on the canonical rendering
`canon(units)`, so whitespace-only and reflow-only edits change no chunk key for any chunker (design.md §2).

Unit identity = (normalized text, par_end, is_heading, sep_class).
"""
from __future__ import annotations

import hashlib, os, re, unicodedata
from dataclasses import dataclass
from functools import lru_cache

T_UNIT_MAX = 480          # units above this many tokens are split into token windows
GLUE_MIN = 8              # prose sentences shorter than this (tokens) are glued to the next sentence
TOKENIZER_NAME = "BAAI/bge-small-en-v1.5"


# ------------------------------------------------------------------ tokenizer
class _Tok:
    """Thin wrapper over the `tokenizers` fast tokenizer of bge-small (avoids importing transformers/torch)."""

    def __init__(self, name: str):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        self.backend_tokenizer = Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))
        self.backend_tokenizer.no_truncation(); self.backend_tokenizer.no_padding()

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        e = self.backend_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return {"input_ids": e.ids, "offset_mapping": e.offsets}


@lru_cache(maxsize=1)
def tokenizer() -> _Tok:
    return _Tok(TOKENIZER_NAME)


_TOK_CACHE: dict[str, int] = {}


def ntok(text: str) -> int:
    """Number of bge-small tokens (no special tokens). Cached by text."""
    v = _TOK_CACHE.get(text)
    if v is None:
        v = len(tokenizer().backend_tokenizer.encode(text, add_special_tokens=False).ids) if text else 0
        if len(_TOK_CACHE) > 2_000_000:
            _TOK_CACHE.clear()
        _TOK_CACHE[text] = v
    return v


def ntok_many(texts: list[str]) -> list[int]:
    miss = [t for t in set(texts) if t not in _TOK_CACHE and t]
    if miss:
        enc = tokenizer().backend_tokenizer.encode_batch(miss, add_special_tokens=False)
        for t, e in zip(miss, enc):
            _TOK_CACHE[t] = len(e.ids)
    return [_TOK_CACHE.get(t, 0) if t else 0 for t in texts]


def token_offsets(text: str) -> list[tuple[int, int]]:
    return tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]


# ------------------------------------------------------------------ normalization
_WS = re.compile(r"[ \t  -​　]+")


def normalize_line(line: str) -> str:
    return _WS.sub(" ", line).rstrip()


def normalize(text: str) -> str:
    """NFC, CRLF -> LF, collapse runs of horizontal whitespace (leading indentation kept as spaces), strip trailing."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    out = []
    for line in text.split("\n"):
        line = line.expandtabs(4)
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        out.append(" " * indent + normalize_line(stripped))
    return "\n".join(out)


# ------------------------------------------------------------------ units
SEP_SPACE, SEP_NL, SEP_BLANK, SEP_END = " ", "\n", "\n\n", ""
SEP_CLASS = {SEP_SPACE: 1, SEP_NL: 2, SEP_BLANK: 3, SEP_END: 0}


@dataclass(frozen=True)
class Unit:
    text: str          # normalized unit text (prose reflowed; code/table lines verbatim after normalization)
    kind: str          # prose | heading | list | code | table | yaml | directive | window
    block: int         # block (paragraph) index
    par_end: bool      # last unit of its block
    is_heading: bool
    sep: str           # separator rendered after this unit in canon()
    tok: int

    @property
    def identity(self) -> tuple:
        return (self.text, self.par_end, self.is_heading, SEP_CLASS[self.sep])

    @property
    def ihash(self) -> bytes:
        return hashlib.blake2b(repr(self.identity).encode(), digest_size=8).digest()


def canon(units: list[Unit]) -> str:
    return "".join(u.text + u.sep for u in units)


def unit_offsets(units: list[Unit]) -> list[tuple[int, int]]:
    """Char span of each unit's text inside canon(units)."""
    pos, out = 0, []
    for u in units:
        out.append((pos, pos + len(u.text))); pos += len(u.text) + len(u.sep)
    return out


# sentence splitting -----------------------------------------------------------
_ABBR = {"e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "no", "fig", "approx", "inc",
         "ltd", "co", "corp", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
         "u.s", "u.k", "a.m", "p.m", "cf", "al", "ca", "vol", "pp", "ed", "eds", "mt", "ft", "gen", "gov", "lt", "col"}
_SENT_END = re.compile(r"""[.!?]["'”’)\]]*\s+(?=["'“‘(\[]?[A-Z0-9])""")


def split_sentences(par: str) -> list[str]:
    out, start = [], 0
    for m in _SENT_END.finditer(par):
        end = m.start() + 1
        while end < len(par) and par[end] in "\"'”’)]":
            end += 1
        cand = par[start:end]
        word = re.search(r"(\S+)$", cand[:-1] if cand.endswith(".") else cand)
        w = (word.group(1).lower().rstrip(".") if word else "")
        if cand.endswith(".") and (w in _ABBR or (len(w) == 1 and w.isalpha()) or re.fullmatch(r"([a-z]\.)+[a-z]", w or "")):
            continue
        if par[start:end].strip():
            out.append(par[start:end].strip())
        start = m.end()
    if par[start:].strip():
        out.append(par[start:].strip())
    return out


# block parsing -------------------------------------------------------------------
_MD_HEAD = re.compile(r"^#{1,6}\s+\S")
_LIST = re.compile(r"^(\s*)([-*+]|\d+[.)]|#\.)\s+\S")
_FENCE = re.compile(r"^\s*(```|~~~)")
_RST_UNDER = re.compile(r"^([=\-~^\"'`#*+:.<>_])\1{2,}\s*$")
_DIRECTIVE = re.compile(r"^\s*\.\. [\w:-]+::|^\s*\.\. _[^:]+:|^\s*\.\. \|")
_TABLE = re.compile(r"^\s*(\|.*\||\+[-=+]+\+)\s*$")
_CODE_DIRECTIVE = re.compile(r"^\s*\.\. (code-block|code|sourcecode|testcode|testsetup|testoutput|doctest|literalinclude|productionlist|highlight)::")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def parse_blocks(text: str) -> list[tuple[str, list[str]]]:
    """Split normalized text into blocks (kind, lines). Kinds: prose, heading, list, code, table, yaml, directive."""
    lines = normalize(text).split("\n")
    blocks: list[tuple[str, list[str]]] = []
    i, n = 0, len(lines)
    # front matter
    if n and lines[0].strip() == "---":
        j = 1
        while j < n and lines[j].strip() != "---":
            j += 1
        if j < n:
            blocks.append(("yaml", lines[: j + 1])); i = j + 1
    literal_next = False  # rst: an indented block after '::' or a code directive is code
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1; continue
        if _FENCE.match(line):
            fence = _FENCE.match(line).group(1); j = i + 1
            while j < n and not lines[j].strip().startswith(fence):
                j += 1
            blocks.append(("code", lines[i: min(j + 1, n)])); i = j + 1; literal_next = False; continue
        if literal_next and _indent(line) > 0:
            base = _indent(line); j = i
            while j < n and (not lines[j].strip() or _indent(lines[j]) >= base):
                j += 1
            while j > i and not lines[j - 1].strip():
                j -= 1
            blocks.append(("code", [l for l in lines[i:j]])); i = j; literal_next = False; continue
        literal_next = False
        if line.lstrip().startswith(">>> "):
            j = i
            while j < n and lines[j].strip():
                j += 1
            blocks.append(("code", lines[i:j])); i = j; continue
        # rst heading: (overline) title underline
        if i + 1 < n and _RST_UNDER.match(lines[i + 1]) and len(lines[i + 1].strip()) >= len(line.strip()) and not _RST_UNDER.match(line):
            blocks.append(("heading", [line.strip(), lines[i + 1].strip()])); i += 2; continue
        if _RST_UNDER.match(line) and i + 2 < n and _RST_UNDER.match(lines[i + 2]) and lines[i + 1].strip():
            blocks.append(("heading", [line.strip(), lines[i + 1].strip(), lines[i + 2].strip()])); i += 3; continue
        if _MD_HEAD.match(line):
            blocks.append(("heading", [line.strip()])); i += 1; continue
        if _TABLE.match(line):
            j = i
            while j < n and _TABLE.match(lines[j]):
                j += 1
            blocks.append(("table", lines[i:j])); i = j; continue
        if _DIRECTIVE.match(line):
            if _CODE_DIRECTIVE.match(line):
                literal_next = True
            # directive line plus its option lines (":opt: value") form one directive block
            j = i + 1
            while j < n and lines[j].strip().startswith(":") and _indent(lines[j]) > _indent(line):
                j += 1
            blocks.append(("directive", lines[i:j])); i = j; continue
        if _LIST.match(line):
            items, j = [], i
            while j < n and lines[j].strip():
                if _LIST.match(lines[j]) and _indent(lines[j]) <= _indent(line) + 1:
                    items.append([lines[j].strip()])
                elif _LIST.match(lines[j]):
                    items.append([lines[j].strip()])
                else:
                    items[-1].append(lines[j].strip())
                j += 1
            blocks.append(("list", [" ".join(it) for it in items])); i = j
            if blocks[-1][1][-1].endswith("::"):
                literal_next = True
            continue
        # prose paragraph: consecutive non-blank lines with the same indentation and no other block start
        j = i + 1; ind = _indent(line)
        while j < n and lines[j].strip() and _indent(lines[j]) == ind and not (
                _FENCE.match(lines[j]) or _MD_HEAD.match(lines[j]) or _LIST.match(lines[j]) or _TABLE.match(lines[j])
                or _DIRECTIVE.match(lines[j]) or (j + 1 < n and _RST_UNDER.match(lines[j + 1]) and not _RST_UNDER.match(lines[j]))):
            j += 1
        par = " ".join(l.strip() for l in lines[i:j])
        blocks.append(("prose", [par])); i = j
        if par.endswith("::"):
            literal_next = True
    return blocks


def _windows(text: str, max_tok: int) -> list[str]:
    offs = token_offsets(text)
    out, k = [], 0
    while k < len(offs):
        e = min(k + max_tok, len(offs))
        a = offs[k][0]; b = offs[e][0] if e < len(offs) else len(text)
        piece = text[a:b].strip()
        if piece:
            out.append(piece)
        k = e
    return out


def segment(text: str, glue_min: int = GLUE_MIN, unit_max: int = T_UNIT_MAX) -> list[Unit]:
    """Deterministic unit segmentation of one document version."""
    raw: list[tuple[str, str, int]] = []   # (text, kind, block)
    for b, (kind, lines) in enumerate(parse_blocks(text)):
        if kind == "prose":
            sents = split_sentences(lines[0])
            toks = ntok_many(sents)
            glued: list[str] = []
            carry = ""
            for s, t in zip(sents, toks):
                if carry:
                    s = carry + " " + s; carry = ""
                if ntok(s) < glue_min:
                    carry = s; continue
                glued.append(s)
            if carry:
                if glued:
                    glued[-1] = glued[-1] + " " + carry
                else:
                    glued.append(carry)
            raw.extend((s, "prose", b) for s in glued)
        elif kind == "heading":
            raw.append(("\n".join(lines), "heading", b))
        else:
            raw.extend((l if kind in ("code", "yaml", "table") else l.strip(), kind, b) for l in lines if l.strip())
    # split oversized units into token windows
    pieces: list[tuple[str, str, int]] = []
    toks = ntok_many([r[0] for r in raw])
    for (t, kind, b), k in zip(raw, toks):
        if k > unit_max:
            pieces.extend((w, "window", b) for w in _windows(t, unit_max))
        else:
            pieces.append((t, kind, b))
    units: list[Unit] = []
    toks = ntok_many([p[0] for p in pieces])
    for idx, ((t, kind, b), k) in enumerate(zip(pieces, toks)):
        last = idx == len(pieces) - 1
        nb = None if last else pieces[idx + 1][2]
        par_end = last or nb != b
        if last:
            sep = SEP_END
        elif par_end:
            sep = SEP_BLANK
        elif kind in ("prose", "window"):
            sep = SEP_SPACE
        else:
            sep = SEP_NL
        units.append(Unit(t, kind, b, par_end, kind == "heading", sep, k))
    return units


def key(text: str, salt: str = "") -> str:
    """Chunk key = H(model, encoder input [, context salt])."""
    return hashlib.blake2b(f"{TOKENIZER_NAME}\x00{salt}\x00{text}".encode(), digest_size=12).hexdigest()

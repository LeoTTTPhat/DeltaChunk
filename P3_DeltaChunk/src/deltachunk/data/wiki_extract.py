"""EditStream-RAG WIKI stream from the simplewiki full-history 7z dump.

Pass 1 (`scan`): stream `bsdtar -xOf <7z>` line by line (MediaWiki export XML; one tag per line except <text>).
  * ns=0, non-redirect pages; page-level salted hash h in [0,1) assigns samples A [0,.2), B [.2,.4), C [.4,.6),
    background [.6,.85) (final revision only). Per sample, keep the CAND lowest-h qualifying pages (bounded heap).
  * identity reverts: a revision whose sha1 equals one of the previous 15 retained revisions drops every revision after
    that one, itself included (the reverted ones and the revert);
  * bots: case-insensitive `bot\\b` on the user name, plus a known-account list; bot revisions are dropped (their
    changes fold into the next human revision);
  * qualification: >= MIN_REV retained revisions and final wikitext >= 2 KB; the last KEEP_RAW raw revisions are stored.
Pass 2 (`convert`): parallel mwparserfromhell conversion (headings as `#`, list items as `- `, refs/templates/tables/
  files/categories removed); text-identical consecutive versions dropped and counted; last <= 300 retained; >= 40 kept.

usage: python -m deltachunk.data.wiki_extract scan|convert [--procs 8]
"""
from __future__ import annotations

import argparse, glob, gzip, hashlib, heapq, html, json, os, pickle, re, subprocess, sys, time
from multiprocessing import Pool

P3 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DUMP = os.path.join(P3, "data/raw/simplewiki-20261101-pages-meta-history.xml.7z")
RAWDIR = os.environ.get("WIKI_RAWDIR", os.path.join(P3, "data/wiki_raw"))
SALT = "editstream-2026"
SAMPLES = {"A": (0.0, 0.2), "B": (0.2, 0.4), "C": (0.4, 0.6)}
BG = (0.6, 0.85)
CAND = 1300            # candidates kept per sample in pass 1 (1,000 needed after conversion)
MIN_REV, KEEP_RAW, KEEP, MIN_FINAL_BYTES = 40, 400, 300, 2048
REVERT_WINDOW = 15
BOT_RE = re.compile(r"bot\b", re.I)
KNOWN_BOTS = {"conversion script", "maintenance script", "mediawiki default", "mediawiki message delivery"}


def page_hash(page_id: int, salt: str = SALT) -> float:
    return int(hashlib.sha256(f"{salt}:{page_id}".encode()).hexdigest()[:15], 16) / 16 ** 15


def is_bot(user: str | None) -> bool:
    return bool(user) and (BOT_RE.search(user) is not None or user.strip().lower() in KNOWN_BOTS)


def filter_revisions(revs: list[dict]) -> tuple[list[dict], dict]:
    """revs: oldest first, dicts with sha1, user. Returns retained revisions and filter counts."""
    st = {"revs": len(revs), "revert_dropped": 0, "bot_dropped": 0}
    kept: list[dict] = []
    for r in revs:
        hit = None
        for j in range(len(kept) - 1, max(-1, len(kept) - 1 - REVERT_WINDOW), -1):
            if r["sha1"] and kept[j]["sha1"] == r["sha1"]:
                hit = j; break
        if hit is not None and hit < len(kept) - 1:
            st["revert_dropped"] += len(kept) - 1 - hit + 1
            del kept[hit + 1:]
            continue
        kept.append(r)
    out = []
    for r in kept:
        if is_bot(r.get("user")):
            st["bot_dropped"] += 1
        else:
            out.append(r)
    return out, st


def scan(args):
    t0 = time.time()
    os.makedirs(RAWDIR, exist_ok=True)
    for s in SAMPLES:
        os.makedirs(os.path.join(RAWDIR, s), exist_ok=True)
    heaps = {s: [] for s in SAMPLES}  # max-heap via (-h, pid)
    bg_out = gzip.open(os.path.join(RAWDIR, "bg.jsonl.gz"), "wt")
    stats = dict(pages=0, ns0_nonredirect=0, qualifying=0, revs=0, revert_dropped=0, bot_dropped=0, bg=0, bytes=0)
    if args.dump.endswith(".xml"):
        proc = None; f = open(args.dump, "rb")
    else:
        proc = subprocess.Popen([sys.executable, "-m", "deltachunk.data.sevenzip_cat", args.dump], stdout=subprocess.PIPE, bufsize=1 << 20)
        f = proc.stdout
    last_log = t0
    page = None; rev = None; in_text = False; text_buf = []; want = False
    U = lambda x: html.unescape(x.decode("utf-8", errors="replace"))
    for raw in f:
        stats["bytes"] += len(raw)
        if in_text:
            end = raw.find(b"</text>")
            if end >= 0:
                if want:
                    text_buf.append(raw[:end]); rev["text"] = U(b"".join(text_buf))
                in_text = False; text_buf = []
            elif want:
                text_buf.append(raw)
            continue
        s = raw.strip()
        if s == b"<page>":
            page = {"revs": []}; want = False
        elif page is None:
            continue
        elif s.startswith(b"<title>"):
            page["title"] = U(s[7:-8])
        elif s.startswith(b"<ns>"):
            page["ns"] = int(s[4:-5])
        elif s.startswith(b"<id>") and "id" not in page and rev is None:
            page["id"] = int(s[4:-5])
            h = page_hash(page["id"]); page["h"] = h
            want = False
            if page.get("ns") == 0:
                for sname, (lo, hi) in SAMPLES.items():
                    if lo <= h < hi and (len(heaps[sname]) < CAND or -heaps[sname][0][0] > h):
                        want = True; page["sample"] = sname
                if BG[0] <= h < BG[1]:
                    want = True; page["sample"] = "bg"
        elif s.startswith(b"<redirect"):
            page["redirect"] = True
        elif s == b"<revision>":
            rev = {}
        elif s == b"</page>":
            stats["pages"] += 1
            if page.get("ns") == 0 and not page.get("redirect"):
                stats["ns0_nonredirect"] += 1
                if want:
                    finish_page(page, heaps, bg_out, stats)
            page = None; want = False
            now = time.time()
            if now - last_log > 60:
                last_log = now
                print(json.dumps(dict(t=round(now - t0), gb_xml=round(stats["bytes"] / 1e9, 2), pages=stats["pages"],
                                      qualifying=stats["qualifying"], heaps={k: len(v) for k, v in heaps.items()},
                                      mb_per_s=round(stats["bytes"] / 1e6 / (now - t0), 1))), flush=True)
        elif rev is None:
            continue
        elif s.startswith(b"<text"):
            if b'deleted="deleted"' in s.split(b">", 1)[0]:
                rev["text_deleted"] = True   # suppressed revision (not blanking): dropped in finish_page / convert
            if s.endswith(b"/>"):
                rev["text"] = ""
            else:
                i = raw.find(b">", raw.find(b"<text")) + 1
                rest = raw[i:]
                end = rest.find(b"</text>")
                if end >= 0:
                    rev["text"] = U(rest[:end]) if want else ""
                else:
                    in_text = True; text_buf = [rest] if want else []
        elif s == b"</revision>":
            if want and rev.get("text_deleted"):
                stats["text_deleted"] = stats.get("text_deleted", 0) + 1   # revision-deleted text: not a content state
                rev = None; stats["revs"] += 1
                continue
            if want:
                rev.setdefault("text", ""); rev.setdefault("sha1", hashlib.sha1(rev["text"].encode()).hexdigest())
                if page.get("sample") == "bg":
                    page["revs"] = [rev]
                else:
                    page["revs"].append(rev)
            stats["revs"] += 1
            rev = None
        elif not want:
            continue
        elif s.startswith(b"<id>") and "id" not in rev:
            rev["id"] = int(s[4:-5])
        elif s.startswith(b"<timestamp>"):
            rev["ts"] = s[11:-12].decode()
        elif s.startswith(b"<username>"):
            rev["user"] = U(s[10:-11])
        elif s.startswith(b"<ip>"):
            rev["user"] = "ip:" + s[4:-5].decode()
        elif s.startswith(b"<sha1>"):
            rev["sha1"] = s[6:-7].decode()
        elif s.startswith(b"<sha1 />") or s == b"<sha1/>":
            rev["sha1"] = ""

    rc = proc.wait() if proc else 0
    bg_out.close()
    for sname, hp in heaps.items():
        json.dump(sorted([(-nh, pid) for nh, pid in hp]), open(os.path.join(RAWDIR, f"{sname}_index.json"), "w"))
    stats["seconds"] = round(time.time() - t0, 1); stats["exit"] = rc
    json.dump(stats, open(os.path.join(RAWDIR, "scan_stats.json"), "w"), indent=1)
    print(json.dumps(stats), flush=True)


def finish_page(page, heaps, bg_out, stats):
    revs = page["revs"]
    if not revs:
        return
    if page["sample"] == "bg":
        last = revs[-1]
        if len(last["text"]) >= 1024 and not last["text"].lstrip().lower().startswith("#redirect"):
            bg_out.write(json.dumps({"page_id": page["id"], "title": page["title"], "rev_id": last["id"], "text": last["text"]}) + "\n")
            stats["bg"] += 1
        return
    kept, st = filter_revisions(revs)
    stats["revert_dropped"] += st["revert_dropped"]; stats["bot_dropped"] += st["bot_dropped"]
    if len(kept) < MIN_REV or len(kept[-1]["text"].encode()) < MIN_FINAL_BYTES:
        return
    if kept[-1]["text"].lstrip().lower().startswith("#redirect"):
        return
    stats["qualifying"] += 1
    sname, h = page["sample"], page["h"]
    hp = heaps[sname]
    rec = {"page_id": page["id"], "title": page["title"], "h": h, "filter": st, "n_retained_raw": len(kept),
           "revs": [{"rev_id": r["id"], "ts": r["ts"], "user": r.get("user"), "text": r["text"]} for r in kept[-KEEP_RAW:]]}
    path = os.path.join(RAWDIR, sname, f"{page['id']}.pkl.gz")
    with gzip.open(path, "wb", compresslevel=3) as fo:
        pickle.dump(rec, fo)
    heapq.heappush(hp, (-h, page["id"]))
    if len(hp) > CAND:
        _, pid = heapq.heappop(hp)
        try: os.remove(os.path.join(RAWDIR, sname, f"{pid}.pkl.gz"))
        except OSError: pass


# ---------------------------------------------------------------- pass 2: wikitext -> text
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
_HEAD = re.compile(r"^(={1,6})\s*(.+?)\s*\1\s*$", re.M)
_DROP_LINK = re.compile(r"^(file|image|category|media|[a-z]{2,3}(-[a-z]+)?):", re.I)


def wikitext_to_text(w: str) -> str:
    import mwparserfromhell as mw
    w = _COMMENT.sub("", w); w = _REF.sub("", w)
    w = _HEAD.sub(lambda m: f"\n\x02{len(m.group(1))}\x02{m.group(2)}\n", w)
    code = mw.parse(w)
    for t in code.filter_tags(recursive=False):
        if str(t.tag).lower() in ("table", "gallery", "math", "syntaxhighlight", "timeline", "imagemap", "score"):
            try: code.remove(t)
            except ValueError: pass
    for l in code.filter_wikilinks(recursive=False):
        if _DROP_LINK.match(str(l.title).strip()):
            try: code.remove(l)
            except ValueError: pass
    for t in code.filter_templates(recursive=False):
        try: code.remove(t)
        except ValueError: pass
    for t in code.filter_tags(recursive=False):
        if t.wiki_markup in ("*", "#", ":", ";"):
            try: code.replace(t, "\x03")
            except ValueError: pass
    txt = code.strip_code(normalize=True, collapse=False)
    txt = txt.replace("\xa0", " ")
    out = []
    for line in txt.split("\n"):
        s = line.strip()
        if s.startswith("\x02"):
            m = re.match(r"\x02(\d)\x02(.*)", s)
            if m:
                title = m.group(2).strip()
                if title:
                    out.append(""); out.append("#" * int(m.group(1)) + " " + title); out.append("")
                continue
        if s.startswith("\x03"):
            s = "- " + s.lstrip("\x03").strip()
            if s == "- ":
                continue
        if s.startswith(("{|", "|}", "|-", "!")):
            continue
        out.append(s)
    text = "\n".join(out)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _convert_one(path):
    with gzip.open(path, "rb") as f:
        rec = pickle.load(f)
    rows, prev, dropped = [], None, 0
    for r in rec["revs"]:
        t = wikitext_to_text(r["text"])
        if t == prev:
            dropped += 1; continue
        rows.append(dict(doc_id=str(rec["page_id"]), title=rec["title"], rev_id=r["rev_id"], ts=r["ts"], user=r["user"], text=t)); prev = t
    rows = rows[-KEEP:]
    for i, r in enumerate(rows): r["rev_idx"] = i
    ok = len(rows) >= MIN_REV and len(rows[-1]["text"].encode()) >= MIN_FINAL_BYTES
    return rec["page_id"], rec["h"], rows, dict(rec["filter"], identical_dropped=dropped, n_retained_raw=rec["n_retained_raw"], ok=ok)


def _convert_bg(line):
    d = json.loads(line)
    return dict(doc_id=str(d["page_id"]), title=d["title"], rev_id=d["rev_id"], text=wikitext_to_text(d["text"]))


def convert(args):
    import pandas as pd
    t0 = time.time(); out_dir = os.path.join(P3, "data/streams"); os.makedirs(out_dir, exist_ok=True)
    allstats = {}
    with Pool(args.procs) as pool:
        for sname in SAMPLES:
            files = glob.glob(os.path.join(RAWDIR, sname, "*.pkl.gz"))
            res = sorted(pool.map(_convert_one, files, chunksize=4), key=lambda x: x[1])
            sel = [r for r in res if r[3]["ok"]][: args.n]
            rows = [row for r in sel for row in r[2]]
            df = pd.DataFrame(rows); df["ts"] = (pd.to_datetime(df["ts"], utc=True) - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)  # unit-safe (pandas 3 parses to us)
            df.to_parquet(os.path.join(out_dir, "wiki.parquet" if sname == "A" else f"wiki_{sname}.parquet"), index=False)
            st = dict(candidates=len(files), qualifying_after_conversion=sum(r[3]["ok"] for r in res), docs=len(sel), versions=len(df),
                      revs_in_selected=sum(r[3]["revs"] for r in sel), revert_dropped=sum(r[3]["revert_dropped"] for r in sel),
                      bot_dropped=sum(r[3]["bot_dropped"] for r in sel), identical_dropped=sum(r[3]["identical_dropped"] for r in sel))
            allstats[sname] = st; print(sname, st, flush=True)
        bgf = os.path.join(RAWDIR, "bg.jsonl.gz")
        if os.path.exists(bgf):
            with gzip.open(bgf, "rt") as f:
                bg = pool.map(_convert_bg, f, chunksize=64)
            bg = [b for b in bg if len(b["text"]) >= 500]
            pd.DataFrame(bg).to_parquet(os.path.join(out_dir, "wiki_bg.parquet"), index=False)
            allstats["bg"] = dict(docs=len(bg))
    scan_stats = json.load(open(os.path.join(RAWDIR, "scan_stats.json"))) if os.path.exists(os.path.join(RAWDIR, "scan_stats.json")) else {}
    allstats.update(scan=scan_stats, convert_seconds=round(time.time() - t0, 1), procs=args.procs)
    json.dump(allstats, open(os.path.join(P3, "results/e0_filter_stats.json"), "w"), indent=1)


def merge(args):
    """Only ~1/3 of the pass-1 candidates keep >= 40 retained versions after conversion (template-only edits become
    text-identical and are dropped), so WIKI-A is the union of the hash samples A, B, C (each a contiguous salted-hash
    range, hence an unbiased page sample), in the order A, B, C by hash, truncated to `n` documents."""
    import pandas as pd
    out_dir = os.path.join(P3, "data/streams"); parts = []
    for f in ("wiki.parquet", "wiki_B.parquet", "wiki_C.parquet"):
        src = os.path.join(out_dir, f)
        if f == "wiki.parquet" and os.path.exists(os.path.join(out_dir, "wiki_sampleA.parquet")):
            src = os.path.join(out_dir, "wiki_sampleA.parquet")
        parts.append(pd.read_parquet(src))
    if not os.path.exists(os.path.join(out_dir, "wiki_sampleA.parquet")):
        os.replace(os.path.join(out_dir, "wiki.parquet"), os.path.join(out_dir, "wiki_sampleA.parquet"))
    docs, rows = [], []
    for name, df in zip("ABC", parts):
        order = list(dict.fromkeys(df.doc_id))
        for d in order:
            if len(docs) < args.n:
                docs.append((name, d))
    keep = {d for _, d in docs}
    wiki = pd.concat([df[df.doc_id.isin(keep)].assign(sample=name) for name, df in zip("ABC", parts)], ignore_index=True)
    assert wiki.ts.between(978307200, 1830297600).all(), "timestamps must be Unix seconds (2001..2028)"
    wiki.to_parquet(os.path.join(out_dir, "wiki.parquet"), index=False)
    st = json.load(open(os.path.join(P3, "results/e0_filter_stats.json")))
    st["wiki_merged"] = dict(docs=int(wiki.doc_id.nunique()), versions=len(wiki), from_sample={k: int(v) for k, v in wiki.groupby("sample").doc_id.nunique().items()},
                             note=merge.__doc__.split("\n")[0])
    json.dump(st, open(os.path.join(P3, "results/e0_filter_stats.json"), "w"), indent=1)
    print(st["wiki_merged"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["scan", "convert", "merge"])
    ap.add_argument("--dump", default=DUMP); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    {"scan": scan, "convert": convert, "merge": merge}[a.cmd](a)


if __name__ == "__main__":
    main()

import gzip, os, pickle, types
from deltachunk.data import wiki_extract as W

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "mini_history.xml")


def test_scan_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "RAWDIR", str(tmp_path))
    monkeypatch.setattr(W, "SAMPLES", {"A": (0.0, 1.0)})
    monkeypatch.setattr(W, "BG", (2.0, 3.0))
    monkeypatch.setattr(W, "MIN_REV", 1); monkeypatch.setattr(W, "MIN_FINAL_BYTES", 1)
    W.scan(types.SimpleNamespace(dump=FIX))
    files = os.listdir(tmp_path / "A")
    assert files == ["7.pkl.gz"]  # redirect page 8 excluded
    rec = pickle.load(gzip.open(tmp_path / "A" / "7.pkl.gz"))
    assert rec["title"] == "Test & Page"
    assert [r["rev_id"] for r in rec["revs"]] == [100, 104]          # vandal+revert dropped, bot dropped
    assert rec["filter"] == {"revs": 5, "revert_dropped": 2, "bot_dropped": 1}
    assert rec["revs"][0]["text"].endswith("Line <b>two</b>.")      # XML unescaped, multi-line text


def test_filter_bot_regex():
    assert W.is_bot("CleanupBot") and W.is_bot("cluebot ng") and W.is_bot("Conversion script")
    assert not W.is_bot("Bottomley") and not W.is_bot("Alice")


def test_wikitext_to_text():
    w = ("{{Infobox|a=1}}\n'''Paris''' is the [[capital city|capital]] of [[France]].<ref>{{cite web|x}}</ref> It is big.\n\n"
         "== History ==\nText [[File:E.jpg|thumb|The tower]] more.\n* item one\n* item [[two]]\n\n"
         "{| class=wikitable\n|-\n| a || b\n|}\n[[Category:Cities]]\n<!-- c -->\n=== Sub ===\nEnd.")
    t = W.wikitext_to_text(w)
    assert t.startswith("Paris is the capital of France. It is big.")
    assert "\n## History\n" in t and "\n### Sub\n" in t
    assert "- item one\n- item two" in t
    assert "thumb" not in t and "Category" not in t and "cite" not in t and "wikitable" not in t

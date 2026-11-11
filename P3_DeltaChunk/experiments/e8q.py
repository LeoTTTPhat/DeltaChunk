"""E8 quality points (Tier B): E4 retrieval quality for parameter variants of the calibrated chunkers.

Runs experiments/e4.py unchanged, with variant chunker names resolved here:
  '<BASE>@<k>=<v>[,<k>=<v>...]'  = chunker BASE with its per-stream calibrated parameters at --T, overridden by k=v
                                   (e.g. 'SentCDC@lam=1' = SentCDC at the calibrated r with paragraph bias lambda=1).
The variant parameters are served through the calibration dicts' .get() only, so e4's checkpoint hash (which covers
the calibration items) is the same as the primary E4 run at the same --T/--bg: base chunkers (REC, SentCDC, ...)
resume from the primary run's checkpoints and only the variants are computed. The target-size points (T=200, 320)
need no wrapper: `e4.py --T 200 --chunkers REC,STRUCT-M,SentCDC --tag _e8q_T200`.
Round 2: '<BASE>@<name>' without '=' (e.g. 'KARA@match', 'REC@nm') takes per-stream parameters from the JSON file in
$R2_MATCH_PARAMS (default results/round2/matched_params.json, key params[stream][variant]), written by
experiments/r2_match_params.py. Existing variant names behave exactly as before.
usage: python experiments/e8q.py --chunkers REC,SentCDC,SentCDC@lam=1 --tag _e8q_lam [any e4.py argument]
"""
import json, os, sys

import e4
from deltachunk.chunkers import make as _make


def _parse(name):
    base, _, ov = name.partition("@")
    kv = {}
    if "=" not in ov:          # per-stream lookup variant; parameters come from _VDict.get
        return base, kv
    for part in ov.split(","):
        k, _, v = part.partition("=")
        kv[k] = float(v) if "." in v or k in ("lam", "p") else int(v)
    return base, kv


_MATCH = {}


def _match_params(stream, name):
    if not _MATCH:
        f = os.environ.get("R2_MATCH_PARAMS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                            "results", "round2", "matched_params.json"))
        _MATCH.update(json.load(open(f))["params"])
    return _MATCH[stream][name]


class _VDict(dict):
    """Calibration dict whose .get() also resolves variant names; items()/hash unchanged."""
    stream = None

    def get(self, k, default=None):
        if isinstance(k, str) and "@" in k:
            base, kv = _parse(k)
            if "=" not in k.partition("@")[2]:
                return {**(dict.get(self, base) or {}), **_match_params(self.stream, k)}
            return {**(dict.get(self, base) or {}), **kv}
        return dict.get(self, k, default)


def _make_variant(name, cal=None):
    if "@" in name:
        base, kv = _parse(name)
        return _make(base, {**(cal or {}), **kv})
    return _make(name, cal)


_cal0 = e4.calibration


def _calibration(stream=None):
    out = {}
    for T, P in _cal0(stream).items():
        d = _VDict(P); d.stream = stream; out[T] = d
    return out


e4.make = _make_variant
e4.calibration = _calibration

if __name__ == "__main__":
    sys.argv[0] = "e4.py"
    e4.main()

"""Unit-identity diff, edit regions and edit classes (design.md §2)."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from deltachunk.text import Unit


@dataclass
class Region:
    a: int      # first old unit
    b: int      # one past last old unit  (ℓ = b - a)
    a2: int     # first new unit
    b2: int     # one past last new unit  (ℓ' = b2 - a2)

    @property
    def l_old(self) -> int:
        return self.b - self.a

    @property
    def l_new(self) -> int:
        return self.b2 - self.a2


def regions(old: list[Unit], new: list[Unit]) -> list[Region]:
    sm = SequenceMatcher(None, [u.ihash for u in old], [u.ihash for u in new], autojunk=False)
    return [Region(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]


def edit_class(old_raw: str, new_raw: str, old: list[Unit], new: list[Unit], regs: list[Region] | None = None) -> str:
    """noop (raw identical) | ws (whitespace/reflow only: identical units) | structure (unit texts identical, flags or
    separators differ) | single | multi."""
    if old_raw == new_raw:
        return "noop"
    if [u.identity for u in old] == [u.identity for u in new]:
        return "ws"
    if [u.text for u in old] == [u.text for u in new]:
        return "structure"
    regs = regions(old, new) if regs is None else regs
    return "single" if len(regs) == 1 else "multi"

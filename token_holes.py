"""Find "holes" in the token tables: unregistered codes wedged between registered
ones, which often flag a token we forgot to implement.

The token space is a 256-entry table (see catalog._TABLE).  Each slot is either a
single-byte Token, a two-byte prefix page (a sub-table of up to 256 Tokens), or
empty.  A *hole* is an empty slot strictly between the lowest and highest filled
slot of its page — interior space the calculator may actually use, as opposed to
the unused lead/tail space every page has.

Small holes (a code or two between neighbours) are the likely genuine omissions;
big runs are usually just unused address space.  This tool prints the gaps with
their bracketing tokens so you can judge which is which.

Usage:
    python token_holes.py            # all interior holes, grouped by page
    python token_holes.py --max N    # only holes spanning N or fewer codes
"""
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import catalog
from titoken import Token


def _name(item) -> str:
    """Short label for a table slot: a token's text, a prefix marker, or empty."""
    if isinstance(item, Token):
        return repr(item.text)
    if isinstance(item, list):
        return '<2-byte page>'
    return '<empty>'


def _gaps(filled: list[int]):
    """Yield (start, end) inclusive runs of missing codes between consecutive
    entries of a sorted list of filled indices."""
    for a, b in zip(filled, filled[1:]):
        if b - a > 1:
            yield a + 1, b - 1


def _report(slots: list, *, base: int, width: int, occupied, max_gap) -> int:
    """Print interior holes for one page.  `slots[i]` is the entry at low byte i;
    a code renders as `base | i` in `width` hex digits; `occupied(item)` decides
    whether a slot counts as filled.  Returns the number of hole groups printed."""
    filled = [i for i, item in enumerate(slots) if occupied(item)]
    if len(filled) < 2:
        return 0
    runs = [(lo, hi) for lo, hi in _gaps(filled)
            if max_gap is None or hi - lo + 1 <= max_gap]
    if not runs:
        return 0

    print(f"\npage 0x{base >> 8:02X}__" if width == 4 else "\nsingle-byte 0x__",
          end="")
    print(f": {len(filled)} tokens, "
          f"low 0x{base | filled[0]:0{width}X}, high 0x{base | filled[-1]:0{width}X}")
    for lo, hi in runs:
        size = hi - lo + 1
        span = (f"0x{base | lo:0{width}X}" if size == 1
                else f"0x{base | lo:0{width}X}-0x{base | hi:0{width}X}")
        print(f"  {span:>15}  ({size:>3} hole{'s' if size > 1 else ' '})  "
              f"after {_name(slots[lo - 1])}, before {_name(slots[hi + 1])}")
    return len(runs)


def main() -> None:
    max_gap = None
    if '--max' in sys.argv:
        max_gap = int(sys.argv[sys.argv.index('--max') + 1])

    table = catalog._TABLE
    groups = 0

    # Single-byte page: a slot is occupied by a Token *or* a 2-byte prefix page.
    print("═══ single-byte tokens ═══")
    groups += _report(table, base=0x00, width=2,
                      occupied=lambda x: x is not None, max_gap=max_gap)

    # Two-byte pages: each prefix byte whose slot holds a sub-table.
    print("\n═══ two-byte tokens ═══")
    for b0, item in enumerate(table):
        if isinstance(item, list):
            groups += _report(item, base=b0 << 8, width=4,
                              occupied=lambda x: isinstance(x, Token), max_gap=max_gap)

    print(f"\n{groups} hole group(s)"
          + (f" of ≤{max_gap} codes" if max_gap is not None else ""))


if __name__ == '__main__':
    main()

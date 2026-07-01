# tibasic

A TI-84+ (TI-BASIC) calculator emulator in Python. Fidelity to real hardware behavior is
the top priority.

## Essentials

- **Indent with tabs, not spaces.** Match the surrounding file.
- **Windows environment.** The Python launcher is `py`, not `python3`. Prefix commands with
  `PYTHONUTF8=1` (or `$env:PYTHONUTF8=1` in PowerShell) — the terminal's cp1252 can't encode
  the block/glyph characters and will crash on things like `Bitmap.disp()` otherwise.
- **Run tests:** `py -m pytest -q` (~1250 tests, <5s; run the whole suite after changes).
- Command reference specs are in `commands/*.txt` (named by hex token, e.g.
  `98_storepic.txt`) — check the relevant one when implementing or fixing a command.

## Working style

Discuss design before non-trivial changes — the user often has a cleaner idea. Recommend a
direction rather than surveying options. When the user pushes back, re-examine rather than
defend.

## Architecture

For a full tour of the token / parser / preparse / environment / bitmap / tifile design and
the fidelity conventions, read **ARCHITECTURE.md** before non-trivial changes.

# Architecture

A TI-84+ (TI-BASIC) calculator emulator in Python: an interpreter that tokenizes,
parses, and executes TI-BASIC programs with high fidelity to real hardware — down to
quirks like which errors trigger a regraph and how pictures are bit-packed on disk.

**Fidelity to the real calculator is the top priority.** When in doubt, the real
device's behavior wins. Command reference specs live in `commands/*.txt` (scraped docs,
named by hex token, e.g. `98_storepic.txt`) — check these when implementing or fixing a
command; they capture the edge cases.

## The load-bearing pieces

- **Tokens are the unit of everything.** `catalog.py::_generate()` yields every `Token`
  with its hex code, display bytes, and behavior. A token IS its own accessor/parser hook
  — variable tokens (`tokenbase.py`: `VariableToken`/`Accessor`) carry `resolve`/`store`/
  `invoke` and take `env` as a parameter (they're stateless and env-less; the catalog
  builds one of each, but tokens are value objects comparing by code — a fresh
  `StringToken(3)` equals the catalog's Str4). Codes
  ≤0xFF are one byte; >0xFF are two. `Flag` (an `IntFlag`) classifies role
  (FUNCTION/COMMAND/INFIX/…) and variable kind — the parser dispatches on flags, not code
  ranges.

- **`parser.py`** is a Pratt parser. `Parser` drives statements; `ArgParser` handles the
  comma-separated args of one call. Trailing-comma model: after each arg the following
  comma is eaten, so `peek()` between args shows the *next* arg's first token (letting
  callers dispatch on type before consuming).

- **`preparse.py`** is the declarative arg-schema layer. Most commands are plain functions
  annotated with vocabulary types (`Real`, `Thunk`, `TiList`, `NumericVar`, `Env`, …) and
  wrapped by `@preparse_cmd` / `@preparse_func` / `@preparse_cmd_func` (these differ only
  in their finalizer: `end_cmd` / `end_func` / `end_paren_cmd`). Argument type validation
  happens in `ArgParser.take()` via the annotation's `require_X` guard — **before** the
  function body runs. For custom parsing (peeking raw tokens, variadics, or tokens that
  can't start an expression such as Pic/GDB variable tokens), use `@special_func` and drive
  the `ArgParser` yourself.

- **`environment.py::Environment`** holds all interpreter state (numerics, matrices, lists,
  graph, window, modes, screen). The render model is *pull-based*: `env.screen` says which
  surface is currently shown; a frontend calls `console.present()` to read it. Drawing
  commands wrap their pixel work in `with env.draw_to_graph():` — the context manager
  regraphs if the graph is stale on entry, and on **normal** exit sets `screen = GRAPH` and
  calls `present()`. An exception in the body leaves screen/frontend state untouched.

- **`bitmap.py::Bitmap`** is the 96×64 monochrome LCD as a dense buffer (one byte per pixel,
  0/1); `GraphScreen` drives it as pixels and the home/menu screens rasterize their glyph
  grid into one. Coordinates are `(row, col)`, origin top-left, matching TI's Pxl- argument
  order. Bit-packing is strictly a serialization concern (see `tifile.py`).

- **`tifile.py`** reads/writes `.8x*` files (programs, lists, pictures). Real TI files have
  a 57-byte header + a 9-byte VAT entry (in-RAM symbol-table overhead, *not* part of the
  variable data) + a 2-byte data-length prefix. File-writable variables are `FileVar`
  accessors (tokenbase); their `name_bytes()` is the *meaningful* name bytes only — the
  8-byte-field padding is the file layer's job (`write_accessor`/`read_accessor`). Pictures are either 63 rows ("graph area" only,
  756 bytes) or 64 rows (native TI-83+/84+, 768 bytes) — derive the row count from the
  length prefix.

## Conventions (not guessable — follow them)

- **Tabs, not spaces, for indentation.** Match the surrounding file.
- Match surrounding style: comment density, naming, idiom. Docstrings explain the *why* and
  any hardware quirks, not the mechanics.
- **Fidelity quirks are deliberate.** e.g. Pxl- validates argument *types* before regraphing
  but checks domain (range) errors *after* — so an invalid-type arg won't regraph, but an
  out-of-range one will. Preserve this "validate type → regraph → check domain" ordering.
- Pxl- addresses rows 0–62 only; the buffer is a full 64 rows (StorePic snapshots all 64).

## Testing

- Run the suite: `py -m pytest -q` (~1250 tests, <5s — run the whole thing after changes).
- `test_tibasic.py` helpers: `toks(code)` builds a token list from a space-separated string
  (each segment is looked up by its display text — mind the spaces, e.g. `'Pxl-On( 0,0'`);
  `run(src, env=None)` executes source and returns the env; `calc(items, env=None)` returns
  the resulting value; `var(env, name)` reads a variable back.

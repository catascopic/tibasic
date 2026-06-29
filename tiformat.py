"""Rendering of TI values to their on-screen text.

Two display styles, matching the calculator:

  * Output( writes a value "the way you'd type it in" — linear, comma-separated,
    a matrix inlined onto one line (see `output_text`).
  * Disp / Pause show a value "as the result of a calculation" — lists are
    space-separated, a matrix spreads over one line per row with its columns
    aligned, and the whole block is right-aligned on the screen (`disp_lines`).

Both produce TI *display bytes* (one byte per glyph/cell), the form the home
screen stores: a canvas frontend draws them through the font tables, the terminal
decodes them back to characters.  Numbers are formatted as ASCII strings first
(`ti83_format`) and encoded at the leaves; a TiString is already glyph bytes, so
its tokens' display bytes splice straight in.
"""
from decimal import Decimal, ROUND_HALF_UP

from core import TiList, TiMatrix, TiString
from tokenbase import encode

_SIG = 10  # the TI-83+ displays 10 significant figures in Normal mode

# Separator/bracket display bytes, encoded through the charset rather than written
# as ASCII literals: most punctuation shares its ASCII code, but '[' does not —
# ASCII 0x5B is θ on the TI, whose '[' glyph is 0xC1 — so b'[' would be wrong.
_LBRACE, _RBRACE = encode('{'), encode('}')
_LBRACK, _RBRACK = encode('['), encode(']')
_COMMA, _SPACE = encode(','), encode(' ')


def ti83_format(x) -> str:
	"""Format a real number the way a TI-83+ shows it in Normal/Float mode.

	10 significant figures, ties rounded away from zero.  Scientific notation
	when |x| ≥ 1e10 or 0 < |x| < 1e-3.  Positive exponents carry no '+' sign,
	and a pure fraction drops its leading zero (.5, not 0.5).
	"""
	d = Decimal(str(x))
	if d == 0:
		return "0"

	# Round to 10 significant figures.  adjusted() is the power of ten of the
	# leading digit, so exp-(_SIG-1) is the place value of the last kept digit.
	exp = d.adjusted()
	d = d.quantize(Decimal(1).scaleb(exp - (_SIG - 1)), rounding=ROUND_HALF_UP)
	exp = d.adjusted()  # re-read: rounding may have bumped it (9.999… → 10)

	if exp >= 10 or exp < -3:
		return _plain(d.scaleb(-exp)) + f"e{exp}"  # shift mantissa into [1, 10)
	return _plain(d)


def _plain(d: Decimal) -> str:
	"""Fixed-point string with trailing zeros and a pure-fraction leading zero removed."""
	s = format(d, 'f')
	if '.' in s:
		s = s.rstrip('0').rstrip('.')
	if s.startswith('0.'):
		return s[1:]
	if s.startswith('-0.'):
		return '-' + s[2:]
	return s


# ── Scalars ───────────────────────────────────────────────────────────────────

def format_complex(value: complex) -> str:
	"""a+bi / a-bi, TI's complex notation.  The imaginary coefficient is always
	shown explicitly (1i, not a bare i), and the real part is dropped only when
	it's exactly zero."""
	im = ti83_format(abs(value.imag))
	sign = '-' if value.imag < 0 else '+'
	if value.real == 0:
		return f"{sign if value.imag < 0 else ''}{im}i"
	return f"{ti83_format(value.real)}{sign}{im}i"


def format_scalar(value) -> str:
	"""Real or complex scalar, as TI would show it — the shared piece between a
	bare value and one element of a list."""
	return format_complex(value) if isinstance(value, complex) else ti83_format(value)


def _scalar_bytes(value) -> bytes:
	"""A scalar's display bytes: format to ASCII, then encode to the charset."""
	return encode(format_scalar(value))


def _string_bytes(value: TiString) -> bytes:
	"""A TiString's display bytes — its tokens' glyph bytes, spliced straight in
	(no encode round-trip, since a typeable token is already one display byte)."""
	return b''.join(t.display for t in value.tokens)


# ── Output( : linear, comma-separated, "the way you'd type it" ─────────────────

def output_text(value) -> bytes:
	"""Render `value` as a single linear run of display bytes for Output(.

	Lists and matrices use comma separators with no spaces, a matrix written
	inline as [[…][…]] — exactly the keystrokes that would re-enter the value.
	"""
	if isinstance(value, TiString):
		return _string_bytes(value)
	if isinstance(value, TiList):
		return _LBRACE + _COMMA.join(_scalar_bytes(v) for v in value.data) + _RBRACE
	if isinstance(value, TiMatrix):
		return _LBRACK + b''.join(
			_LBRACK + _COMMA.join(_scalar_bytes(v) for v in row) + _RBRACK for row in value.data
		) + _RBRACK
	return _scalar_bytes(value)


# ── Disp / Pause : calculation-result style, right-aligned ────────────────────

def value_lines(value) -> list[bytes]:
	"""The full content line(s) for a value, in display bytes — unaligned and not
	clipped to any width.

	This is the raw form a scroll view pages a window over (see scrollview.py);
	disp_lines right-aligns these onto the screen.  A list is one space-separated
	line, a matrix one line per row with columns aligned, a scalar/string a single
	line.
	"""
	if isinstance(value, TiString):
		return [_string_bytes(value)]
	if isinstance(value, TiList):
		return [_LBRACE + _SPACE.join(_scalar_bytes(v) for v in value.data) + _RBRACE]
	if isinstance(value, TiMatrix):
		return _matrix_disp_lines(value)
	return [_scalar_bytes(value)]


def disp_lines(value, width: int) -> list[bytes]:
	"""Render `value` as the screen line(s) Disp/Pause would show, in display bytes.

	Lists are space-separated and a matrix spreads over one line per row with
	its columns aligned.  Numbers, lists, and matrices are right-aligned to
	`width`; strings are left-aligned (returned as-is, written from column 0).
	"""
	lines = value_lines(value)
	if isinstance(value, TiString):
		return lines           # strings sit at the left margin, never right-aligned
	return _right_align(lines, width)


def _matrix_disp_lines(mat: TiMatrix) -> list[bytes]:
	"""One line per row, columns left-aligned to a common per-column width, with
	the bracket nesting TI uses: [[1 2]  on the first row, closing  [3 4]]  on
	the last.  Column widths ignore magnitude/decimals — purely the rendered
	text length of each entry.
	"""
	if not mat.data:
		return [_LBRACK + _RBRACK]
	cells = [[_scalar_bytes(v) for v in row] for row in mat.data]
	widths = [max(len(row[c]) for row in cells) for c in range(mat.cols)]
	lines = []
	for i, row in enumerate(cells):
		body  = _SPACE.join(cell.ljust(w) for cell, w in zip(row, widths))
		left  = _LBRACK + _LBRACK if i == 0 else _SPACE + _LBRACK
		right = _RBRACK + _RBRACK if i == mat.rows - 1 else _RBRACK
		lines.append(left + body + right)
	return lines


def _right_align(lines: list[bytes], width: int) -> list[bytes]:
	"""Indent every line by the same amount so the block sits flush against the
	right edge, preserving internal (e.g. matrix-column) alignment.  Left as-is
	when it's already too wide to fit, so Disp can truncate it normally.

	One display byte is one cell, so byte length is the on-screen width.
	"""
	overhang = width - max(len(line) for line in lines)
	if overhang <= 0:
		return lines
	pad = b' ' * overhang
	return [pad + line for line in lines]

"""Rendering of TI values to their on-screen text.

Two display styles, matching the calculator:

  * Output( writes a value "the way you'd type it in" — linear, comma-separated,
    a matrix inlined onto one line (see `output_text`).
  * Disp / Pause show a value "as the result of a calculation" — lists are
    space-separated, a matrix spreads over one line per row with its columns
    aligned, and the whole block is right-aligned on the screen (`disp_lines`).

Number formatting (`ti83_format`) is shared by both.
"""
from decimal import Decimal, ROUND_HALF_UP

from core import TiList, TiMatrix, TiString

_SIG = 10  # the TI-83+ displays 10 significant figures in Normal mode


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


# ── Output( : linear, comma-separated, "the way you'd type it" ─────────────────

def output_text(value) -> str:
	"""Render `value` as a single linear string for Output(.

	Lists and matrices use comma separators with no spaces, a matrix written
	inline as [[…][…]] — exactly the keystrokes that would re-enter the value.
	"""
	if isinstance(value, TiString):
		return str(value)
	if isinstance(value, TiList):
		return '{' + ','.join(format_scalar(v) for v in value.data) + '}'
	if isinstance(value, TiMatrix):
		return '[' + ''.join(
			'[' + ','.join(ti83_format(v) for v in row) + ']' for row in value.data
		) + ']'
	return format_scalar(value)


# ── Disp / Pause : calculation-result style, right-aligned ────────────────────

def disp_lines(value, width: int) -> list[str]:
	"""Render `value` as the screen line(s) Disp/Pause would show.

	Lists are space-separated and a matrix spreads over one line per row with
	its columns aligned.  Numbers, lists, and matrices are right-aligned to
	`width`; strings are left-aligned (returned as-is, written from column 0).
	"""
	if isinstance(value, TiString):
		return [str(value)]
	if isinstance(value, TiList):
		lines = ['{' + ' '.join(format_scalar(v) for v in value.data) + '}']
	elif isinstance(value, TiMatrix):
		lines = _matrix_disp_lines(value)
	else:
		lines = [format_scalar(value)]
	return _right_align(lines, width)


def _matrix_disp_lines(mat: TiMatrix) -> list[str]:
	"""One line per row, columns left-aligned to a common per-column width, with
	the bracket nesting TI uses: [[1 2]  on the first row, closing  [3 4]]  on
	the last.  Column widths ignore magnitude/decimals — purely the rendered
	text length of each entry.
	"""
	if not mat.data:
		return ['[]']
	cells = [[ti83_format(v) for v in row] for row in mat.data]
	widths = [max(len(row[c]) for row in cells) for c in range(mat.cols)]
	lines = []
	for i, row in enumerate(cells):
		body  = ' '.join(cell.ljust(w) for cell, w in zip(row, widths))
		left  = '[[' if i == 0 else ' ['
		right = ']]' if i == mat.rows - 1 else ']'
		lines.append(f"{left}{body}{right}")
	return lines


def _right_align(lines: list[str], width: int) -> list[str]:
	"""Indent every line by the same amount so the block sits flush against the
	right edge, preserving internal (e.g. matrix-column) alignment.  Left as-is
	when it's already too wide to fit, so Disp can truncate it normally.
	"""
	overhang = width - max(len(line) for line in lines)
	if overhang <= 0:
		return lines
	pad = ' ' * overhang
	return [pad + line for line in lines]

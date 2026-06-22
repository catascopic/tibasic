import math

import distributions as dist
from preparse import preparse_cmd, preparse_cmd_func, Env, TiListComplex, Real, Thunk, special_func, no_arg_command, TiCall
from errors import DataTypeError, DomainError
from modes import Screen
from graphscreen import GraphScreen
from fonts import SMALL_FONT, LARGE_FONT
from core import TiString, py_int
from graph import (
	MAX_ROW, MAX_COL, _round_half_up,
	_x_to_col, _y_to_row, _col_to_x, _graph_to_pixel,
	_bresenham, _in_bounds, _plot_segment,
	sample_function as _function_sampler,
	trace_curve as _trace_curve,
)


class _GraphDrawing(TiCall):
	"""Wraps a drawing command so it brings up the graph *before* drawing — regraphing
	the functions on the HOME→GRAPH transition so they sit beneath the new drawing
	(regraph clears and re-plots, so it has to run first or it would erase the draw).

	If the command then raises (e.g. bad arguments), the screen transition is rolled
	back, so a draw that fails doesn't leave the graph showing.  Queries (pxl-Test()
	and ClrDraw — which clears without displaying — are not wrapped."""

	def __init__(self, inner: TiCall):
		super().__init__(inner.func)
		self._inner = inner

	def call_with_parser(self, args):
		env = args.env
		prev_screen = env.screen
		env.draw_to_graph()
		try:
			return self._inner.call_with_parser(args)
		except Exception:
			env.screen = prev_screen
			raise

# Pt-On/Off/Change mark pixel offsets (Δrow, Δcol) relative to centre.
# mark 2/6 = 3×3 filled box (9 pixels)
# mark 3/7 = 3×3 cross / plus sign (5 pixels)
# anything else = dot (1 pixel)
_CROSS  = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
_BOX    = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_MARK_OFFSETS = {2.0: _BOX, 3.0: _CROSS, 6.0: _BOX, 7.0: _CROSS}


def _validate(row, col):
	"""Check the Pxl-addressable range, then return Python ints for screen indexing."""
	if not _in_bounds(row, col):
		raise DomainError(f"Pixel out of range: row={row}, column={col}")
	return py_int(row), py_int(col)


@preparse_cmd_func
def pxl_on(env: Env, row: Real, col: Real) -> None:
	env.graph.set(*_validate(row, col))


@preparse_cmd_func
def pxl_off(env: Env, row: Real, col: Real) -> None:
	env.graph.set_off(*_validate(row, col))


@preparse_cmd_func
def pxl_change(env: Env, row: Real, col: Real) -> None:
	env.graph.toggle(*_validate(row, col))


@preparse_cmd_func
def pxl_test(env: Env, row: Real, col: Real) -> float:
	return float(env.graph.get(*_validate(row, col)))


@no_arg_command
def clr_draw(env) -> None:
	# ClrDraw clears the drawing but doesn't itself display the graph; it only
	# regraphs the functions if the graph is already the screen being shown.
	env.graph.clear()
	if env.screen is Screen.GRAPH:
		env.regraph()


def _pt_action(env, x, y, mark, action) -> None:
	row, col = _graph_to_pixel(env, x, y)
	if _in_bounds(row, col):
		try:
			points = _MARK_OFFSETS[mark]
		except KeyError:
			action(env.graph, row, col)
		else:
			for dr, dc in points:
				r = row + dr
				c = col + dc
				if _in_bounds(r, c):
					action(env.graph, r, c)


@preparse_cmd_func
def pt_on(env: Env, x: Real, y: Real, mark: Real = 1.0) -> None:
	_pt_action(env, x, y, mark, GraphScreen.set)


@preparse_cmd_func
def pt_off(env: Env, x: Real, y: Real, mark: Real = 1.0) -> None:
	_pt_action(env, x, y, mark, GraphScreen.set_off)


@preparse_cmd_func
def pt_change(env: Env, x: Real, y: Real, mark: Real = 1.0) -> None:
	_pt_action(env, x, y, mark, GraphScreen.toggle)


@preparse_cmd
def vertical(env: Env, x: Real) -> None:
	"""Vertical X — draw a full-height line at graph x-coordinate X."""
	col = _x_to_col(env, x)
	if 0 <= col <= MAX_COL:
		for row in range(MAX_ROW + 1):
			env.graph.set(row, col, True)


@preparse_cmd
def horizontal(env: Env, y: Real) -> None:
	"""Horizontal Y — draw a full-width line at graph y-coordinate Y."""
	row = _y_to_row(env, y)
	if 0 <= row <= MAX_ROW:
		for col in range(MAX_COL + 1):
			env.graph.set(row, col, True)


@preparse_cmd_func
def line(env: Env, x1: Real, y1: Real, x2: Real, y2: Real, erase: Real = 1) -> None:
	"""Line(X1,Y1,X2,Y2[,erase]) — draw (or erase) a line between two graph points.

	erase=0 turns pixels off; any other value (default 1) turns them on.
	Off-screen pixels are clipped silently.
	"""
	on = (erase != 0)
	r0, c0 = _graph_to_pixel(env, x1, y1)
	r1, c1 = _graph_to_pixel(env, x2, y2)
	for r, c in _bresenham(r0, c0, r1, c1):
		if _in_bounds(r, c):
			env.graph.set(r, c, on)


@preparse_cmd_func
def circle(env: Env, x: Real, y: Real, r: Real, _fast: TiListComplex = None) -> None:
	"""Circle(X,Y,r[,{i}]) — draw a circle (or ellipse) at graph (X,Y) with graph radius r.

	The optional 4th argument enables the 'fast circle' routine on real hardware
	(Bresenham 8-fold symmetry); the calculator requires it to be a complex list
	(e.g. {i}).  We validate it for fidelity but ignore its value — we always use
	the parametric approach, which handles non-square windows correctly.
	Negative radius is treated as its absolute value.  Off-screen pixels are clipped.
	"""
	w = env.window
	xmin = w.xmin
	xmax = w.xmax
	ymin = w.ymin
	ymax = w.ymax
	cy, cx = _graph_to_pixel(env, x, y)
	rx = abs(r) * MAX_COL / (xmax - xmin)
	ry = abs(r) * MAX_ROW / (ymax - ymin)
	# Step finely enough that no pixel is skipped (~4 steps per pixel of circumference).
	n = max(8, math.ceil(4 * math.pi * max(rx, ry)) + 1)
	for i in range(n):
		theta = 2 * math.pi * i / n
		col = cx + _round_half_up(rx * math.cos(theta))
		row = cy - _round_half_up(ry * math.sin(theta))
		if _in_bounds(row, col):
			env.graph.set(row, col)


# ── Function graphing (DrawF / DrawInv) and distribution shading ────────────────
# The curve sampler/tracer and coordinate math live in plot.py (shared with the
# Y= function grapher, Environment.regraph).


def _shade_under(env, f, lo: float, hi: float) -> None:
	"""Fill the area between the curve y=f(x) and the x-axis for lo ≤ x ≤ hi."""
	axis_row = _y_to_row(env, 0.0)
	for col in range(MAX_COL + 1):
		x = _col_to_x(env, col)
		if x < lo or x > hi:
			continue
		y = f(x)
		if y is None:
			continue
		top, bot = sorted((_y_to_row(env, y), axis_row))
		for row in range(max(top, 0), min(bot, MAX_ROW) + 1):
			env.graph.set(row, col)


@preparse_cmd
def draw_f(env: Env, formula: Thunk) -> None:
	"""DrawF expr — graph an expression in X as Y=f(X) (Func mode, regardless of mode)."""
	_trace_curve(env, _function_sampler(env, formula.eval))


@preparse_cmd
def draw_inv(env: Env, formula: Thunk) -> None:
	"""DrawInv expr — graph the inverse of expr: X becomes vertical, Y horizontal."""
	_trace_curve(env, _function_sampler(env, formula.eval), inv=True)


@preparse_cmd_func
def shade_norm(env: Env, lower: Real, upper: Real, mu: Real = 0, sigma: Real = 1) -> None:
	"""ShadeNorm(lower,upper[,μ,σ]) — draw the normal curve, shade the interval's area."""
	f = lambda x: dist.normalpdf(x, mu, sigma)
	_trace_curve(env, f)
	_shade_under(env, f, lower, upper)


@preparse_cmd_func
def shade_t(env: Env, lower: Real, upper: Real, df: Real) -> None:
	"""Shade_t(lower,upper,df) — draw the Student-t curve, shade the interval's area."""
	f = lambda x: dist.tpdf(x, df)
	_trace_curve(env, f)
	_shade_under(env, f, lower, upper)


@preparse_cmd_func
def shade_chi_sq(env: Env, lower: Real, upper: Real, df: Real) -> None:
	"""Shadeχ²(lower,upper,df) — draw the chi-square curve, shade the interval's area."""
	f = lambda x: dist.chi_sq_pdf(x, df)
	_trace_curve(env, f)
	_shade_under(env, f, lower, upper)


@preparse_cmd_func
def shade_f(env: Env, lower: Real, upper: Real, df1: Real, df2: Real) -> None:
	"""Shade𝐅(lower,upper,df1,df2) — draw the F curve, shade the interval's area."""
	f = lambda x: dist.f_pdf(x, df1, df2)
	_trace_curve(env, f)
	_shade_under(env, f, lower, upper)


def _shade_pixel(pattern: int, patres: int, row: int, col: int) -> bool:
	"""Return True if (row, col) falls on a shading line for the given pattern/resolution.

	pattern 1 — vertical lines:            col % (patres + 1) == 0
	pattern 2 — horizontal lines:          row % (patres + 1) == 0
	pattern 3 — negative-slope 45° lines:  (row - col) % (patres + 1) == 0
	pattern 4 — positive-slope 45° lines:  (row + col) % (patres + 1) == 0

	patres is the number of blank pixels between shading lines; patres=1 leaves
	one blank pixel between each bar, patres=8 leaves eight.
	"""
	step = patres + 1
	if pattern == 2:
		return row % step == 0
	elif pattern == 3:
		return (row - col) % step == 0
	elif pattern == 4:
		return (row + col) % step == 0
	else:                          # pattern 1 (default) or out-of-range
		return col % step == 0


@preparse_cmd_func
def shade(env: Env, lower: Thunk, upper: Thunk,
          xleft: Real = None, xright: Real = None,
          pattern: Real = 1, patres: Real = 1) -> None:
	"""Shade(lowerfunc,upperfunc[,Xleft,Xright,pattern,patres]) — shade between two curves.

	Draws both boundary curves on the graph, then fills the region where
	lowerfunc(X) < upperfunc(X), restricted to Xleft ≤ X ≤ Xright (defaulting to
	the window's Xmin/Xmax).

	pattern (1–4) selects the shading line direction:
	  1 = vertical (default)   2 = horizontal
	  3 = negative-slope 45°   4 = positive-slope 45°

	patres (1–8) is the number of blank pixels between shading lines: 1 leaves
	one blank pixel between each bar, 8 leaves eight.
	"""
	w = env.window
	lo = w.xmin if xleft is None else xleft
	hi = w.xmax if xright is None else xright
	flo = _function_sampler(env, lower.eval)
	fhi = _function_sampler(env, upper.eval)
	pat = max(1, min(4, py_int(pattern)))
	res = py_int(patres)
	if res < 0:
		raise DomainError(f"Shade(: patres must be ≥ 0, got {res}")
	# Draw both boundary curves.
	_trace_curve(env, flo)
	_trace_curve(env, fhi)
	# Fill the region between them.
	for col in range(0, MAX_COL + 1):
		x = _col_to_x(env, col)
		if x < lo or x > hi:
			continue
		ylo = flo(x)
		yhi = fhi(x)
		if ylo is None or yhi is None or ylo > yhi:
			continue
		top = _y_to_row(env, yhi)   # upper function → smaller row number
		bot = _y_to_row(env, ylo)   # lower function → larger row number
		for row in range(max(top, 0), min(bot, MAX_ROW) + 1):
			if _shade_pixel(pat, res, row, col):
				env.graph.set(row, col, True)


def _numeric_derivative(f, x: float, h: float = 1e-3):
	"""Central-difference slope of f at x, or None if either sample is undefined.

	Uses h=0.001 to match the calculator's nDeriv default tolerance, which is the
	same routine Tangent( uses to find the slope of the tangent line.
	"""
	fp = f(x + h)
	fm = f(x - h)
	if fp is None or fm is None:
		return None
	return (fp - fm) / (2 * h)


@preparse_cmd_func
def tangent(env: Env, formula: Thunk, value: Real) -> None:
	"""Tangent(expr,value) — graph expr and draw the line tangent to it at X=value.

	The slope is found numerically (central difference, matching nDeriv), and the
	tangent line is drawn across the full window from Xmin to Xmax.
	"""
	f = _function_sampler(env, formula.eval)
	_trace_curve(env, f)
	m = _numeric_derivative(f, value)
	y0 = f(value)                 # evaluated last so X/Y exit holding the tangent point
	if m is None or y0 is None:
		return
	w = env.window
	xmin, xmax = w.xmin, w.xmax
	tan = lambda x: y0 + m * (x - value)
	r0, c0 = _graph_to_pixel(env, xmin, tan(xmin))
	r1, c1 = _graph_to_pixel(env, xmax, tan(xmax))
	_plot_segment(env, r0, c0, r1, c1, True)


# ── Text( ────────────────────────────────────────────────────────────────────

def _num_to_chars(value: float) -> bytes:
	"""Format a real number as TI display bytes."""
	if value.is_integer() and abs(value) < 1e10:
		s = str(int(abs(value)))
	else:
		s = f'{abs(value):.10g}'
	if value < 0:
		yield 0x1A        # ⁻ (negation glyph)

	i = 0
	while i < len(s):
		ch = s[i]
		if ch in 'eE':
			yield 0x1B    # ᴇ
			i += 1
			if i < len(s) and s[i] == '+':
				i += 1             # skip '+' in exponent — TI omits it
			elif i < len(s) and s[i] == '-':
				yield 0x1A  # ⁻ before negative exponent digits
				i += 1
		else:
			yield ord(ch) # digits and '.' are the same bytes as ASCII
			i += 1


def _get_text_chars(value) -> bytes:
	"""Convert a real number or TiString to its display byte sequence.

	Complex, list, and matrix values are rejected (ERR:DATA TYPE on the calculator).
	"""
	if isinstance(value, float):
		yield from _num_to_chars(value)

	elif isinstance(value, TiString):
		for token in value.tokens:
			yield from token.display
	else:
		raise DataTypeError(f"Text(: expected a real number or string, got {type(value).__name__}")


def _blit_char(graph, row: int, col: int, glyph: bytes, height: int) -> int:
	"""Draw one glyph at (row, col) in overwrite mode; return the column after it.

	Every pixel of the glyph's width×height box is written — glyph bits where they
	fall, off everywhere else — so a 0-bit clears whatever was beneath it.  The
	caller guarantees the box is on-screen; inter-character and below-glyph spacing
	(the large font's 1px separator) is added by the caller.
	"""
	width = len(glyph)
	for dc in range(width):
		c = col + dc
		for dr in range(height):
			graph.set(row + dr, c, bool((glyph[dc] >> (height - 1 - dr)) & 1))
	return col + width


@special_func
def text(args):
	"""Text(row,col,val[,val...]) — draw values on the graph screen in small font.

	Text(-1,row,col,val[,val...]) uses the large font instead.
	"""	
	first = args.expr()
	large_mode = first == -1

	if large_mode:
		font   = LARGE_FONT
		height = 7
		row    = py_int(args.expr())
		col    = py_int(args.expr())
	else:
		font   = SMALL_FONT
		height = 6
		row    = py_int(first)
		col    = py_int(args.expr())

	if row < 0 or row + height > MAX_ROW + 1:
		raise DomainError("Text(: not enough vertical space")

	# Collect and validate every value before touching the screen, so an invalid
	# argument (e.g. a complex number) can't leave a partially-drawn string. #
	glyphs = []
	while args.has_next:
		for char in _get_text_chars(args.expr()):
			glyphs.append(font[char])
	args.end_paren_cmd()

	graph = args.env.graph
	cur_col = col
	for glyph in glyphs:
		if cur_col + len(glyph) > MAX_COL + 1:
			break
		next_col = _blit_char(graph, row, cur_col, glyph, height)
		if large_mode:
			# Large font carries a 1px separator: blank the column to the right of
			# the glyph and the padding row beneath, but only if they fit in the drawable area.
			if next_col <= MAX_COL:
				for r in range(row, row + height + 1):
					graph.set(r, next_col, False)
			if row + height <= MAX_ROW:
				for c in range(cur_col, next_col):
					graph.set(row + height, c, False)
			next_col += 1
		cur_col = next_col


# Wrap every command that marks the graph so that running it displays the graph
# (regraphing on the transition).  pxl-Test( is a query and ClrDraw clears without
# displaying, so neither appears here.
for _name in (
	'pxl_on', 'pxl_off', 'pxl_change',
	'pt_on', 'pt_off', 'pt_change',
	'vertical', 'horizontal', 'line', 'circle',
	'draw_f', 'draw_inv', 'tangent',
	'shade', 'shade_norm', 'shade_t', 'shade_chi_sq', 'shade_f',
	'text',
):
	globals()[_name] = _GraphDrawing(globals()[_name])

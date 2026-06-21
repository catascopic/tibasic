import math
import random
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, date, timedelta
from typing import Any, ClassVar

from core import TiList, is_complex_val
from core import Variable, NumericVariable, RealVariable, ListVariable, UserList, MatrixVariable, StringVariable, EquationVariable, require_real, require_int, py_int
from errors import TiError, DataTypeError, DomainError, IllegalNestError, InvalidCommandError, InvalidDimError, UndefinedError, NonRealAnsError
from modes import AngleMode, NumberMode, GraphMode, ComplexMode, DrawMode, GraphOrder, Screen
from graphscreen import GraphScreen
from graph import (
	trace_curve, trace_parametric, draw_axes, draw_grid,
	sample_function, sample_parametric, sample_polar, sample_sequence,
)
from iodevice import HomeScreenIO
from terminal import ScriptedConsole
from titoken import Token


# Sentinel parked in the sequence cache while a term is being computed; seeing it
# again means the definition refers to itself (e.g. u(n)=u(n)).
_SEQ_COMPUTING = object()


class Environment:

	def __init__(self, console=None):
		# VARIABLES
		self.numerics   = [NumericVariable()  for _ in range(27)]  # A–Z, θ
		self.lists      = [ListVariable()     for _ in range(6)]   # L1–L6
		self.matrices   = [MatrixVariable()   for _ in range(10)]  # [A]–[J]
		self.strings    = [StringVariable()   for _ in range(10)]  # Str1–Str0
		# Graph equations and their on/off selection (see GraphFunctions).  The per-mode
		# variable lists below are flat views the catalog tokens index into.
		self.graph_functions = GraphFunctions()
		self.function   = self.graph_functions.variables(GraphMode.FUNC)  # Y1–Y0
		self.parametric = self.graph_functions.variables(GraphMode.PAR)   # X1T–Y6T
		self.polar      = self.graph_functions.variables(GraphMode.POL)   # r1–r6
		self.sequence   = self.graph_functions.variables(GraphMode.SEQ)   # u, v, w
		self.user_lists = {}
		# self.stat        = [None] * 0x3D # stat vars
		self.n = NumericVariable()
		self.ans = 0
		# MODES
		self.angle_mode    = AngleMode.RAD
		self.number_mode   = NumberMode.NORMAL
		self.fix_digits    = None          # None = Float, 0–9 = Fix N
		self.graph_mode    = GraphMode.FUNC
		self.screen        = Screen.HOME   # which screen is currently displayed
		self.complex_mode  = ComplexMode.REAL
		self.draw_mode     = DrawMode.CONNECTED
		self.graph_order   = GraphOrder.SEQUENTIAL
		self.coord_on      = True
		self.polar_gc      = False
		self.axes_on       = True
		self.grid_on       = False
		self.label_on      = False
		self.expr_on       = True
		self.diagnostic_on = False
		self.dt_fmt        = 1
		self.tm_fmt        = 12
		self.clock_on      = True
		# Window / graphing variables (Xscl, Xmin, Xmax, …) plus the hidden Z-copy
		# that ZoomSto/ZoomRcl snapshot and restore (the "Z" system vars aren't
		# separate variables — they're a saved Window).
		self.window = Window()
		self.zoom_window = Window()
		# Table-screen variables (TblStart, ΔTbl, TblInput).  The table itself isn't
		# implemented, but programs can still store to and read these back.
		self.table = TableVars()
		# LCD pixel buffer (used by Pxl-/Pt-/Line/etc. drawing commands)
		self.graph = GraphScreen()
		# Text I/O device.  Commands talk to it semantically (io.disp/output/pause/…);
		# it owns how that's realized — the default is the faithful 8×16 home screen
		# painted by a Console backend (see iodevice.HomeScreenIO).
		self.io = HomeScreenIO(console or ScriptedConsole())
		# TVM finance variables (used by bal(, ΣPrn(, ΣInt(, tvm_Pmt, etc.)
		self.n_tvm = RealVariable()   # 𝐍 (number of payments)
		self.i_pct = RealVariable()   # I% (interest rate per period, as percentage)
		self.pv    = RealVariable()   # PV (present value)
		self.pmt   = RealVariable()   # PMT (payment amount)
		self.fv    = RealVariable()   # FV (future value)
		self.py    = RealVariable(1.0)  # P/Y (payments per year)
		self.cy    = RealVariable(1.0)  # C/Y (compounding periods per year)
		# Programs
		self.programs: dict[str, "Program"] = {}      # name -> stored Program
		self.execution_stack: list[object] = []       # in-flight Executions (innermost last)
		# Internal data
		self._seq_cache: dict = {}      # memoized sequence terms within one evaluation pass
		self._seq_active = False        # True while a top-level sequence evaluation owns the cache
		self._datetime_offset = timedelta(0)  # virtual_time = system_time + offset
		self._nest_depth: dict[object, int] = defaultdict(lambda: 0)  # tracks nesting depth for ILLEGAL NEST guards

	def submit(self, tokens: list[Token]):
		"""Interpret a token stream as if entered on the home screen."""
		# TODO: Should there be some kind of flag that makes newline characters raise an error?
		# On the calculator, it's impossible to get a newline character on the home screen (arguably that's what Enter does)
		# And actually, maybe if you treat NEWLINE as pressing Enter, everything works as intended
		from parser import Parser
		try:
			Parser(tokens, self).parse()
		except StopSignal:
			# It's correct to catch here because doing `prgmTEST:1->A` (where TEST just runs Stop) will not reach the following store command.
			pass
		self.io.finish()

	def to_rad(self, x: float):
		"""Convert x from the current angle mode to radians (for trig input)."""
		return x if self.in_radians else x * (math.pi / 180)

	def from_rad(self, r: float):
		"""Convert r (radians) to the current angle mode (for inverse trig output)."""
		return r if self.in_radians else r * (180 / math.pi)

	def from_deg(self, x: float):
		"""Convert x (in degrees) to the current angle mode (for DMS literals)."""
		return x * (math.pi / 180) if self.in_radians else x
	
	@property
	def in_radians(self):
		return self.angle_mode is AngleMode.RAD
	
	@property
	def real_only(self):
		return self.complex_mode is ComplexMode.REAL

	# ── I/O convenience accessors ────────────────────────────────────────────────
	# The home-screen device exposes its grid and Console backend; these shortcuts
	# let callers reach them (and swap the Console) without going through .io.  They
	# assume the default HomeScreenIO and don't apply to a grid-less device.

	@property
	def home(self):
		return self.io.home

	@property
	def console(self):
		return self.io.console

	@console.setter
	def console(self, value):
		self.io = HomeScreenIO(value)
		
	@property
	def x(self) -> Variable:
		return self.numerics[23]

	@property
	def y(self) -> Variable:
		return self.numerics[24]

	@property
	def t(self) -> Variable:
		return self.numerics[19]

	@property
	def theta(self) -> Variable:
		return self.numerics[26]

	def guard_real(self, inputs, result):
		"""Raise NonRealAnsError if real-mode is active and a real-input operation produced a complex result."""
		if self.real_only and is_complex_val(result) and not any(is_complex_val(x) for x in inputs):
			raise NonRealAnsError(repr(result))
		return result

	def set_random_seed(self, value):
		random.seed(require_real(value))

	# ── Graph display ────────────────────────────────────────────────────────────
	# The graph is only (re)plotted when it's displayed — never eagerly.  display_graph
	# is the explicit "show the graph" action (DispGraph); draw_to_graph is what a
	# drawing command calls so the graph comes up with the functions under it.

	def regraph(self):
		"""Redraw the graph from scratch: clear it, then plot the current mode's
		selected, defined functions.

		Function, parametric, polar, and sequence modes are all plotted (sequence uses
		the default Time plot — Web and uv/vw/uvw phase plots aren't supported yet).
		The grid (GridOn) and then the axes (with Xscl/Yscl tick marks) are drawn first,
		beneath the curves, when grid_on / axes_on are set; labels aren't drawn.

		Each selected, defined function is traced through the shared plotters in graph.py,
		honoring Connected/Dot draw mode.  As with DrawF, this leaves X/Y (and T, θ, or n)
		holding the last sampled point.
		"""
		self.graph.clear()
		if self.grid_on:
			draw_grid(self)
		if self.axes_on:
			draw_axes(self)
		plotter = {
			GraphMode.FUNC: self._plot_functions,
			GraphMode.PAR:  self._plot_parametric,
			GraphMode.POL:  self._plot_polar,
			GraphMode.SEQ:  self._plot_sequence,
		}.get(self.graph_mode)
		if plotter is not None:
			plotter()

	def _selected_functions(self, mode: GraphMode):
		"""The GraphFunctions in `mode` that are turned on (FnOn/stored)."""
		return [f for f in self.graph_functions.groups[mode] if f.selected]

	def _plot_functions(self):
		"""Func mode: each Yn sampled column-by-column as Y=f(X) (like DrawF)."""
		for func in self._selected_functions(GraphMode.FUNC):
			equation = func.equations[0].value
			if equation is not None:
				trace_curve(self, sample_function(self, lambda eq=equation: eq.eval(self)))

	def _plot_parametric(self):
		"""Par mode: sweep T over [Tmin,Tmax] by Tstep, plotting (XnT(T), YnT(T)).
		Both halves of the pair must be defined for the curve to plot."""
		w = self.window
		tmin, tmax, tstep = w.tmin.resolve(), w.tmax.resolve(), w.tstep.resolve()
		for func in self._selected_functions(GraphMode.PAR):
			x_eq, y_eq = func.equations[0].value, func.equations[1].value
			if x_eq is not None and y_eq is not None:
				point = sample_parametric(
					self, lambda eq=x_eq: eq.eval(self), lambda eq=y_eq: eq.eval(self))
				trace_parametric(self, point, tmin, tmax, tstep)

	def _plot_polar(self):
		"""Pol mode: sweep θ over [θmin,θmax] by θstep, plotting (r·cosθ, r·sinθ)."""
		w = self.window
		tmin, tmax, tstep = w.theta_min.resolve(), w.theta_max.resolve(), w.theta_step.resolve()
		for func in self._selected_functions(GraphMode.POL):
			r_eq = func.equations[0].value
			if r_eq is not None:
				point = sample_polar(self, lambda eq=r_eq: eq.eval(self))
				trace_parametric(self, point, tmin, tmax, tstep)

	def _plot_sequence(self):
		"""Seq mode, Time plot (the default): plot the point (n, s(n)) for each selected
		sequence over n = PlotStart … nMax stepping by PlotStep.

		Web and uv/vw/uvw phase plots aren't drawn — they'd need a Seq plot-type setting
		the emulator doesn't model yet, so only the default Time plot is produced.  The
		whole sweep shares one memo cache (see _sequence_pass) so a recursive sequence is
		evaluated bottom-up rather than recomputed from nMin at every point.
		"""
		w = self.window
		nmax, start, step = w.n_max.resolve(), w.plot_start.resolve(), w.plot_step.resolve()
		with self._sequence_pass():
			for index, func in enumerate(self.graph_functions.groups[GraphMode.SEQ]):
				if func.selected and func.equations[0].value is not None:
					trace_parametric(self, sample_sequence(self, index), start, nmax, step)

	# ── Sequence evaluation (Seq mode u/v/w) ─────────────────────────────────────
	# A sequence term is either an explicit initial value from its u(nMin) list or,
	# beyond those, the recurrence formula evaluated with n bound to the index — which
	# may refer back to earlier terms (u(n-1), v(n-2), …).  Terms are memoized within
	# an evaluation pass so a recursive definition stays linear and self-reference is
	# caught instead of recursing forever.

	@contextmanager
	def _sequence_pass(self):
		"""Own a fresh memo cache for the duration of a top-level sequence evaluation,
		shared by every nested term it pulls in.  A new top-level call starts clean, so
		a redefined sequence is never read from a stale cache."""
		top_level = not self._seq_active
		if top_level:
			self._seq_cache = {}
			self._seq_active = True
		try:
			yield
		finally:
			if top_level:
				self._seq_active = False

	def eval_sequence(self, index: int, at):
		"""Evaluate sequence 𝑢/𝑣/𝑤 (index 0/1/2) at term number `at`, rounded to an
		integer n.  Returns the term's value; raises DomainError below nMin or on a
		self-referential definition, and UndefinedError if the sequence has no formula
		past its initial terms."""
		with self._sequence_pass():
			return self._eval_sequence(index, py_int(at))

	def _eval_sequence(self, index: int, k: int):
		func = self.graph_functions.groups[GraphMode.SEQ][index]
		n_min = py_int(self.window.n_min.resolve())
		if k < n_min:
			raise DomainError(f"Sequence index {k} is below nMin ({n_min})")
		# Initial terms come straight from the u(nMin) list (element i is the value at
		# n_min + i — chronological order).
		initial = func.initial
		if initial is not None and k - n_min < len(initial.data):
			return initial.data[k - n_min]
		equation = func.equations[0].value
		if equation is None:
			raise UndefinedError(f"Sequence {'uvw'[index]} is not defined")
		key = (index, k)
		cache = self._seq_cache
		if key in cache:
			term = cache[key]
			if term is _SEQ_COMPUTING:
				raise DomainError(f"Sequence {'uvw'[index]} references itself at n={k}")
			return term
		cache[key] = _SEQ_COMPUTING
		saved_n = self.n.value
		self.n.value = float(k)
		try:
			term = equation.eval(self)
		finally:
			self.n.value = saved_n
		cache[key] = term
		return term

	def store_sequence_initial(self, index: int, value) -> None:
		"""Set sequence 𝑢/𝑣/𝑤's u(nMin) initial-value list ({…}→u(nMin)).  A scalar is
		wrapped as a one-element list (a single initial term)."""
		func = self.graph_functions.groups[GraphMode.SEQ][index]
		func.initial = value if isinstance(value, TiList) else TiList([value])

	def display_graph(self):
		"""DispGraph — make the graph the active screen and re-plot the functions."""
		self.screen = Screen.GRAPH
		self.regraph()

	def draw_to_graph(self):
		"""A drawing command is about to modify the graph: display it, re-plotting on
		the transition so the functions sit beneath the drawing.  No re-plot if the
		graph is already up — that would erase earlier drawing."""
		if self.screen is not Screen.GRAPH:
			self.display_graph()

	# ── Zoom memory (ZoomSto / ZoomRcl) ──────────────────────────────────────────
	# The Z-window system variables (ZXmin, Zθstep, …) are a saved snapshot of the
	# whole window rather than separate variables.

	def zoom_store(self):
		"""ZoomSto — save the current window into the Zoom memory."""
		self.zoom_window = self.window.copy()

	def zoom_recall(self):
		"""ZoomRcl — restore the window saved by the last ZoomSto."""
		self.window = self.zoom_window.copy()

	# ── Virtual clock ────────────────────────────────────────────────────────────

	def now(self) -> datetime:
		"""Current datetime adjusted by any offset set via setDate/setTime."""
		return datetime.now() + self._datetime_offset

	def set_date(self, year, month, day):
		now = datetime.now()
		v = now + self._datetime_offset
		new_v = datetime(py_int(year), py_int(month), py_int(day), v.hour, v.minute, v.second)
		self._datetime_offset = new_v - now

	def set_time(self, hour, minute, second):
		now = datetime.now()
		v = now + self._datetime_offset
		new_v = datetime(v.year, v.month, v.day, py_int(hour), py_int(minute), py_int(second))
		self._datetime_offset = new_v - now

	# ── Nullary helpers (used by nullary= fields in tokens) ──────────────────────

	def get_date(self):
		d = self.now()
		return TiList([float(d.year), float(d.month), float(d.day)])

	def get_time(self):
		t = self.now()
		return TiList([float(t.hour), float(t.minute), float(t.second)])

	def start_tmr(self):
		return float(int(self.now().timestamp()))

	def get_dt_fmt(self):
		return float(self.dt_fmt)

	def get_tm_fmt(self):
		return float(self.tm_fmt)

	def is_clock_on(self):
		return float(self.clock_on)

	def get_key(self):
		return float(self.io.get_key())

	def rand(self):
		return random.random()

	@contextmanager
	def nest_guard(self, key: object, max_depth: int = 0):
		if self._nest_depth[key] > max_depth:
			raise IllegalNestError(key)
		self._nest_depth[key] += 1
		try:
			yield
		finally:
			self._nest_depth[key] -= 1

	def _iter_values(self):
		# Variable display names, reconstructed from each storage list's index so
		# this doesn't depend on catalog's token tables.  Order/spelling mirror the
		# TI charset: A–Z then θ; L₁–L₆ (subscript digits); [A]–[J]; Str1–Str0.
		numeric_names = [chr(0x41 + i) for i in range(26)] + ['θ']
		named = (
			zip(numeric_names, self.numerics),
			((f"L{chr(0x2081 + i)}", var) for i, var in enumerate(self.lists)),
			((f"[{chr(0x41 + i)}]",  var) for i, var in enumerate(self.matrices)),
			((f"Str{(i + 1) % 10}",  var) for i, var in enumerate(self.strings)),
		)
		for group in named:
			for name, var in group:
				if var.value is not None:
					yield name, var.value
		for name, lst in self.user_lists.items():
			yield f"${name}", lst
		yield "Ans", self.ans

	def dump(self):
		for name, value in self._iter_values():
			print(f"{name}= {int(value) if isinstance(value, float) and value.is_integer() else value!r}")

	def __repr__(self):
		return f"ENV({','.join(f"{name}={value!r}" for name, value in self._iter_values())})"

	def current_execution(self):
		"""Return the innermost in-flight Execution.

		Raises InvalidCommandError if called outside a program (e.g. from the
		home screen), matching the calculator's ERR:INVALID for control-flow
		commands like Return, Goto, and End.
		"""
		if not self.execution_stack:
			raise InvalidCommandError("This command cannot be used outside a program")
		return self.execution_stack[-1]

	def print_screen(self):
		pass


# ── Window variables ──────────────────────────────────────────────────────────

class _DeltaVariable(RealVariable):
	"""ΔX / ΔY — not stored, but derived from the window bounds as
	(hi − lo) / divisions.  Storing one adjusts `hi` (keeping `lo` and the pixel
	count fixed) so the relation holds, matching the calculator.

	Holds a back-reference to its Window so it can read `lo` and write `hi`; this
	is the only window variable that touches its siblings, so it's the only one
	that needs the reference.
	"""

	def __init__(self, window: "Window", lo_attr: str, hi_attr: str, divisions: int):
		self.window = window
		self.lo_attr = lo_attr
		self.hi_attr = hi_attr
		self.divisions = divisions

	def resolve(self):
		lo = getattr(self.window, self.lo_attr).resolve()
		hi = getattr(self.window, self.hi_attr).resolve()
		return (hi - lo) / self.divisions

	def store(self, new_value) -> None:
		delta = require_real(new_value)
		lo = getattr(self.window, self.lo_attr).resolve()
		getattr(self.window, self.hi_attr).store(lo + self.divisions * delta)


class _IntWindowVariable(RealVariable):
	"""A window variable constrained to whole numbers (nMin, nMax)."""

	def normalize(self, value):
		return require_int(value)


class _XresVariable(RealVariable):
	"""Xres — Function-graph resolution; an integer 1–8."""

	def normalize(self, value):
		v = require_int(value)
		if not (1 <= v <= 8):
			raise DomainError(f"Xres must be an integer 1-8, got {v:g}")
		return v


class _FactorVariable(RealVariable):
	"""XFact / YFact — Zoom In/Out scaling factors; must be ≥ 1."""

	def normalize(self, value):
		v = require_real(value)
		if v < 1:
			raise DomainError(f"Zoom factor must be ≥ 1, got {v:g}")
		return v


class Window:
	"""The TI graph/window variables — everything that defines a graphing setup.

	Most are plain RealVariables; the ones with side effects use the variable
	subclasses above (ΔX/ΔY derived, Xres/nMin/nMax integers, XFact/YFact ≥ 1).
	`copy()` snapshots it for the Zoom memory (see Environment.zoom_store).
	"""

	# The independently-stored variables — what copy() snapshots.  ΔX/ΔY are
	# omitted: they're derived from the bounds, so a snapshot of the bounds carries
	# them implicitly.
	_STORED = (
		'xmin', 'xmax', 'ymin', 'ymax', 'xscl', 'yscl', 'xres',
		'tmin', 'tmax', 'tstep', 'theta_min', 'theta_max', 'theta_step',
		'n_min', 'n_max', 'plot_start', 'plot_step',
		'x_fact', 'y_fact',
	)

	def __init__(self):
		# Screen bounds and axis scales
		self.xmin       = RealVariable(-10.0)
		self.xmax       = RealVariable(10.0)
		self.ymin       = RealVariable(-10.0)
		self.ymax       = RealVariable(10.0)
		self.xscl       = RealVariable(1.0)
		self.yscl       = RealVariable(1.0)
		self.xres       = _XresVariable(1.0)
		# ΔX/ΔY: derived from the bounds (94 columns / 62 rows of intervals)
		self.delta_x    = _DeltaVariable(self, 'xmin', 'xmax', 94)
		self.delta_y    = _DeltaVariable(self, 'ymin', 'ymax', 62)
		# Parametric (T) and polar (θ) sweep ranges
		self.tmin       = RealVariable()
		self.tmax       = RealVariable()
		self.tstep      = RealVariable()
		self.theta_min  = RealVariable()
		self.theta_max  = RealVariable()
		self.theta_step = RealVariable()
		# Sequence (n) range — integers — and which n values are graphed
		self.n_min      = _IntWindowVariable(1.0)
		self.n_max      = _IntWindowVariable(10.0)
		self.plot_start = RealVariable(1.0)
		self.plot_step  = RealVariable(1.0)
		# Note: the recursive sequence initial conditions u(nMin)/v(nMin)/w(nMin) are
		# NOT window variables — they're lists stored per sequence on GraphFunction.initial
		# (set via {…}→u(nMin), read via u(nMin)), with no user-facing variable, mirroring
		# how the calculator keeps them outside the addressable variable space.
		# Zoom In/Out factors
		self.x_fact     = _FactorVariable(4.0)
		self.y_fact     = _FactorVariable(4.0)

	def copy(self) -> "Window":
		"""A snapshot of the stored window variables (for ZoomSto/ZoomRcl).

		Copies raw values, not the Variable objects, so the clone's derived ΔX/ΔY
		stay bound to the clone rather than the original.
		"""
		clone = Window()
		for name in self._STORED:
			getattr(clone, name).value = getattr(self, name).value
		return clone


class TableVars:
	"""Table-screen variables.  The table feature isn't implemented; these exist
	only so programs can store to and read them back."""

	def __init__(self):
		self.tbl_start = RealVariable()       # TblStart
		self.delta_tbl = RealVariable(1.0)    # ΔTbl
		self.tbl_input = ListVariable()       # TblInput — a 7-element list


# ── Graph functions (the plottable equations + their on/off selection) ──────────

class GraphFunction:
	"""One selectable graph function: its equation variable(s) plus an on/off
	`selected` flag.  Function-, polar-, and sequence-mode functions have a single
	equation; a parametric function is an X/Y pair that shares this one flag, so
	storing to either equation selects the pair together."""

	def __init__(self, equation_count: int):
		self.selected = False
		# Storing to any of this function's equations selects it (Y1=… re-enables Y1),
		# matching the calculator; the flag stays here, off the EquationVariable.
		self.equations = [EquationVariable(on_store=self._select) for _ in range(equation_count)]
		# Seq mode only: the u(nMin) initial-value list (a TiList), defining the explicit
		# starting terms a recursive sequence builds on.  None until set; unused by the
		# other graph modes.
		self.initial = None

	def _select(self) -> None:
		self.selected = True


class GraphFunctions:
	"""All plottable equations, grouped by graph mode, each carrying its own on/off
	state.  The graph (when displayed) plots the current mode's selected, defined
	functions; selection is tracked here rather than on the equation variables, and
	survives a mode switch (storing Y1 in Polar mode selects it for when you return
	to Function mode).  FnOn/FnOff toggle the current mode's selection.
	"""

	# graph mode → (function count, equations per function)
	_LAYOUT = {
		GraphMode.FUNC: (10, 1),   # Y1–Y0
		GraphMode.PAR:  (6, 2),    # X1T/Y1T – X6T/Y6T (pairs)
		GraphMode.POL:  (6, 1),    # r1–r6
		GraphMode.SEQ:  (3, 1),    # u, v, w
	}

	def __init__(self):
		self.groups = {
			mode: [GraphFunction(per_func) for _ in range(count)]
			for mode, (count, per_func) in self._LAYOUT.items()
		}

	def variables(self, mode: GraphMode) -> list:
		"""Flat list of a mode's equation variables, in token order (X1T, Y1T, X2T,
		… for parametric) — the view the catalog tokens index into."""
		return [eq for func in self.groups[mode] for eq in func.equations]

	def set_selected(self, mode: GraphMode, on: bool, numbers=()) -> None:
		"""FnOn/FnOff: select (on=True) or deselect the listed 1-based functions in
		`mode`, or all of them when `numbers` is empty.  Number 0 means the 10th
		function (Y0), matching the calculator's 1-9,0 numbering."""
		funcs = self.groups[mode]
		if not numbers:
			for func in funcs:
				func.selected = on
			return
		for n in numbers:
			index = n - 1 if n >= 1 else 9        # FnOn 0 → the 10th function (Y0)
			if not (0 <= index < len(funcs)):
				raise DomainError(f"FnOn/FnOff: function number out of range: {n}")
			funcs[index].selected = on


class ReturnSignal(Exception):
	"""Raised by Return to exit the current sub-program and return to the caller."""


class StopSignal(Exception):
	"""Raised by Stop to terminate all program execution immediately."""

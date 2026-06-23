import math
import random
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any, ClassVar

import math as _math

from core import TiList, TiEquation, is_complex_val
from core import require_real, require_int, py_int
from errors import TiError, DataTypeError, DomainError, IllegalNestError, InvalidCommandError, InvalidDimError, UndefinedError, NonRealAnsError
from modes import AngleMode, NumberMode, GraphMode, ComplexMode, DrawMode, GraphOrder, Screen
from graphscreen import GraphScreen
from graph import (
	draw_axes, draw_grid,
	trace_curve, trace_parametric,
	sample_function, sample_parametric, sample_polar, sample_sequence,
	_param_values, MAX_COL,
)
from graphmodes import HANDLERS as GRAPH_MODE_HANDLERS
from accessors import NumericVar
from iodevice import HomeScreenIO
from terminal import ScriptedConsole
from titoken import Token


class GraphStyle(Enum):
	LINE        = 'line'
	THICK       = 'thick'
	SHADE_ABOVE = 'shade_above'
	SHADE_BELOW = 'shade_below'
	TRACE       = 'trace'
	ANIMATE     = 'animate'
	DOT         = 'dot'


@dataclass
class FuncData:
	"""One Y= function slot: equation, on/off, style.  Knows how to plot itself."""
	equation: 'TiEquation | None' = None
	selected: bool = False
	style: GraphStyle = GraphStyle.LINE

	def is_defined(self) -> bool:
		return self.equation is not None

	def plot(self, env) -> None:
		eq = self.equation
		trace_curve(env, sample_function(env, lambda: eq.eval(env)))

	def fit_points(self, env) -> list:
		"""Y values sampled across the current X window, for ZoomFit."""
		w = env.window
		xmin = w.xmin
		delta = (w.xmax - xmin) / MAX_COL
		eq = self.equation
		f = sample_function(env, lambda: eq.eval(env))
		return [y for i in range(MAX_COL + 1) if (y := f(xmin + i * delta)) is not None]


@dataclass
class ParData:
	"""One parametric pair (XnT + YnT): both halves, shared on/off and style."""
	x_eq: 'TiEquation | None' = None
	y_eq: 'TiEquation | None' = None
	selected: bool = False
	style: GraphStyle = GraphStyle.LINE

	def is_defined(self) -> bool:
		return self.x_eq is not None and self.y_eq is not None

	def plot(self, env) -> None:
		w = env.window
		x_eq, y_eq = self.x_eq, self.y_eq
		trace_parametric(env,
			sample_parametric(env, lambda: x_eq.eval(env), lambda: y_eq.eval(env)),
			w.tmin, w.tmax, w.tstep)

	def fit_points(self, env) -> list:
		"""(x, y) pairs sampled over T, for ZoomFit."""
		w = env.window
		x_eq, y_eq = self.x_eq, self.y_eq
		point = sample_parametric(env, lambda: x_eq.eval(env), lambda: y_eq.eval(env))
		return [xy for t in _param_values(w.tmin, w.tmax, w.tstep) if (xy := point(t)) is not None]


@dataclass
class PolarData:
	"""One polar equation slot (r=): equation, on/off, style."""
	equation: 'TiEquation | None' = None
	selected: bool = False
	style: GraphStyle = GraphStyle.LINE

	def is_defined(self) -> bool:
		return self.equation is not None

	def plot(self, env) -> None:
		w = env.window
		eq = self.equation
		trace_parametric(env,
			sample_polar(env, lambda: eq.eval(env)),
			w.theta_min, w.theta_max, w.theta_step)

	def fit_points(self, env) -> list:
		"""(x, y) pairs sampled over θ, for ZoomFit."""
		w = env.window
		eq = self.equation
		point = sample_polar(env, lambda: eq.eval(env))
		return [xy for t in _param_values(w.theta_min, w.theta_max, w.theta_step) if (xy := point(t)) is not None]


@dataclass
class SeqData:
	"""One sequence slot (u/v/w): index, equation, on/off, style."""
	seq_index: int
	equation: 'TiEquation | None' = None
	selected: bool = False
	style: GraphStyle = GraphStyle.LINE

	def is_defined(self) -> bool:
		return self.equation is not None

	def plot(self, env) -> None:
		w = env.window
		trace_parametric(env,
			sample_sequence(env, self.seq_index),
			w.plot_start, w.n_max, w.plot_step)

	def fit_points(self, env) -> list:
		"""(x, y) pairs sampled over n, for ZoomFit."""
		w = env.window
		point = sample_sequence(env, self.seq_index)
		return [xy for t in _param_values(w.plot_start, w.n_max, w.plot_step) if (xy := point(t)) is not None]


# Sentinel parked in the sequence cache while a term is being computed; seeing it
# again means the definition refers to itself (e.g. u(n)=u(n)).
_SEQ_COMPUTING = object()


_NUMERIC_NAMES = tuple(chr(0x41 + i) for i in range(26)) + ('theta',)
_MATRIX_NAMES  = tuple(chr(0x41 + i) for i in range(10))   # A–J


class NumericVars:
	"""Storage for the real/complex variables A–Z and θ: a fixed set of named slots,
	None until assigned.  This is *just the data* — the access logic (auto-init, type
	checks) lives in accessors.NumericVar — so a value can be read directly as
	env.numerics.A (or env.numerics.theta), which is handy in tests."""

	__slots__ = _NUMERIC_NAMES

	def __init__(self):
		for name in self.__slots__:
			setattr(self, name, None)

	def __repr__(self):
		live = {n: getattr(self, n) for n in self.__slots__ if getattr(self, n) is not None}
		return f"NumericVars({live})"


class MatrixVars:
	"""Storage for matrix variables [A]–[J]: a fixed set of named slots, None until
	assigned.  Access logic (UndefinedError on resolve, deep-copy on store) lives in
	accessors.MatrixVar; env.matrices.A reads the raw TiMatrix | None directly."""

	__slots__ = _MATRIX_NAMES

	def __init__(self):
		for name in self.__slots__:
			setattr(self, name, None)

	def __repr__(self):
		live = {n: getattr(self, n) for n in self.__slots__ if getattr(self, n) is not None}
		return f"MatrixVars({live})"


class Environment:

	def __init__(self, console=None):
		# VARIABLES
		self.numerics   = NumericVars()     # A–Z, θ  (named slots, None until assigned)
		self.matrices   = MatrixVars()      # [A]–[J] (named slots, None until assigned)
		self.lists      = [None] * 6        # L1–L6  (TiList | None)
		self.strings    = [None] * 10       # Str1–Str0 (TiString | None)
		# Plottable functions, one slot per selectable entry.  Each carries its equation,
		# on/off selection flag, and draw style, and knows how to plot itself.
		self.function   = [FuncData()    for _ in range(10)]     # Y1–Y0
		self.parametric = [ParData()     for _ in range(6)]      # X1T/Y1T – X6T/Y6T
		self.polar      = [PolarData()   for _ in range(6)]      # r1–r6
		self.sequence   = [SeqData(i)    for i in range(3)]      # u, v, w
		self.sequence_initial: list = [None, None, None]         # u/v/w(nMin) lists
		self.user_lists = {}
		# self.stat        = [None] * 0x3D # stat vars
		self.n: float = 0.0
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
		# Window / graphing variables (Xscl, Xmin, Xmax, …) and the ZoomSto snapshot.
		# The snapshot is a second Window whose fields ARE the Z-window system variables
		# (ZXmin, ZTmax, …): the Z-tokens read/write it directly (see catalog), and
		# ZoomSto/ZoomRcl copy between the two.
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
		self.n_tvm: float = 0.0   # 𝐍 (number of payments)
		self.i_pct: float = 0.0   # I% (interest rate per period, as percentage)
		self.pv:    float = 0.0   # PV (present value)
		self.pmt:   float = 0.0   # PMT (payment amount)
		self.fv:    float = 0.0   # FV (future value)
		self.py:    float = 1.0   # P/Y (payments per year)
		self.cy:    float = 1.0   # C/Y (compounding periods per year)
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
		
	# X/Y/T/θ are ordinary numeric variables; these return a bound Reference (which
	# exposes .get()/.set(), .resolve(), .scoped()) so the graphers and CALC commands can
	# read/write them without knowing the storage layout.
	@property
	def x(self):
		return NumericVar('X').reference(self)

	@property
	def y(self):
		return NumericVar('Y').reference(self)

	@property
	def t(self):
		return NumericVar('T').reference(self)

	@property
	def theta(self):
		return NumericVar('theta').reference(self)

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
		self.graph_mode_handler.plot(self)

	@property
	def graph_mode_handler(self):
		"""The strategy object for the current graph mode (see graphmodes.py).  Owns
		plotting, the ZoomFit extent, and the ZStandard reset of mode-specific vars."""
		return GRAPH_MODE_HANDLERS[self.graph_mode]

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
		n_min = py_int(self.window.n_min)
		if k < n_min:
			raise DomainError(f"Sequence index {k} is below nMin ({n_min})")
		# Initial terms come straight from the u(nMin) list (element i is the value at
		# n_min + i — chronological order).
		initial = self.sequence_initial[index]
		if initial is not None and k - n_min < len(initial.data):
			return initial.data[k - n_min]
		equation = self.sequence[index].equation
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
		saved_n = self.n
		self.n = float(k)
		try:
			term = equation.eval(self)
		finally:
			self.n = saved_n
		cache[key] = term
		return term

	def store_sequence_initial(self, index: int, value) -> None:
		"""Set sequence 𝑢/𝑣/𝑤's u(nMin) initial-value list ({…}→u(nMin)).  A scalar is
		wrapped as a one-element list (a single initial term)."""
		self.sequence_initial[index] = value if isinstance(value, TiList) else TiList([value])

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
	# zoom_window holds the Z-window system variables (ZXmin, Zθstep, …); ZoomSto
	# copies the live window into them and ZoomRcl copies them back.

	def zoom_store(self):
		"""ZoomSto — copy the current window into the Z-window variables."""
		self.zoom_window = self.window.copy()

	def zoom_recall(self):
		"""ZoomRcl — restore the window from the Z-window variables."""
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
		for disp, attr in zip(numeric_names, _NUMERIC_NAMES):
			value = getattr(self.numerics, attr)
			if value is not None:
				yield disp, value
		for i, val in enumerate(self.lists):
			if val is not None:
				yield f"L{chr(0x2081 + i)}", val
		for name in _MATRIX_NAMES:
			val = getattr(self.matrices, name)
			if val is not None:
				yield f"[{name}]", val
		for i, val in enumerate(self.strings):
			if val is not None:
				yield f"Str{(i + 1) % 10}", val
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

class Window:
	"""The TI graph/window variables — plain floats.

	Most variables are plain float | None (None = unset); the validation logic for
	the special ones (Xres, nMin/nMax, XFact/YFact, ΔX/ΔY) lives in the accessor
	subclasses in accessors.py and is enforced only on the TI-BASIC store path.
	Internal Python code (zoom.py, graphmodes.py) reads and writes them directly.

	ΔX and ΔY are computed properties, not stored: they follow from the x/y bounds
	and the pixel grid (94 column intervals / 62 row intervals).
	"""

	_STORED = (
		'xmin', 'xmax', 'ymin', 'ymax', 'xscl', 'yscl', 'xres',
		'tmin', 'tmax', 'tstep', 'theta_min', 'theta_max', 'theta_step',
		'n_min', 'n_max', 'plot_start', 'plot_step',
		'x_fact', 'y_fact',
	)

	def __init__(self):
		# Screen bounds and axis scales
		self.xmin       = -10.0
		self.xmax       =  10.0
		self.ymin       = -10.0
		self.ymax       =  10.0
		self.xscl       =  1.0
		self.yscl       =  1.0
		self.xres       =  1.0
		# Parametric (T) sweep range
		self.tmin       =  0.0
		self.tmax       =  2 * _math.pi
		self.tstep      =  _math.pi / 24
		# Polar (θ) sweep range
		self.theta_min  =  0.0
		self.theta_max  =  2 * _math.pi
		self.theta_step =  _math.pi / 24
		# Sequence (n) range — stored as int-valued floats
		self.n_min      =  1.0
		self.n_max      = 10.0
		self.plot_start =  1.0
		self.plot_step  =  1.0
		# Zoom In/Out factors
		self.x_fact     =  4.0
		self.y_fact     =  4.0

	@property
	def delta_x(self) -> float:
		"""ΔX — the graph width per pixel column: (Xmax − Xmin) / 94."""
		return (self.xmax - self.xmin) / 94

	@property
	def delta_y(self) -> float:
		"""ΔY — the graph height per pixel row: (Ymax − Ymin) / 62."""
		return (self.ymax - self.ymin) / 62

	def copy(self) -> "Window":
		"""Snapshot all stored window variables (for ZoomSto/ZoomRcl)."""
		clone = Window()
		for name in self._STORED:
			setattr(clone, name, getattr(self, name))
		return clone


class TableVars:
	"""Table-screen variables.  The table feature isn't implemented; these exist
	only so programs can store to and read them back."""

	def __init__(self):
		self.tbl_start: float | None = None   # TblStart
		self.delta_tbl: float = 1.0           # ΔTbl
		self.tbl_input = None                 # TblInput — a 7-element list (TiList | None)


class ReturnSignal(Exception):
	"""Raised by Return to exit the current sub-program and return to the caller."""


class StopSignal(Exception):
	"""Raised by Stop to terminate all program execution immediately."""


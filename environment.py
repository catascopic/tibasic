import math
import random
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta

import math as _math

from core import TiList, is_complex_val, require_real, require_int, py_int, require_list
from errors import DataTypeError, IllegalNestError, InvalidCommandError, InvalidDimError, NonRealAnsError
from modes import AngleMode, NumberMode, GraphMode, ComplexMode, DrawMode, GraphOrder, Screen
from graphscreen import GraphScreen
from graph import (
	draw_axes, draw_grid,
	GraphStyle, FuncData, ParData, PolarData, SeqData,
)
from graphmodes import HANDLERS as GRAPH_MODE_HANDLERS
from homescreen import HomeScreen
from terminal import ScriptedConsole
from tokenbase import Token, Accessor


_NUMERIC_NAMES = tuple(chr(0x41 + i) for i in range(26)) + ('theta',)

class AlphaList:
	
	def __init__(self, size):
		self.data = [None] * size

	def __getitem__(self, i):
		return self.data[i]
	
	def __setitem__(self, i, value):
		self.data[i] = value
	
	def __len__(self):
		return len(self.data)
	
	def __iter__(self):
		return iter(self.data)
	
	def _alpha_index(self, index):
		if index >= len(self.data):
			raise IndexError(f"item {_NUMERIC_NAMES[index]} ({index}) out of range")
		return index

	def items(self):
		return zip(_NUMERIC_NAMES, self.data)

	def __repr__(self):
		return f"AlphaList({''.join(f'{n}={v}' for n, v in self.items() if v is not None)})"


def _make_prop(index):
	def fget(self):
		return self.data[self._alpha_index(index)]
	def fset(self, value):
		self.data[self._alpha_index(index)] = value
	return property(fget, fset)

for _i, _name in enumerate(_NUMERIC_NAMES):
	setattr(AlphaList, _name, _make_prop(_i))


class Environment:

	def __init__(self, console=None):
		# VARIABLES
		self.numerics   = AlphaList(27)  # A–Z, θ  (index- or name-addressable)
		self.matrices   = AlphaList(10)  # [A]–[J]
		self.lists      = [None] * 6       # L1–L6  (TiList | None)
		self.strings    = [None] * 10      # Str1–Str0 (TiString | None)
		self.pics       = [None] * 10      # Pic0–Pic9, by number (Bitmap | None)
		self.function   = [FuncData()  for _ in range(10)]  # Y1–Y0
		self.parametric = [ParData()   for _ in range(6)]   # X1T/Y1T – X6T/Y6T
		self.polar      = [PolarData() for _ in range(6)]   # r1–r6
		self.sequence   = [SeqData()   for _ in range(3)]   # u, v, w
		self.user_lists = {}
		self.n = 0.0
		self.ans = 0.0
		# MODES
		self.angle_mode    = AngleMode.RAD
		self.number_mode   = NumberMode.NORMAL
		self.fix_digits    = None          # None = Float, 0–9 = Fix N
		self._graph_mode   = GraphMode.FUNC   # backing field; use the property after init
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
		self.window = Window()
		# Table-screen variables (TblStart, ΔTbl, TblInput).  The table itself isn't
		# implemented, but programs can still store to and read these back.
		self.table = TableVars()
		# The two display surfaces — peer calculator state.  The graph is a 64×96
		# pixel buffer (Pxl-/Pt-/Line/…); the home screen is the 16×8 character grid
		# (Disp/Output(/Input).  Commands mutate these directly and ping the console.
		self.graph = GraphScreen()
		self.home = HomeScreen()
		# The transient menu modal: a MenuScreen while a Menu( is up (Screen.MENU),
		# None otherwise.  Set/cleared by the Menu( command around the blocking call.
		self.menu = None
		# The frontend: it renders the model and supplies input.  Assigning it attaches
		# this env (see the `console` setter) so it can read home/graph on present().
		self.console = console or ScriptedConsole()
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
		self.console.finish()

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

	@property
	def graph_mode(self):
		return self._graph_mode

	@graph_mode.setter
	def graph_mode(self, value):
		if value is not self._graph_mode:
			self.graph.valid = False
		self._graph_mode = value

	# ── Frontend ─────────────────────────────────────────────────────────────────
	# A console renders the model (home/graph) and supplies input.  Assigning one
	# back-references this env onto it, so its methods can read the screens without
	# being handed them; swapping the console (terminal ↔ free-form ↔ scripted) is
	# just an assignment.

	@property
	def console(self):
		return self._console

	@console.setter
	def console(self, value):
		self._console = value
		value.env = self

	def guard_real(self, inputs, result):
		"""Raise NonRealAnsError if real-mode is active and a real-input operation produced a complex result."""
		if self.real_only and is_complex_val(result) and not any(is_complex_val(x) for x in inputs):
			raise NonRealAnsError(repr(result))
		return result

	def set_random_seed(self, value):
		random.seed(require_real(value))

	# ── Graph display ────────────────────────────────────────────────────────────
	# The graph is only (re)plotted when it's displayed — never eagerly.

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
		self.graph.valid = True

	@property
	def graph_mode_handler(self):
		"""The strategy object for the current graph mode (see graphmodes.py).  Owns
		plotting, the ZoomFit extent, and the ZStandard reset of mode-specific vars."""
		return GRAPH_MODE_HANDLERS[self.graph_mode]

	def display_graph(self):
		"""DispGraph — make the graph the active screen and re-plot the functions if stale."""
		self.screen = Screen.GRAPH
		if not self.graph.valid:
			self.regraph()

	@contextmanager
	def draw_to_graph(self):
		"""Context manager for drawing commands: regraph if stale, then on normal exit
		switch to the graph screen and notify the frontend.  On exception, screen and
		frontend state are left unchanged."""
		if not self.graph.valid:
			self.regraph()
		yield
		self.screen = Screen.GRAPH
		self.console.present()

	def zoom_store(self):
		"""ZoomSto — copy the live window variables into the Z-window variables."""
		w = self.window
		for name in Window._ZOOM_VARS:
			setattr(w, 'z' + name, getattr(w, name))

	def zoom_recall(self):
		"""ZoomRcl — restore the live window variables from the Z-window variables."""
		w = self.window
		for name in Window._ZOOM_VARS:
			setattr(w, name, getattr(w, 'z' + name))

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
		return float(self.console.read_key())

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
		for name, value in self.numerics.items():
			if value is not None:
				yield name, value
		for i, val in enumerate(self.lists):
			if val is not None:
				yield f"L{chr(0x2081 + i)}", val
		for name, value in self.matrices.items():
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
			print(f"{name}={int(value) if isinstance(value, float) and value.is_integer() else value!r}")

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

	def print_screen(self, path):
		"""Save the active screen to `path` as a BMP — the graph's pixels, the home
		grid, or a Menu( modal, rasterized through the font, whichever is displayed."""
		surface = {Screen.GRAPH: self.graph, Screen.MENU: self.menu}.get(self.screen, self.home)
		surface.print_screen(path)


# ── Window variables ──────────────────────────────────────────────────────────

class Window:
	"""The TI graph/window variables — plain floats.

	Most variables are plain float | None (None = unset); the validation logic for
	the special ones (Xres, nMin/nMax, XFact/YFact, ΔX/ΔY) lives in the accessor
	subclasses in tokentypes.py and is enforced only on the TI-BASIC store path.
	Internal Python code (zoom.py, graphmodes.py) reads and writes them directly.

	ΔX and ΔY are computed properties, not stored: they follow from the x/y bounds
	and the pixel grid (94 column intervals / 62 row intervals).
	"""

	# Variable names that ZoomSto/ZoomRcl copies between the live window and
	# the Z-vars (the subset with Z-token counterparts in the catalog).
	_ZOOM_VARS = (
		'xmin', 'xmax', 'ymin', 'ymax', 'xscl', 'yscl', 'xres',
		'tmin', 'tmax', 'tstep', 'theta_min', 'theta_max', 'theta_step',
		'n_min', 'n_max', 'plot_start', 'plot_step',
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
		# Z-variables (ZXmin, ZXmax, …) — initialized to the same defaults as the
		# regular window vars so ZoomRcl works sensibly before the first ZoomSto.
		for _name in self._ZOOM_VARS:
			setattr(self, 'z' + _name, getattr(self, _name))

	@property
	def delta_x(self) -> float:
		"""ΔX — the graph width per pixel column: (Xmax − Xmin) / 94."""
		return (self.xmax - self.xmin) / 94

	@property
	def delta_y(self) -> float:
		"""ΔY — the graph height per pixel row: (Ymax − Ymin) / 62."""
		return (self.ymax - self.ymin) / 62


class TableVars:
	"""Table-screen variables.  The table feature isn't implemented; these exist
	only so programs can store to and read them back."""

	def __init__(self):
		self.tbl_start: float | None = None   # TblStart
		self.delta_tbl: float = 1.0           # ΔTbl
		self.tbl_input = None                 # TblInput — a 7-element list (TiList | None)


class UserList(Accessor):
	"""A user-defined list ʟNAME — a dict slot in env.user_lists, keyed by name.

	Built synthetically by the parser when it sees the ʟ prefix; has no catalog
	entry since the name is determined at parse time, not compile time.
	"""

	def __init__(self, name: str):
		self.name = name

	def _get(self, env):
		return env.user_lists.get(self.name)

	def _set(self, env, value):
		env.user_lists[self.name] = value

	def resolve(self, env):
		value = super().resolve(env)
		if not value.data:
			raise InvalidDimError("empty list")
		return value

	def store(self, env, value):
		self._set(env, require_list(value).copy())

	def is_invokable(self):
		return True

	def invoke(self, arg_parser):
		index = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[index]

	def delete(self, env):
		env.user_lists.pop(self.name, None)

	def __repr__(self):
		return f"UserList({self.name!r})"


class ReturnSignal(Exception):
	"""Raised by Return to exit the current sub-program and return to the caller."""


class StopSignal(Exception):
	"""Raised by Stop to terminate all program execution immediately."""


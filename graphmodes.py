"""Per-graph-mode behavior, as a small strategy hierarchy.

Each of the four graphing modes (Function, Parametric, Polar, Sequence) decides how
ZoomFit measures extent and what ZStandard resets — behavior that used to live as
parallel 4-way branches in environment.py and zoom.py.  The handlers are stateless
singletons (see HANDLERS); all mutable state stays on the environment.

Plotting and point sampling live on the individual function data objects (FuncData,
ParData, PolarData, SeqData in graph.py).  Each handler just iterates its list,
delegates to the function objects, and handles the mode-specific window reset.
"""
import math
from contextlib import nullcontext

from graph import sequence_pass
from modes import GraphMode, AngleMode
from errors import WindowRangeError, DomainError


def _trig_window(env):
	"""(Tmax/θmax, Tstep/θstep) for ZStandard — 2π / π⁄24 in Radian, 360 / 7.5 in Degree."""
	if env.angle_mode is AngleMode.RAD:
		return 2 * math.pi, math.pi / 24
	return 360.0, 7.5


class GraphModeHandler:
	"""Strategy for one graphing mode.  Subclasses supply `_fns` and `standard_window`;
	`plot`, `fit_bounds`, and `set_selected` are implemented here or on SweptMode."""

	def _fns(self, env) -> list:
		"""The list of function data objects for this mode."""
		raise NotImplementedError

	def _pass(self, env):
		"""Context manager wrapping the full plot/fit sweep (SeqMode uses a memo cache)."""
		return nullcontext()

	def standard_window(self, env) -> None:
		"""ZStandard: reset this mode's own window variables."""

	def set_selected(self, env, on: bool, numbers=()) -> None:
		"""FnOn/FnOff: select or deselect the listed 1-based functions, or all when
		`numbers` is empty.  Number 0 means the last function, matching 1-9,0 numbering."""
		fns = self._fns(env)
		indices = range(len(fns)) if not numbers else [
			(n - 1 if n >= 1 else len(fns) - 1) for n in numbers
		]
		for i in indices:
			if not (0 <= i < len(fns)):
				raise DomainError("FnOn/FnOff: function number out of range")
			fns[i].selected = on

	def plot(self, env) -> None:
		"""Draw all selected, defined functions onto env.graph."""
		with self._pass(env):
			for fn in self._fns(env):
				if fn.selected and fn.is_defined():
					fn.plot(env)

	def fit_bounds(self, env) -> tuple:
		"""ZoomFit: return (xmin, xmax, ymin, ymax) enclosing every plotted point.

		xmin/xmax are None when the mode leaves the x-range alone (Func).  Raises
		WindowRangeError when there's nothing to fit or the range collapses to a point.
		"""
		raise NotImplementedError


class FuncMode(GraphModeHandler):
	"""Function mode: each Yn is sampled column-by-column as Y=f(X)."""

	def _fns(self, env):
		return env.function

	def fit_bounds(self, env):
		ys = [y for fn in env.function if fn.selected and fn.is_defined()
		        for y in fn.fit_points(env)]
		if not ys or min(ys) == max(ys):
			raise WindowRangeError("ZoomFit: no Y range to fit")
		return (None, None, min(ys), max(ys))   # Func keeps the current X range

	def standard_window(self, env):
		env.window.xres = 1.0


class SweptMode(GraphModeHandler):
	"""Parametric, Polar, and Sequence modes: a path traced as a parameter advances.

	Subclasses supply only `_fns`, `standard_window`, and optionally `_pass`;
	`fit_bounds` is shared here since all three modes return an (x, y) bounding box.
	"""

	def fit_bounds(self, env):
		pts = []
		with self._pass(env):
			for fn in self._fns(env):
				if fn.selected and fn.is_defined():
					pts += fn.fit_points(env)
		xs = [x for x, _ in pts]
		ys = [y for _, y in pts]
		if not pts or min(xs) == max(xs) or min(ys) == max(ys):
			raise WindowRangeError("ZoomFit: no range to fit")
		return (min(xs), max(xs), min(ys), max(ys))


class ParMode(SweptMode):
	"""Parametric mode: sweep T, plotting (XnT(T), YnT(T))."""

	def _fns(self, env):
		return env.parametric

	def standard_window(self, env):
		w = env.window
		full, step = _trig_window(env)
		w.tmin, w.tmax, w.tstep = 0.0, full, step


class PolMode(SweptMode):
	"""Polar mode: sweep θ, plotting (r·cosθ, r·sinθ)."""

	def _fns(self, env):
		return env.polar

	def standard_window(self, env):
		w = env.window
		full, step = _trig_window(env)
		w.theta_min, w.theta_max, w.theta_step = 0.0, full, step


class SeqMode(SweptMode):
	"""Sequence mode, Time plot: plot (n, s(n)) over n = PlotStart … nMax by PlotStep.

	The whole sweep shares one memo cache (env._sequence_pass) so a recursive sequence
	is evaluated bottom-up rather than recomputed at every point.
	"""

	def _fns(self, env):
		return env.sequence

	def _pass(self, env):
		return sequence_pass(env)

	def standard_window(self, env):
		w = env.window
		w.n_min, w.n_max = 1.0, 10.0
		w.plot_start, w.plot_step = 1.0, 1.0


HANDLERS = {
	GraphMode.FUNC: FuncMode(),
	GraphMode.PAR:  ParMode(),
	GraphMode.POL:  PolMode(),
	GraphMode.SEQ:  SeqMode(),
}

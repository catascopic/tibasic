"""Per-graph-mode behavior, as a small strategy hierarchy.

Each of the four graphing modes (Function, Parametric, Polar, Sequence) decides how
its equations are plotted, how ZoomFit measures their extent, and what ZStandard
resets — behavior that used to live as parallel 4-way branches in environment.py and
zoom.py.  The handlers are stateless singletons (see HANDLERS); all mutable state
stays on the environment, which is passed in.

The modes split into two shapes:
  * FuncMode      — one Y per pixel column (Y=f(X)), traced with trace_curve.
  * SweptMode     — a path traced as a parameter advances (Par/Pol/Seq), via
                    trace_parametric; subclasses differ only in the sampler and which
                    window variables give the (start, stop, step) sweep.

This module imports only graph.py (the cycle-free plotting layer), modes, and errors,
so it stays out of the preparse⇄environment import cycle.
"""
import math
from contextlib import nullcontext

from modes import GraphMode, AngleMode
from errors import WindowRangeError
from graph import (
	MAX_COL, _param_values,
	trace_curve, trace_parametric,
	sample_function, sample_parametric, sample_polar, sample_sequence,
)


def _trig_window(env):
	"""(Tmax/θmax, Tstep/θstep) for ZStandard — 2π / π⁄24 in Radian mode, 360 / 7.5 in
	Degree mode."""
	if env.angle_mode is AngleMode.RAD:
		return 2 * math.pi, math.pi / 24
	return 360.0, 7.5


class GraphModeHandler:
	"""Strategy for one graphing mode.  Subclasses implement plotting, the ZoomFit
	extent, and the ZStandard reset of the mode's own window variables.

	`function_count` and `equations_per_function` describe the mode's layout — how many
	selectable functions it has and how many equation variables each owns (2 for a
	parametric X/Y pair, 1 otherwise).  GraphFunctions builds its storage from these.
	"""

	function_count: int
	equations_per_function: int = 1

	def plot(self, env) -> None:
		"""Draw the mode's selected, defined functions onto env.graph."""
		raise NotImplementedError

	def fit_bounds(self, env) -> tuple:
		"""ZoomFit: return (xmin, xmax, ymin, ymax) enclosing every plotted point.

		xmin/xmax are None when the mode leaves the x-range alone (Func).  Raises
		WindowRangeError when there's nothing to fit or the range collapses to a point.
		"""
		raise NotImplementedError

	def standard_window(self, env) -> None:
		"""ZStandard: reset this mode's own window variables (the shared bounds and
		scales are reset by the caller)."""


class FuncMode(GraphModeHandler):
	"""Function mode: each Yn is sampled column-by-column as Y=f(X) (like DrawF)."""

	function_count = 10            # Y1–Y0

	def plot(self, env):
		for _, (eq,) in env.graph_functions.selected_defined(GraphMode.FUNC):
			trace_curve(env, sample_function(env, lambda eq=eq: eq.eval(env)))

	def fit_bounds(self, env):
		w = env.window
		xmin = w.xmin
		delta = (w.xmax - xmin) / MAX_COL
		ys = []
		for _, (eq,) in env.graph_functions.selected_defined(GraphMode.FUNC):
			f = sample_function(env, lambda eq=eq: eq.eval(env))
			ys += [y for i in range(MAX_COL + 1) if (y := f(xmin + i * delta)) is not None]
		if not ys or min(ys) == max(ys):
			raise WindowRangeError("ZoomFit: no Y range to fit")
		return (None, None, min(ys), max(ys))   # Func keeps the current X range

	def standard_window(self, env):
		env.window.xres = 1.0


class SweptMode(GraphModeHandler):
	"""A path traced as a parameter advances (Parametric, Polar, Sequence).

	Subclasses supply only point_fns (a sampler per selected, defined function) and
	sweep_range (the window's start/stop/step); plot and fit_bounds are shared.
	"""

	def point_fns(self, env):
		"""Yield a point(t) -> (x, y) | None callable for each selected, defined function."""
		raise NotImplementedError

	def sweep_range(self, env) -> tuple:
		"""(start, stop, step) for the parameter sweep, from the window variables."""
		raise NotImplementedError

	def _pass(self, env):
		"""Context manager active while points are evaluated (Seq owns the memo cache)."""
		return nullcontext()

	def plot(self, env):
		start, stop, step = self.sweep_range(env)
		with self._pass(env):
			for point in self.point_fns(env):
				trace_parametric(env, point, start, stop, step)

	def fit_bounds(self, env):
		start, stop, step = self.sweep_range(env)
		pts = []
		with self._pass(env):
			for point in self.point_fns(env):
				pts += [xy for t in _param_values(start, stop, step) if (xy := point(t)) is not None]
		xs = [x for x, _ in pts]
		ys = [y for _, y in pts]
		if not pts or min(xs) == max(xs) or min(ys) == max(ys):
			raise WindowRangeError("ZoomFit: no range to fit")
		return (min(xs), max(xs), min(ys), max(ys))


class ParMode(SweptMode):
	"""Parametric mode: sweep T, plotting (XnT(T), YnT(T)); both halves must be defined."""

	function_count = 6             # X1T/Y1T – X6T/Y6T
	equations_per_function = 2     # the X/Y pair (selected_defined yields it only when both are set)

	def point_fns(self, env):
		for _, (x_eq, y_eq) in env.graph_functions.selected_defined(GraphMode.PAR):
			yield sample_parametric(env, lambda e=x_eq: e.eval(env), lambda e=y_eq: e.eval(env))

	def sweep_range(self, env):
		w = env.window
		return w.tmin, w.tmax, w.tstep

	def standard_window(self, env):
		w = env.window
		full, step = _trig_window(env)
		w.tmin, w.tmax, w.tstep = 0.0, full, step


class PolMode(SweptMode):
	"""Polar mode: sweep θ, plotting (r·cosθ, r·sinθ)."""

	function_count = 6             # r1–r6

	def point_fns(self, env):
		for _, (r_eq,) in env.graph_functions.selected_defined(GraphMode.POL):
			yield sample_polar(env, lambda e=r_eq: e.eval(env))

	def sweep_range(self, env):
		w = env.window
		return w.theta_min, w.theta_max, w.theta_step

	def standard_window(self, env):
		w = env.window
		full, step = _trig_window(env)
		w.theta_min, w.theta_max, w.theta_step = 0.0, full, step


class SeqMode(SweptMode):
	"""Sequence mode, Time plot: plot (n, s(n)) over n = PlotStart … nMax by PlotStep.

	The whole sweep shares one memo cache (see Environment._sequence_pass) so a
	recursive sequence is evaluated bottom-up rather than recomputed at every point.
	Web and uv/vw/uvw phase plots aren't modeled — only the default Time plot.
	"""

	function_count = 3             # u, v, w

	def point_fns(self, env):
		for index, _ in env.graph_functions.selected_defined(GraphMode.SEQ):
			yield sample_sequence(env, index)

	def sweep_range(self, env):
		w = env.window
		return w.plot_start, w.n_max, w.plot_step

	def _pass(self, env):
		return env._sequence_pass()

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

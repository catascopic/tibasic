"""Mode-setting commands (Radian, Degree, Float, Fix, Func, FnOn, ClockOn, …).

The mode *enums* live in modes.py (low-level); these are the commands that set
them, so they sit up here with the rest of the command layer.
"""
from preparse import no_arg_command, no_arg_bunch
from preparse import preparse_cmd, Real, Env
from errors import DomainError
from core import py_int
from modes import AngleMode, NumberMode, GraphMode, ComplexMode, DrawMode, GraphOrder


def _mode(attr, value):
	@no_arg_command
	def cmd(env: 'Environment'):
		setattr(env, attr, value)
	return cmd


# Angle mode
radian = _mode('angle_mode', AngleMode.RAD)
degree = _mode('angle_mode', AngleMode.DEG)

# Number display mode
normal = _mode('number_mode', NumberMode.NORMAL)
sci    = _mode('number_mode', NumberMode.SCI)
eng    = _mode('number_mode', NumberMode.ENG)

# Display precision
float_ = _mode('fix_digits', None)

@preparse_cmd
def fix(env: Env, n: Real):
	n = py_int(n)
	if not 0 <= n <= 9:
		raise DomainError(f"Fix: argument must be 0–9, got {n}")
	env.fix_digits = n

# Graph type
func  = _mode('graph_mode', GraphMode.FUNC)
param = _mode('graph_mode', GraphMode.PAR)
polar = _mode('graph_mode', GraphMode.POL)
seq   = _mode('graph_mode', GraphMode.SEQ)

# Plot evaluation order
sequential = _mode('graph_order', GraphOrder.SEQUENTIAL)
simul      = _mode('graph_order', GraphOrder.SIMUL)

# Coordinate system display
polar_gc = _mode('polar_gc', True)
rect_gc  = _mode('polar_gc', False)

# Display settings (on/off pairs)
coord_on  = _mode('coord_on',  True)
coord_off = _mode('coord_on',  False)
axes_on   = _mode('axes_on',   True)
axes_off  = _mode('axes_on',   False)
grid_on   = _mode('grid_on',   True)
grid_off  = _mode('grid_on',   False)
label_on  = _mode('label_on',  True)
label_off = _mode('label_on',  False)
expr_on   = _mode('expr_on',   True)
expr_off  = _mode('expr_on',   False)
diagnostic_on  = _mode('diagnostic_on', True)
diagnostic_off = _mode('diagnostic_on', False)

# Draw style
connected = _mode('draw_mode', DrawMode.CONNECTED)
dot       = _mode('draw_mode', DrawMode.DOT)

# Complex number mode
real       = _mode('complex_mode', ComplexMode.REAL)
a_plus_bi  = _mode('complex_mode', ComplexMode.A_PLUS_BI)
re_theta_i = _mode('complex_mode', ComplexMode.RE_THETA_I)

# TODO: FULL/HORIZ/G-T

# Clock — these can bunch (ClockOnClockOn is valid)
@no_arg_bunch
def clock_on(env): env.clock_on = True

@no_arg_bunch
def clock_off(env): env.clock_on = False


# Equation selection
def _fn_select(env, on: bool, numbers):
	env.graph_mode_handler.set_selected(env, on, [py_int(n) for n in numbers])

@preparse_cmd
def fn_on(env: Env, *numbers: Real):
	"""FnOn [function#,...] — select (turn on) the listed functions in the current
	graph mode, or all of them with no arguments."""
	_fn_select(env, True, numbers)

@preparse_cmd
def fn_off(env: Env, *numbers: Real):
	"""FnOff [function#,...] — deselect (turn off) the listed functions in the current
	graph mode, or all of them with no arguments."""
	_fn_select(env, False, numbers)

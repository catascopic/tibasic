from __future__ import annotations
from enum import Enum, auto
from decorators import forms_func
from errors import DomainError
from tiobjects import require_int


class AngleMode(Enum):
	RAD = auto()
	DEG = auto()

class NumberMode(Enum):
	NORMAL = auto()
	SCI    = auto()
	ENG    = auto()

class GraphMode(Enum):
	FUNC = auto()
	PAR  = auto()
	POL  = auto()
	SEQ  = auto()

class ComplexMode(Enum):
	REAL       = auto()
	A_PLUS_BI  = auto()
	RE_THETA_I = auto()

class DrawMode(Enum):
	CONNECTED = auto()
	DOT       = auto()

class GraphOrder(Enum):
	SEQUENTIAL = auto()
	SIMUL      = auto()


def _mode(attr, value):
	@forms_func
	def cmd(a):
		a.end_cmd()
		setattr(a.env, attr, value)
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

@forms_func
def fix(a):
	n = require_int(a.expr())
	if not 0 <= n <= 9:
		raise DomainError(f"Fix: argument must be 0–9, got {n}")
	a.env.fix_digits = n
	a.end_cmd()

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

# Clock
clock_on  = _mode('clock_on', True)
clock_off = _mode('clock_on', False)

from enum import Enum, auto, StrEnum


class AngleMode(Enum):
	RAD = auto()
	DEG = auto()

class NumberMode(Enum):
	NORMAL = auto()
	SCI    = auto()
	ENG    = auto()

class GraphMode(StrEnum):
	FUNC = 'function'
	PAR  = 'parametric'
	POL  = 'polar'
	SEQ  = 'sequence'

class Screen(Enum):
	"""Which screen is currently displayed.  Not a persistent mode setting — runtime
	state that tracks the last screen shown, so e.g. ClrDraw knows whether to regraph.
	(The table isn't implemented; TABLE only records that DispTable switched away.)"""
	HOME  = auto()
	GRAPH = auto()
	TABLE = auto()

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

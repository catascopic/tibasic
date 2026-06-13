from enum import Enum, auto


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

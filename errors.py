"""
TI-BASIC runtime and parse errors.

Each class carries a `code` attribute matching the calculator's ERR: screen
(e.g. ERR:DIM MISMATCH), plus a human-readable message as the exception value.
"""
from typing import ClassVar


class TiError(Exception):
	"""Base class for all TI-BASIC errors."""
	code: ClassVar[str]

	def __init__(self, message: str = '', *, pos: int | None = None):
		super().__init__(message)
		self.pos = pos  # token-stream index where the error occurred, if known
		self.located = False  # True once a parser has displayed the error location


class TiSyntaxError(TiError):
	"""ERR:SYNTAX — token sequence that the parser cannot understand."""
	code = 'SYNTAX'


class ArgumentError(TiError):
	"""ERR:ARGUMENT — wrong number of arguments to a function or command."""
	code = 'ARGUMENT'


class DataTypeError(TiError):
	"""ERR:DATA TYPE — value is the wrong type for this operation
	(e.g. a string where a number is required, or a complex result in real mode)."""
	code = 'DATA TYPE'


class DomainError(TiError):
	"""ERR:DOMAIN — argument is outside the domain of a function
	(e.g. √ of a negative, log of zero, step of zero in seq)."""
	code = 'DOMAIN'


class DimMismatchError(TiError):
	"""ERR:DIM MISMATCH — two lists or matrices have incompatible dimensions
	for the requested operation."""
	code = 'DIM MISMATCH'


class InvalidDimError(TiError):
	"""ERR:INVALID DIM — a dimension value is out of range or not an integer."""
	code = 'INVALID DIM'


class UndefinedError(TiError):
	"""ERR:UNDEFINED — a variable or function has not been defined."""
	code = 'UNDEFINED'


class SingularMatrixError(TiError):
	"""ERR:SINGULAR MAT — matrix is singular and cannot be inverted."""
	code = 'SINGULAR MAT'


class DivideByZeroError(TiError):
	"""ERR:DIVIDE BY 0 — division by zero."""
	code = 'DIVIDE BY 0'


class StatError(TiError):
	"""ERR:STAT — invalid input to a statistics function
	(e.g. empty list, all-identical values when variance is required)."""
	code = 'STAT'


class IncrementError(TiError):
	"""ERR:INCREMENT — the step/increment value is zero, or its sign is inconsistent
	with the start and end values (e.g. positive step but start > end)."""
	code = 'INCREMENT'


class IllegalNestError(TiError):
	"""ERR:ILLEGAL NEST — a function that cannot be nested has been called recursively."""
	code = 'ILLEGAL NEST'


class LabelError(TiError):
	"""ERR:LABEL — a Goto references a label that does not exist in the program."""
	code = 'LABEL'


class NonRealAnsError(TiError):
	"""ERR:NONREAL ANS — a computation produced a complex result while in Real mode."""
	code = 'NONREAL ANS'


class InvalidCommandError(TiError):
	"""ERR:INVALID — a command was executed in a context where it is not allowed
	(e.g. a block-level command run from the home screen instead of a program)."""
	code = 'INVALID'


class TiMemoryError(TiError):
	"""ERR:MEMORY — the calculator ran out of memory.

	In our interpreter this is raised when Python's call stack overflows (e.g.
	Y₁=Y₁ → infinite recursion), matching the real calculator's ERR:MEMORY."""
	code = 'MEMORY'


class TiOverflowError(TiError):
	"""ERR:OVERFLOW — a calculation went beyond the bounds of (1e100, 1e-100)."""
	code = 'OVERFLOW'


class BoundError(TiError):
	"""ERR:BOUND — a lower bound is greater than its upper bound (e.g. fMin/fMax)."""
	code = 'BOUND'


class BadGuessError(TiError):
	"""ERR:BAD GUESS — solve(: the guess is outside the bounds or the function is
	undefined there."""
	code = 'BAD GUESS'


class NoSignChangeError(TiError):
	"""ERR:NO SIGN CHG — solve(: no sign change was found over the search interval,
	so no root could be bracketed."""
	code = 'NO SIGN CHG'


class ToleranceError(TiError):
	"""ERR:TOL NOT MET — an iterative routine could not reach the requested
	tolerance (e.g. fMin/fMax)."""
	code = 'TOL NOT MET'


class WindowRangeError(TiError):
	"""ERR:WINDOW RANGE — the graphing window is invalid (Xmin ≥ Xmax, Ymin ≥ Ymax),
	or a zoom produced an empty range (e.g. ZoomFit on a constant function)."""
	code = 'WINDOW RANGE'

"""
TI-BASIC runtime and parse errors.

Each class carries a `code` attribute matching the calculator's ERR: screen
(e.g. ERR:DIM MISMATCH), plus a human-readable message as the exception value.
"""


class TiError(Exception):
	"""Base class for all TI-BASIC errors."""
	code = 'ERROR'


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

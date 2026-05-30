"""
Control-flow signals used to unwind the call stack.

These are not TiErrors (they don't represent ERR: screens); they are
internal Python exceptions that communicate flow events between the
parser, Program objects, and the top-level entry point.
"""


class ReturnSignal(Exception):
	"""Raised by Return to exit the current sub-program and return to the caller."""


class StopSignal(Exception):
	"""Raised by Stop to terminate all program execution immediately."""

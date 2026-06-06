"""Declarative argument schema for ArgParser.take().

A schema is a sequence of ArgSpec values naming which ArgParser parse method
handles each positional argument, e.g.

    formula, var, start, end, step = a.take(thunk, numeric_var, expr, expr, optional(expr, 1))

Because take() parses exactly the declared arguments (rather than greedily
consuming everything), the existing trailing-comma machinery reports the right
errors for free: a missing required argument raises ArgumentError from the
parse method, and a surplus argument is caught by the following end_* call.

This module deliberately has no imports: ArgParser.take() duck-types on the
attributes below, so the heavyweight parser/forms modules can import the spec
vocabulary without any circular-import risk.
"""
from __future__ import annotations


class ArgSpec:
	__slots__ = ('method', 'optional', 'default', 'variadic')

	def __init__(self, method: str, optional: bool = False, default=None, variadic: bool = False):
		self.method = method      # name of the ArgParser parse method to call
		self.optional = optional  # if missing, yield `default` instead of raising
		self.default = default
		self.variadic = variadic  # greedily consume the rest; yields a list

	def __repr__(self):
		flags = ''.join(s for s, on in (('?', self.optional), ('*', self.variadic)) if on)
		return f"{self.method}{flags}"


# Spec vocabulary — names match ArgParser parse methods.
expr         = ArgSpec('expr')
thunk        = ArgSpec('thunk')
numeric_var  = ArgSpec('numeric_var')
list_var     = ArgSpec('list_var')
matrix_var   = ArgSpec('matrix_var')
string_var   = ArgSpec('string_var')
equation_var = ArgSpec('equation_var')
label_name   = ArgSpec('label_name')
program_name = ArgSpec('program_name')


def optional(spec: ArgSpec, default=None) -> ArgSpec:
	"""Mark a spec optional: if the argument is absent, yield `default`."""
	return ArgSpec(spec.method, optional=True, default=default)


def rest(spec: ArgSpec) -> ArgSpec:
	"""Greedily consume all remaining arguments with `spec`; yields a list.

	Must be the last entry in a schema.
	"""
	return ArgSpec(spec.method, variadic=True)

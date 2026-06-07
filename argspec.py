"""Declarative argument schema for @preparse, expressed as type annotations.

A @preparse function declares each parameter's parser by annotating it with one
of the vocabulary aliases below.  @preparse reads the signature (see
`schema_from_signature`) and builds a tuple of ArgSpec values:

    @preparse(FUNC)
    def gcd(a: vectorized[numeric], b: vectorized[numeric]) -> float: ...

Optionality and arity fall straight out of the signature:

  * a parameter with a default is optional; when the caller omits it, take()
    omits it too, so the function's own default applies;
  * a ``*args`` parameter is variadic and consumes the rest as a list.

Each alias is an ``Annotated[T, ArgSpec(...)]``.  The ArgSpec metadata names the
ArgParser parse method; the base type ``T`` is the value the core function
actually receives, so the annotations stay truthful to a type checker too.

This module imports nothing from the project at runtime (only stdlib), so the
heavyweight parser/forms modules can import the vocabulary with no circular-import
risk.  The base types live behind TYPE_CHECKING purely for static tooling.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, replace
from numbers import Number
from typing import Annotated, Any, get_args, TYPE_CHECKING

if TYPE_CHECKING:
	from environment import Environment, Variable
	from parser import Thunk


@dataclass(frozen=True)
class ArgSpec:
	method: str                    # name of the ArgParser parse method that extracts the value
	validate: str = ''             # name of a value validator @preparse applies (real/numeric/integer)
	optional: bool = False         # if absent, the slot is omitted from the call
	variadic: bool = False         # greedily consume the rest; yields a list
	vectorize: bool = False        # map element-wise over a TiList in this slot
	matrix: bool = False           # also map element-wise over a TiMatrix (implies vectorize)

	def __repr__(self):
		flags = ''.join(s for s, on in (
			('?', self.optional), ('*', self.variadic),
			('~', self.vectorize), ('#', self.matrix),
		) if on)
		name = f"{self.method}:{self.validate}" if self.validate else self.method
		return f"{name}{flags}"


def _as_spec(annotation) -> ArgSpec:
	"""Return the ArgSpec for a schema entry: a bare ArgSpec, or one wrapped in
	an Annotated alias (e.g. ``expr`` → ``Annotated[Any, ArgSpec('expr')]``)."""
	if isinstance(annotation, ArgSpec):
		return annotation
	for meta in get_args(annotation):
		if isinstance(meta, ArgSpec):
			return meta
	raise TypeError(f"Not an argument spec: {annotation!r}")


# ── Vocabulary ──────────────────────────────────────────────────────────────
# Each alias is Annotated[<value type the core receives>, ArgSpec('<method>')].
# The value types all extract via the `expr` parser method and differ only in the
# value validator @preparse applies (so a TiList/TiMatrix in a vectorized slot is
# validated element-wise, not rejected wholesale).
# `env` is special: it is injected from ArgParser.env without consuming a token.

expr    = Annotated[Any,           ArgSpec('expr')]
numeric = Annotated[Number,        ArgSpec('expr', validate='numeric')]  # require_num per value
real    = Annotated[float,         ArgSpec('expr', validate='real')]     # require_real per value
integer = Annotated[float,         ArgSpec('expr', validate='integer')]  # require_int per value; call int() when a Python int is needed
thunk   = Annotated['Thunk',       ArgSpec('thunk')]

num_var      = Annotated['Variable', ArgSpec('numeric_var')]
list_var     = Annotated['Variable', ArgSpec('list_var')]
matrix_var   = Annotated['Variable', ArgSpec('matrix_var')]
string_var   = Annotated['Variable', ArgSpec('string_var')]
equation_var = Annotated['Variable', ArgSpec('equation_var')]
any_var      = Annotated['Variable', ArgSpec('any_var')]
list_var_prefix_optional = Annotated['Variable', ArgSpec('list_var_prefix_optional')]

label_name   = Annotated[str, ArgSpec('label_name')]
program_name = Annotated[str, ArgSpec('program_name')]

env     = Annotated['Environment', ArgSpec('env')]
PassEnv = env  # readable alias for use in new-style annotations

# Back-compat alias: old code imports `numeric_var`.
numeric_var = num_var


def _mark(item, **flags) -> Annotated:
	"""Return a copy of an alias with extra ArgSpec flags set (vectorize/matrix)."""
	spec = _as_spec(item)
	args = get_args(item)
	base = args[0] if args else Any
	return Annotated[base, replace(spec, **flags)]


class vectorized:
	"""``vectorized[numeric]`` marks a parameter as a mapped axis: a TiList in
	that slot is iterated element-wise (zipped with the other vectorized slots),
	while scalars and unmarked parameters are broadcast unchanged."""
	def __class_getitem__(cls, item):
		return _mark(item, vectorize=True)


class matrix_vectorized:
	"""Like ``vectorized``, but the slot also maps element-wise over a TiMatrix
	via TiMatrix.transform.  At most one matrix_vectorized argument may receive a
	matrix per call."""
	def __class_getitem__(cls, item):
		return _mark(item, vectorize=True, matrix=True)


# ── Schema extraction from a function signature ─────────────────────────────

def schema_from_signature(func) -> tuple:
	"""Build an ArgSpec schema from *func*'s annotated parameters.

	Each parameter must be annotated with a vocabulary alias.  A parameter with
	a default becomes optional; a ``*args`` parameter becomes variadic.  The
	return annotation is ignored.

	Annotations are read with ``eval_str=True`` so this works whether or not the
	defining module uses ``from __future__ import annotations`` — the alias names
	just have to be importable in that module (they are).  The aliases' own
	forward-ref base types are never resolved.
	"""
	annotations = inspect.get_annotations(func, eval_str=True)
	schema = []
	for name, p in inspect.signature(func).parameters.items():
		if name not in annotations:
			raise TypeError(
				f"@preparse: {func.__name__}: parameter {name!r} has no annotation"
			)
		spec = _as_spec(annotations[name])
		if p.kind is inspect.Parameter.VAR_POSITIONAL:
			spec = replace(spec, variadic=True)
		elif p.default is not inspect.Parameter.empty:
			spec = replace(spec, optional=True)
		schema.append(spec)
	return tuple(schema)

"""Declarative argument schema for @preparse, expressed as type annotations.

A @preparse function declares each parameter's parser by annotating it with one
of the vocabulary aliases below.  @preparse reads the signature (see
`schema_from_signature`) and builds a tuple of ArgSpec values:

    @preparse(FUNC)
    def gcd(a: vectorized[numeric], b: vectorized[numeric]) -> float: ...

Optionality and arity fall straight out of the signature — there is no separate
`optional(...)` / `rest(...)` to keep in sync:

  * a parameter with a default is optional; when the caller omits it, take()
    omits it too, so the function's own default applies;
  * a ``*args`` parameter is variadic and consumes the rest as a list.

Each alias is an ``Annotated[T, ArgSpec(...)]``.  The ArgSpec metadata names the
ArgParser parse method; the base type ``T`` is the value the core function
actually receives, so the annotations stay truthful to a type checker too.

Legacy positional schemas — ``@preparse(env, expr, optional(expr))`` — still
work: anywhere a spec is consumed it is normalized via `_as_spec`, which accepts
both bare ArgSpec values and the Annotated aliases.

This module imports nothing from the project at runtime (only stdlib), so the
heavyweight parser/forms modules can import the vocabulary with no circular-import
risk.  The base types live behind TYPE_CHECKING purely for static tooling.
"""
from __future__ import annotations

import inspect
from numbers import Number
from typing import Annotated, Any, get_args, TYPE_CHECKING

if TYPE_CHECKING:
	from environment import Environment, Variable
	from parser import Thunk


class ArgSpec:
	__slots__ = ('method', 'optional', 'variadic', 'vectorize', 'matrix')

	def __init__(
		self,
		method: str,
		optional: bool = False,
		variadic: bool = False,
		vectorize: bool = False,
		matrix: bool = False,
	):
		self.method = method        # name of the ArgParser parse method to call
		self.optional = optional    # if absent, the slot is omitted from the call
		self.variadic = variadic    # greedily consume the rest; yields a list
		self.vectorize = vectorize  # map element-wise over a TiList in this slot
		self.matrix = matrix        # also map element-wise over a TiMatrix (implies vectorize)

	def __repr__(self):
		flags = ''.join(s for s, on in (
			('?', self.optional), ('*', self.variadic),
			('~', self.vectorize), ('#', self.matrix),
		) if on)
		return f"{self.method}{flags}"


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
# `env` is special: it is injected from ArgParser.env without consuming a token.

expr    = Annotated[Any,           ArgSpec('expr')]
numeric = Annotated[Number,        ArgSpec('expr')]   # readable alias for expr
real    = Annotated[float,         ArgSpec('real')]   # enforces require_real at parse time
integer = Annotated[int,           ArgSpec('expr')]   # readable alias; core does py_int
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
	merged = {'vectorize': spec.vectorize, 'matrix': spec.matrix, **flags}
	return Annotated[base, ArgSpec(spec.method, **merged)]


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


# ── Legacy positional-schema helpers ────────────────────────────────────────
# New-style schemas express these through the signature (a default → optional,
# ``*args`` → variadic), but the old positional form is still accepted.

def optional(spec) -> ArgSpec:
	"""Mark a spec optional (legacy positional form). No default is stored: an
	omitted optional simply isn't passed, so the function's signature default
	applies."""
	s = _as_spec(spec)
	return ArgSpec(s.method, optional=True, vectorize=s.vectorize, matrix=s.matrix)


def rest(spec) -> ArgSpec:
	"""Greedily consume all remaining arguments (legacy positional form); yields
	a list.  Must be the last entry in a schema."""
	s = _as_spec(spec)
	return ArgSpec(s.method, variadic=True, vectorize=s.vectorize, matrix=s.matrix)


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
			spec = ArgSpec(spec.method, variadic=True,
			               vectorize=spec.vectorize, matrix=spec.matrix)
		elif p.default is not inspect.Parameter.empty:
			spec = ArgSpec(spec.method, optional=True,
			               vectorize=spec.vectorize, matrix=spec.matrix)
		schema.append(spec)
	return tuple(schema)

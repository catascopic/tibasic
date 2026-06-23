"""How tokens read, write, and reference the environment.

An `Accessor` is a stateless flyweight that knows *how* to get/set one symbol — a
numeric variable, a list, π, rand, a window setting — in any Environment.  A Token
holds one (`Token.accessor`), and the parser calls `resolve`/`store` on it directly,
passing the environment, so there's no per-environment variable object to allocate and
no `env` threaded through every call site.

When a command needs to *hold* a variable rather than its value (For, fnInt, Input,
DelVar, …), `Accessor.reference(env)` binds the accessor to an environment as a
`Reference`.  A Reference exposes a small mutable-variable surface (`resolve`,
`store`, `delete`, `scoped`, and a `.value` get/set) so those commands can treat any
symbol uniformly.

Every symbol has a dedicated accessor type — numeric, matrix, list, string,
user-list, equation, sequence, window, TVM, and table variables — so values live
directly in the Environment with no wrapper objects.
"""
from abc import ABC
from contextlib import contextmanager

from core import TiList, TiString, TiEquation
from core import require_num, require_matrix, require_list, require_string, require_real, require_int, py_int
from errors import TiSyntaxError, UndefinedError, InvalidDimError, DomainError, DataTypeError
from graph import eval_sequence
from modes import GraphMode


# Maps each graph mode to the equation-list attr whose changes matter in that mode.
_EQ_ATTR_FOR_MODE = {
	GraphMode.FUNC: 'function',
	GraphMode.PAR:  'parametric',
	GraphMode.POL:  'polar',
	GraphMode.SEQ:  'sequence',
}

# Maps each window attr to the set of graph modes where changing it invalidates the graph.
_ALL_MODES = frozenset({GraphMode.FUNC, GraphMode.PAR, GraphMode.POL, GraphMode.SEQ})
_WINDOW_ATTR_MODES: dict[str, frozenset] = {
	'xmin': _ALL_MODES, 'xmax': _ALL_MODES,
	'ymin': _ALL_MODES, 'ymax': _ALL_MODES,
	'xscl': _ALL_MODES, 'yscl': _ALL_MODES,
	'xres':        frozenset({GraphMode.FUNC}),
	'tmin':        frozenset({GraphMode.PAR}),
	'tmax':        frozenset({GraphMode.PAR}),
	'tstep':       frozenset({GraphMode.PAR}),
	'theta_min':   frozenset({GraphMode.POL}),
	'theta_max':   frozenset({GraphMode.POL}),
	'theta_step':  frozenset({GraphMode.POL}),
	'n_min':       frozenset({GraphMode.SEQ}),
	'n_max':       frozenset({GraphMode.SEQ}),
	'plot_start':  frozenset({GraphMode.SEQ}),
	'plot_step':   frozenset({GraphMode.SEQ}),
}


class Accessor(ABC):
	"""Stateless description of how to access one symbol.  Subclasses override the
	pieces that differ; the defaults make a read-only value (store raises).

	`resolve`/`store` are the validating, user-facing operations (auto-init, type
	checks); `_get`/`_set` are the raw, unchecked accessors used internally by
	`Reference` (e.g. by `scoped`).

	A symbol is read in expression context as either `resolve` or `invoke`:
	`invocable` accessors (lists, matrices, equations, sequences) own a trailing
	parenthesised argument — `L1(2)`, `[A](r,c)`, `Y1(x)`, `u(n)` — which the parser
	routes to `invoke(parser)` with the `(` already eaten.  Every other symbol is
	read with `resolve`, so a following `(` stays implicit multiplication (`A(2)`
	== A·2).
	"""

	kind = None        # discriminator for callers that branch on type (e.g. Input: 'string')
	invocable = False  # True ⇒ a trailing '(' is a call/index (see invoke), not implicit mult

	def _get(self, env):
		raise NotImplementedError(f"{type(self).__name__} does not support _get")

	def _set(self, env, value):
		raise NotImplementedError(f"{type(self).__name__} does not support _set")

	@property
	def label(self) -> str:
		"""Human-readable name for error messages; defaults to the repr."""
		return repr(self)

	def resolve(self, env):
		"""Read the value, erroring if the slot is empty.  Subclasses with auto-init
		(NumericVar), extra validation (ListVar), or computed values (rand, ΔX) override."""
		value = self._get(env)
		if value is None:
			raise UndefinedError(f"{self.label} is not defined")
		return value

	def store(self, env, value):
		raise TiSyntaxError(f"Cannot store to {self}")

	def invoke(self, arg_parser):
		raise TiSyntaxError(f"Cannot invoke {self}")

	def delete(self, env):
		raise TiSyntaxError(f"Cannot delete {self}")

	def reference(self, env) -> "Reference":
		return Reference(env, self)


class Deletable:
	"""Mixin for accessors whose `DelVar` clears the slot to its undefined state —
	`None`.  List it before `Accessor` in the bases so this `delete` overrides the
	base's raising one (e.g. `class StringVar(Deletable, Accessor)`)."""

	def delete(self, env):
		self._set(env, None)


class Reference:
	"""An accessor bound to an environment — what commands receive when they take a
	variable rather than its value.  Exposes a uniform mutable-variable surface so
	consumers (For/fnInt/Input/DelVar/…) work the same for every symbol kind."""

	__slots__ = ('env', 'accessor')

	def __init__(self, env, accessor: Accessor):
		self.env = env
		self.accessor = accessor

	def resolve(self):
		return self.accessor.resolve(self.env)

	def store(self, value):
		self.accessor.store(self.env, value)

	def delete(self):
		self.accessor.delete(self.env)

	def get(self):
		return self.accessor._get(self.env)

	def set(self, value):
		self.accessor._set(self.env, value)

	@contextmanager
	def scoped(self):
		"""Save the raw value, run the block, restore it — for temporarily binding a
		variable while evaluating a sub-expression (fnInt, solve, Σ, …)."""
		saved = self.accessor._get(self.env)
		try:
			yield
		finally:
			self.accessor._set(self.env, saved)

	def __repr__(self):
		return f"Reference({self.accessor!r})"


class NumericVar(Deletable, Accessor):
	"""A real/complex variable A–Z, θ — a named slot in env.numerics (NumericVars).

	An undefined numeric reads as 0 (and is initialized to 0 on first resolve),
	matching the calculator; `store` accepts any number (real or complex).
	"""

	__slots__ = ('name',)

	def __init__(self, name: str):
		self.name = name

	def _get(self, env):
		return getattr(env.numerics, self.name)

	def _set(self, env, value):
		setattr(env.numerics, self.name, value)

	def resolve(self, env):
		value = self._get(env)
		if value is None:
			value = 0.0
			self._set(env, value)
		return value

	def store(self, env, value):
		self._set(env, require_num(value))

	def __repr__(self):
		return f"NumericVar({self.name!r})"


class ComputedAccessor(Accessor):
	"""A read-only 0-arg value: a constant (π, 𝑒, 𝑖) or a computed query (getKey,
	getDate, …).  `resolve(env)` calls the wrapped function; storing is an error."""

	__slots__ = ('fn',)

	def __init__(self, fn):
		self.fn = fn          # (env) -> value

	def resolve(self, env):
		return self.fn(env)


class RandAccessor(Accessor):
	"""`rand` — resolves to a fresh random number; storing seeds the generator."""

	def resolve(self, env):
		return env.rand()

	def store(self, env, value):
		env.set_random_seed(value)


class UserListVar(Accessor):
	"""A user-defined list ᴸNAME — a dict slot in env.user_lists, keyed by name.

	Mirrors ListVar, but the storage is a name-keyed dict (no fixed slots) so an
	undefined list is simply an absent key.  Preserves the complex flag of the
	previous value on store.
	"""

	__slots__ = ('name',)
	invocable = True

	def __init__(self, name: str):
		self.name = name

	@property
	def label(self):
		return f"user list {self.name!r}"

	def _get(self, env):
		return env.user_lists.get(self.name)

	def _set(self, env, value):
		env.user_lists[self.name] = value

	def resolve(self, env):
		value = super().resolve(env)      # errors if undefined
		if not value.data:
			raise InvalidDimError("empty list")
		return value

	def store(self, env, value):
		self._set(env, require_list(value).copy())

	def invoke(self, arg_parser):
		index = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[index]

	def delete(self, env):
		env.user_lists.pop(self.name, None)

	def __repr__(self):
		return f"UserListVar({self.name!r})"


class MatrixVar(Deletable, Accessor):
	"""A matrix variable [A]–[J] — a named slot in env.matrices (MatrixVars).

	Undefined matrices raise UndefinedError on resolve; `store` deep-copies so the
	stored value is isolated from the caller.
	"""

	__slots__ = ('name',)
	invocable = True

	def __init__(self, name: str):
		self.name = name

	@property
	def label(self):
		return f"[{self.name}]"

	def _get(self, env):
		return getattr(env.matrices, self.name)

	def _set(self, env, value):
		setattr(env.matrices, self.name, value)

	def store(self, env, value):
		self._set(env, require_matrix(value).copy())

	def invoke(self, arg_parser):
		row = py_int(arg_parser.expr(), InvalidDimError)
		col = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[(row, col)]

	def __repr__(self):
		return f"MatrixVar({self.name!r})"


class ListVar(Deletable, Accessor):
	"""A built-in list L1–L6 — a 0-based slot in env.lists (list[TiList | None]).

	Preserves the complex flag of the previous value on store, matching the
	calculator's behaviour for complex lists.
	"""

	__slots__ = ('index',)
	invocable = True

	def __init__(self, index: int):
		self.index = index

	@property
	def label(self):
		return f"L{self.index + 1}"

	def _get(self, env):
		return env.lists[self.index]

	def _set(self, env, value):
		env.lists[self.index] = value

	def resolve(self, env):
		value = super().resolve(env)      # errors if undefined
		if not value.data:
			raise InvalidDimError("empty list")
		return value

	def store(self, env, value):
		self._set(env, require_list(value).copy())

	def invoke(self, arg_parser):
		index = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[index]

	def __repr__(self):
		return f"ListVar({self.index!r})"


class StringVar(Deletable, Accessor):
	"""A string variable Str1–Str0 — a 0-based slot in env.strings (list[TiString | None]).

	`kind = 'string'` lets Input/Prompt store the raw typed text rather than
	evaluating it as an expression.
	"""

	__slots__ = ('index',)
	kind = 'string'

	def __init__(self, index: int):
		self.index = index

	@property
	def label(self):
		return f"Str{(self.index + 1) % 10}"

	def _get(self, env):
		return env.strings[self.index]

	def _set(self, env, value):
		env.strings[self.index] = value

	def store(self, env, value):
		self._set(env, require_string(value))

	def __repr__(self):
		return f"StringVar({self.index!r})"


class WindowVar(Accessor):
	"""A plain real-valued window variable (Xmin, ZXmin, Tmax, …) — a named float
	attr on env.window.  Z-variables (ZXmin, ZTmax, …) use the z-prefixed attr name
	directly (e.g. WindowVar('zxmin')).

	`resolve` raises UndefinedError for variables that were never assigned (Z-vars
	before ZoomSto, etc.); `store` enforces require_real.
	"""

	__slots__ = ('attr',)

	def __init__(self, attr: str):
		self.attr = attr

	@property
	def label(self):
		return f"window variable {self.attr!r}"

	def _get(self, env):
		return getattr(env.window, self.attr)

	def _set(self, env, value):
		setattr(env.window, self.attr, value)
		
	def _store_check_valid(self, env, value):
		modes = _WINDOW_ATTR_MODES.get(self.attr)
		if modes and env.graph_mode in modes and value != self._get(env):
			env.graph.valid = False
		self._set(env, value)

	def store(self, env, value):
		self._store_check_valid(env, require_real(value))

	def __repr__(self):
		return f"WindowVar({self.attr!r})"


class XresVar(WindowVar):
	"""Xres — function-graph resolution; a whole number 1–8."""

	def store(self, env, value):
		require_int(value)
		if not (1 <= value <= 8):
			raise DomainError(f"Xres must be an integer 1-8, got {value:g}")
		self._store_check_valid(env, value)


class IntWindowVar(WindowVar):
	"""nMin, nMax — window variables constrained to whole numbers."""

	def store(self, env, value):
		self._store_check_valid(env, require_int(value))


class FactorWindowVar(WindowVar):
	"""XFact, YFact — zoom-in/out scaling factors; must be ≥ 1."""

	def store(self, env, value):
		v = require_real(value)
		if v < 1:
			raise DomainError(f"Zoom factor must be ≥ 1, got {v:g}")
		self._set(env, v)


class DeltaWindowVar(Accessor):
	"""ΔX / ΔY — computed from the window bounds; not stored directly.

	resolve: (hi − lo) / divisions.
	store(δ): adjusts the hi bound so the relation holds, leaving lo fixed.
	"""

	__slots__ = ('lo_attr', 'hi_attr', 'divisions')

	def __init__(self, lo_attr: str, hi_attr: str, divisions: int):
		self.lo_attr = lo_attr
		self.hi_attr = hi_attr
		self.divisions = divisions

	def resolve(self, env):
		lo = getattr(env.window, self.lo_attr)
		hi = getattr(env.window, self.hi_attr)
		return (hi - lo) / self.divisions

	def store(self, env, value):
		delta = require_real(value)
		lo = getattr(env.window, self.lo_attr)
		new_hi = lo + self.divisions * delta
		modes = _WINDOW_ATTR_MODES.get(self.hi_attr)
		if modes and env.graph_mode in modes and new_hi != getattr(env.window, self.hi_attr):
			env.graph.valid = False
		setattr(env.window, self.hi_attr, new_hi)

	def __repr__(self):
		return f"DeltaWindowVar({self.lo_attr!r}, {self.hi_attr!r}, {self.divisions})"


class EnvVar(Accessor):
	"""A plain real variable stored directly on env (TVM variables, stat n, …)."""

	__slots__ = ('attr',)

	def __init__(self, attr: str):
		self.attr = attr

	@property
	def label(self):
		return repr(self.attr)

	def _get(self, env):
		return getattr(env, self.attr)

	def _set(self, env, value):
		setattr(env, self.attr, value)

	def store(self, env, value):
		self._set(env, require_real(value))

	def __repr__(self):
		return f"EnvVar({self.attr!r})"


class TableVar(Accessor):
	"""A plain real variable stored on env.table (TblStart, ΔTbl)."""

	__slots__ = ('attr',)

	def __init__(self, attr: str):
		self.attr = attr

	@property
	def label(self):
		return f"table variable {self.attr!r}"

	def _get(self, env):
		return getattr(env.table, self.attr)

	def _set(self, env, value):
		setattr(env.table, self.attr, value)

	def store(self, env, value):
		self._set(env, require_real(value))

	def __repr__(self):
		return f"TableVar({self.attr!r})"


def _normalize_eq(value) -> TiEquation:
	"""Coerce a TiString or TiEquation to TiEquation; raise DataTypeError otherwise."""
	if isinstance(value, TiString):
		return TiEquation(value.tokens)
	if isinstance(value, TiEquation):
		return value
	raise DataTypeError(f"Expected equation or string, got {value!r}")


@contextmanager
def scoped_numeric(env, name: str, value):
	"""Temporarily set numeric variable `name` to `value`, restoring it on exit."""
	
	saved = getattr(env.numerics, name)
	setattr(env.numerics, name, value)
	try:
		yield
	finally:
		saved = setattr(env.numerics, name, saved)


class EquationVar(Deletable, Accessor):
	"""A graph equation (Y1–Y0, X1T/Y1T–X6T/Y6T, r1–r6).

	An equation is not one of TI's runtime value types, so reading it as a value
	(`resolve`) evaluates the formula at the current X.  `Y1(x)` is function
	composition — resolve with X temporarily set to x.  The raw `TiEquation` is
	reachable only through `get`/`set`, which is why copying a formula out
	(Equ►String) needs a dedicated command.  `store` normalises a string/equation
	and selects the function.
	"""

	__slots__ = ('attr', 'index')
	invocable = True
	indep = None

	def __init__(self, attr: str, index: int):
		self.attr = attr
		self.index = index

	def _get(self, env):
		return getattr(env, self.attr)[self.index].equation

	def _set(self, env, value):
		getattr(env, self.attr)[self.index].equation = value

	def resolve(self, env):
		eq = self._get(env)
		if eq is None:
			raise UndefinedError("Equation is not defined")
		return eq.eval(env)

	def invoke(self, arg_parser):
		# The '(' is already eaten: Y1(x) composes by resolving with X set to x.
		x = arg_parser.expr()
		arg_parser.end_func()
		env = arg_parser.env
		with scoped_numeric(env, self.indep, require_num(x)):
			return self.resolve(env)

	def store(self, env, value):
		new_eq = _normalize_eq(value)
		old_eq = self._get(env)
		getattr(env, self.attr)[self.index].selected = True
		if new_eq != old_eq and _EQ_ATTR_FOR_MODE.get(env.graph_mode) == self.attr:
			env.graph.valid = False
		self._set(env, new_eq)

	def __repr__(self):
		return f"EquationVar({self.attr!r}, {self.index})"


class FuncEquationVar(EquationVar):
	__slots__ = ()
	indep = 'X'

	def __init__(self, index: int):
		super().__init__('function', index)


class ParEquationVar(EquationVar):
	"""One half of a parametric pair (XnT or YnT).

	Inherits `invoke`/`resolve`/`store` from `EquationVar` but addresses `x_eq`
	or `y_eq` on the `ParData` rather than the generic `.equation` field.
	Both halves share the same index (the pair index) so selecting either half
	selects the pair.
	"""

	__slots__ = ('half',)
	indep = 'T'

	def __init__(self, index: int, half: str):
		super().__init__('parametric', index)
		self.half = half   # 'x' or 'y'

	def _get(self, env):
		return getattr(env.parametric[self.index], self.half)

	def _set(self, env, value):
		setattr(env.parametric[self.index], self.half, value)

	def __repr__(self):
		return f"ParEquationVar({self.index}, {self.half!r})"


class PolarEquationVar(EquationVar):
	__slots__ = ()
	indep = 'theta'

	def __init__(self, index: int):
		super().__init__('polar', index)


class SequenceVar(EquationVar):
	"""A recursive sequence variable 𝑢/𝑣/𝑤.

	Reading it as a value (`resolve`) evaluates the sequence at the current n;
	`u(n)` evaluates at an explicit index.  The raw `TiEquation` is reachable
	through `_get`/`_set`.
	"""

	def __init__(self, index: int):
		super().__init__('sequence', index)

	def resolve(self, env):
		return eval_sequence(env, self.index, env.n)

	def invoke(self, arg_parser):
		n = arg_parser.expr()
		arg_parser.end_func()
		return eval_sequence(arg_parser.env, self.index, n)

	def __repr__(self):
		return f"SequenceVar({chr(0x75 + self.index)})"


class SequenceInitialVar(Deletable, Accessor):
	"""The u(nMin)/v(nMin)/w(nMin) token — stores and reads the initial-value list
	for a recursive sequence."""

	__slots__ = ('index',)

	def __init__(self, index: int):
		self.index = index

	def _get(self, env):
		return env.sequence[self.index].initial

	def _set(self, env, value):
		env.sequence[self.index].initial = value

	def store(self, env, value):
		if isinstance(value, TiList):
			if len(value.data) > 2:
				raise InvalidDimError("u/v/w(nMin) list may have at most 2 elements")
		else:
			value = TiList([require_real(value)])
		if value != env.sequence[self.index].initial and env.graph_mode is GraphMode.SEQ:
			env.graph.valid = False
		env.sequence[self.index].initial = value

	def __repr__(self):
		return f"SequenceInitialVar({self.index})"

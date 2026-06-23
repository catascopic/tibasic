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
from abc import ABC, abstractmethod
from contextlib import contextmanager

from core import require_num, require_matrix, require_list, require_string, require_real, require_int, py_int
from core import TiString, TiEquation
from errors import TiSyntaxError, UndefinedError, InvalidDimError, DomainError, DataTypeError
from modes import GraphMode


class Accessor(ABC):
	"""Stateless description of how to access one symbol.  Subclasses override the
	pieces that differ; the defaults make a read-only value (store raises).

	`resolve`/`store` are the validating, user-facing operations (auto-init, type
	checks); `get`/`set` are the raw, unchecked accessors used internally (e.g. by
	`scoped`).  By default `get`/`set` fall back to `resolve`/`store`.

	A symbol is read in expression context as either `resolve` or `invoke`:
	`invocable` accessors (lists, matrices, equations, sequences) own a trailing
	parenthesised argument — `L1(2)`, `[A](r,c)`, `Y1(x)`, `u(n)` — which the parser
	routes to `invoke(parser)` with the `(` already eaten.  Every other symbol is
	read with `resolve`, so a following `(` stays implicit multiplication (`A(2)`
	== A·2).
	"""

	kind = None        # discriminator for callers that branch on type (e.g. Input: 'string')
	invocable = False  # True ⇒ a trailing '(' is a call/index (see invoke), not implicit mult

	def get(self):
		raise ValueError(f"cannot set {self}")

	def set(self, value):
		raise ValueError(f"cannot get {self}")
	
	@abstractmethod
	def resolve(self, env):
		pass

	def store(self, env, value):
		raise TiSyntaxError(f"Cannot store to {self}")

	def invoke(self, arg_parser):
		raise TiSyntaxError(f"Cannot invoke {self}")

	def delete(self, env):
		raise TiSyntaxError(f"Cannot delete {self}")

	def reference(self, env) -> "Reference":
		return Reference(env, self)


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
		return self.accessor.get(self.env)

	def set(self, value):
		self.accessor.set(self.env, value)

	@contextmanager
	def scoped(self):
		"""Save the value, run the block, restore it — for fnInt/solve/Σ/… which bind
		the variable while evaluating a formula."""
		saved = self.resolve()
		try:
			yield
		finally:
			self.accessor.set(self.env, saved)

	def __repr__(self):
		return f"Reference({self.accessor!r})"


class NumericVar(Accessor):
	"""A real/complex variable A–Z, θ — a named slot in env.numerics (NumericVars).

	An undefined numeric reads as 0 (and is initialized to 0 on first resolve),
	matching the calculator; `store` accepts any number (real or complex).
	"""

	__slots__ = ('name',)

	def __init__(self, name: str):
		self.name = name

	def get(self, env):
		return getattr(env.numerics, self.name)

	def set(self, env, value):
		setattr(env.numerics, self.name, value)

	def resolve(self, env):
		value = self.get(env)
		if value is None:
			value = 0.0
			self.set(env, value)
		return value

	def store(self, env, value):
		self.set(env, require_num(value))

	def delete(self, env):
		self.set(env, None)

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

	def get(self, env):
		return env.user_lists.get(self.name)

	def set(self, env, value):
		env.user_lists[self.name] = value

	def resolve(self, env):
		try:
			value = env.user_lists[self.name]
		except KeyError:
			raise UndefinedError(f"User list {self.name!r} is not defined")
		if not value.data:
			raise InvalidDimError("empty list")
		return value

	def store(self, env, value):
		lst = require_list(value)
		current = self.get(env)
		was_complex = current is not None and current.is_complex
		new_value = lst.copy()
		if was_complex:
			new_value._upgrade_to_complex()
		self.set(env, new_value)

	def invoke(self, arg_parser):
		index = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[index]

	def delete(self, env):
		env.user_lists.pop(self.name, None)

	def __repr__(self):
		return f"UserListVar({self.name!r})"


class MatrixVar(Accessor):
	"""A matrix variable [A]–[J] — a named slot in env.matrices (MatrixVars).

	Undefined matrices raise UndefinedError on resolve; `store` deep-copies so the
	stored value is isolated from the caller.
	"""

	__slots__ = ('name',)
	invocable = True

	def __init__(self, name: str):
		self.name = name

	def get(self, env):
		return getattr(env.matrices, self.name)

	def set(self, env, value):
		setattr(env.matrices, self.name, value)

	def resolve(self, env):
		value = self.get(env)
		if value is None:
			raise UndefinedError(f"[{self.name}] is not defined")
		return value

	def store(self, env, value):
		self.set(env, require_matrix(value).copy())

	def invoke(self, arg_parser):
		row = py_int(arg_parser.expr(), InvalidDimError)
		col = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[(row, col)]

	def delete(self, env):
		self.set(env, None)

	def __repr__(self):
		return f"MatrixVar({self.name!r})"


class ListVar(Accessor):
	"""A built-in list L1–L6 — a 0-based slot in env.lists (list[TiList | None]).

	Preserves the complex flag of the previous value on store, matching the
	calculator's behaviour for complex lists.
	"""

	__slots__ = ('index',)
	invocable = True

	def __init__(self, index: int):
		self.index = index

	def get(self, env):
		return env.lists[self.index]

	def set(self, env, value):
		env.lists[self.index] = value

	def resolve(self, env):
		value = self.get(env)
		if value is None:
			raise UndefinedError(f"L{self.index + 1} is not defined")
		if not value.data:
			raise InvalidDimError("empty list")
		return value

	def store(self, env, value):
		lst = require_list(value)
		current = self.get(env)
		was_complex = current is not None and current.is_complex
		new_value = lst.copy()
		if was_complex:
			new_value._upgrade_to_complex()
		self.set(env, new_value)

	def invoke(self, arg_parser):
		index = py_int(arg_parser.expr(), InvalidDimError)
		arg_parser.end_func()
		return self.resolve(arg_parser.env)[index]

	def delete(self, env):
		self.set(env, None)

	def __repr__(self):
		return f"ListVar({self.index!r})"


class StringVar(Accessor):
	"""A string variable Str1–Str0 — a 0-based slot in env.strings (list[TiString | None]).

	`kind = 'string'` lets Input/Prompt store the raw typed text rather than
	evaluating it as an expression.
	"""

	__slots__ = ('index',)
	kind = 'string'

	def __init__(self, index: int):
		self.index = index

	def get(self, env):
		return env.strings[self.index]

	def set(self, env, value):
		env.strings[self.index] = value

	def resolve(self, env):
		value = self.get(env)
		if value is None:
			raise UndefinedError(f"Str{(self.index + 1) % 10} is not defined")
		return value

	def store(self, env, value):
		self.set(env, require_string(value))

	def delete(self, env):
		self.set(env, None)

	def __repr__(self):
		return f"StringVar({self.index!r})"


class WindowVar(Accessor):
	"""A plain real-valued window variable (Xmin, Tmax, …) — a named float attr on a
	Window held by env.  `target` names that Window: 'window' for the live settings,
	'zoom_window' for the ZoomSto snapshot's Z-variables (ZXmin, ZTmax, …).

	`resolve` raises UndefinedError for variables that were never assigned (tmin, tstep,
	etc. on a fresh env without ZStandard); `store` enforces require_real.
	"""

	__slots__ = ('attr', 'target')

	def __init__(self, attr: str, target: str = 'window'):
		self.attr = attr
		self.target = target

	def get(self, env):
		return getattr(getattr(env, self.target), self.attr)

	def set(self, env, value):
		setattr(getattr(env, self.target), self.attr, value)

	def resolve(self, env):
		v = self.get(env)
		if v is None:
			raise UndefinedError(f"Window variable {self.attr!r} is not defined")
		return v

	def store(self, env, value):
		self.set(env, require_real(value))

	def __repr__(self):
		return f"WindowVar({self.attr!r}, {self.target!r})"


class XresVar(WindowVar):
	"""Xres — function-graph resolution; a whole number 1–8."""

	def store(self, env, value):
		v = require_int(value)
		if not (1 <= v <= 8):
			raise DomainError(f"Xres must be an integer 1-8, got {v:g}")
		self.set(env, v)


class IntWindowVar(WindowVar):
	"""nMin, nMax — window variables constrained to whole numbers."""

	def store(self, env, value):
		self.set(env, require_int(value))


class FactorWindowVar(WindowVar):
	"""XFact, YFact — zoom-in/out scaling factors; must be ≥ 1."""

	def store(self, env, value):
		v = require_real(value)
		if v < 1:
			raise DomainError(f"Zoom factor must be ≥ 1, got {v:g}")
		self.set(env, v)


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
		setattr(env.window, self.hi_attr, lo + self.divisions * delta)

	def __repr__(self):
		return f"DeltaWindowVar({self.lo_attr!r}, {self.hi_attr!r}, {self.divisions})"


class EnvVar(Accessor):
	"""A plain real variable stored directly on env (TVM variables, stat n, …)."""

	__slots__ = ('attr',)

	def __init__(self, attr: str):
		self.attr = attr

	def get(self, env):
		return getattr(env, self.attr)

	def set(self, env, value):
		setattr(env, self.attr, value)

	def resolve(self, env):
		v = self.get(env)
		if v is None:
			raise UndefinedError(f"{self.attr!r} is not defined")
		return v

	def store(self, env, value):
		self.set(env, require_real(value))

	def __repr__(self):
		return f"EnvVar({self.attr!r})"


class TableVar(Accessor):
	"""A plain real variable stored on env.table (TblStart, ΔTbl)."""

	__slots__ = ('attr',)

	def __init__(self, attr: str):
		self.attr = attr

	def get(self, env):
		return getattr(env.table, self.attr)

	def set(self, env, value):
		setattr(env.table, self.attr, value)

	def resolve(self, env):
		v = self.get(env)
		if v is None:
			raise UndefinedError(f"Table variable {self.attr!r} is not defined")
		return v

	def store(self, env, value):
		self.set(env, require_real(value))

	def __repr__(self):
		return f"TableVar({self.attr!r})"


def _normalize_eq(value) -> TiEquation:
	"""Coerce a TiString or TiEquation to TiEquation; raise DataTypeError otherwise."""
	if isinstance(value, TiString):
		return TiEquation(value.tokens)
	if isinstance(value, TiEquation):
		return value
	raise DataTypeError(f"Expected equation or string, got {value!r}")


class EquationVar(Accessor):
	"""A graph equation (Y1–Y0, X1T/Y1T–X6T/Y6T, r1–r6).

	An equation is not one of TI's runtime value types, so reading it as a value
	(`resolve`) evaluates the formula at the current X.  `Y1(x)` is function
	composition — resolve with X temporarily set to x.  The raw `TiEquation` is
	reachable only through `get`/`set`, which is why copying a formula out
	(Equ►String) needs a dedicated command.  `store` normalises a string/equation
	and selects the function.
	"""

	__slots__ = ('mode', 'eq_index', 'func_index')
	invocable = True

	def __init__(self, mode: GraphMode, eq_index: int, func_index: int):
		self.mode = mode
		self.eq_index = eq_index
		self.func_index = func_index

	def get(self, env):
		return env.graph_functions.equations[self.mode][self.eq_index]

	def set(self, env, value):
		env.graph_functions.equations[self.mode][self.eq_index] = value

	def resolve(self, env):
		eq = self.get(env)
		if eq is None:
			raise UndefinedError("Equation is not defined")
		return eq.eval(env)

	def invoke(self, arg_parser):
		# The '(' is already eaten: Y1(x) composes by resolving with X set to x.
		x = arg_parser.expr()
		arg_parser.end_func()
		env = arg_parser.env
		saved_x = env.numerics.X
		env.numerics.X = require_num(x)
		try:
			return self.resolve(env)
		finally:
			env.numerics.X = saved_x

	def store(self, env, value):
		self.set(env, _normalize_eq(value))
		env.graph_functions.selected[self.mode][self.func_index] = True

	def delete(self, env):
		self.set(env, None)

	def __repr__(self):
		return f"EquationVar({self.mode.name}, {self.eq_index}, {self.func_index})"


class SequenceVar(Accessor):
	"""A recursive sequence variable 𝑢/𝑣/𝑤.

	Reading it as a value (`resolve`) evaluates the sequence at the current n;
	`u(n)` evaluates at an explicit index.  The raw `TiEquation` is reachable
	through `get`/`set`; `store_initial` handles the `{…}→u(nMin)` initial-value
	store path.
	"""

	__slots__ = ('seq_index',)
	invocable = True

	def __init__(self, seq_index: int):
		self.seq_index = seq_index

	def get(self, env):
		return env.graph_functions.equations[GraphMode.SEQ][self.seq_index]

	def set(self, env, value):
		env.graph_functions.equations[GraphMode.SEQ][self.seq_index] = value

	def resolve(self, env):
		return env.eval_sequence(self.seq_index, env.n)

	def invoke(self, arg_parser):
		# The '(' is already eaten: u(n) evaluates at the explicit index n.
		n = arg_parser.expr()
		arg_parser.end_func()
		return arg_parser.env.eval_sequence(self.seq_index, n)

	def store(self, env, value):
		self.set(env, _normalize_eq(value))
		env.graph_functions.selected[GraphMode.SEQ][self.seq_index] = True

	def store_initial(self, env, value):
		"""Handle `{…}→u(nMin)`: store the sequence's initial-value list."""
		env.store_sequence_initial(self.seq_index, value)

	def delete(self, env):
		self.set(env, None)

	def __repr__(self):
		return f"SequenceVar({self.seq_index})"

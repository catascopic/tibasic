"""How tokens read, write, and reference the environment.

An `Accessor` is a stateless flyweight that knows *how* to get/set one symbol — a
numeric variable, a list, π, rand, a window setting — in any Environment.  A Token
holds one (`Token.accessor`), and the parser calls `resolve`/`store` on it directly,
passing the environment, so there's no per-environment Variable object to allocate and
no `env` threaded through every call site.

When a command needs to *hold* a variable rather than its value (For, fnInt, Input,
DelVar, …), `Accessor.reference(env)` binds the accessor to an environment as a
`Reference`.  A Reference deliberately mirrors the old Variable surface (`resolve`,
`store`, `delete`, `scoped`, and a `.value` get/set) so those commands keep working
while the underlying storage migrates.

Numeric, matrix, list, string, window, TVM, and table variables all have dedicated
accessor types.  Equation variables and user-defined lists still use `core.Variable`
objects reached through the `LegacyAccessor` bridge.
"""
from contextlib import contextmanager

from core import require_num, require_matrix, require_list, require_string, require_real, require_int
from errors import TiSyntaxError, UndefinedError, InvalidDimError, DomainError


class Accessor:
	"""Stateless description of how to access one symbol.  Subclasses override the
	pieces that differ; the defaults make a read-only value (store raises).

	`resolve`/`store` are the validating, user-facing operations (auto-init, type
	checks); `get`/`set` are the raw, unchecked accessors used internally (e.g. by
	`scoped`).  By default `get`/`set` fall back to `resolve`/`store`.
	"""

	kind = None     # a discriminator for callers that must branch on type (e.g. Input: 'string')

	def resolve(self, env):
		raise NotImplementedError

	def store(self, env, value):
		raise TiSyntaxError("Invalid store target")

	def get(self, env):
		return self.resolve(env)

	def set(self, env, value):
		self.store(env, value)

	def delete(self, env):
		pass

	def reference(self, env) -> "Reference":
		return Reference(env, self)


class Reference:
	"""An accessor bound to an environment — what commands receive when they take a
	variable rather than its value.  Mirrors the old Variable surface so consumers
	(For/fnInt/Input/DelVar/…) need no changes."""

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

	@property
	def value(self):
		return self.accessor.get(self.env)

	@value.setter
	def value(self, new_value):
		self.accessor.set(self.env, new_value)

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


class LegacyAccessor(Accessor):
	"""Bridge to a not-yet-converted core.Variable object (lists, matrices, strings,
	equations, window vars).  `lookup(env)` returns the Variable; everything forwards
	to it.  `kind` lets callers that must discriminate (Input: string vs numeric) do
	so without isinstance on the storage class.
	"""

	__slots__ = ('lookup', 'kind')

	def __init__(self, lookup, kind=None):
		self.lookup = lookup   # (env) -> core.Variable
		self.kind = kind

	def resolve(self, env):
		return self.lookup(env).resolve()

	def store(self, env, value):
		self.lookup(env).store(value)

	def get(self, env):
		return self.lookup(env).value

	def set(self, env, value):
		self.lookup(env).value = value

	def delete(self, env):
		self.lookup(env).delete()

	def __repr__(self):
		return f"LegacyAccessor(kind={self.kind!r})"


class MatrixVar(Accessor):
	"""A matrix variable [A]–[J] — a named slot in env.matrices (MatrixVars).

	Undefined matrices raise UndefinedError on resolve; `store` deep-copies so the
	stored value is isolated from the caller.
	"""

	__slots__ = ('name',)

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
	"""A plain real-valued window variable (Xmin, Tmax, …) — a named float attr on env.window.

	`resolve` raises UndefinedError for variables that were never assigned (tmin, tstep,
	etc. on a fresh env without ZStandard); `store` enforces require_real.
	"""

	__slots__ = ('attr',)

	def __init__(self, attr: str):
		self.attr = attr

	def get(self, env):
		return getattr(env.window, self.attr)

	def set(self, env, value):
		setattr(env.window, self.attr, value)

	def resolve(self, env):
		v = self.get(env)
		if v is None:
			raise UndefinedError(f"Window variable {self.attr!r} is not defined")
		return v

	def store(self, env, value):
		self.set(env, require_real(value))

	def __repr__(self):
		return f"WindowVar({self.attr!r})"


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

import math
import string
import itertools

VALID_NAMES = frozenset(itertools.chain(string.ascii_uppercase, ('theta',)))

class Ref:
    def __init__(self, store: 'NumericVars', name: str):
        self._store = store
        self._name = name

    # Raw get/set — no validation, mirrors direct variable access on the calculator
    def get(self) -> float | None:
        return getattr(self._store, self._name)

    def set(self, value: float) -> None:
        setattr(self._store, self._name, value)

    # resolve/store — validated equivalents
    def resolve(self) -> float:
        """Get the value, raising if the variable has never been assigned."""
        value = self.get()
        if value is None:
            raise ValueError(f"Variable {self._name} is undefined")
        return value

    def store(self, value: object) -> None:
        """Validate then set. Accepts int/float, rejects non-numeric, NaN, and infinity."""
        if not isinstance(value, (int, float)):
            raise TypeError(f"TI-84 numeric variables must be real numbers, got {type(value).__name__}")
        if math.isnan(value):
            raise ValueError("TI-84 does not support NaN")
        if math.isinf(value):
            raise ValueError("TI-84 does not support infinity")
        self.set(float(value))

    def __repr__(self):
        return f"Ref({self._name!r} = {self.get()!r})"


class NumericVars:
    __slots__ = tuple(VALID_NAMES)

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

    def ref(self, name: str) -> Ref:
        if name not in VALID_NAMES:
            raise KeyError(f"'{name}' is not a valid TI-84 variable (expected A-Z or theta)")
        return Ref(self, name)

    def __repr__(self):
        set_vars = {s: getattr(self, s) for s in self.__slots__ if getattr(self, s) is not None}
        return f"NumericVars({set_vars})"
		
		



class NumericVar:
	def __init__(self, name):
		self.name = name
	
	def get(self, env):
		return getattr(env.numerics, self.name)

	def set(self, env, value):
		return setattr(env.numerics, self.name, value)
	
	def store(self, env, value):
		self.set(env, require_number(value))
	
	def resolve(self, env):
		value = self.get(self)
		if value is not None:
			return value
		self.set(env, 0.0)
		return 0.0
	
	def ref(env):
		return Ref(env, self)


class Reference:
	def __init__(self, env, var):
		self.env = env
		self.var = var
	
	[get/set/store/resolve]

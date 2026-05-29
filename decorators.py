from functools import wraps
from itertools import repeat
from typing import TYPE_CHECKING

from tiobjects import TiList

if TYPE_CHECKING:
	from parser import ArgParser


# I'm thinking of a new paradigm for calculator functions: every function is expected to take ArgParser.
# This was basically already happening, except that pure functions were being wrapped by tokens (using the "pure" parameter as opposed to "func"), but I think it might be better to do away with this and just put decorators on every function, which turn it into a function that accepts ArgParser.

# Ideally we'd have these decorators:
# (A) takes regular function, adapts it so it's called with ArgParser (_make_pure_function in tokens used to do this)
# (B) same thing, but passes in env (@env_func in forms used to do this)
# (C) turns function into vectorized
# (D) turns function into vectorized, but skips the first param (env)
# I'd like (C) to be decoupled from (A), so that I can apply @vectorized to functions without necessarily turning them into ArgParse functions. As seen below, I use env_vectorized as sort of the base-level vectorizor, which probably isn't correct.  I'd like a standalone vectorize decorator, since the _rand_int_single function needs this.
# If there can be a standalone vectorized function, there should probably decorators for convenience that do what the following decorators do (pure-vectorized, pure-normal, env-vectorized, env-normal).
	
# Also: What's even the point of TYPE_CHECKING? It seems like I still have to put ArgParser in quotes

# Ideally we'd start with this, then build off of it:
def vectorized(func):
	@wraps(func)
	def apply(args):
		len_check = set()
		vec = []
		for a in args:
			if isinstance(a, TiList):
				len_check.add(len(a))
				vec.append(a)
			else:
				vec.append(repeat(a))
		if not len_check:
			return func(*args)
		if len(len_check) == 1:
			return TiList([func(*v) for v in zip(*vec)])
		raise DimMismatchError(f"Dim mismatch: {len_check}")
	return apply

# One possible solution could be to make a factory for the vectorized decorator which takes a flag that controls whether the first argument (env) is exempt from vectorization.
# Alternatively, maybe make a helper (non-decorator) function that does a vectorized call, and it takes a function and arguments. env functions would pass partial(func, env). For example:

def call_vectorized(func, *args):
	len_check = set()
	vec = []
	for a in args:
		if isinstance(a, TiList):
			len_check.add(len(a))
			vec.append(a)
		else:
			vec.append(repeat(a))
	if not len_check:
		return func(*args)
	if len(len_check) == 1:
		return TiList([func(*v) for v in zip(*vec)])
	raise DimMismatchError(f"Dim mismatch: {len_check}")

# Here's what I currently have. I want to have these four decorators, I just want the vectorized logic to also exist as a standalone decorator without code duplication.

def env_vectorized(func):
	@wraps(func)
	def apply(parser: 'ArgParser'):
		args = parser.parse_args()
		len_check = set()
		vec = []
		for a in args:
			if isinstance(a, TiList):
				len_check.add(len(a))
				vec.append(a)
			else:
				vec.append(repeat(a))
		if not len_check:
			return func(parser.env, *args)
		if len(len_check) == 1:
			return TiList([func(parser.env, *v) for v in zip(*vec)])
		raise DimMismatchError(f"Dim mismatch: {len_check}")
	return apply


def env_func(func):
	@wraps(func)
	def apply(a: 'ArgParser'):
		return func(a.env, *a.parse_args())
	return apply


def pure_vectorized(func):
	@wraps(func)
	def apply(_env, *args):
		return func(*args)
	return env_vectorized(apply)


def pure_func(func):
	@wraps(func)
	def apply(a: 'ArgParser'):
		return func(*a.parse_args())
	return apply




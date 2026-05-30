from __future__ import annotations
from functools import partial, wraps
from itertools import repeat
from typing import TYPE_CHECKING

from tiobjects import TiList
from errors import DimMismatchError

if TYPE_CHECKING:
	from parser import ArgParser


def _call_vectorized(func, args):
	"""Core vectorize logic: broadcast func(*args) element-wise over any TiList args."""
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


def vectorized(func):
	"""Standalone decorator: broadcast f(*args) element-wise over TiList arguments."""
	@wraps(func)
	def apply(*args):
		return _call_vectorized(func, args)
	return apply


def pure_vectorized(func):
	"""ArgParser handler: parse args, vectorize, call func(*args)."""
	vec = vectorized(func)
	@wraps(func)
	def apply(a: ArgParser):
		return vec(*a.parse_args())
	return apply


def env_vectorized(func):
	"""ArgParser handler: parse args, vectorize, call func(env, *args)."""
	@wraps(func)
	def apply(a: ArgParser):
		return _call_vectorized(partial(func, a.env), a.parse_args())
	return apply


def env_func(func):
	"""ArgParser handler: parse args, call func(env, *args)."""
	@wraps(func)
	def apply(a: ArgParser):
		return func(a.env, *a.parse_args())
	return apply


def pure_func(func):
	"""ArgParser handler: parse args, call func(*args)."""
	@wraps(func)
	def apply(a: ArgParser):
		return func(*a.parse_args())
	return apply

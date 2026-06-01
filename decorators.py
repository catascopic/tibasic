from __future__ import annotations
from functools import partial, wraps, update_wrapper
from itertools import repeat
from typing import TYPE_CHECKING

from tiobjects import TiList
from errors import DimMismatchError

if TYPE_CHECKING:
	from parser import ArgParser


def _call_vectorized(func, args):
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
	@wraps(func)
	def apply(*args):
		return _call_vectorized(func, args)
	return apply


def _vectorized_env(func):
	@wraps(func)
	def apply(env, *args):
		return _call_vectorized(partial(func, env), args)
	return apply


class _TiBase:
	def __init__(self, func):
		self._func = func
		update_wrapper(self, func)

	def __call__(self, *args):
		return self._func(*args)


class pure_func(_TiBase):
	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		a.end_func()
		return self(*args)


class env_func(_TiBase):
	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		a.end_func()
		return self(a.env, *args)


class cmd_env_func(_TiBase):
	def call_with_parser(self, a: ArgParser):
		args = a.parse_args()
		a.end_paren_cmd()
		return self(a.env, *args)


class forms_func:
	def __init__(self, func):
		self._func = func
		update_wrapper(self, func)

	def __call__(self, *args):
		return self._func(*args)

	def call_with_parser(self, a: ArgParser):
		return self._func(a)


def pure_vectorized(func):
	return pure_func(vectorized(func))


def env_vectorized(func):
	return env_func(_vectorized_env(func))

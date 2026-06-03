from dataclasses import dataclass
from collections.abc import Callable

from environment import Variable


@dataclass(slots=True, frozen=True, eq=False)
class Token:
	code: bytes
	char: str | None
	text: str
	bp: tuple[int, int] | None = None
	operator:  Callable | None = None  # (lhs, rhs) -> value
	postfix:   Callable | None = None  # (operand) -> value (prefix or postfix)
	function:  Callable | None = None  # (ArgParser) -> value for function tokens
	command:   Callable | None = None  # (ArgParser) -> None for command tokens
	nullary:   Callable | None = None  # (env) -> value for read-only computed constants
	converter: Callable | None = None  # (value) -> value for ►DMS, ►Dec, ►Frac and others
	variable:  Callable | None = None  # Variable flyweight for storable typed variables

	# ── Token type predicates ──────────────────────────────────────────────────

	def is_digit(self) -> bool:
		return 0x30 <= self.code[0] <= 0x39

	def is_numeric_var(self) -> bool:
		return 0x41 <= self.code[0] < 0x5C or self.code == b'\x62\x21'

	def is_list_var(self) -> bool:
		return self.code[0] == 0x5D

	def is_list_start(self):
		return self.code[0] in {0x5D, 0xEB}

	def is_matrix_var(self) -> bool:
		return self.code[0] == 0x5C

	def is_equation_var(self) -> bool:
		return self.code[0] == 0x5E

	def is_string_var(self) -> bool:
		return self.code[0] == 0xAA

	def is_stat_var(self) -> bool:
		return self.code[0] == 0x62

	def is_window_var(self) -> bool:
		return self.code[0] == 0x63

	def is_name_char(self):
		return self.is_numeric_var() or self.is_digit()

	def __repr__(self):
		return f'0x{int.from_bytes(self.code):0{2 * len(self.code)}X}:{self.text!r}'


class EofToken:
	"""Sentinel returned by Parser.peek() at end of input.

	All type predicates return False; all callable/variable fields are None.
	Duck-type compatible with Token for all predicate and attribute access patterns
	used in the parser.
	"""
	text      = '<END-OF-INPUT>'
	char      = None
	bp        = None
	operator  = None
	postfix   = None
	function  = None
	command   = None
	nullary   = None
	converter = None
	variable  = None

	def is_digit(self) -> bool:        return False
	def is_numeric_var(self) -> bool:  return False
	def is_list_var(self) -> bool:     return False
	def is_list_start(self) -> bool:   return False
	def is_matrix_var(self) -> bool:   return False
	def is_equation_var(self) -> bool: return False
	def is_string_var(self) -> bool:   return False
	def is_stat_var(self) -> bool:     return False
	def is_window_var(self) -> bool:   return False
	def is_name_char(self) -> bool:    return False

	def __repr__(self) -> str:         return '<EOF>'


EOF_TOKEN = EofToken()

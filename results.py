"""Statement result types and store-target types returned by the parser."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

Value = float | complex | str | list  # list = TI list or matrix (list-of-rows)

# ── Statement results ──────────────────────────────────────────────────────────

@dataclass
class ExprResult:
	value: Value

@dataclass
class StoreResult:
	value: Value
	target: 'StoreTarget'

@dataclass
class IfResult:
	cond: bool

@dataclass
class WhileResult:
	cond: bool

@dataclass
class RepeatResult:
	pass

@dataclass
class ForResult:
	var: str
	start: float
	end: float
	step: float

@dataclass
class EndResult:
	pass

@dataclass
class LblResult:
	label: str

@dataclass
class GotoResult:
	label: str

@dataclass
class ReturnResult:
	pass

@dataclass
class StopResult:
	pass

@dataclass
class PauseResult:
	value: Value | None

@dataclass
class DispResult:
	values: list[Value]

@dataclass
class InputResult:
	prompt: str | None
	target: 'StoreTarget'

@dataclass
class PromptResult:
	targets: list['StoreTarget']

@dataclass
class OutputResult:
	row: float
	col: float
	value: Value

@dataclass
class MenuResult:
	title: Value
	options: list[tuple[Value, str]]  # (display_name, label_name)

@dataclass
class ClrHomeResult:
	pass

@dataclass
class DispGraphResult:
	pass

# ── Store targets ──────────────────────────────────────────────────────────────

@dataclass
class RealVarTarget:
	name: str  # single letter or 'θ'

@dataclass
class ListVarTarget:
	code: bytes  # 0x5Dxx

@dataclass
class UserListTarget:
	name: str

@dataclass
class MatrixVarTarget:
	code: bytes  # 0x5Cxx

@dataclass
class StringVarTarget:
	code: bytes  # 0xAAxx

@dataclass
class ListElementTarget:
	list_ref: bytes | str  # built-in code or user-list name
	index: Value

@dataclass
class MatrixElementTarget:
	code: bytes
	row: Value
	col: Value

@dataclass
class AnsTarget:
	pass

StoreTarget = (RealVarTarget | ListVarTarget | UserListTarget | MatrixVarTarget
	| StringVarTarget | ListElementTarget | MatrixElementTarget | AnsTarget)

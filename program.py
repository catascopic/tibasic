from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import Thunk

from environment import Environment, Variable
from errors import TiSyntaxError, IncrementError, LabelError
from signals import ReturnSignal
from tiobjects import require_real
from catalog import THEN, ELSE, LBL


class Program:
	"""
	Wraps a stored token stream as an executable program.

	Owns the Parser (and thus the current execution position), the block
	stack that tracks active For/While/Repeat/If-Then blocks, and the
	per-execution flags used by control-flow commands.

	Programs are pushed onto env.program_stack for the duration of their
	execution so that control-flow commands can access the current program
	via env.current_program.
	"""

	def __init__(self, tokens: list, env: Environment):
		from parser import Parser  # lazy: program → parser → tokens → forms → program
		self._parser = Parser(tokens, env)
		self.env = env
		self.block_stack: list[Block] = []

	# ── Execution loop ────────────────────────────────────────────────────────

	def run(self) -> None:
		"""Execute all statements in the token stream until EOF."""
		self.env.program_stack.append(self)
		try:
			self._parser.run()
		except ReturnSignal:
			pass
		finally:
			finished = self.env.program_stack.pop()
			if finished is not self:
				raise ValueError(f"Program stack out of order: expected {self}; got {finished}")

	# ── Block stack ───────────────────────────────────────────────────────────

	def push_block(self, block: Block) -> None:
		self.block_stack.append(block)

	def pop_block(self) -> Block:
		if not self.block_stack:
			raise TiSyntaxError("End/Else without matching block")
		return self.block_stack.pop()

	def jump(self, pos: int) -> None:
		"""Set the parser's execution position (used by loop blocks on_end)."""
		self._parser.pos = pos

	# ── Control-flow commands ─────────────────────────────────────────────────

	def begin_if(self, cond: bool) -> None:
		if self._parser.eat_if(THEN):
			self._parser.end_statement()
			if cond:
				self.push_block(ThenBlock())
				# no special handling required for Else here
			else:
				found = self._parser.skip_block(else_mode=True)
				if found is ELSE:
					# handle Else as if it's an If-Else block that's closed by End
					self.push_block(ThenBlock())
		elif not cond:
			self._parser.skip_statement()

	def begin_else(self) -> None:
		if not isinstance(self.pop_block(), ThenBlock):
			raise TiSyntaxError("Else without matching Then")
		self._parser.skip_block()

	def begin_while(self, condition: Thunk) -> None:
		if condition.eval():
			self.push_block(WhileBlock(self._parser.pos, condition))
		else:
			self._parser.skip_block()

	def begin_repeat(self, condition: Thunk) -> None:
		self.push_block(RepeatBlock(self._parser.pos, condition))

	def begin_for(self, var: Variable, start: float, end: float, step: float) -> None:
		var.set(self.env, start)
		if check_for_condition(start, end, step):
			self.push_block(ForBlock(self._parser.pos, var, end, step))
		else:
			self._parser.skip_block()

	def end_block(self) -> None:
		self.pop_block().on_end(self)

	def is_gt(self, var: Variable, threshold: float) -> None:
		new_val = require_real(var.get(self.env)) + 1
		var.set(self.env, new_val)
		if new_val > threshold:
			self._parser.skip_statement()

	def ds_lt(self, var: Variable, threshold: float) -> None:
		new_val = require_real(var.get(self.env)) - 1
		var.set(self.env, new_val)
		if new_val < threshold:
			self._parser.skip_statement()

	# ── Label search ──────────────────────────────────────────────────────────

	def goto(self, name: str) -> None:
		"""Jump to the first Lbl <name> in the token stream.

		Scans from the beginning, skipping statements with skip_statement()
		(which correctly ignores LBL tokens inside string literals).
		Raises LabelError if the label is not found.
		"""
		p = self._parser
		current_pos = p.pos
		p.pos = 0
		while p.has_next:
			if p.eat_if(LBL):
				label = p.parse_label_name()
				p.end_statement()
				if label == name:
					return
			else:
				p.skip_statement()
		raise LabelError(
			f"Label not found: {name!r}",
			pos=current_pos - 2 - len(name)  # kind of a hack
		)


# ── For-loop continuation helper ─────────────────────────────────────────────

def check_for_condition(value: float, end: float, step: float) -> bool:
	"""Return True if the For loop should continue (value has not exceeded end)."""
	if step > 0:
		return value <= end + 1e-10
	elif step < 0:
		return value >= end - 1e-10
	else:
		raise IncrementError("For: step cannot be zero")


class Block(ABC):
	def on_end(self, prgm: 'Program'):
		"""This method should not be abstract; the default action is to do nothing."""


@dataclass
class ForBlock(Block):
	pos: int
	var: Variable
	end: float
	step: float

	def on_end(self, prgm: Program) -> None:
		new_val = self.var.get(prgm.env) + self.step
		self.var.set(prgm.env, new_val)
		if check_for_condition(new_val, self.end, self.step):
			prgm.jump(self.pos)
			prgm.push_block(self)


@dataclass
class WhileBlock(Block):
	pos: int
	condition: Thunk

	def on_end(self, prgm: Program) -> None:
		if self.condition.eval():
			prgm.jump(self.pos)
			prgm.push_block(self)


@dataclass
class RepeatBlock(Block):
	pos: int
	condition: Thunk

	def on_end(self, prgm: Program) -> None:
		if not self.condition.eval():
			prgm.jump(self.pos)
			prgm.push_block(self)


@dataclass
class ThenBlock(Block):
	"""Marker for an active If/Then or Else block.  End simply pops it (no-op on_end)."""
	pass

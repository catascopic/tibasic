from __future__ import annotations
from abc import ABC, abstractmethod
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
	via env.current_program().
	"""

	def __init__(self, tokens: list, env: Environment):
		from parser import Parser  # lazy: program → parser → tokens → forms → program
		self._parser = Parser(tokens, env)
		self._env = env
		self._block_stack: list[Block] = []

	def run(self):
		"""Execute all statements in the token stream until EOF."""
		self._env.program_stack.append(self)
		try:
			self._parser.run()
		except ReturnSignal:
			pass
		finally:
			finished = self._env.program_stack.pop()
			if finished is not self:
				raise ValueError(f"Program stack out of order: expected {self}; got {finished}")

	def push_block(self, block: Block):
		self._block_stack.append(block)

	def begin_if(self, condition: bool):
		if self._parser.eat_if(THEN):
			self._parser.end_statement()
			if condition:
				self.push_block(ThenBlock())
				# no special handling required for Else here
			else:
				found = self._parser.skip_block(else_mode=True)
				if found is ELSE:
					# handle Else as if it's an If-Else block that's closed by End
					self.push_block(ThenBlock())
		elif not condition:
			self._parser.skip_statement()

	def begin_else(self):
		if not self._block_stack:
			raise TiSyntaxError("Else without matching block")
		block = self._block_stack.pop()
		if not isinstance(block, ThenBlock):
			raise TiSyntaxError(f"Expected Then block to match Else; got {block}")
		self._parser.skip_block()

	def begin_while(self, condition: Thunk):
		if condition.eval():
			self.push_block(WhileBlock(self._parser.pos, condition))
		else:
			self._parser.skip_block()

	def begin_repeat(self, condition: Thunk):
		self.push_block(RepeatBlock(self._parser.pos, condition))

	def begin_for(self, var: Variable, start: float, end: float, step: float):
		var.value = start
		if check_for_condition(start, end, step):
			self.push_block(ForBlock(self._parser.pos, var, end, step))
		else:
			self._parser.skip_block()

	def end_block(self):
		if not self._block_stack:
			raise TiSyntaxError("End without matching block")
		block = self._block_stack[-1]		
		if block.on_end():
			self._parser.pos = block.pos
		else:
			self._block_stack.pop()

	def is_gt(self, var: Variable, threshold: float):
		var.value = require_real(var.resolve()) + 1
		if var.value > threshold:
			self._parser.skip_statement()

	def ds_lt(self, var: Variable, threshold: float):
		var.value = require_real(var.resolve()) - 1
		if var.value < threshold:
			self._parser.skip_statement()

	def goto(self, name: str):
		"""Jump to the first Lbl <name> in the token stream.

		Scans from the beginning, skipping statements with skip_statement()
		(which correctly ignores LBL tokens inside string literals).
		Raises LabelError if the label is not found.
		"""
		if self._block_stack:
			pass  # TODO: emit warning for jumping out of a block
		
		p = self._parser
		lbl_pos = p.pos
		p.pos = 0
		while p.has_next:
			if p.eat_if(LBL):
				label = p.parse_label_name()
				p.end_statement()
				if label == name:
					return
			else:
				p.skip_statement()
		# Missing label: return to the invalid Goto
		p.pos = lbl_pos
		raise LabelError(f"Label not found: {name!r}")


def check_for_condition(value: float, end: float, step: float) -> bool:
	if step > 0:
		return value <= end + 1e-10
	if step < 0:
		return value >= end - 1e-10
	raise IncrementError("For: step cannot be zero")


class Block(ABC):
	@abstractmethod
	def on_end(self) -> bool:
		"""Returns whether the program should re-run the block."""


@dataclass
class LoopBlock(Block, ABC):
	pos: int


@dataclass
class ForBlock(LoopBlock):
	var: Variable
	end: float
	step: float

	def on_end(self):
		self.var.value = self.var.resolve() + self.step
		return check_for_condition(self.var.value, self.end, self.step)


@dataclass
class WhileBlock(LoopBlock):
	condition: Thunk

	def on_end(self):
		return self.condition.eval()


@dataclass
class RepeatBlock(LoopBlock):
	condition: Thunk

	def on_end(self):
		return not self.condition.eval()


class ThenBlock(Block):
	def on_end(self):
		return False

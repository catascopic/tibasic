from abc import ABC, abstractmethod
from dataclasses import dataclass

from core import Thunk, require_real
from tokenbase import Reference
from environment import Environment, ReturnSignal
from errors import TiSyntaxError, IncrementError, LabelError
from tokenbase import THEN, ELSE, LBL
from parser import Parser


class Program:
	"""A stored TI-BASIC program: a token stream with an optional name.

	The static, reusable form kept in env.programs.  Running it builds a fresh
	Execution (one invocation's runtime state) and executes it; the same Program
	can be run any number of times.
	"""

	def __init__(self, tokens: list, name: str | None = None):
		self.tokens = tokens
		self.name = name

	def run(self, env: Environment) -> None:
		"""Execute this program once in `env`."""
		Execution(self, env).run()

	def __repr__(self) -> str:
		return f"Program({self.name!r}, {len(self.tokens)} tokens)"


class Execution:
	"""The runtime state of one program invocation.

	Owns the Parser (and thus the current execution position), the block stack
	that tracks active For/While/Repeat/If-Then blocks, and the control-flow
	mechanics that act on them.

	An Execution is pushed onto env.execution_stack for the duration of the
	invocation, so control-flow commands can reach the current one via
	env.current_execution(); a prgm-call stacks a new Execution on top.
	"""

	def __init__(self, program: Program, env: Environment):
		self.program = program
		self._parser = Parser(program.tokens, env)
		self._env = env
		self._block_stack: list[Block] = []

	def run(self):
		"""Execute all statements in the token stream until EOF."""
		self._env.execution_stack.append(self)
		try:
			self._parser.parse()
		except ReturnSignal:
			pass
		finally:
			finished = self._env.execution_stack.pop()
			if finished is not self:
				raise ValueError(f"Execution stack out of order: expected {self}; got {finished}")

	def push_block(self, block: 'Block'):
		self._block_stack.append(block)

	def begin_if(self, condition: bool):
		if self._parser.eat_if(THEN):
			self._parser.end_statement()
			if condition:
				self.push_block(ThenBlock())
				# no special handling required for Else here
			else:
				found = self._parser.skip_block(else_mode=True)
				if found.code == ELSE:
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

	def begin_for(self, var: Reference, start: float, end: float, step: float):
		var.set(start)
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

	def is_gt(self, var: Reference, threshold: float):
		new = require_real(var.resolve()) + 1
		var.set(new)
		if new > threshold:
			self._parser.skip_statement()

	def ds_lt(self, var: Reference, threshold: float):
		new = require_real(var.resolve()) - 1
		var.set(new)
		if new < threshold:
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
	var: Reference
	end: float
	step: float

	def on_end(self):
		new = self.var.resolve() + self.step
		self.var.set(new)
		return check_for_condition(new, self.end, self.step)


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

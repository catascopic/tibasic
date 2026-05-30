from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import Parser, Thunk

from environment import Environment, Variable
from errors import TiSyntaxError, IncrementError, LabelError
from signals import ReturnSignal
from tokens import COLON, NEWLINE, QUOTE


# ── For-loop continuation helper ─────────────────────────────────────────────

def _for_continues(val: float, end_val: float, step: float) -> bool:
	"""Return True if the For loop should continue (val has not exceeded end_val)."""
	if step > 0:
		return val <= end_val + 1e-10
	elif step < 0:
		return val >= end_val - 1e-10
	else:
		raise IncrementError("For: step cannot be zero")


# ── Block types ───────────────────────────────────────────────────────────────

class Block:
	"""Base class for all control-flow blocks.

	Each concrete subclass overrides on_end to implement the End logic for
	that block type, eliminating isinstance checks in end_cmd.
	"""

	def on_end(self, prog: Program, parser: Parser) -> None:
		"""Handle End for this block.  Default implementation is a no-op (ThenBlock)."""
		pass


@dataclass
class ForBlock(Block):
	"""State for an active For( loop."""
	pos: int         # token index of the separator before the loop body
	var: Variable    # loop variable
	end_val: float   # loop exits when var exceeds this (or drops below it for negative step)
	step: float      # added to var at each End

	def on_end(self, prog: Program, parser: Parser) -> None:
		new_val = self.var.get(parser.env) + self.step
		self.var.set(parser.env, new_val)
		if _for_continues(new_val, self.end_val, self.step):
			prog.push_block(self)  # keep alive for the next iteration
			parser.pos = self.pos  # jump back to separator before body


@dataclass
class WhileBlock(Block):
	"""State for an active While loop."""
	pos: int          # token index of the separator before the loop body
	condition: Thunk  # re-evaluated at End; True → repeat, False → exit

	def on_end(self, prog: Program, parser: Parser) -> None:
		if self.condition.eval():
			prog.push_block(self)
			parser.pos = self.pos


@dataclass
class RepeatBlock(Block):
	"""State for an active Repeat loop."""
	pos: int          # token index of the separator before the loop body
	condition: Thunk  # evaluated at End; True → exit, False → repeat

	def on_end(self, prog: Program, parser: Parser) -> None:
		if not self.condition.eval():
			prog.push_block(self)
			parser.pos = self.pos


@dataclass
class ThenBlock(Block):
	"""Marker for an active If/Then or Else block.  End simply pops it (no-op on_end)."""
	pass


# ── Program ───────────────────────────────────────────────────────────────────

class Program:
	"""Wraps a stored token stream as an executable program.

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
		self._env = env
		self.block_stack: list[Block] = []
		self._pending_if_result: bool | None = None  # set by If, read by Then

	# ── Execution loop ────────────────────────────────────────────────────────

	def run(self) -> None:
		"""Execute all statements in the token stream until EOF."""
		self._env.program_stack.append(self)
		try:
			self._parser.run()
		except ReturnSignal:
			pass  # normal sub-program return; stop executing this program
		finally:
			self._env.program_stack.pop()

	# ── Block stack ───────────────────────────────────────────────────────────

	def push_block(self, block: Block) -> None:
		self.block_stack.append(block)

	def pop_block(self) -> Block:
		if not self.block_stack:
			raise TiSyntaxError("End without matching block")
		return self.block_stack.pop()

	def peek_block(self) -> Block | None:
		return self.block_stack[-1] if self.block_stack else None

	# ── Label search ──────────────────────────────────────────────────────────

	def goto(self, name: str) -> None:
		"""Jump to the first Lbl <name> in the token stream.

		Scans from the beginning, respecting string literals.
		Raises LabelError if the label is not found.
		"""
		from tokens import LBL  # lazy: avoids circular import at module load time
		p = self._parser
		in_string = False
		i = 0
		while i < len(p.tokens):
			t = p.tokens[i]
			if in_string:
				if t is QUOTE or t is NEWLINE:
					in_string = False
			elif t is QUOTE:
				in_string = True
			elif t is LBL:
				# next 1–2 name-char tokens form the label name
				label = ''
				j = i + 1
				while j < len(p.tokens) and p.tokens[j].is_name_char() and len(label) < 2:
					label += p.tokens[j].char
					j += 1
				if label == name:
					p.pos = j  # resume execution after the label
					return
			i += 1
		raise LabelError(f"Label not found: {name!r}")

	# ── Properties ───────────────────────────────────────────────────────────

	@property
	def env(self) -> Environment:
		return self._env

	@property
	def parser(self) -> Parser:
		return self._parser

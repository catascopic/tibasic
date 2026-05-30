from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import Thunk

from environment import Environment, Variable
from errors import TiSyntaxError, ReturnSignal
from parser import Parser, EOF_TOKEN
from tokens import COLON, NEWLINE, QUOTE


# ── Block Types ────────────────────────────────────────────────────────

class Block:
	"""Base class for all blocks."""
	pass

@dataclass
class ForBlock(Block):
	"""State for an active For( loop."""
	pos: int         # token index of the first token in the loop body
	var: Variable    # loop variable
	end_val: float   # loop exits when var exceeds this (or goes below it for negative step)
	step: float      # added to var at each End

@dataclass
class WhileBlock(Block):
	"""State for an active While loop."""
	pos: int         # token index of the first token in the loop body
	condition: Thunk # re-evaluated at End; True → jump back to body, False → exit

@dataclass
class RepeatBlock(Block):
	"""State for an active Repeat loop."""
	pos: int         # token index of the first token in the loop body
	condition: Thunk # evaluated at End; True → exit loop, False → jump back to body

@dataclass
class ThenBlock(Block):
	"""Marker for an active If/Then or Else block.  End simply pops it."""
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
		from errors import LabelError
		from tokens import LBL  # imported lazily; token defined when control-flow tokens are added
		p = self._parser
		in_string = False
		i = 0
		while i < len(p.tokens):
			t = p.tokens[i]
			if in_string:
				if t is QUOTE:
					in_string = False
				elif t is NEWLINE:
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

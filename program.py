from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from parser import Thunk

from environment import Environment, Variable
from errors import TiSyntaxError, IncrementError, LabelError
from signals import ReturnSignal
from tiobjects import require_real
from tokens import COLON, NEWLINE, QUOTE


# ── For-loop continuation helper ─────────────────────────────────────────────

def check_for_condition(val: float, end_val: float, step: float) -> bool:
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

	def on_end(self, prog: 'Program') -> None:
		"""Handle End for this block.  Default implementation is a no-op (ThenBlock)."""
		pass


@dataclass
class ForBlock(Block):
	"""State for an active For( loop."""
	pos: int         # token index of the separator before the loop body
	var: Variable    # loop variable
	end_val: float   # loop exits when var exceeds this (or drops below it for negative step)
	step: float      # added to var at each End

	def on_end(self, prog: 'Program') -> None:
		new_val = self.var.get(prog.env) + self.step
		self.var.set(prog.env, new_val)
		if check_for_condition(new_val, self.end_val, self.step):
			prog.push_block(self)       # keep alive for the next iteration
			prog.jump_to(self.pos)      # jump back to separator before body


@dataclass
class WhileBlock(Block):
	"""State for an active While loop."""
	pos: int          # token index of the separator before the loop body
	condition: Thunk  # re-evaluated at End; True → repeat, False → exit

	def on_end(self, prog: 'Program') -> None:
		if self.condition.eval():
			prog.push_block(self)
			prog.jump_to(self.pos)


@dataclass
class RepeatBlock(Block):
	"""State for an active Repeat loop."""
	pos: int          # token index of the separator before the loop body
	condition: Thunk  # evaluated at End; True → exit, False → repeat

	def on_end(self, prog: 'Program') -> None:
		if not self.condition.eval():
			prog.push_block(self)
			prog.jump_to(self.pos)


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
		self.env = env
		self.block_stack: list[Block] = []
		self.pending_if_result: bool | None = None  # set by If, read by Then

	# ── Execution loop ────────────────────────────────────────────────────────

	def run(self) -> None:
		"""Execute all statements in the token stream until EOF."""
		self.env.program_stack.append(self)
		try:
			self._parser.run()
		except ReturnSignal:
			pass  # normal sub-program return; stop executing this program
		finally:
			self.env.program_stack.pop()

	# ── Block stack ───────────────────────────────────────────────────────────

	def push_block(self, block: Block) -> None:
		self.block_stack.append(block)

	def pop_block(self) -> Block:
		if not self.block_stack:
			raise TiSyntaxError("End without matching block")
		return self.block_stack.pop()

	def peek_block(self) -> Block | None:
		return self.block_stack[-1] if self.block_stack else None

	def jump_to(self, pos: int) -> None:
		"""Set the parser's execution position (used by loop blocks on_end)."""
		self._parser.pos = pos

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
				# Use parse_label_name to read the 1–2 name-char tokens after LBL
				p.pos = i + 1
				label = p.parse_label_name()
				if label == name:
					return  # p.pos now points past the label name — correct resume point
				i = p.pos  # skip over whatever name chars were consumed
				continue
			i += 1
		raise LabelError(f"Label not found: {name!r}")

	# ── Control-flow commands ─────────────────────────────────────────────────
	# Each method contains the logic that was previously scattered across the
	# command functions in forms.py.  The command functions now just parse
	# their arguments and delegate here.

	def begin_if(self, cond: bool) -> None:
		"""Handle If: peek ahead for Then; otherwise skip the next statement if False."""
		p = self._parser
		saved = p.pos
		p.eat_statement_sep()
		next_tok = p.peek()
		p.pos = saved
		from tokens import THEN
		if next_tok is THEN:
			self.pending_if_result = cond
		elif not cond:
			p.eat_statement_sep()
			p.skip_statement()

	def begin_then(self) -> None:
		"""Handle Then: open a Then-block (True branch) or skip to Else / End (False branch)."""
		result = self.pending_if_result
		self.pending_if_result = None
		if result is None:
			raise TiSyntaxError("Then without If")
		if result:
			self.push_block(ThenBlock())
		else:
			from tokens import ELSE
			found = self._parser.skip_block(also_stop_at_else=True)
			if found is ELSE:
				self.push_block(ThenBlock())  # execute else-body; End will pop this block

	def begin_else(self) -> None:
		"""Handle Else: pop the Then-block and skip to the matching End."""
		block = self.pop_block()
		if not isinstance(block, ThenBlock):
			raise TiSyntaxError("Else without matching Then")
		self._parser.skip_block()

	def begin_while(self, condition: 'Thunk') -> None:
		"""Handle While: evaluate condition; enter loop body or skip to End."""
		if condition.eval():
			self.push_block(WhileBlock(pos=self._parser.pos, condition=condition))
		else:
			self._parser.skip_block()

	def begin_repeat(self, condition: 'Thunk') -> None:
		"""Handle Repeat: always enter the body; End will check the condition."""
		self.push_block(RepeatBlock(pos=self._parser.pos, condition=condition))

	def begin_for(self, var: Variable, start: float, end_val: float, step: float) -> None:
		"""Handle For(: initialise the loop variable; enter body or skip to End."""
		var.set(self.env, start)
		if check_for_condition(start, end_val, step):
			self.push_block(ForBlock(pos=self._parser.pos, var=var, end_val=end_val, step=step))
		else:
			self._parser.skip_block()

	def end_block(self) -> None:
		"""Handle End: dispatch to the innermost block's on_end."""
		self.pop_block().on_end(self)

	def is_gt(self, var: Variable, threshold: float) -> None:
		"""Handle IS>(: increment var; skip the next statement if var > threshold."""
		new_val = require_real(var.get(self.env)) + 1
		var.set(self.env, new_val)
		if new_val > threshold:
			self._parser.eat_statement_sep()
			self._parser.skip_statement()

	def ds_lt(self, var: Variable, threshold: float) -> None:
		"""Handle DS<(: decrement var; skip the next statement if var < threshold."""
		new_val = require_real(var.get(self.env)) - 1
		var.set(self.env, new_val)
		if new_val < threshold:
			self._parser.eat_statement_sep()
			self._parser.skip_statement()

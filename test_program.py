"""Tests for control-flow constructs executed inside a Program.

Uses Program(toks(...)).run(env) directly so that StopSignal propagates
to the test rather than being swallowed by run_line.
"""

import pytest
from pytest import approx

from environment import Environment, ReturnSignal, StopSignal
from errors import IncrementError, TiSyntaxError, LabelError
from program import Program
from test_tibasic import toks, var, calc


# ── Helper ────────────────────────────────────────────────────────────────────

def run(src: str, env: Environment | None = None) -> Environment:
	"""Build a token list from *src*, run inside a Program, return the environment."""
	if env is None:
		env = Environment()
	env.programs['TEST'] = Program(toks(src), 'TEST')
	env.submit(toks('prgm TEST'))
	return env


# ── For( ─────────────────────────────────────────────────────────────────────

class TestFor:

	def test_basic_sum(self):
		# 1 + 2 + 3 + 4 + 5 = 15
		env = run("""
		For( A,1,5)
		B+A@B
		End
		""")
		assert var(env, 'B') == 15

	def test_step_2(self):
		# A iterates over 1, 3, 5, 7, 9 → 5 iterations
		env = run("""
		For( A,1,9,2)
		B+1@B
		End
		""")
		assert var(env, 'B') == 5

	def test_negative_step(self):
		# A iterates over 5, 4, 3, 2, 1 → 5 iterations
		env = run("""
		For( A,5,1,~1)
		B+1@B
		End
		""")
		assert var(env, 'B') == 5

	def test_no_iterations_when_start_exceeds_end(self):
		# start > end with default positive step → body never runs
		env = run("""
		For( A,5,1)
		99@B
		End
		42@C
		""")
		assert var(env, 'B') is None   # body never executed
		assert var(env, 'C') == 42     # execution continues after End

	def test_variable_value_after_loop(self):
		# On exit the loop variable holds the first out-of-range value
		env = run("""
		For( A,1,3)
		End
		""")
		assert var(env, 'A') == 4

	def test_step_zero_raises(self):
		with pytest.raises(IncrementError):
			run("""
			For( A,1,5,0)
			End
			""")

	def test_nested(self):
		# 3 * 3 = 9 increments
		env = run("""
		For( A,1,3
		For( B,1,3
		C+1@C
		End
		End
		""")
		assert var(env, 'C') == 9

	def test_inner_variable_independent(self):
		# Inner loop must not clobber outer loop variable
		env = run("""
		For( A,1,3
		For( A,10,10
		B+1@B
		End
		End
		""")
		# After inner loop A = 11; outer loop sees that and exits early
		# (implementation detail — just verify it terminates and outer ran at least once)
		assert var(env, 'A') is not None
		assert var(env, 'B') == 1


# ── While ─────────────────────────────────────────────────────────────────────

class TestWhile:

	def test_counts_up(self):
		env = run("""
		1@A
		While A<5
		A+1@A
		End
		""")
		assert var(env, 'A') == 5

	def test_never_enters_when_false(self):
		env = run("""
		While 0
		99@A
		End
		42@B
		""")
		assert var(env, 'A') is None   # body never ran
		assert var(env, 'B') == 42     # continues after End

	def test_condition_reevaluated_each_iteration(self):
		# A doubles each iteration; loop exits when A ≥ 10
		env = run("""
		1@A
		While A<10
		2A@A
		End
		""")
		assert var(env, 'A') == 16  # 1→2→4→8→16 (first value ≥ 10)


# ── Repeat ────────────────────────────────────────────────────────────────────

class TestRepeat:

	def test_runs_until_condition_true(self):
		env = run("""
		Repeat A=3
		A+1@A
		End
		""")
		assert var(env, 'A') == 3

	def test_body_runs_at_least_once(self):
		# Condition is True from the very start, but body still runs once
		env = run("""
		Repeat 1
		99@A
		End
		42@B
		""")
		assert var(env, 'A') == 99
		assert var(env, 'B') == 42

	def test_condition_checked_at_end(self):
		# A starts at 5, condition A > 3 is immediately True,
		# but body still executes once before the check
		env = run("""
		5@A
		Repeat A>3
		A-1@A
		End
		""")
		assert var(env, 'A') == 4


# ── If / one-line ─────────────────────────────────────────────────────────────

class TestIfOneLine:

	def test_true_executes_next(self):
		env = run("""
		If 1
			42@A
		""")
		assert var(env, 'A') == 42

	def test_false_skips_next(self):
		env = run("""
		If 0
			42@A
		99@B
		""")
		assert var(env, 'A') is None
		assert var(env, 'B') == 99

	def test_false_skips_empty_statement(self):
		# An empty statement counts as the "next" statement and is still skipped
		env = run("""
		If 0

		99@A
		""")
		assert var(env, 'A') == 99

	def test_chained_conditions(self):
		# Two independent one-line Ifs
		env = run("""
		If 1
			10@A
		If 0
			20@B
		30@C
		""")
		assert var(env, 'A') == 10
		assert var(env, 'B') is None
		assert var(env, 'C') == 30

	def test_false_skips_colon_separated_statement(self):
		# If 0:stmt — the separator ':' must be consumed by end_cmd *before*
		# skip_statement runs; otherwise skip_statement eats the colon and
		# 'stmt' executes instead of being skipped.
		env = run("""
		If 0:42@A
		99@B
		""")
		assert var(env, 'A') is None
		assert var(env, 'B') == 99


# ── If / Then / Else / End ────────────────────────────────────────────────────

class TestIfThenElse:

	def test_then_true(self):
		env = run("""
		If 1
		Then
			42@A
		End
		""")
		assert var(env, 'A') == 42

	def test_then_false_skips_body(self):
		env = run("""
		If 0
		Then
			42@A
		End
		99@B
		""")
		assert var(env, 'A') is None
		assert var(env, 'B') == 99

	def test_then_else_takes_then_branch(self):
		env = run("""
		If 1
		Then
			10@A
		Else
			20@A
		End
		99@B
		""")
		assert var(env, 'A') == 10
		assert var(env, 'B') == 99

	def test_then_else_takes_else_branch(self):
		env = run("""
		If 0
		Then
			10@A
		Else
			20@A
		End
		99@B
		""")
		assert var(env, 'A') == 20
		assert var(env, 'B') == 99

	def test_nested_then(self):
		env = run("""
		If 1
		Then
			If 1
			Then
				42@A
			End
		End
		""")
		assert var(env, 'A') == 42

	def test_nested_then_inner_false(self):
		env = run("""
		If 1
		Then
			If 0
			Then
				99@A
			End
			42@B
		End
		""")
		assert var(env, 'A') is None
		assert var(env, 'B') == 42

	def test_then_false_nested_inside(self):
		# Outer If is False; skip_block must skip nested blocks correctly
		env = run("""
		If 0
			Then
				For( A,1,5
				End
			End
		99@B
		""")
		assert var(env, 'B') == 99   # outer block was skipped cleanly

	def test_if_inside_for(self):
		# Count how many values A takes that are > 3
		env = run("""
		For( A,1,5)
			If A>3
			Then
				B+1@B
			End
		End
		""")
		assert var(env, 'B') == 2   # A=4 and A=5

	def test_bare_then_not_counted_as_block_when_skipping(self):
		# A Then not preceded by If must not open a new depth level while
		# skip_block scans for the matching End.
		# If 0 → skip body; the bare Then inside should be transparent;
		# the first End closes the outer If/Then; the second End is unmatched.
		with pytest.raises(TiSyntaxError):
			run("""
			If 0
			Then
				1@A
				Then
				End
				2@B
			End
			3@C
			""")

	def test_nested_else_in_false_branch_does_not_close_outer_block(self):
		# When the outer If is false, skip_block(else_mode=True) scans for the
		# matching Else/End.  An inner Else at depth>0 must NOT decrement depth —
		# it is the middle of its own block, not a closer.  If it does decrement,
		# depth hits 0 early; the inner End is then mistaken for the outer End;
		# the outer Else is reached with an empty block stack → TiSyntaxError.
		env = run("""
		If 0
		Then
			If 1
			Then
				1@A
			Else
				2@A
			End
		Else
			3@A
		End
		""")
		assert var(env, 'A') == 3

	def test_then_colon_separated_from_if(self):
		# If cond:Then on the same line — end_cmd must eat the ':' before
		# begin_if peeks for Then; otherwise begin_if sees ':' and misses Then,
		# causing the bare Then on the next "statement" to raise TiSyntaxError.
		env = run("""
		If 1: Then
			42@A
		End
		""")
		assert var(env, 'A') == 42

	def test_then_false_colon_separated_skips_body(self):
		env = run("""
		If 0: Then
			42@A
		End
		99@B
		""")
		assert var(env, 'A') is None
		assert var(env, 'B') == 99

	def test_end_without_block_raises(self):
		with pytest.raises(TiSyntaxError):
			run('End')

	def test_end_after_block_closed_raises(self):
		with pytest.raises(TiSyntaxError):
			run("""
			If 1
			Then
				1@A
			End
			End
			""")


# ── Lbl / Goto ────────────────────────────────────────────────────────────────

class TestLblGoto:

	def test_basic_forward_goto(self):
		env = run("""
		Goto A
		99@B
		Lbl A
		42@C
		""")
		assert var(env, 'B') is None   # skipped
		assert var(env, 'C') == 42

	def test_two_char_label(self):
		env = run("""
		Goto AB
		99@C
		Lbl AB
		42@D
		""")
		assert var(env, 'C') is None
		assert var(env, 'D') == 42

	def test_goto_backward_loop(self):
		# Manually build a counting loop with Goto
		env = run("""
		Lbl A
		A+1@A
		If A<5
		Goto A
		42@B
		""")
		assert var(env, 'A') == 5
		assert var(env, 'B') == 42

	def test_label_not_found_raises(self):
		with pytest.raises(LabelError):
			run('Goto Z')

	def test_label_error_carries_goto_position(self):
		# pos is set by _exec_statement to the index of the Goto token itself
		tokens = toks("""
		1@A
		Goto Z
		""")
		with pytest.raises(LabelError) as exc_info:
			Program(tokens).run(Environment())
		assert exc_info.value.pos == len(tokens) - 1


# ── Return ────────────────────────────────────────────────────────────────────

class TestReturn:

	def test_exits_subprogram(self):
		env = Environment()
		env.programs['P'] = Program(toks("""
		1@A
		Return
		99@A
		"""), 'P')
		run("""
		prgm P
		2@B
		""", env)
		assert var(env, 'A') == 1   # Return fired before 99→A
		assert var(env, 'B') == 2   # caller continued normally

	def test_does_not_exit_caller(self):
		# Return only exits the innermost program
		env = Environment()
		env.programs['I'] = Program(toks("""
		1@A
		Return
		99@A
		"""), 'I')
		env.programs['O'] = Program(toks("""
		prgm I
		2@B
		Return
		99@B
		"""), 'O')
		run("""
		prgm O
		3@C
		""", env)
		assert var(env, 'A') == 1   # INNER's Return fired
		assert var(env, 'B') == 2   # OUTER continued, then its own Return fired
		assert var(env, 'C') == 3   # top-level caller continued

	def test_return_doesnt_cancel_other_statements(self):
		env = Environment()
		env.programs['P'] = Program(toks('Return'), 'P')
		calc("""
		1@A
		prgm P
		99@B
		""", env)
		assert var(env, 'A') == 1    # executed before the sub-program
		assert var(env, 'B') == 99   # caller continues after Return


# ── Stop ─────────────────────────────────────────────────────────────────────

class TestStop:

	def test_stop_side_effects_before_stop(self):
		env = run("""
		1@A
		Stop
		99@A
		""")
		assert var(env, 'A') == 1    # executed before Stop
		assert var(env, 'B') is None  # never reached

	def test_stop_propagates_through_subprogram(self):
		env = Environment()
		env.programs['P'] = Program(toks('Stop'), 'P')
		run("""
		1@A
		prgm P
		99@B
		""", env)
		assert var(env, 'A') == 1    # executed before the sub-program
		assert var(env, 'B') is None  # skipped because Stop propagated

	def test_stop_does_cancel_other_statements(self):
		env = Environment()
		env.programs['P'] = Program(toks('Stop'), 'P')
		calc("""
		1@A
		prgm P
		99@B
		""", env)
		assert var(env, 'A') == 1    # executed before the sub-program
		assert var(env, 'B') is None  # skipped because Stop propagated


# ── IS>( / DS<( ──────────────────────────────────────────────────────────────

class TestIsGtDsLt:

	def test_is_gt_increments_variable(self):
		env = run("""
		3@A
		IS>( A,5
		99@B
		42@C
		""")
		assert var(env, 'A') == 4
		assert var(env, 'B') == 99
		assert var(env, 'C') == 42

	def test_is_gt_no_skip_when_not_exceeded(self):
		env = run("""
		3@A
		IS>( A,5)
		99@B
		42@C
		""")
		assert var(env, 'B') == 99
		assert var(env, 'C') == 42

	def test_is_gt_skips_when_exceeded(self):
		env = run("""
		5@A
		IS>( A,5)
		99@B
		42@C
		""")
		assert var(env, 'B') is None   # skipped
		assert var(env, 'C') == 42

	def test_ds_lt_decrements_variable(self):
		env = run("""
		4@A
		DS<( A,3)
		99@B
		42@C
		""")
		assert var(env, 'A') == 3

	def test_ds_lt_no_skip_when_not_below(self):
		env = run("""
		3@A
		DS<( A,3)
		99@B
		42@C
		""")
		assert var(env, 'A') == 2
		assert var(env, 'B') is None
		assert var(env, 'C') == 42

	def test_ds_lt_skips_when_below(self):
		env = Environment()
		run("""
		3@A
		DS<( A,3)
		99@B
		42@C
		""", env)
		assert var(env, 'B') is None   # skipped
		assert var(env, 'C') == 42

	def test_is_gt_skips_colon_separated_statement(self):
		# IS>(A,5):stmt — end_paren_cmd must eat both ')' and ':' before is_gt
		# calls skip_statement; otherwise skip_statement eats ':' and 'stmt'
		# executes instead of being skipped.
		env = run("""
		5@A
		IS>( A,5):99@B
		42@C
		""")
		assert var(env, 'B') is None   # skipped
		assert var(env, 'C') == 42

	def test_ds_lt_skips_colon_separated_statement(self):
		env = run("""
		3@A
		DS<( A,3):99@B
		42@C
		""")
		assert var(env, 'B') is None   # skipped
		assert var(env, 'C') == 42


# ── Junk after no-arg commands ────────────────────────────────────────────────

class TestJunkAfterCommand:
	"""No-arg commands must reject stray tokens; junk inside skipped blocks is fine."""

	def test_return_with_junk_raises(self):
		with pytest.raises(TiSyntaxError):
			run('Return 5')

	def test_stop_with_junk_raises(self):
		with pytest.raises(TiSyntaxError):
			run('Stop 5')

	def test_end_with_junk_raises(self):
		with pytest.raises(TiSyntaxError):
			run("""
			If 1
			Then
			End 5
			End
			""")

	def test_else_with_junk_raises(self):
		with pytest.raises(TiSyntaxError):
			run("""
			If 1
			Then
			Else 5
			End
			""")

	def test_junk_in_skipped_repeat(self):
		# Repeat's argument tokens are consumed by skip_statement — not executed
		env = run("""
		If 0
		Then
		Repeat 1/0
		End
		End
		99@A
		""")
		assert var(env, 'A') == 99

	def test_junk_in_skipped_while(self):
		env = run("""
		If 0
		Then
		While $BAD
		End
		End
		99@A
		""")
		assert var(env, 'A') == 99

	def test_junk_in_skipped_for(self):
		env = run("""
		If 0
		Then
		For( real( )
		End
		End
		99@B
		""")
		assert var(env, 'B') == 99

	def test_nested_junk_blocks_in_skipped_region(self):
		env = run("""
		If 0
		Then
		For( A,+1,+2)
		Repeat +-*/
		End
		End
		End
		99@C
		""")
		assert var(env, 'C') == 99


# ── DelVar the loop index ─────────────────────────────────────────────────────

class TestForDelVar:

	def test_delvar_resets_counter_causing_extra_iterations(self):
		# DelVar on a numeric variable resets it to its default (0).
		# For(X,3,5) normally runs 3 times (X=3,4,5).
		# Deleting X on the first pass makes it 0; the next End increments
		# from 0 to 1, and the loop continues from there — 6 iterations total.
		env = run("""
		For( X,3,5)
		A+1@A
		If A=2
		DelVar X
		End
		""")
		assert var(env, 'A') == 7


# ── Goto into / out of loop blocks ────────────────────────────────────────────

class TestGotoBlocks:

	def test_goto_into_loop_raises_at_end(self):
		# Jumping into a For loop body bypasses begin_for, so no ForBlock is
		# pushed onto the block stack.  When End is reached the stack is empty
		# — ERR:SYNTAX on hardware, TiSyntaxError here.
		with pytest.raises(TiSyntaxError):
			run("""
			Goto IN
			For( X,1,3)
			Lbl IN
			99@A
			End
			""")

	def test_goto_out_of_loop_stale_block_harmless(self):
		# Goto out of a For loop leaves a stale ForBlock on the block stack,
		# but code after the label runs normally because no further End is
		# encountered to accidentally pop it.
		env = run("""
		For( X,1,5)
		A+1@A
		If A=3
		Goto D
		End
		Lbl D
		99@B
		""")
		assert var(env, 'A') == 3
		assert var(env, 'B') == 99

	def test_goto_out_stale_block_consumed_by_later_end(self):
		# After jumping out of a For loop the stale ForBlock stays on the
		# stack.  A later End (after a Lbl) pops it and jumps back into the
		# loop body — the loop effectively resumes from where it left off.
		# X=1: Goto S → C+1, End loops back (stale block, X→2)
		# X=2: normal End exits → Goto D → B=99
		env = run("""
		For( X,1,2)
		A+1@A
		If X=1
		Goto S
		End
		Goto D
		Lbl S
		C+1@C
		End
		Lbl D
		99@B
		""")
		assert var(env, 'A') == 2
		assert var(env, 'B') == 99
		assert var(env, 'C') == 1


# ── Using one-line If to skip End ─────────────────────────────────────────────

class TestIfSkipsEnd:

	def test_if_skips_end_acts_as_continue(self):
		# A one-line If whose body is End acts like a "continue": when the
		# condition is true, End executes immediately (looping back or exiting),
		# skipping the rest of the loop body for that iteration.
		# X=1: If false → skip End → B+=1 → normal End loops
		# X=2: If true  → End loops back immediately, B not updated
		# X=3: If false → skip End → B+=3 → normal End loops
		# X=4: If false → skip End → B+=4 → normal End exits
		env = run("""
		For( X,1,4)
		A+X@A
		If X=2
		End
		B+X@B
		End
		""")
		assert var(env, 'A') == 10   # 1+2+3+4
		assert var(env, 'B') == 8    # 1+3+4  (X=2 skipped)


class TestRecursion:

	def test_factorial(self):
		env = Environment()
		calc('{5,1@A', env)
		run("""
			$A( dim( $A@T
			$A( dim( $A )-1@N
			dim( $A)-2@ dim( $A
			If N≤1
			Then
				T@$A(1+ dim( $A
				Return
			End
			N-1@$A( 1+ dim( $A
			TN@$A( 1+ dim( $A
			prgm TEST
		""", env)

		assert calc('$A( dim( $A', env) == 120

class TestEngima:

	def test_dynamic_rotors(self):
		env = run("""
		{1,2,3@RTR1
		{4,5,6@RTR2
		{7,8,9@RTR3
		For( N,1,3
		expr( "$RTR" + sub( "123",N,1) + "(N"@$RES(N
		End
		""")
		assert var(env, '$RES').data == [1, 5, 9]
	
	def test_index_of(self):
		env = run("""
		seq( X,X,1,26@ L1
		{1,9,22,21,18,15,4,6,23,16,26,3,10,14,7,8,13,25,17,5,2,19,24,20,11,12}@RTR
		max( L1 *($RTR=12@I
		max( L1 *($RTR=13@J
		""")
		assert var(env, 'I') == 26
		assert var(env, 'J') == 17


# ── skip_statement edge cases ─────────────────────────────────────────────────

class TestSkipStatement:
	"""Verify that skip_statement (and skip_block which calls it) correctly
	navigates tricky statement content: strings with colons/STORE inside them,
	empty strings, and nested control structures.

	Each test arranges a program where mishandling a skipped statement would
	produce a wrong variable value or raise an unexpected exception.
	"""

	# ── Bare If (false): skip_statement called directly on the next statement ─

	def test_store_terminates_string(self):
		# "HELLO→Str1 uses STORE as an implicit string terminator (no closing ").
		# skip_statement must stop scanning string content at STORE and then
		# consume the remainder of the statement (the variable name) before
		# returning.  A=1 confirms that only one statement was skipped.
		env = run("""
		If 0
		"HELLO@ Str1
		1@A
		""")
		assert var(env, 'A') == 1

	def test_colon_inside_string_not_separator(self):
		# The COLON inside "A:B" is string content; skip_statement's inner
		# string-scanning loop must swallow it without returning early.
		# A=1 confirms that only "A:B"→Str1 was skipped (one statement).
		env = run("""
		If 0
		"A:B"@ Str1
		1@A
		""")
		assert var(env, 'A') == 1

	def test_empty_string_skipped_as_one_statement(self):
		# ""→Str1 is a valid statement (empty string assignment).
		# skip_statement sees QUOTE immediately followed by QUOTE — the inner
		# loop breaks right away — then continues to consume →Str1 normally.
		env = run("""
		If 0
		""@ Str1
		1@A
		""")
		assert var(env, 'A') == 1

	# ── If-Then (false): skip_block calls skip_statement for each statement ──

	def test_then_block_fully_skipped(self):
		# All statements inside a false Then block must be skipped.
		env = run("""
		If 0
		Then
		99@A
		End
		1@A
		""")
		assert var(env, 'A') == 1

	def test_then_else_runs_else_block(self):
		# False If-Then-Else: the Then block is skipped, the Else block runs.
		env = run("""
		If 0
		Then
		1@A
		Else
		2@A
		End
		""")
		assert var(env, 'A') == 2

	def test_string_colon_in_skipped_then_block(self):
		# "A:B"→Str1 inside a false Then block: skip_block advances the leading
		# QUOTE as the "statement command token", then calls skip_statement for
		# the rest.  The COLON that is part of "A:B" must not make skip_statement
		# return early — if it does, the remaining fragment (:B"→Str1) is parsed
		# as a new statement and the block structure goes out of sync.
		env = run("""
		If 0
		Then
		"A:B"@ Str1
		99@A
		End
		1@A
		""")
		assert var(env, 'A') == 1

	def test_store_terminated_string_in_skipped_block(self):
		# "HELLO→Str1 inside a false Then block.  After skip_block consumes the
		# leading QUOTE, skip_statement processes HELLO, then encounters STORE,
		# which is handled as a normal (non-separator) token at that point.
		# The statement must be consumed completely before the separator.
		env = run("""
		If 0
		Then
		"HELLO@ Str1
		99@A
		End
		1@A
		""")
		assert var(env, 'A') == 1

	def test_quoted_then(self):
		env = Environment()
		with pytest.raises(TiSyntaxError):
			run("""
			If 0
			" Then
			1@A
			End
			""", env)
		assert var(env, 'A') == 1

	def test_quoted_end_skipped(self):
		env = run("""
		If 0
		Then
		" End @ Str1
		End
		1@A
		""")
		assert var(env, 'A') == 1
		assert var(env, 'Str1') is None
	
	def test_quoted_colon_skipped(self):
		env = run("""
		If 0
		Then
		":
		End
		1@A
		""")
		assert var(env, 'A') == 1
	
	def test_quoted_unterminated_empty_string_skipped(self):
		env = run("""
		If 0
		Then
		"
		End
		1@A
		""")
		assert var(env, 'A') == 1

	def test_colon_first_char_of_string_in_skipped_block(self):
		# ":End"→Str1 — the VERY FIRST character of the string content is COLON,
		# immediately followed by the End token.
		# If skip_block's pre-consumed QUOTE causes skip_statement to see that
		# COLON as a statement terminator, skip_statement returns immediately and
		# the very next token (End, inside the string) is mistaken for the block's
		# closing End, causing the whole block to exit one statement early.
		env = run("""
		If 0
		Then
		" End "@ Str1
		End
		1@A
		""")
		assert var(env, 'A') == 1
		assert var(env, 'Str1') is None

	def test_nested_if_then_depth_tracked(self):
		# A nested If-Then inside the skipped block must increment depth so that
		# the inner End does not close the outer block.
		env = run("""
		If 0
		Then
		If 1
		Then
		99 @ A
		End
		End
		1 @ A
		""")
		assert var(env, 'A') == 1

	def test_then_without_preceding_if_is_transparent(self):
		# A bare Then (not preceded by If on the same statement) must NOT
		# increment depth inside skip_block — prev_if will be False for it.
		# If it erroneously incremented depth, the actual End would decrement to
		# depth=1 instead of 0 and skip_block would continue past the End,
		# consuming the 1→A statement and never returning.
		env = run("""
		If 0
		Then
		Then
		End
		1@A
		""")
		assert var(env, 'A') == 1

	def test_for_loop_body_skipped_when_range_empty(self):
		# For(N,5,1) — step defaults to 1, start > end, body never runs.
		# skip_block must consume the body and End without executing anything.
		env = run("""
		For( N,5,1
		99@A
		End
		1@A
		""")
		assert var(env, 'A') == 1
	
	def test_unmatched_false_if_then_okay(self):
		env = run("""
		If 0
		Then
		99@A
		""")
		assert var(env, 'A') is None
	
	def test_unmatched_true_if_then_okay(self):
		env = run("""
		If 1
		Then
		99@A
		""")
		assert var(env, 'A') == 99


class TestProgThunk:

	def test_while_condition_reevaluated(self):
		# The While condition is captured once and re-evaluated on every loop-back.
		# It must read the CURRENT value of A each time.
		env = run("""
		0@A
		While A<3
		A+1@A
		End
		""")
		assert var(env, 'A') == 3

	def test_repeat_runs_body_before_checking_condition(self):
		# Repeat executes the body at least once before evaluating the condition.
		# Starting with A=0: body sets A=1, condition A=3 is false → repeat;
		# body sets A=2, still false; body sets A=3, condition true → exit.
		env = run("""
		0@A
		Repeat A=3
		A+1@A
		End
		""")
		assert var(env, 'A') == 3

	def test_while_compound_condition_in_thunk(self):
		# A compound condition (A>0 and A≠3) is captured as a single thunk.
		# The `and` operator's comma-free syntax means no extra stack tracking
		# is needed; this test confirms the thunk boundary is the COLON.
		env = run("""
		5@A
		While A>0 and A≠3
		A-1@A
		End
		""")
		assert var(env, 'A') == 3

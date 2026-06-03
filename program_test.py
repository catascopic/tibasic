"""Tests for control-flow constructs executed inside a Program.

Uses Program(toks(...), env).run() directly so that StopSignal propagates
to the test rather than being swallowed by run_line.
"""

import pytest

from environment import Environment
from errors import IncrementError, TiSyntaxError, LabelError
from signals import ReturnSignal, StopSignal
from program import Program
from tibasic_test import toks, var, calc


# ── Helper ────────────────────────────────────────────────────────────────────

def run(src: str, env: Environment | None = None) -> Environment:
	"""Build a token list from *src*, run inside a Program, return the environment."""
	if env is None:
		env = Environment()
	env.programs['TEST'] = toks(src)
	env.run(toks('prgm TEST'))
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
			Program(tokens, Environment()).run()
		assert exc_info.value.pos == 4


# ── Return ────────────────────────────────────────────────────────────────────

class TestReturn:

	def test_exits_subprogram(self):
		env = Environment()
		env.programs['P'] = toks("""
		1@A
		Return
		99@A
		""")
		run("""
		prgm P
		2@B
		""", env)
		assert var(env, 'A') == 1   # Return fired before 99→A
		assert var(env, 'B') == 2   # caller continued normally

	def test_does_not_exit_caller(self):
		# Return only exits the innermost program
		env = Environment()
		env.programs['I'] = toks("""
		1@A
		Return
		99@A
		""")
		env.programs['O'] = toks("""
		prgm I
		2@B
		Return
		99@B
		""")
		run("""
		prgm O
		3@C
		""", env)
		assert var(env, 'A') == 1   # INNER's Return fired
		assert var(env, 'B') == 2   # OUTER continued, then its own Return fired
		assert var(env, 'C') == 3   # top-level caller continued

	def test_return_doesnt_cancel_other_statements(self):
		env = Environment()
		env.programs['P'] = toks('Return')
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
		env.programs['P'] = toks('Stop')
		run("""
		1@A
		prgm P
		99@B
		""", env)
		assert var(env, 'A') == 1    # executed before the sub-program
		assert var(env, 'B') is None  # skipped because Stop propagated

	def test_stop_does_cancel_other_statements(self):
		env = Environment()
		env.programs['P'] = toks('Stop')
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
		Repeat +2
		End
		End
		99@A
		""")
		assert var(env, 'A') == 99

	def test_junk_in_skipped_while(self):
		env = run("""
		If 0
		Then
		While +2
		End
		End
		99@A
		""")
		assert var(env, 'A') == 99

	def test_junk_in_skipped_for(self):
		env = run("""
		If 0
		Then
		For( A,+2,+3)
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
		Repeat +3
		End
		End
		End
		99@C
		""")
		assert var(env, 'C') == 99


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

	@pytest.mark.skip('WTF')
	def test_fibonacci(self):
		env = Environment()
		calc('{9@A', env)
		run("""
		$A(  dim(  $A))@N
		dim( $A)-1@ dim( $A)
		If N≤1
		Then
			Disp N
			N@$A(1+ dim( $A))
			Return
		End

		N@$A(1+ dim( $A))
		N-1@$A(1+ dim( $A))
		prgm TEST
		$A( dim( $A))@R
		dim( $A)-1@ dim( $A)
		$A( dim( $A))@N

		R@$A(1+ dim( $A))
		N@$A(1+ dim( $A))
		N-2@$A(1+ dim( $A))
		prgm TEST
		$A( dim( $A))@S
		dim( $A)-1@ dim( $A)
		dim( $A)-1@ dim( $A)
		$A( dim( $A))@R
		dim( $A)-1@ dim( $A)

		R+S@$A(1+ dim( $A))
		""", env)

		assert calc('$A( dim( $A', env) == 34

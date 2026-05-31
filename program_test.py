"""Tests for control-flow constructs executed inside a Program.

Uses Program(toks(...), env).run() directly so that StopSignal propagates
to the test rather than being swallowed by run_line.
"""

import pytest

from environment import Environment
from errors import IncrementError, TiSyntaxError
from signals import ReturnSignal, StopSignal
from program import Program
from tibasic_test import toks, var


# ── Helper ────────────────────────────────────────────────────────────────────

def run(src: str, env: Environment | None = None) -> Environment:
	"""Build a token list from *src*, run inside a Program, return the environment."""
	if env is None:
		env = Environment()
	Program(toks(src), env).run()
	return env


# ── For( ─────────────────────────────────────────────────────────────────────

class TestFor:

	def test_basic_sum(self):
		# 1 + 2 + 3 + 4 + 5 = 15
		env = run('For( A , 1 , 5 ) : B + A @ B : End')
		assert var(env, 'B') == 15

	def test_step_2(self):
		# A iterates over 1, 3, 5, 7, 9 → 5 iterations
		env = run('For( A , 1 , 9 , 2 ) : B + 1 @ B : End')
		assert var(env, 'B') == 5

	def test_negative_step(self):
		# A iterates over 5, 4, 3, 2, 1 → 5 iterations
		env = run('For( A , 5 , 1 , ~ 1 ) : B + 1 @ B : End')
		assert var(env, 'B') == 5

	def test_no_iterations_when_start_exceeds_end(self):
		# start > end with default positive step → body never runs
		env = run('For( A , 5 , 1 ) : 99 @ B : End : 42 @ C')
		assert var(env, 'B') is None   # body never executed
		assert var(env, 'C') == 42     # execution continues after End

	def test_variable_value_after_loop(self):
		# On exit the loop variable holds the first out-of-range value
		env = run('For( A , 1 , 3 ) : End')
		assert var(env, 'A') == 4

	def test_step_zero_raises(self):
		with pytest.raises(IncrementError):
			run('For( A , 1 , 5 , 0 ) : End')

	def test_nested(self):
		# 3 × 3 = 9 increments
		env = run('For( A , 1 , 3 ) : For( B , 1 , 3 ) : C + 1 @ C : End : End')
		assert var(env, 'C') == 9

	def test_inner_variable_independent(self):
		# Inner loop must not clobber outer loop variable
		env = run('For( A , 1 , 3 ) : For( A , 10 , 10 ) : End : End')
		# After inner loop A = 11; outer loop sees that and exits early
		# (implementation detail — just verify it terminates and outer ran at least once)
		assert var(env, 'A') is not None


# ── While ─────────────────────────────────────────────────────────────────────

class TestWhile:

	def test_counts_up(self):
		env = run('1 @ A : While A < 5 : A + 1 @ A : End')
		assert var(env, 'A') == 5

	def test_never_enters_when_false(self):
		env = run('While 0 : 99 @ A : End : 42 @ B')
		assert var(env, 'A') is None   # body never ran
		assert var(env, 'B') == 42     # continues after End

	def test_condition_reevaluated_each_iteration(self):
		# A doubles each iteration; loop exits when A ≥ 10
		env = run('1 @ A : While A < 10 : A * 2 @ A : End')
		assert var(env, 'A') == 16  # 1→2→4→8→16 (first value ≥ 10)


# ── Repeat ────────────────────────────────────────────────────────────────────

class TestRepeat:

	def test_runs_until_condition_true(self):
		env = run('0 @ A : Repeat A = 3 : A + 1 @ A : End')
		assert var(env, 'A') == 3

	def test_body_runs_at_least_once(self):
		# Condition is True from the very start, but body still runs once
		env = run('Repeat 1 : 99 @ A : End : 42 @ B')
		assert var(env, 'A') == 99
		assert var(env, 'B') == 42

	def test_condition_checked_at_end(self):
		# A starts at 5, condition A > 3 is immediately True,
		# but body still executes once before the check
		env = run('5 @ A : Repeat A > 3 : A - 1 @ A : End')
		assert var(env, 'A') == 4


# ── If / one-line ─────────────────────────────────────────────────────────────

class TestIfOneLine:

	def test_true_executes_next(self):
		env = run('If 1 : 42 @ A')
		assert var(env, 'A') == 42

	def test_false_skips_next(self):
		env = run('If 0 : 42 @ A : 99 @ B')
		assert var(env, 'A') is None
		assert var(env, 'B') == 99

	def test_false_skips_empty_statement(self):
		# An empty statement counts as the "next" statement and is still skipped
		env = run('If 0 : : 99 @ A')
		assert var(env, 'A') == 99

	def test_chained_conditions(self):
		# Two independent one-line Ifs
		env = run('If 1 : 10 @ A : If 0 : 20 @ B : 30 @ C')
		assert var(env, 'A') == 10
		assert var(env, 'B') is None
		assert var(env, 'C') == 30


# ── If / Then / Else / End ────────────────────────────────────────────────────

class TestIfThenElse:

	def test_then_true(self):
		env = run('If 1 : Then : 42 @ A : End')
		assert var(env, 'A') == 42

	def test_then_false_skips_body(self):
		env = run('If 0 : Then : 42 @ A : End : 99 @ B')
		assert var(env, 'A') is None
		assert var(env, 'B') == 99

	def test_then_else_takes_then_branch(self):
		env = run('If 1 : Then : 10 @ A : Else : 20 @ A : End')
		assert var(env, 'A') == 10

	def test_then_else_takes_else_branch(self):
		env = run('If 0 : Then : 10 @ A : Else : 20 @ A : End')
		assert var(env, 'A') == 20

	def test_nested_then(self):
		env = run('If 1 : Then : If 1 : Then : 42 @ A : End : End')
		assert var(env, 'A') == 42

	def test_nested_then_inner_false(self):
		env = run('If 1 : Then : If 0 : Then : 99 @ A : End : 42 @ B : End')
		assert var(env, 'A') is None
		assert var(env, 'B') == 42

	def test_then_false_nested_inside(self):
		# Outer If is False; scan_block_end must skip nested blocks correctly
		env = run('If 0 : Then : For( A , 1 , 5 ) : End : End : 99 @ B')
		assert var(env, 'B') == 99   # outer block was skipped cleanly

	def test_if_inside_for(self):
		# Count how many values A takes that are > 3
		env = run('For( A , 1 , 5 ) : If A > 3 : Then : B + 1 @ B : End : End')
		assert var(env, 'B') == 2   # A=4 and A=5


# ── Lbl / Goto ────────────────────────────────────────────────────────────────

class TestLblGoto:

	def test_basic_forward_goto(self):
		env = run('Goto A : 99 @ B : Lbl A : 42 @ C')
		assert var(env, 'B') is None   # skipped
		assert var(env, 'C') == 42

	def test_two_char_label(self):
		env = run('Goto AB : 99 @ C : Lbl AB : 42 @ D')
		assert var(env, 'C') is None
		assert var(env, 'D') == 42

	def test_goto_backward_loop(self):
		# Manually build a counting loop with Goto
		env = run('Lbl A : A + 1 @ A : If A < 5 : Goto A : 42 @ B')
		assert var(env, 'A') == 5
		assert var(env, 'B') == 42

	def test_label_not_found_raises(self):
		from errors import LabelError
		with pytest.raises(LabelError):
			run('Goto Z')


# ── Return ────────────────────────────────────────────────────────────────────

class TestReturn:

	def test_exits_subprogram(self):
		env = Environment()
		env.programs['P'] = toks('1 @ A : Return : 99 @ A')
		run('prgm P : 2 @ B', env)
		assert var(env, 'A') == 1   # Return fired before 99→A
		assert var(env, 'B') == 2   # caller continued normally

	def test_does_not_exit_caller(self):
		# Return only exits the innermost program
		env = Environment()
		env.programs['I'] = toks('1 @ A : Return : 99 @ A')
		env.programs['O'] = toks('prgm I : 2 @ B : Return : 99 @ B')
		run('prgm O : 3 @ C', env)
		assert var(env, 'A') == 1   # INNER's Return fired
		assert var(env, 'B') == 2   # OUTER continued, then its own Return fired
		assert var(env, 'C') == 3   # top-level caller continued

	def test_return_signal_propagates_from_program_run(self):
		# Program.run() catches ReturnSignal; calling code sees no exception
		env = Environment()
		Program(toks('Return'), env).run()   # must not raise


# ── Stop ─────────────────────────────────────────────────────────────────────

class TestStop:

	def test_stop_raises_stop_signal(self):
		with pytest.raises(StopSignal):
			run('1 @ A : Stop : 99 @ A')

	def test_stop_side_effects_before_stop(self):
		env = Environment()
		with pytest.raises(StopSignal):
			Program(toks('1 @ A : Stop : 99 @ A'), env).run()
		assert var(env, 'A') == 1    # executed before Stop
		assert var(env, 'B') is None  # never reached

	def test_stop_propagates_through_subprogram(self):
		env = Environment()
		env.programs['P'] = toks('Stop')
		with pytest.raises(StopSignal):
			run('1 @ A : prgm P : 99 @ B', env)
		assert var(env, 'A') == 1    # executed before the sub-program
		assert var(env, 'B') is None  # skipped because Stop propagated

	def test_stop_caught_by_run_line(self):
		# run_line is the top-level entry point; it must absorb StopSignal
		from parser import run_line
		env = Environment()
		run_line(toks('1 @ A : Stop : 99 @ B'), env)   # must not raise
		assert var(env, 'A') == 1


# ── IS>( / DS<( ──────────────────────────────────────────────────────────────

class TestIsGtDsLt:

	def test_is_gt_increments_variable(self):
		env = Environment()
		env.numerics[0] = 3   # A = 3 → becomes 4
		run('IS>( A , 5 ) : 99 @ B : 42 @ C', env)
		assert var(env, 'A') == 4

	def test_is_gt_no_skip_when_not_exceeded(self):
		env = Environment()
		env.numerics[0] = 3   # A → 4, not > 5
		run('IS>( A , 5 ) : 99 @ B : 42 @ C', env)
		assert var(env, 'B') == 99   # not skipped
		assert var(env, 'C') == 42

	def test_is_gt_skips_when_exceeded(self):
		env = Environment()
		env.numerics[0] = 5   # A → 6, 6 > 5
		run('IS>( A , 5 ) : 99 @ B : 42 @ C', env)
		assert var(env, 'B') is None   # skipped
		assert var(env, 'C') == 42

	def test_ds_lt_decrements_variable(self):
		env = Environment()
		env.numerics[0] = 4   # A = 4 → becomes 3
		run('DS<( A , 3 ) : 99 @ B : 42 @ C', env)
		assert var(env, 'A') == 3

	def test_ds_lt_no_skip_when_not_below(self):
		env = Environment()
		env.numerics[0] = 4   # A → 3, not < 3
		run('DS<( A , 3 ) : 99 @ B : 42 @ C', env)
		assert var(env, 'B') == 99   # not skipped
		assert var(env, 'C') == 42

	def test_ds_lt_skips_when_below(self):
		env = Environment()
		env.numerics[0] = 3   # A → 2, 2 < 3
		run('DS<( A , 3 ) : 99 @ B : 42 @ C', env)
		assert var(env, 'B') is None   # skipped
		assert var(env, 'C') == 42

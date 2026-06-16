"""Tests for the home screen model and the I/O commands (Disp, Output(, ClrHome)."""
import pytest

from homescreen import HomeScreen
from console import ScriptedConsole
from errors import DomainError, DataTypeError, TiSyntaxError, InvalidCommandError
from environment import Environment
from program import Program
from test_tibasic import run, toks, var


def run_with(src: str, inputs):
	"""Run src with a ScriptedConsole feeding `inputs` to Input/Prompt."""
	env = Environment()
	env.console = ScriptedConsole(inputs=inputs)
	env.run(toks(src))
	return env


def run_program(src: str, *, choices=(), inputs=()):
	"""Run src as a program (so control-flow commands like Menu( have a context)."""
	env = Environment()
	env.console = ScriptedConsole(inputs=inputs, choices=choices)
	Program(toks(src), env).run()
	return env


# ── HomeScreen model ──────────────────────────────────────────────────────────

class TestHomeScreen:
	def test_blank_grid_shape(self):
		lines = HomeScreen().render().split('\n')
		assert len(lines) == 8
		assert all(line == ' ' * 16 for line in lines)

	def test_output_writes_at_position(self):
		h = HomeScreen()
		h.output(2, 3, 'AB')
		assert h.render().split('\n')[2] == '   AB' + ' ' * 11

	def test_output_wraps_to_next_row(self):
		h = HomeScreen()
		h.output(0, 14, 'ABCD')        # 14,15 on row 0; 0,1 on row 1
		rows = h.render().split('\n')
		assert rows[0].endswith('AB')
		assert rows[1].startswith('CD')

	def test_output_clips_at_bottom(self):
		h = HomeScreen()
		h.output(7, 14, 'ABCDEF')      # only AB fit; CDEF fall off the bottom
		assert h.render().split('\n')[7].endswith('AB')

	def test_disp_appends_and_advances(self):
		h = HomeScreen()
		h.disp('one')
		h.disp('two')
		rows = h.render().split('\n')
		assert rows[0].startswith('one') and rows[1].startswith('two')
		assert h.cursor_row == 2

	def test_disp_clips_to_width(self):
		h = HomeScreen()
		h.disp('X' * 20)
		assert h.render().split('\n')[0] == 'X' * 16

	def test_disp_scrolls_past_bottom(self):
		h = HomeScreen()
		for i in range(9):
			h.disp(str(i))
		rows = h.render().split('\n')
		assert rows[0].startswith('1')   # '0' scrolled off the top
		assert rows[7].startswith('8')

	def test_clear_resets_grid_and_cursor(self):
		h = HomeScreen()
		h.disp('stuff')
		h.clear()
		assert h.render() == '\n'.join([' ' * 16] * 8)
		assert h.cursor_row == 0


# ── Disp / Output( / ClrHome through the interpreter ──────────────────────────

def _line(env, n):
	return env.home.render().split('\n')[n]


class TestDisp:
	def test_number(self):
		assert _line(run('Disp 5'), 0).startswith('5')

	def test_string(self):
		assert _line(run('Disp "HELLO'), 0) == 'HELLO' + ' ' * 11

	def test_multiple_values_stack(self):
		env = run('Disp 1 : Disp 2')
		assert _line(env, 0).startswith('1') and _line(env, 1).startswith('2')

	def test_complex_not_yet_supported(self):
		with pytest.raises(DataTypeError): run('Disp i')

	def test_each_disp_renders_a_frame(self):
		env = run('Disp 1 : Disp 2')
		assert len(env.console.frames) == 2


class TestOutput:
	def test_positions_value(self):
		assert _line(run('Output( 2,3,42'), 1) == '  42' + ' ' * 12

	def test_one_indexed_top_left(self):
		assert _line(run('Output( 1,1,9'), 0).startswith('9')

	def test_row_out_of_range_raises(self):
		with pytest.raises(DomainError): run('Output( 9,1,5')

	def test_col_out_of_range_raises(self):
		with pytest.raises(DomainError): run('Output( 1,17,5')


class TestClrHome:
	def test_clears_and_resets(self):
		env = run('Disp 7')
		run('ClrHome', env)
		assert env.home.render() == '\n'.join([' ' * 16] * 8)
		assert env.home.cursor_row == 0


class TestPause:
	def test_bare_pause_blocks_and_renders(self):
		env = run_program('Pause')
		assert env.console.frames == [env.home.render()]

	def test_value_displayed_like_disp(self):
		env = run_program('Pause 5')
		assert env.home.render().split('\n')[0].startswith('5')

	def test_value_stored_to_ans(self):
		assert run_program('Pause 5').ans == 5

	def test_string_value_stored_to_ans(self):
		env = run_program('Pause "HI')
		assert str(env.ans) == 'HI'

	def test_bare_pause_does_not_touch_ans(self):
		env = Environment()
		env.console = ScriptedConsole()
		env.ans = 99
		Program(toks('Pause'), env).run()
		assert env.ans == 99

	def test_outside_program_raises(self):
		with pytest.raises(InvalidCommandError): run('Pause')

	def test_complex_value_raises(self):
		with pytest.raises(DataTypeError): run_program('Pause i')


_MENU_PROG = """Menu( "PICK","ONE",A,"TWO",B
Lbl A
1@X
Return
Lbl B
2@X"""


class TestMenu:
	def test_routes_to_first_option(self):
		assert var(run_program(_MENU_PROG, choices=[0]), 'X') == 1

	def test_routes_to_second_option(self):
		assert var(run_program(_MENU_PROG, choices=[1]), 'X') == 2

	def test_outside_program_raises(self):
		with pytest.raises(InvalidCommandError):
			run('Menu( "T","O",A')

	def test_non_string_option_raises(self):
		with pytest.raises(DataTypeError):
			run_program('Menu( 5,"O",A\nLbl A', choices=[0])


class TestInput:
	def test_reads_number_into_var(self):
		assert var(run_with('Input X', ['5']), 'X') == 5

	def test_with_prompt(self):
		assert var(run_with('Input "AGE?",A', ['30']), 'A') == 30

	def test_evaluates_expression(self):
		assert var(run_with('Input X', ['2+3']), 'X') == 5

	def test_reads_list(self):
		assert var(run_with('Input L1', ['{1,2,3}']), 'L1').data == [1, 2, 3]

	def test_prompt_passed_to_console(self):
		env = Environment()
		env.console = ScriptedConsole(inputs=['1'])
		env.run(toks('Input "PICK",X'))
		# (ScriptedConsole ignores the prompt, but the command must not choke on it)
		assert var(env, 'X') == 1

	def test_unsupported_character_raises(self):
		with pytest.raises(TiSyntaxError):
			run_with('Input X', ['sin'])

	def test_graph_form_not_supported(self):
		with pytest.raises(TiSyntaxError):
			run_with('Input', [])


# ── ScriptedConsole ───────────────────────────────────────────────────────────

class TestScriptedConsole:
	def test_captures_frames(self):
		c = ScriptedConsole()
		h = HomeScreen()
		h.disp('hi')
		c.update(h)
		assert c.frames == [h.render()]

	def test_read_key_empty_is_zero(self):
		assert ScriptedConsole().read_key() == 0

	def test_read_key_drains_queue(self):
		c = ScriptedConsole(keys=[25, 34])
		assert (c.read_key(), c.read_key(), c.read_key()) == (25, 34, 0)

	def test_read_value_without_input_errors(self):
		with pytest.raises(RuntimeError):
			ScriptedConsole().read_value('?')

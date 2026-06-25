"""Tests for the home screen model and the I/O commands (Disp, Output(, ClrHome)."""
import re
import pytest

from homescreen import HomeScreen
from terminal import ScriptedConsole, TerminalConsole
from errors import DomainError, DataTypeError, TiSyntaxError, InvalidCommandError
from environment import Environment
from program import Program
from test_tibasic import run, toks, var


def run_with(src: str, inputs):
	"""Run src with a ScriptedConsole feeding `inputs` to Input/Prompt."""
	env = Environment()
	env.console = ScriptedConsole(inputs=inputs)
	env.submit(toks(src))
	return env


def run_program(src: str, *, choices=(), inputs=()):
	"""Run src as a program (so control-flow commands like Menu( have a context)."""
	env = Environment()
	env.console = ScriptedConsole(inputs=inputs, choices=choices)
	Program(toks(src)).run(env)
	return env


# ── HomeScreen model ──────────────────────────────────────────────────────────

class TestHomeScreen:
	def test_blank_grid_shape(self):
		lines = HomeScreen().render().split('\n')
		assert len(lines) == 8
		assert all(line == ' ' * 16 for line in lines)

	def test_output_writes_at_position(self):
		h = HomeScreen()
		h.output(2, 3, b'AB')
		assert h.render().split('\n')[2] == '   AB' + ' ' * 11

	def test_output_wraps_to_next_row(self):
		h = HomeScreen()
		h.output(0, 14, b'ABCD')        # 14,15 on row 0; 0,1 on row 1
		rows = h.render().split('\n')
		assert rows[0].endswith('AB')
		assert rows[1].startswith('CD')

	def test_output_clips_at_bottom(self):
		h = HomeScreen()
		h.output(7, 14, b'ABCDEF')      # only AB fit; CDEF fall off the bottom
		assert h.render().split('\n')[7].endswith('AB')

	def test_disp_appends_and_advances(self):
		h = HomeScreen()
		h.disp(b'one')
		h.disp(b'two')
		rows = h.render().split('\n')
		assert rows[0].startswith('one') and rows[1].startswith('two')
		assert h.cursor_row == 2

	def test_disp_truncates_with_ellipsis(self):
		h = HomeScreen()
		h.disp(b'X' * 20)
		assert h.render().split('\n')[0] == 'X' * 15 + '…'

	def test_disp_exact_width_no_ellipsis(self):
		h = HomeScreen()
		h.disp(b'X' * 16)
		assert h.render().split('\n')[0] == 'X' * 16

	def test_output_wraps_long_text_across_rows(self):
		h = HomeScreen()
		h.output(0, 0, b'X' * 20)
		rows = h.render().split('\n')
		assert rows[0] == 'X' * 16
		assert rows[1].startswith('X' * 4)

	def test_echo_wraps_instead_of_truncating(self):
		h = HomeScreen()
		h.echo(b'X' * 20)
		rows = h.render().split('\n')
		assert rows[0] == 'X' * 16
		assert rows[1].startswith('X' * 4)
		assert h.cursor_row == 2

	def test_echo_empty_still_advances_cursor(self):
		h = HomeScreen()
		h.echo(b'')
		assert h.cursor_row == 1

	def test_echo_exact_width_advances_one_row(self):
		h = HomeScreen()
		h.echo(b'X' * 32)   # exactly 2 full rows
		assert h.cursor_row == 2

	def test_echo_may_fill_bottom_row_unlike_disp(self):
		# Input/Prompt are allowed to leave the bottom row filled — only Disp
		# guarantees a trailing blank line.
		h = HomeScreen()
		for i in range(7):
			h.disp(str(i).encode())
		h.echo(b'LAST')
		assert h.render().split('\n')[7].startswith('LAST')

	def test_disp_scrolls_past_bottom(self):
		# Disp guarantees a blank trailing line, so filling the bottom row
		# triggers an extra scroll beyond just "make room" — '8' ends up on the
		# second-to-last row, with the last row blank, not holding '8' itself.
		h = HomeScreen()
		for i in range(9):
			h.disp(str(i).encode())
		rows = h.render().split('\n')
		assert rows[0].startswith('2')   # '0' and '1' both scrolled off
		assert rows[6].startswith('8')
		assert rows[7] == ' ' * 16

	def test_disp_always_leaves_a_blank_bottom_line(self):
		h = HomeScreen()
		for i in range(20):       # many more than fit; screen scrolls repeatedly
			h.disp(str(i).encode())
		assert h.render().split('\n')[7] == ' ' * 16
		assert h.cursor_row == 7

	def test_clear_resets_grid_and_cursor(self):
		h = HomeScreen()
		h.disp(b'stuff')
		h.clear()
		assert h.render() == '\n'.join([' ' * 16] * 8)
		assert h.cursor_row == 0


# ── Disp / Output( / ClrHome through the interpreter ──────────────────────────

def _line(env, n):
	return env.home.render().split('\n')[n]


class TestDisp:
	def test_number_right_aligned(self):
		assert _line(run('Disp 5'), 0) == '5'.rjust(16)

	def test_string_left_aligned(self):
		assert _line(run('Disp "HELLO'), 0) == 'HELLO' + ' ' * 11

	def test_multiple_values_stack(self):
		env = run('Disp 1 : Disp 2')
		assert _line(env, 0) == '1'.rjust(16) and _line(env, 1) == '2'.rjust(16)

	def test_complex_right_aligned(self):
		assert _line(run('Disp 3+4i'), 0) == '3+4i'.rjust(16)

	def test_pure_imaginary_drops_zero_real_part(self):
		assert _line(run('Disp i'), 0) == '1i'.rjust(16)

	def test_list_space_separated_right_aligned(self):
		assert _line(run('Disp {1,2,3}'), 0) == '{1 2 3}'.rjust(16)

	def test_matrix_one_line_per_row(self):
		env = run('Disp [[1,2][3,4]]')
		# Block right-aligned: both rows indented equally (16 - 7) so columns/brackets line up.
		assert _line(env, 0).rstrip() == ' ' * 9 + '[[1 2]'
		assert _line(env, 1)          == ' ' * 9 + ' [3 4]]'

	def test_matrix_columns_left_aligned_to_common_width(self):
		# Columns are padded to their widest entry, left-justified, ignoring magnitude.
		env = run('Disp [[1,22][333,4]]')
		assert _line(env, 0).rstrip() == ' ' * 6 + '[[1   22]'
		assert _line(env, 1)          == ' ' * 6 + ' [333 4 ]]'

	def test_each_disp_renders_a_frame(self):
		env = run('Disp 1 : Disp 2')
		assert len(env.console.frames) == 2


class TestOutput:
	def test_positions_value(self):
		assert _line(run('Output( 2,3,42'), 1) == '  42' + ' ' * 12

	def test_one_indexed_top_left(self):
		assert _line(run('Output( 1,1,9'), 0).startswith('9')

	def test_list_comma_separated_not_aligned(self):
		# Output renders a list the way you'd type it: commas, no padding.
		assert _line(run('Output( 1,1,{1,2,3}'), 0).startswith('{1,2,3}')

	def test_matrix_inline_comma_separated(self):
		# A matrix is written linearly with comma separators and no spaces.
		assert _line(run('Output( 1,1,[[1,2][3,4]]'), 0).startswith('[[1,2][3,4]]')

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
		assert env.home.render().split('\n')[0] == '5'.rjust(16)

	def test_value_stored_to_ans(self):
		assert run_program('Pause 5').ans == 5

	def test_zero_value_is_shown_and_stored(self):
		# Guards against treating a falsy-but-present value (0) as "no argument".
		env = run_program('Pause 0')
		assert env.ans == 0
		assert env.home.render().split('\n')[0] == '0'.rjust(16)

	def test_string_value_stored_to_ans(self):
		env = run_program('Pause "HI')
		assert str(env.ans) == 'HI'

	def test_bare_pause_does_not_touch_ans(self):
		env = Environment()
		env.console = ScriptedConsole()
		env.ans = 99
		Program(toks('Pause')).run(env)
		assert env.ans == 99

	def test_outside_program_raises(self):
		with pytest.raises(InvalidCommandError): run('Pause')

	def test_complex_displayed_like_disp(self):
		# Pause shows any value Disp can — including complex, right-aligned.
		env = run_program('Pause 3+4i')
		assert env.home.render().split('\n')[0] == '3+4i'.rjust(16)

	def test_list_displayed_like_disp(self):
		env = run_program('Pause {1,2,3}')
		assert env.home.render().split('\n')[0] == '{1 2 3}'.rjust(16)


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


class TestTerminalMenuRendering:
	"""Unit tests for TerminalConsole's menu-screen rendering/key logic, which
	don't need msvcrt or a real terminal — pure data in, data out."""

	def _console(self):
		c = TerminalConsole.__new__(TerminalConsole)
		c._last_home = None
		c._last_run_frame = -1
		return c

	@staticmethod
	def _plain(row):
		return re.sub(r'\x1b\[\d+m', '', row)

	def test_eight_rows_of_sixteen_visible_chars(self):
		rows = self._console()._menu_rows('T', ['A', 'B'], selected=0)
		assert len(rows) == HomeScreen.ROWS
		assert all(len(self._plain(r)) == HomeScreen.COLS for r in rows)

	def test_title_row_is_inverted(self):
		row = self._console()._menu_rows('PICK', ['A'], selected=0)[0]
		# Only "PICK" itself is inverted — the trailing padding to fill the row
		# is plain, not part of the inverted span.
		assert row == '\033[7mPICK\033[27m' + ' ' * 12
		assert self._plain(row) == 'PICK' + ' ' * 12

	def test_selected_prefix_inverted_others_not(self):
		rows = self._console()._menu_rows('T', ['ONE', 'TWO', 'THREE'], selected=1)
		assert '\033[7m' not in rows[1] and self._plain(rows[1]) == '1:ONE' + ' ' * 11
		assert rows[2].startswith('\033[7m2:\033[27m')
		assert '\033[7m' not in rows[3]

	def test_long_option_truncated_to_fit(self):
		row = self._console()._menu_rows('T', ['X' * 30], selected=0)[1]
		assert self._plain(row) == '1:' + 'X' * 14   # 16 - len('1:')

	def test_blank_rows_pad_to_eight(self):
		rows = self._console()._menu_rows('T', ['A'], selected=0)
		assert rows[2:] == [' ' * HomeScreen.COLS] * 6

	def test_poll_recognizes_up_down_enter_digit(self):
		c = self._console()
		assert c._poll_menu_key(_FakeMsvcrt(['\x00', 'H']), 3) == 'up'
		assert c._poll_menu_key(_FakeMsvcrt(['\x00', 'P']), 3) == 'down'
		assert c._poll_menu_key(_FakeMsvcrt(['\r']), 3) == 'enter'
		assert c._poll_menu_key(_FakeMsvcrt(['2']), 3) == 1

	def test_poll_ignores_digit_beyond_option_count(self):
		c = self._console()
		assert c._poll_menu_key(_FakeMsvcrt(['9']), 3) is None

	def test_poll_returns_none_when_idle(self):
		c = self._console()
		assert c._poll_menu_key(_FakeMsvcrt([]), 3) is None


class _FakeMsvcrt:
	"""A one-shot fake: kbhit() is True exactly once per queued event, then
	False forever — enough to drive a single _poll_menu_key call deterministically."""

	def __init__(self, events):
		self.events = list(events)

	def kbhit(self):
		return bool(self.events)

	def getwch(self):
		return self.events.pop(0)


class TestInput:
	def test_reads_number_into_var(self):
		assert var(run_with('Input X', ['5']), 'X') == 5

	def test_with_prompt(self):
		assert var(run_with('Input "AGE?",A', ['30']), 'A') == 30

	def test_string_var_without_comma_is_target(self):
		# Str1 with no comma → it's the target variable, not a prompt
		env = run_with('Input Str1', ['ABC'])
		assert str(var(env, 'Str1')) == 'ABC'

	def test_string_input_stores_raw_no_quotes_needed(self):
		# Input into a string var: typed text is stored verbatim (no quotes required)
		env = run_with('Input Str1', ['ABC'])
		assert str(var(env, 'Str1')) == 'ABC'

	def test_string_input_includes_typed_quotes(self):
		# If the user types quotes, they become part of the stored string
		env = run_with('Input Str1', ['"ABC"'])
		assert str(var(env, 'Str1')) == '"ABC"'

	def test_evaluates_expression(self):
		assert var(run_with('Input X', ['2+3']), 'X') == 5

	def test_reads_list(self):
		assert var(run_with('Input L1', ['{1,2,3}']), 'L1').data == [1, 2, 3]

	def test_prompt_passed_to_console(self):
		env = Environment()
		env.console = ScriptedConsole(inputs=['1'])
		env.submit(toks('Input "PICK",X'))
		# (ScriptedConsole ignores the prompt, but the command must not choke on it)
		assert var(env, 'X') == 1

	def test_unsupported_character_raises(self):
		with pytest.raises(TiSyntaxError):
			run_with('Input X', ['sin'])

	def test_long_response_wraps_not_truncates(self):
		# 1 (prompt '?') + 21 chars = 22, past one 16-wide row — should wrap
		# across rows like Output(, not get Disp's truncate-with-ellipsis.
		typed = '1+1+1+1+1+1+1+1+1+1+1'
		env = run_with('Input X', [typed])
		lines = env.home.render().split('\n')
		assert lines[0] == ('?' + typed)[:16]
		assert lines[1].startswith(('?' + typed)[16:])
		assert '…' not in lines[0] and '…' not in lines[1]

	def test_graph_form_not_supported(self):
		with pytest.raises(TiSyntaxError):
			run_with('Input', [])


# ── ScriptedConsole ───────────────────────────────────────────────────────────

class TestScriptedConsole:
	def test_captures_frames(self):
		c = ScriptedConsole()
		h = HomeScreen()
		h.disp(b'hi')
		c.update(h)
		assert c.frames == [h.render()]

	def test_read_key_empty_is_zero(self):
		assert ScriptedConsole().read_key() == 0

	def test_read_key_drains_queue(self):
		c = ScriptedConsole(keys=[25, 34])
		assert (c.read_key(), c.read_key(), c.read_key()) == (25, 34, 0)

	def test_read_tokens_without_input_errors(self):
		with pytest.raises(ValueError):
			ScriptedConsole().read_tokens(None, HomeScreen())

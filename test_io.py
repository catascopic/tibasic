"""Tests for the home screen model and the I/O commands (Disp, Output(, ClrHome)."""
import re
import pytest

from homescreen import HomeScreen
from terminal import ScriptedConsole, TerminalConsole, PixelConsole
from errors import DomainError, DataTypeError, TiSyntaxError, InvalidCommandError
from environment import Environment
from program import Program
from core import TiList, TiString
from modes import Screen
from menuscreen import MenuScreen
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
		lines = str(HomeScreen()).split('\n')
		assert len(lines) == 8
		assert all(line == ' ' * 16 for line in lines)

	def test_output_writes_at_position(self):
		h = HomeScreen()
		h.output(2, 3, b'AB')
		assert str(h).split('\n')[2] == '   AB' + ' ' * 11

	def test_output_wraps_to_next_row(self):
		h = HomeScreen()
		h.output(0, 14, b'ABCD')        # 14,15 on row 0; 0,1 on row 1
		rows = str(h).split('\n')
		assert rows[0].endswith('AB')
		assert rows[1].startswith('CD')

	def test_output_clips_at_bottom(self):
		h = HomeScreen()
		h.output(7, 14, b'ABCDEF')      # only AB fit; CDEF fall off the bottom
		assert str(h).split('\n')[7].endswith('AB')

	def test_disp_appends_and_advances(self):
		h = HomeScreen()
		h.write_line(b'one')
		h.write_line(b'two')
		rows = str(h).split('\n')
		assert rows[0].startswith('one') and rows[1].startswith('two')
		assert h.cursor_row == 2

	def test_disp_truncates_with_ellipsis(self):
		h = HomeScreen()
		h.write_line(b'X' * 20)
		assert str(h).split('\n')[0] == 'X' * 15 + '…'

	def test_disp_exact_width_no_ellipsis(self):
		h = HomeScreen()
		h.write_line(b'X' * 16)
		assert str(h).split('\n')[0] == 'X' * 16

	def test_output_wraps_long_text_across_rows(self):
		h = HomeScreen()
		h.output(0, 0, b'X' * 20)
		rows = str(h).split('\n')
		assert rows[0] == 'X' * 16
		assert rows[1].startswith('X' * 4)

	def test_echo_wraps_instead_of_truncating(self):
		h = HomeScreen()
		h.echo(b'X' * 20)
		rows = str(h).split('\n')
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
			h.write_line(str(i).encode())
		h.echo(b'LAST')
		assert str(h).split('\n')[7].startswith('LAST')

	def test_disp_scrolls_past_bottom(self):
		# Disp guarantees a blank trailing line, so filling the bottom row
		# triggers an extra scroll beyond just "make room" — '8' ends up on the
		# second-to-last row, with the last row blank, not holding '8' itself.
		h = HomeScreen()
		for i in range(9):
			h.write_line(str(i).encode())
		rows = str(h).split('\n')
		assert rows[0].startswith('2')   # '0' and '1' both scrolled off
		assert rows[6].startswith('8')
		assert rows[7] == ' ' * 16

	def test_disp_always_leaves_a_blank_bottom_line(self):
		h = HomeScreen()
		for i in range(20):       # many more than fit; screen scrolls repeatedly
			h.write_line(str(i).encode())
		assert str(h).split('\n')[7] == ' ' * 16
		assert h.cursor_row == 7

	def test_clear_resets_grid_and_cursor(self):
		h = HomeScreen()
		h.write_line(b'stuff')
		h.clear()
		assert str(h) == '\n'.join([' ' * 16] * 8)
		assert h.cursor_row == 0

	def test_output_to_bottom_does_not_advance_disp_scroll(self):
		# ClrHome, Output( to the bottom row, then Disp: the screen mustn't scroll
		# until the 8th Disp — Output's positioned write doesn't count toward how
		# many lines Disp has taken up.
		h = HomeScreen()
		h.output(7, 0, b'BOTTOM')           # window row 7 (0-indexed)
		for i in range(7):                  # 7 disps fill rows 0..6, no scroll
			h.write_line(str(i).encode())
		rows = str(h).split('\n')
		assert rows[0].startswith('0')      # nothing scrolled off yet
		assert rows[7].startswith('BOTTOM') # Output's row still showing
		h.write_line(b'7')                  # the 8th disp writes row 7 and scrolls
		rows = str(h).split('\n')
		assert rows[0].startswith('1')      # '0' has now scrolled out of view
		assert rows[6].startswith('7')
		assert rows[7] == ' ' * 16          # blank bottom line restored

	def test_clrhome_preserves_scrollback(self):
		# ClrHome scrolls the screen out of view without dropping it from the buffer.
		h = HomeScreen()
		h.write_line(b'keep')
		h.clear()
		assert str(h).strip() == ''     # window blank
		assert h.lines and bytes(h.lines[0]).startswith(b'keep')   # still in scrollback


# ── Disp / Output( / ClrHome through the interpreter ──────────────────────────

def _line(env, n):
	return str(env.home).split('\n')[n]


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


class TestHomeScreenLogs:
	"""The host-only Disp `values` log (actual TiValues) and the byte scrollback —
	what lets the model serve a full-width free-form view and keep history."""

	def test_disp_records_actual_values(self):
		env = run('Disp 5 : Disp {1,2,3} : Disp "HI')
		vals = env.home.values
		assert len(vals) == 3 and vals[0] == 5
		assert isinstance(vals[1], TiList) and vals[1].data == [1, 2, 3]
		assert isinstance(vals[2], TiString) and str(vals[2]) == 'HI'

	def test_scrollback_retains_lines_that_left_the_window(self):
		env = run('\n'.join(f'Disp {i}' for i in range(12)))
		assert len(env.home.lines) == 12                          # every line kept
		assert str(env.home).split('\n')[0].strip() == '5'    # window has scrolled

	def test_clrhome_scrolls_out_of_view_but_keeps_history(self):
		env = run('Disp 7')
		run('ClrHome', env)
		assert str(env.home).strip() == ''      # window blank
		assert env.home.values[0] == 7              # values log kept
		assert env.home.lines                       # byte scrollback kept

	def test_matrix_disp_is_one_value_two_lines(self):
		env = run('Disp [[1,2][3,4]]')
		assert len(env.home.values) == 1            # one value...
		rows = str(env.home).split('\n')
		assert rows[0].rstrip().endswith('[[1 2]')  # ...spread over two grid lines
		assert rows[1].rstrip().endswith('[3 4]]')


class TestPresent:
	def test_each_drawing_command_presents_a_frame(self):
		# One present() per drawing command — what lets a frontend animate a drawing
		# being built up.  Four Pxl-On calls → four captured graph frames.
		env = Environment()
		env.console = ScriptedConsole()
		Program(toks('For( I,0,3\nPxl-On( I,I\nEnd')).run(env)
		assert len(env.console.frames) == 4


class TestPrintScreen:
	def test_home_writes_a_bmp(self, tmp_path):
		env = run('Disp "HI')
		path = tmp_path / 'home.bmp'
		env.print_screen(str(path))
		assert path.read_bytes()[:2] == b'BM'

	def test_home_and_graph_share_lcd_dimensions(self, tmp_path):
		# Both rasterize to the same 96×64 LCD, so the BMPs are byte-identical in size.
		env = run('Disp "HI')
		home_bmp = tmp_path / 'home.bmp'
		env.print_screen(str(home_bmp))
		env.screen = Screen.GRAPH
		graph_bmp = tmp_path / 'graph.bmp'
		env.print_screen(str(graph_bmp))
		assert len(home_bmp.read_bytes()) == len(graph_bmp.read_bytes())


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
		assert str(env.home) == '\n'.join([' ' * 16] * 8)
		assert env.home.cursor_row == 0


class TestPause:
	def test_bare_pause_blocks_and_renders(self):
		env = run_program('Pause')
		assert env.console.frames == [str(env.home)]

	def test_value_displayed_like_disp(self):
		env = run_program('Pause 5')
		assert str(env.home).split('\n')[0] == '5'.rjust(16)

	def test_value_stored_to_ans(self):
		assert run_program('Pause 5').ans == 5

	def test_zero_value_is_shown_and_stored(self):
		# Guards against treating a falsy-but-present value (0) as "no argument".
		env = run_program('Pause 0')
		assert env.ans == 0
		assert str(env.home).split('\n')[0] == '0'.rjust(16)

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
		assert str(env.home).split('\n')[0] == '3+4i'.rjust(16)

	def test_list_displayed_like_disp(self):
		env = run_program('Pause {1,2,3}')
		assert str(env.home).split('\n')[0] == '{1 2 3}'.rjust(16)


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

	def test_modal_cleared_after_selection(self):
		# The menu is transient: env.menu is set only for the duration of the call.
		env = run_program(_MENU_PROG, choices=[0])
		assert env.menu is None


def _menu(title, *options, selected=0):
	"""Build a MenuScreen from plain strings — the model takes TiStrings (options
	are TiStrings on the real path), so the test spells them out here."""
	m = MenuScreen(TiString.from_str(title), [TiString.from_str(o) for o in options])
	m.selected = selected
	return m


class TestMenuScreen:
	"""The MenuScreen model: canonical layout + navigation, frontend-independent."""

	def test_down_and_up_wrap_around(self):
		m = _menu('T', 'A', 'B', 'C')
		m.down()
		assert m.selected == 1
		m.up(); m.up()                  # 1 → 0 → wraps to 2
		assert m.selected == 2

	def test_choose_jumps_directly(self):
		m = _menu('T', 'A', 'B', 'C')
		m.choose(2)
		assert m.selected == 2

	def test_rows_layout_and_padding(self):
		rows = _menu('PICK', 'ONE', 'TWO').rows()
		assert len(rows) == 8
		assert rows[0] == 'PICK' + ' ' * 12
		assert rows[1] == '1:ONE' + ' ' * 11
		assert rows[2] == '2:TWO' + ' ' * 11
		assert rows[3] == ' ' * 16

	def test_styled_rows_mark_title_and_selection(self):
		# styled_rows segments are display bytes (the model's native form).
		styled = _menu('T', 'A', 'B', selected=1).styled_rows()
		assert styled[0][0] == (b'T', True)         # title inverted
		assert styled[1][0] == (b'1:', False)       # unselected prefix plain
		assert styled[2][0] == (b'2:', True)        # selected prefix inverted

	def test_print_screen_writes_bmp(self, tmp_path):
		path = tmp_path / 'menu.bmp'
		_menu('PICK', 'ONE', 'TWO').print_screen(str(path))
		data = path.read_bytes()
		assert data[:2] == b'BM'


class TestTerminalMenuRendering:
	"""Unit tests for TerminalConsole's menu-screen rendering/key logic, which
	don't need msvcrt or a real terminal — pure data in, data out."""

	def _console(self):
		c = TerminalConsole.__new__(TerminalConsole)
		c._last_run_frame = -1
		return c

	@staticmethod
	def _plain(row):
		return re.sub(r'\x1b\[\d+m', '', row)

	@staticmethod
	def _menu(title, options, selected=0):
		m = MenuScreen(TiString.from_str(title), [TiString.from_str(o) for o in options])
		m.selected = selected
		return m

	def test_eight_rows_of_sixteen_visible_chars(self):
		rows = self._console()._menu_rows(self._menu('T', ['A', 'B']))
		assert len(rows) == HomeScreen.ROWS
		assert all(len(self._plain(r)) == HomeScreen.COLS for r in rows)

	def test_title_row_is_inverted(self):
		row = self._console()._menu_rows(self._menu('PICK', ['A']))[0]
		# Only "PICK" itself is inverted — the trailing padding to fill the row
		# is plain, not part of the inverted span.
		assert row == '\033[7mPICK\033[27m' + ' ' * 12
		assert self._plain(row) == 'PICK' + ' ' * 12

	def test_selected_prefix_inverted_others_not(self):
		rows = self._console()._menu_rows(self._menu('T', ['ONE', 'TWO', 'THREE'], selected=1))
		assert '\033[7m' not in rows[1] and self._plain(rows[1]) == '1:ONE' + ' ' * 11
		assert rows[2].startswith('\033[7m2:\033[27m')
		assert '\033[7m' not in rows[3]

	def test_long_option_truncated_to_fit(self):
		row = self._console()._menu_rows(self._menu('T', ['X' * 30]))[1]
		assert self._plain(row) == '1:' + 'X' * 14   # 16 - len('1:')

	def test_blank_rows_pad_to_eight(self):
		rows = self._console()._menu_rows(self._menu('T', ['A']))
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


class TestScreenRows:
	"""What each console paints for the active screen (_screen_rows) — pure data:
	TerminalConsole is text-first and skips mid-flight graph repaints; PixelConsole
	rasterizes every screen to 32 rows of 96 half-block pixels."""

	@staticmethod
	def _console(cls):
		c = cls.__new__(cls)     # skip __init__: no VT setup / stdout reconfig in tests
		c.env = Environment()
		return c

	def test_terminal_home_is_text(self):
		rows, width = self._console(TerminalConsole)._screen_rows(False)
		assert width == HomeScreen.COLS
		assert len(rows) == 8 and all(len(r) == 16 for r in rows)

	def test_terminal_graph_skipped_until_settled(self):
		c = self._console(TerminalConsole)
		c.env.screen = Screen.GRAPH
		assert c._screen_rows(False) is None            # mid-flight: leave prior frame
		rows, width = c._screen_rows(True)              # Pause/finish: paint it
		assert width == 96 and len(rows) == 32

	def test_pixel_home_is_rasterized(self):
		rows, width = self._console(PixelConsole)._screen_rows(False)
		assert width == 96
		assert len(rows) == 32 and all(len(r) == 96 for r in rows)

	def test_pixel_graph_paints_mid_flight(self):
		c = self._console(PixelConsole)
		c.env.screen = Screen.GRAPH
		assert c._screen_rows(False) is not None        # never suppressed

	def test_pixel_menu_is_rasterized(self):
		c = self._console(PixelConsole)
		c.env.menu = _menu('PICK', 'ONE')
		c.env.screen = Screen.MENU
		rows, width = c._screen_rows(False)
		assert width == 96 and len(rows) == 32

	def test_home_rasterize_draws_glyph_pixels(self):
		env = Environment()
		blank = sum(px for row in env.home.rasterize().buffer for px in row)
		env.home.write_line(b'A')
		drawn = sum(px for row in env.home.rasterize().buffer for px in row)
		assert blank == 0 and drawn > 0                 # the glyph set some pixels


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

	def test_list_into_numeric_creates_user_list(self):
		# A list entered for a numeric variable lands in ∟<name>, not the numeric var
		env = run_with('Input A', ['{1,2,3}'])
		assert var(env, '$A').data == [1, 2, 3]
		assert var(env, 'A') is None

	def test_prompt_passed_to_console(self):
		env = Environment()
		env.console = ScriptedConsole(inputs=['1'])
		env.submit(toks('Input "PICK",X'))
		# (ScriptedConsole ignores the prompt, but the command must not choke on it)
		assert var(env, 'X') == 1

	def test_garbage_input_is_syntax_error(self):
		# 'sin' tokenizes fine (lowercase letters are typeable) but doesn't parse as an
		# expression — a genuine ERR:SYNTAX from the parser.
		with pytest.raises(TiSyntaxError):
			run_with('Input X', ['sin'])

	def test_untypeable_character_is_host_error(self):
		# A character the keypad can't produce (a tab) can't happen on real hardware; it
		# only arises because a frontend accepts free-form text, so the console reports a
		# plain ValueError rather than ERR:SYNTAX.
		with pytest.raises(ValueError):
			run_with('Input X', ['\t'])

	def test_long_response_wraps_not_truncates(self):
		# 1 (prompt '?') + 21 chars = 22, past one 16-wide row — should wrap
		# across rows like Output(, not get Disp's truncate-with-ellipsis.
		typed = '1+1+1+1+1+1+1+1+1+1+1'
		env = run_with('Input X', [typed])
		lines = str(env.home).split('\n')
		assert lines[0] == ('?' + typed)[:16]
		assert lines[1].startswith(('?' + typed)[16:])
		assert '…' not in lines[0] and '…' not in lines[1]

	def test_graph_form_not_supported(self):
		with pytest.raises(TiSyntaxError):
			run_with('Input', [])

	def test_constant_target_is_rejected(self):
		# π has an accessor but isn't an assignable variable
		with pytest.raises(TiSyntaxError):
			run_with('Input π', ['1'])

	# ── Prompt grammar ──────────────────────────────────────────────────────────
	# The display string is a restricted string expression — a literal, a string
	# variable, Ans, or sub(, joined with +.  When the first argument is a string
	# variable and a comma follows, it's the prompt and the next variable is the
	# target; with no comma the lone string variable is the target.

	def test_string_var_prompt_with_second_target(self):
		# Input Str1,Str2 → Str1 is the prompt, the response goes into Str2
		env = run_with('"HI" @ Str1\nInput Str1 , Str2', ['XYZ'])
		assert str(var(env, 'Str2')) == 'XYZ'
		assert str(var(env, 'Str1')) == 'HI'   # unchanged — it was the prompt

	def test_sub_prompt(self):
		assert var(run_with('Input sub( "AB" ,1,1 ) ,X', ['5']), 'X') == 5

	def test_concatenated_prompt(self):
		assert var(run_with('Input "A" + "B" ,X', ['7']), 'X') == 7

	def test_string_var_concatenated_prompt(self):
		assert var(run_with('"GO" @ Str1\nInput Str1 + "B" ,X', ['9']), 'X') == 9

	def test_nested_sub_prompt(self):
		assert var(run_with('Input sub( sub( "ABC" ,1,2 ) ,1,1 ) ,X', ['3']), 'X') == 3

	# Only the *first* token is checked.  A clock function (getDtStr(/getTmStr() can't
	# begin a prompt — so as the first argument it's read as the target and rejected —
	# but once a valid first token commits to "prompt", the rest is parsed normally, so
	# a clock function later in the expression is simply evaluated (the real calculator
	# rejects it; we deliberately don't, since only the first token is gated).

	def test_clock_function_as_first_arg_is_syntax_error(self):
		with pytest.raises(TiSyntaxError):
			run_with('Input getDtStr( 1 ) ,X', [])

	def test_get_tm_str_as_first_arg_is_syntax_error(self):
		with pytest.raises(TiSyntaxError):
			run_with('Input getTmStr( 1 ) ,X', [])

	def test_clock_function_later_in_prompt_is_allowed(self):
		# Allowed here (unlike hardware): the first token is a literal, so the rest —
		# including getDtStr( — is just evaluated as an ordinary string expression.
		assert var(run_with('Input "A" + getDtStr( 1 ) ,X', ['5']), 'X') == 5

	def test_parenthesized_first_arg_is_syntax_error(self):
		# A leading '(' isn't a prompt-starter, so it's read as the target → not a var
		with pytest.raises(TiSyntaxError):
			run_with('Input ( "A" ) ,X', [])

	def test_number_concatenated_to_prompt_is_data_type_error(self):
		# "A"+5 starts with a valid token, parses normally, then fails at evaluation
		with pytest.raises(DataTypeError):
			run_with('Input "A" + 5 ,X', [])


class TestPrompt:
	def test_stores_value_to_var(self):
		assert var(run_with('Prompt A', ['5']), 'A') == 5

	def test_multiple_vars(self):
		env = run_with('Prompt A,B', ['3', '7'])
		assert var(env, 'A') == 3 and var(env, 'B') == 7

	def test_prompt_echoes_name(self):
		# The home screen echo shows NAME=?<typed> after a Prompt
		env = run_with('Prompt A', ['1'])
		assert str(env.home).split('\n')[0].startswith('A=?')

	def test_user_list_prompt_strips_prefix(self):
		# Prompt ʟNAME should echo NAME=?, not ʟNAME=? (per calculator behavior)
		env = run_with('Prompt ʟNAME', ['{1}'])
		first = str(env.home).split('\n')[0]
		assert first.startswith('NAME=?')
		assert 'ʟ' not in first

	def test_bare_user_list_name_is_rejected(self):
		# Prompt requires the ʟ prefix for user lists; bare multi-char names are an error
		with pytest.raises(Exception):
			run_with('Prompt ABC', ['{1}'])

	def test_list_into_numeric_creates_user_list(self):
		# A list entered for a numeric variable is stored to ∟<name> instead
		env = run_with('Prompt A', ['{1,2,3}'])
		assert var(env, '$A').data == [1, 2, 3]
		assert var(env, 'A') is None    # the numeric A was never written


# ── ScriptedConsole ───────────────────────────────────────────────────────────

class TestScriptedConsole:
	def test_captures_frames(self):
		env = Environment()                 # default console is a ScriptedConsole
		env.home.write_line(b'hi')
		env.console.present()
		assert env.console.frames == [str(env.home)]

	def test_read_key_empty_is_zero(self):
		assert ScriptedConsole().read_key() == 0

	def test_read_key_drains_queue(self):
		c = ScriptedConsole(keys=[25, 34])
		assert (c.read_key(), c.read_key(), c.read_key()) == (25, 34, 0)

	def test_read_tokens_without_input_errors(self):
		with pytest.raises(ValueError):
			ScriptedConsole().read_tokens(None)

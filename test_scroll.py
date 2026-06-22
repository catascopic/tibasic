"""Scrolling a large list/matrix paused on the home screen (ScrollView)."""
import pytest

from core import TiList, TiMatrix
from homescreen import HomeScreen
from environment import Environment
from terminal import ScriptedConsole, TerminalConsole
from program import Program
from scrollview import ScrollView
from test_tibasic import toks

UP, DOWN, ELL = '↑', '↓', '…'
RIGHT = HomeScreen.COLS - 1


def matrix(rows, cols):
	return TiMatrix([[float(r * cols + c) for c in range(cols)] for r in range(rows)])


def grid(value, *, down=0, right=0):
	"""Render `value` after scrolling `down`/`right` steps; return (view, lines)."""
	sv = ScrollView(value)
	for _ in range(down):
		sv.move('down')
	for _ in range(right):
		sv.move('right')
	home = HomeScreen()
	sv.render_into(home)
	return sv, home.render().split('\n')


# ── Scrollability ─────────────────────────────────────────────────────────────

class TestScrollable:
	def test_small_matrix_not_scrollable(self):
		assert not ScrollView(matrix(2, 2)).scrollable

	def test_short_list_not_scrollable(self):
		assert not ScrollView(TiList([1.0, 2.0, 3.0])).scrollable

	def test_tall_matrix_scrollable(self):
		assert ScrollView(matrix(10, 2)).scrollable

	def test_wide_matrix_scrollable(self):
		assert ScrollView(matrix(2, 12)).scrollable

	def test_long_list_scrollable(self):
		assert ScrollView(TiList([float(i) for i in range(40)])).scrollable


# ── Offsets and clamping ──────────────────────────────────────────────────────

class TestMovement:
	def test_starts_top_left(self):
		sv = ScrollView(matrix(10, 8))
		assert (sv.row_off, sv.col_off) == (0, 0)
		assert not sv.can_up and not sv.can_left

	def test_down_clamps_at_bottom(self):
		sv = ScrollView(matrix(10, 2))      # 10 rows, viewport shows 7
		for _ in range(50):
			sv.move('down')
		assert sv.row_off == 10 - ScrollView.VIEW_ROWS    # == 3
		assert not sv.can_down

	def test_up_clamps_at_top(self):
		sv = ScrollView(matrix(10, 2))
		for _ in range(50):
			sv.move('down')
		for _ in range(50):
			sv.move('up')
		assert sv.row_off == 0 and not sv.can_up

	def test_right_clamps_at_edge(self):
		sv = ScrollView(matrix(2, 12))
		width = sv.content_width
		for _ in range(99):
			sv.move('right')
		assert sv.col_off == width - HomeScreen.COLS
		assert not sv.can_right

	def test_list_never_scrolls_vertically(self):
		sv = ScrollView(TiList([float(i) for i in range(40)]))
		assert not sv.can_up and not sv.can_down
		sv.move('down')
		assert sv.row_off == 0


# ── Rendering: indicators and layout ──────────────────────────────────────────

class TestRender:
	def test_bottom_row_always_blank(self):
		_, lines = grid(matrix(10, 8))
		assert lines[7] == ' ' * 16

	def test_down_arrow_at_bottom_right_overrides_ellipsis(self):
		# Top-left of a tall+wide matrix: down arrow at the bottom content row's
		# right edge, replacing the right ellipsis that the other rows show.
		_, lines = grid(matrix(10, 8))
		assert lines[6][RIGHT] == DOWN
		assert lines[0][RIGHT] == ELL        # a row that can't scroll up keeps the ellipsis
		assert lines[0][0] != ELL            # no left ellipsis at col 0

	def test_up_arrow_appears_after_scrolling_down(self):
		_, lines = grid(matrix(10, 8), down=2)
		assert lines[0][RIGHT] == UP

	def test_no_down_arrow_at_bottom(self):
		_, lines = grid(matrix(10, 8), down=99)
		assert lines[0][RIGHT] == UP
		assert lines[6][RIGHT] != DOWN       # can't scroll down any further

	def test_left_ellipsis_on_every_content_row_when_scrolled_right(self):
		sv, lines = grid(matrix(10, 8), right=3)
		assert sv.can_left
		for r in range(ScrollView.VIEW_ROWS):
			assert lines[r][0] == ELL

	def test_no_arrows_for_wide_but_short_matrix(self):
		# Only horizontal overflow: ellipses, but never up/down arrows.
		_, lines = grid(matrix(2, 12))
		joined = '\n'.join(lines)
		assert UP not in joined and DOWN not in joined
		assert ELL in joined

	def test_list_shows_both_ellipses_when_scrolled(self):
		sv, lines = grid(TiList([float(i) for i in range(40)]), right=5)
		assert lines[0][0] == ELL and lines[0][RIGHT] == ELL
		assert UP not in '\n'.join(lines) and DOWN not in '\n'.join(lines)


# ── Pause integration ─────────────────────────────────────────────────────────

def _pause(value):
	env = Environment()
	env.console = ScriptedConsole()
	if isinstance(value, TiMatrix):
		env.matrices.A = value
	else:
		env.lists[0] = value
	name = '[A]' if isinstance(value, TiMatrix) else 'L1'
	Program(toks(f'Pause {name}')).run(env)
	return env


class TestPauseScrolling:
	def test_scrollable_matrix_rendered_as_window(self):
		env = _pause(matrix(10, 8))
		frame = env.console.frames[-1].split('\n')
		assert frame[0].startswith('[[')     # left-aligned window, not right-aligned Disp
		assert frame[7] == ' ' * 16
		assert isinstance(env.ans, TiMatrix)

	def test_small_matrix_still_disp_style(self):
		env = _pause(TiMatrix([[1.0, 2.0], [3.0, 4.0]]))
		frame = env.console.frames[-1].split('\n')
		joined = '\n'.join(frame)
		assert ELL not in joined and DOWN not in joined and UP not in joined
		assert frame[0].rstrip().endswith('[[1 2]')    # right-aligned Disp block

	def test_scrollable_list_rendered_as_window(self):
		env = _pause(TiList([float(i) for i in range(40)]))
		frame = env.console.frames[-1].split('\n')
		assert frame[0].startswith('{')
		assert frame[0][RIGHT] == ELL        # more to the right


# ── Key polling ───────────────────────────────────────────────────────────────

class _FakeMsvcrt:
	def __init__(self, events):
		self.events = list(events)

	def kbhit(self):
		return bool(self.events)

	def getwch(self):
		return self.events.pop(0)


class TestScrollKeys:
	def _console(self):
		return TerminalConsole.__new__(TerminalConsole)

	@pytest.mark.parametrize("scan,direction", [('H', 'up'), ('P', 'down'), ('K', 'left'), ('M', 'right')])
	def test_arrows_map_to_directions(self, scan, direction):
		assert self._console()._poll_scroll_key(_FakeMsvcrt(['\x00', scan])) == direction

	@pytest.mark.parametrize("key", ['\r', ' '])
	def test_enter_or_space_exits(self, key):
		assert self._console()._poll_scroll_key(_FakeMsvcrt([key])) == 'exit'

	def test_idle_returns_none(self):
		assert self._console()._poll_scroll_key(_FakeMsvcrt([])) is None

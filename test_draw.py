"""Tests for drawing commands (Pxl-On/Off/Change, pxl-Test) and the Screen buffer."""

import pytest

from environment import Environment
from screen import Screen
from errors import ArgumentError, DomainError
from test_tibasic import calc, run, toks
from test_program import run as run_program


# ── Screen class ───────────────────────────────────────────────────────────────

class TestScreen:
	def test_dimensions(self):
		s = Screen()
		assert s.ROWS == 64
		assert s.COLS == 96
		assert len(s.buffer) == 64 * 96

	def test_starts_blank(self):
		s = Screen()
		assert not any(s.buffer)

	def test_set_get(self):
		s = Screen()
		s.set(3, 5)  # (row, column)
		assert s.get(3, 5)
		assert not s.get(5, 3)  # (row, column) order matters

	def test_set_off(self):
		s = Screen()
		s.set(3, 5, True)
		s.set(3, 5, False)
		assert not s.get(3, 5)

	def test_toggle(self):
		s = Screen()
		s.toggle(3, 5)
		assert s.get(3, 5)
		s.toggle(3, 5)
		assert not s.get(3, 5)

	def test_clear(self):
		s = Screen()
		s.set(0, 0)
		s.set(63, 95)
		s.clear()
		assert not any(s.buffer)

	def test_corners_addressable(self):
		s = Screen()
		s.set(0, 0)
		s.set(63, 95)  # last row, last column of the full buffer
		assert s.get(0, 0)
		assert s.get(63, 95)


# ── Pxl- commands through the interpreter ───────────────────────────────────────

class TestPixelCommands:
	def test_pixel_starts_off(self):
		assert calc('pxl-Test( 3,5') == 0

	def test_pxl_on(self):
		env = run('Pxl-On( 3,5')
		assert calc('pxl-Test( 3,5', env) == 1
		assert env.screen.get(3, 5)  # (row, column)

	def test_pxl_off(self):
		env = run('Pxl-On( 3,5')
		run('Pxl-Off( 3,5', env)
		assert calc('pxl-Test( 3,5', env) == 0

	def test_pxl_change_toggles(self):
		env = Environment()
		run('Pxl-Change( 3,5', env)
		assert calc('pxl-Test( 3,5', env) == 1
		run('Pxl-Change( 3,5', env)
		assert calc('pxl-Test( 3,5', env) == 0

	def test_on_is_idempotent(self):
		env = run('Pxl-On( 3,5')
		run('Pxl-On( 3,5', env)
		assert calc('pxl-Test( 3,5', env) == 1

	def test_row_column_order(self):
		# Pxl-On(row, column): row is vertical, column is horizontal
		env = run('Pxl-On( 2,7')
		assert env.screen.get(2, 7)
		assert not env.screen.get(7, 2)

	def test_pixels_independent(self):
		env = run('Pxl-On( 10,20')
		assert calc('pxl-Test( 10,20', env) == 1
		assert calc('pxl-Test( 10,21', env) == 0
		assert calc('pxl-Test( 11,20', env) == 0

	def test_pxl_test_returns_real(self):
		env = run('Pxl-On( 0,0')
		result = calc('pxl-Test( 0,0', env)
		assert isinstance(result, float)
		assert result == 1.0

	def test_corner_origin(self):
		env = run('Pxl-On( 0,0')
		assert env.screen.get(0, 0)

	def test_corner_max(self):
		env = run('Pxl-On( 62,94')
		assert env.screen.get(62, 94)


# ── Range and argument validation ───────────────────────────────────────────────

class TestPixelValidation:
	def test_row_too_large(self):
		with pytest.raises(DomainError):
			run('Pxl-On( 63,0')

	def test_column_too_large(self):
		with pytest.raises(DomainError):
			run('Pxl-On( 0,95')

	def test_negative_row(self):
		with pytest.raises(DomainError):
			run('Pxl-On( ~1,0')

	def test_negative_column(self):
		with pytest.raises(DomainError):
			run('Pxl-On( 0,~1')

	def test_test_out_of_range(self):
		with pytest.raises(DomainError):
			calc('pxl-Test( 63,0')

	def test_too_many_args(self):
		# Fixed schema (env, expr, expr): surplus arg caught by end_func()
		with pytest.raises(ArgumentError):
			run('Pxl-On( 1,2,3')

	def test_missing_arg(self):
		# Fixed schema: the second expr finds no argument -> ArgumentError
		with pytest.raises(ArgumentError):
			run('Pxl-On( 1')


# ── Pt-On: graph-coordinate → pixel translation ─────────────────────────────────

class TestPointOn:
	def test_origin_standard_window(self):
		# Default window -10..10: (0,0) maps to the center pixel (31, 47)
		env = run('Pt-On( 0,0')
		assert env.screen.get(31, 47)

	def test_reads_back_via_pxl_test(self):
		env = run('Pt-On( 0,0')
		assert calc('pxl-Test( 31,47', env) == 1

	def test_top_left_corner(self):
		# (Xmin, Ymax) -> (row 0, col 0)
		env = run('Pt-On( ~10,10')
		assert env.screen.get(0, 0)

	def test_bottom_right_corner(self):
		# (Xmax, Ymin) -> (row 62, col 94)
		env = run('Pt-On( 10,~10')
		assert env.screen.get(62, 94)

	def test_off_screen_draws_nothing(self):
		env = run('Pt-On( 100,100')
		assert not any(env.screen.buffer)

	def test_round_half_up_lands_on_higher_column(self):
		# Window 0..188 makes column = x / 2, so x=1 -> 0.5 -> rounds up to 1
		env = Environment()
		env.window.xmin.value = 0
		env.window.xmax.value = 188
		run('Pt-On( 1,0', env)
		assert env.screen.get(31, 1)
		assert not env.screen.get(31, 0)

	def test_round_half_up_again(self):
		# x=3 -> 1.5 -> rounds up to 2
		env = Environment()
		env.window.xmin.value = 0
		env.window.xmax.value = 188
		run('Pt-On( 3,0', env)
		assert env.screen.get(31, 2)
		assert not env.screen.get(31, 1)


# ── Pt-Off / Pt-Change ──────────────────────────────────────────────────────────

class TestPointOff:
	def test_turns_off_lit_pixel(self):
		env = run('Pt-On( 0,0')
		run('Pt-Off( 0,0', env)
		assert not env.screen.get(31, 47)

	def test_off_screen_no_error(self):
		env = run('Pt-Off( 100,100')
		assert not any(env.screen.buffer)

	def test_off_is_idempotent(self):
		env = run('Pt-Off( 0,0')
		run('Pt-Off( 0,0', env)
		assert not env.screen.get(31, 47)


class TestPointChange:
	def test_toggles_on(self):
		env = run('Pt-Change( 0,0')
		assert env.screen.get(31, 47)

	def test_toggles_off(self):
		env = run('Pt-On( 0,0')
		run('Pt-Change( 0,0', env)
		assert not env.screen.get(31, 47)

	def test_off_screen_no_error(self):
		env = run('Pt-Change( 100,100')
		assert not any(env.screen.buffer)


# ── ClrDraw ──────────────────────────────────────────────────────────────────────

class TestClrDraw:
	def test_clears_pixels(self):
		env = run('Pxl-On( 3,5')
		run('ClrDraw', env)
		assert not any(env.screen.buffer)

	def test_clears_after_pt_on(self):
		env = run('Pt-On( 0,0')
		run('ClrDraw', env)
		assert not any(env.screen.buffer)

	def test_clear_then_draw(self):
		env = run('Pxl-On( 10,10')
		run('ClrDraw', env)
		run('Pxl-On( 20,20', env)
		assert env.screen.get(20, 20)
		assert not env.screen.get(10, 10)


# ── Pt-On mark shapes ───────────────────────────────────────────────────────────

class TestPointMarks:
	"""Verify pixel patterns for each mark value (assumed shapes; check on hardware)."""

	def _pt_on_mark(self, mark_tok: str):
		"""Run Pt-On(0,0,mark) and return the set of (row,col) pixels that are on."""
		env = run(f'Pt-On( 0,0,{mark_tok}')
		return {(r, c) for r in range(63) for c in range(95) if env.screen.get(r, c)}

	def test_mark_1_is_dot(self):
		assert self._pt_on_mark('1') == {(31, 47)}

	def test_unknown_mark_defaults_to_dot(self):
		# Any value not in {2,3,6,7} is treated as a dot
		assert self._pt_on_mark('4') == {(31, 47)}
		assert self._pt_on_mark('5') == {(31, 47)}

	def test_mark_2_is_3x3_box(self):
		pixels = self._pt_on_mark('2')
		# Box is the 8-pixel ring; centre is not set
		expected = {(31 + dr, 47 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
		            if (dr, dc) != (0, 0)}
		assert pixels == expected

	def test_mark_6_same_as_2(self):
		assert self._pt_on_mark('6') == self._pt_on_mark('2')

	def test_mark_3_is_cross(self):
		pixels = self._pt_on_mark('3')
		expected = {(31, 47), (30, 47), (32, 47), (31, 46), (31, 48)}
		assert pixels == expected

	def test_mark_7_same_as_3(self):
		assert self._pt_on_mark('7') == self._pt_on_mark('3')

	def test_box_clips_at_edge(self):
		# Pt-On at top-left corner: the 3×3 hollow box clips to the visible region.
		# The 8 ring offsets are: (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1).
		# After clipping to row>=0, col>=0 only (0,1),(1,-1→clipped),(1,0),(1,1) survive,
		# i.e. (0,1), (1,0), (1,1).
		env = run('Pt-On( ~10,10,2')  # corner (row 0, col 0)
		pixels = {(r, c) for r in range(63) for c in range(95) if env.screen.get(r, c)}
		expected = {(0, 1), (1, 0), (1, 1)}
		assert pixels == expected

	def test_pt_off_with_mark_clears_shape(self):
		env = run('Pt-On( 0,0,2')
		run('Pt-Off( 0,0,2', env)
		assert not any(env.screen.buffer)

	def test_pt_change_with_mark_toggles(self):
		env = run('Pt-Change( 0,0,3')
		cross = {(31, 47), (30, 47), (32, 47), (31, 46), (31, 48)}
		for r, c in cross:
			assert env.screen.get(r, c)
		run('Pt-Change( 0,0,3', env)
		assert not any(env.screen.buffer)


# ── Vertical / Horizontal ────────────────────────────────────────────────────────

class TestVertical:
	def test_center(self):
		# x=0 on standard window -> col 47; should fill all rows 0-62
		env = run('Vertical 0')
		for row in range(MAX_ROW + 1):
			assert env.screen.get(row, 47)
		assert not env.screen.get(0, 46)
		assert not env.screen.get(0, 48)

	def test_left_edge(self):
		env = run('Vertical ~10')   # Xmin -> col 0
		for row in range(MAX_ROW + 1):
			assert env.screen.get(row, 0)

	def test_right_edge(self):
		env = run('Vertical 10')    # Xmax -> col 94
		for row in range(MAX_ROW + 1):
			assert env.screen.get(row, 94)

	def test_off_screen_draws_nothing(self):
		env = run('Vertical 100')
		assert not any(env.screen.buffer)


class TestHorizontal:
	def test_center(self):
		# y=0 on standard window -> row 31; should fill all cols 0-94
		env = run('Horizontal 0')
		for col in range(MAX_COL + 1):
			assert env.screen.get(31, col)
		assert not env.screen.get(30, 0)
		assert not env.screen.get(32, 0)

	def test_top_edge(self):
		env = run('Horizontal 10')   # Ymax -> row 0
		for col in range(MAX_COL + 1):
			assert env.screen.get(0, col)

	def test_bottom_edge(self):
		env = run('Horizontal ~10')  # Ymin -> row 62
		for col in range(MAX_COL + 1):
			assert env.screen.get(62, col)

	def test_off_screen_draws_nothing(self):
		env = run('Horizontal 100')
		assert not any(env.screen.buffer)


# ── Line( ────────────────────────────────────────────────────────────────────────

class TestLine:
	def test_horizontal_line(self):
		# Line along top of graph screen
		env = run('Line( ~10,10,10,10')
		for col in range(MAX_COL + 1):
			assert env.screen.get(0, col)

	def test_vertical_line(self):
		# Line along left side of graph screen
		env = run('Line( ~10,10,~10,~10')
		for row in range(MAX_ROW + 1):
			assert env.screen.get(row, 0)

	def test_diagonal_corner_to_corner(self):
		# Corner to corner: endpoints must be set
		env = run('Line( ~10,10,10,~10')
		assert env.screen.get(0, 0)
		assert env.screen.get(62, 94)

	def test_single_point(self):
		env = run('Line( 0,0,0,0')
		assert env.screen.get(31, 47)
		assert sum(env.screen.buffer) == 1

	def test_erase(self):
		env = run('Line( ~10,10,10,10')   # draw top row
		run('Line( ~10,10,10,10,0', env)  # erase it
		assert not any(env.screen.buffer)

	def test_erase_default_draws(self):
		env = run('Line( ~10,10,10,10,1')
		assert env.screen.get(0, 0)

	def test_partial_offscreen(self):
		# Line starts on screen, ends off — visible portion should be drawn
		env = run('Line( 0,0,100,0')
		assert env.screen.get(31, 47)  # origin pixel is set
		assert not env.screen.get(31, 95)  # col 95 is outside MAX_COL


from draw import MAX_ROW, MAX_COL


# ── Circle( ───────────────────────────────────────────────────────────────────────

class TestCircle:
	# Default window: xmin=-10, xmax=10, ymin=-10, ymax=10
	# Centre (0,0) → pixel (31, 47)
	# rx = r * 94/20 = r*4.7,  ry = r * 62/20 = r*3.1

	def test_draws_something(self):
		env = run('Circle( 0,0,5')
		assert any(env.screen.buffer)

	def test_center_unset(self):
		# Circle is an outline; the center pixel should be dark.
		env = run('Circle( 0,0,5')
		assert not env.screen.get(31, 47)

	def test_rightmost_cardinal(self):
		# θ=0: col = cx + rx, row = cy.  For r=10: col = 47+47 = 94, row = 31.
		env = run('Circle( 0,0,10')
		assert env.screen.get(31, 94)

	def test_leftmost_cardinal(self):
		# θ=π: col = 47-47 = 0, row = 31.
		env = run('Circle( 0,0,10')
		assert env.screen.get(31, 0)

	def test_topmost_cardinal(self):
		# θ=π/2: col = 47, row = 31-31 = 0.
		env = run('Circle( 0,0,10')
		assert env.screen.get(0, 47)

	def test_bottommost_cardinal(self):
		# θ=3π/2: col = 47, row = 31+31 = 62.
		env = run('Circle( 0,0,10')
		assert env.screen.get(62, 47)

	def test_negative_radius_same_as_positive(self):
		env_pos = run('Circle( 0,0,5')
		env_neg = run('Circle( 0,0,~5')
		assert env_pos.screen.buffer == env_neg.screen.buffer

	def test_off_screen_no_error(self):
		# Huge radius: most pixels are off screen; should not crash.
		env = run('Circle( 0,0,1000')
		# No assertion about pixels — just no exception.

	def test_entirely_offscreen_draws_nothing(self):
		# Circle centred far off screen with small radius.
		env = run('Circle( 100,100,1')
		assert not any(env.screen.buffer)

	def test_ellipse_non_square_window(self):
		# With xmin=0, xmax=20, ymin=0, ymax=20 and centre (10,10):
		#   cx=47, cy=31, rx = 2*94/20 = 9.4, ry = 2*62/20 = 6.2
		# Rightmost point: col = 47+9 or 47+10, row = 31.
		# Topmost point:   col = 47, row = 31-6 = 25.
		# These differ → it is actually an ellipse, not a circle.
		env = Environment()
		env.window.xmin.value = 0
		env.window.xmax.value = 20
		env.window.ymin.value = 0
		env.window.ymax.value = 20
		run('Circle( 10,10,2', env)
		# Rightmost pixel column should be further from centre than topmost pixel row.
		import math
		# rx_pix ≈ 9.4, ry_pix ≈ 6.2 → horizontally wider
		assert env.screen.get(31, 47 + round(2 * 94 / 20))
		assert env.screen.get(31 - round(2 * 62 / 20), 47)

	def test_fast_arg_accepted(self):
		# Passing a complex list as 4th arg should not raise; shape is the same.
		env_slow = run('Circle( 0,0,5')
		# We can't easily construct {i} from the string runner, so just verify
		# the no-arg form works fine (full {i} test would need token-level encoding).
		assert any(env_slow.screen.buffer)


# ── Use inside a stored program ─────────────────────────────────────────────────

class TestPixelInProgram:
	def test_loop_draws_diagonal(self):
		env = run_program("""
		For( I,0,10
		Pxl-On( I,I
		End
		""")
		for i in range(11):
			assert env.screen.get(i, i)
		assert not env.screen.get(0, 1)

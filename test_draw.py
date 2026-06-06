"""Tests for drawing commands (Pxl-On/Off/Change, pxl-Test) and the Screen buffer."""

import pytest

from environment import Environment
from screen import Screen
from errors import DomainError
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
		# Fixed-arity env_func: surplus args bind past the signature -> TypeError
		with pytest.raises(TypeError):
			run('Pxl-On( 1,2,3')

	def test_missing_arg(self):
		with pytest.raises(TypeError):
			run('Pxl-On( 1')


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

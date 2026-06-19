"""The grid-less FreeFormIO device: Disp/Output/Pause/Input via print()/input()."""
import pytest

from environment import Environment
from iodevice import FreeFormIO
from program import Program
from test_tibasic import toks, var


def freeform_env():
	env = Environment()
	env.io = FreeFormIO()
	return env


def run(src, env):
	env.run(toks(src))
	return env


class TestFreeFormOutput:
	def test_disp_prints_value(self, capsys):
		run('Disp 5', freeform_env())
		assert capsys.readouterr().out == '5\n'

	def test_disp_prints_string(self, capsys):
		run('Disp "HI', freeform_env())
		assert capsys.readouterr().out == 'HI\n'

	def test_disp_matrix_one_line_per_row(self, capsys):
		run('Disp [[1,2][3,4]]', freeform_env())
		assert capsys.readouterr().out == '[[1 2]\n [3 4]]\n'

	def test_output_ignores_position_and_prints(self, capsys):
		# Weakly supported: an out-of-range cell that would raise on the home screen
		# is fine here — the value is just printed without positioning.
		run('Output( 99,99,7', freeform_env())
		assert capsys.readouterr().out == '7\n'


class TestFreeFormPause:
	def test_pause_prints_value_and_marker(self, capsys, monkeypatch):
		monkeypatch.setattr('builtins.input', lambda *a: '')
		Program(toks('Pause 5'), env := freeform_env()).run()
		assert capsys.readouterr().out == '5\n[PAUSED]\n'
		assert env.ans == 5

	def test_bare_pause_just_blocks(self, capsys, monkeypatch):
		monkeypatch.setattr('builtins.input', lambda *a: '')
		Program(toks('Pause'), freeform_env()).run()
		assert capsys.readouterr().out == '[PAUSED]\n'


class TestFreeFormInput:
	def test_input_reads_via_input(self, monkeypatch):
		monkeypatch.setattr('builtins.input', lambda *a: '42')
		assert var(run('Input X', freeform_env()), 'X') == 42

	def test_prompt_reads_each_var(self, monkeypatch):
		answers = iter(['1', '2'])
		monkeypatch.setattr('builtins.input', lambda *a: next(answers))
		env = run('Prompt A,B', freeform_env())
		assert (var(env, 'A'), var(env, 'B')) == (1, 2)


class TestFreeFormKeys:
	def test_get_key_unsupported_returns_zero(self):
		assert FreeFormIO().get_key() == 0

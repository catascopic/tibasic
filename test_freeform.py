"""The grid-less FreeFormConsole: Disp streamed via print(), Input via input()."""
import pytest

from environment import Environment
from terminal import FreeFormConsole
from program import Program
from test_tibasic import toks, var


def freeform_env():
	env = Environment()
	env.console = FreeFormConsole()
	return env


def run(src, env):
	env.submit(toks(src))
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

	def test_output_writes_to_grid_not_stream(self, capsys):
		# Output( is positional, so it writes to the (unpainted) home grid like any
		# other frontend rather than streaming; free-form prints only Disp values,
		# so nothing is emitted, but the grid reflects the write.
		env = run('Output( 1,1,7', freeform_env())
		assert capsys.readouterr().out == ''
		assert env.home.render().split('\n')[0].startswith('7')


class TestFreeFormPause:
	def test_pause_prints_value_and_marker(self, capsys, monkeypatch):
		monkeypatch.setattr('builtins.input', lambda *a: '')
		Program(toks('Pause 5')).run(env := freeform_env())
		assert capsys.readouterr().out == '5\n[PAUSED]\n'
		assert env.ans == 5

	def test_bare_pause_just_blocks(self, capsys, monkeypatch):
		monkeypatch.setattr('builtins.input', lambda *a: '')
		Program(toks('Pause')).run(freeform_env())
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
		assert FreeFormConsole().read_key() == 0

import sys

from environment import Environment
from terminal import TerminalConsole
from iodevice import FreeFormIO
from program import Program
from test_tibasic import toks


def run_demo(source, freeform=False):
	"""Run `source` as a stored program.

	By default it uses the faithful home-screen device painted to the terminal;
	pass freeform=True (or `--freeform` on the command line) for the grid-less
	print()/input() device instead.
	"""
	env = Environment(console=TerminalConsole())
	if freeform:
		env.io = FreeFormIO()
	env.programs['DEMO'] = Program(toks(source), 'DEMO')
	env.submit(toks('prgm DEMO'))


if __name__ == '__main__':
	run_demo("""

Pause randM( 10,10

	""", freeform='--freeform' in sys.argv)

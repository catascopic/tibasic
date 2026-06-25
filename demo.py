import sys

from environment import Environment
from terminal import TerminalConsole, FreeFormConsole
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
		env.console = FreeFormConsole()
	env.programs['DEMO'] = Program(toks(source), 'DEMO')
	env.submit(toks('prgm DEMO'))


if __name__ == '__main__':
	run_demo("""

Lbl 1
Disp "X--------------O
Disp "_X------------O
Disp "__X----------O
Disp "___X--------O
Disp "____X------O
Disp "_____X----O
Disp "______X--O
Disp "_______XO
Disp "_______OX
Disp "______O--X
Disp "_____O----X
Disp "____O------X
Disp "___O--------X
Disp "__O----------X
Disp "_O------------X
Disp "O--------------X
Disp "_O------------X
Disp "__O----------X
Disp "___O--------X
Disp "____O------X
Disp "_____O----X
Disp "______O--X
Disp "_______OX
Disp "_______XO
Disp "______X--O
Disp "_____X----O
Disp "____X------O
Disp "___X--------O
Disp "__X----------O
Disp "_X------------O
Goto 1

	""", freeform='--freeform' in sys.argv)


"""

Disp "X
Disp "_X
Disp "__X
Disp "___X
Disp "____X
Disp "_____X
Disp "______X
Disp "_______X
Disp "________X
Disp "_________X
Disp "__________X
Disp "___________X
Disp "____________X
Disp "_____________X
Disp "______________X
Disp "_______________X
Disp "______________X
Disp "_____________X
Disp "____________X
Disp "___________X
Disp "__________X
Disp "_________X
Disp "________X
Disp "_______X
Disp "______X
Disp "_____X
Disp "____X
Disp "___X
Disp "__X
Disp "_X
prgm DEMO

	"""
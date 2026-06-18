from environment import Environment
from terminal import TerminalConsole
from test_tibasic import toks

def run_demo(source):
	env = Environment(console=TerminalConsole())
	env.programs['DEMO'] = toks(source)
	env.run(toks('prgm DEMO'))


if __name__ == '__main__':
	run_demo("""

Pause randM( 10,10
	
	""")
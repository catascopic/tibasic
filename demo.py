from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

Program(toks("""

Disp [[1/3,2/3][ pi , sqrt( 2

"""), Environment(TerminalConsole())).run()

from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

Program(toks("""

Disp [[1,2]][3.33,4

"""), Environment(TerminalConsole())).run()

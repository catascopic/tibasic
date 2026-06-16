from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

PROGRAM = """
Output( 5,10,"X
For( A,1,10
Disp A
Pause
End
"""

env = Environment()
env.console = TerminalConsole()
Program(toks(PROGRAM), env).run()

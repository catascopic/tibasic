from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

PROGRAM = """
Input "GREETING=", Str1
Disp Str1
"""

env = Environment()
env.console = TerminalConsole()
Program(toks(PROGRAM), env).run()

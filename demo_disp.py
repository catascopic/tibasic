from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

PROGRAM = """
Prompt A, Str1 , L1
Disp A, Str1 , L1
"""

env = Environment()
env.console = TerminalConsole()
Program(toks(PROGRAM), env).run()

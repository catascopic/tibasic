from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

Program(toks("""

For( A,1,10
Disp A[[1,2][3.25,4
End

"""), Environment(TerminalConsole())).run()

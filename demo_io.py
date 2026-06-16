"""Quick interactive demo of the home-screen I/O commands on a TerminalConsole.

Run:  python demo_io.py

Shows a Menu(, branches to the chosen label, reads a number with Input, and writes
the result positioned on the 16x8 home screen.  Then reveals a word one letter at
a time with Output(, pressing Pause between each letter.  Runs as a Program so
Menu('s and Goto's branching have a program context.
"""
from environment import Environment
from terminal import TerminalConsole
from program import Program
from test_tibasic import toks

PROGRAM = """
ClrHome
Menu( "CHOOSE","SQUARE",S,"DOUBLE",D
Lbl S
Input "N?",N
ClrHome
Output( 1,1,"SQUARE=
Output( 1,9,N N
Goto T
Lbl D
Input "N?",N
ClrHome
Output( 1,1,"DOUBLE=
Output( 1,9,2N
Lbl T
Pause
ClrHome
Output( 1,1,"WATCH:
For( I,1, length( "HELLO
Output( 4,I, sub( "HELLO",I,1
Pause
End
Output( 6,1,"DONE
"""

env = Environment()
env.console = TerminalConsole()
Program(toks(PROGRAM), env).run()

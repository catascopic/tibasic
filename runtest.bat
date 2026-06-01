@ECHO off
python -m pytest program_test.py
EXIT /B
PAUSE
python -m pytest tibasic_test.py
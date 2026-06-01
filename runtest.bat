@ECHO off
python -m pytest program_test.py
REM EXIT /B
REM PAUSE
python -m pytest tibasic_test.py
from catalog import TOKENS, NEWLINE
from tifile import TiProgram, write

tokens = sorted(TOKENS, key=lambda t: t.code)

def _iter():
	for token in tokens:
		if token.code > b'\xEF\x16':
			return
		if token is not NEWLINE:
			yield token
			yield NEWLINE

write('ALLTOKEN.8xp', TiProgram('ALLTOKEN', list(_iter())))

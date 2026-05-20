"""Read and write TI-83/84 .8xp program files."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from tokens import Token, TOKENS

_SIGNATURE    = b'**TI83F*\x1a\x0a\x00'
_UNLOCKED = 0x05
_LOCKED  = 0x06
_TWO_BYTE_PREFIXES = {0x5c, 0x5d, 0x5e, 0x60, 0x61, 0x62, 0x63, 0x7e, 0xaa, 0xbb, 0xef}
_LOOKUP = {t.code: t for t in TOKENS}


@dataclass
class TiProgram:
	name:     str
	tokens:   list[Token]
	comment:  str  = ''
	archived: bool = False
	locked:   bool = False


def _decode(program: bytes) -> list:
	it = iter(program)
	while True:
		try:
			b = next(it)
		except StopIteration:
			return
		code = bytes([b, next(it)] if b in _TWO_BYTE_PREFIXES else [b])
		try:
			yield _LOOKUP[code]
		except KeyError as e:
			raise ValueError(f"Unknown token code: {code.hex()}") from e


def read(path) -> TiProgram:
	data = Path(path).read_bytes()
	if data[:8] != b'**TI83F*':
		raise ValueError(f"{path}: not a TI-83/84 variable file")

	comment = data[11:53].rstrip(b'\x00 ').decode('ascii', errors='replace')

	pos = 55
	entry_type = int.from_bytes(data[pos:pos + 2], 'little')  # 0x000B or 0x000D
	pos += 4  # skip entry_type (2) + data_len (2)
	file_type = data[pos]
	pos += 1
	name = data[pos:pos + 8].rstrip(b'\x00').decode('ascii')
	pos += 8

	if entry_type == 0x000d:
		_version = data[pos]
		flag = data[pos + 1]
		pos += 2
	else:
		flag = 0x00

	pos += 2  # skip data_len copy
	prog_len = int.from_bytes(data[pos:pos + 2], 'little')
	pos += 2
	program  = data[pos:pos + prog_len]

	return TiProgram(
		name     = name,
		tokens   = list(_decode(program)),
		comment  = comment,
		archived = bool(flag & 0x80),
		locked   = (file_type == _LOCKED),
	)


def write(path, prog: TiProgram) -> None:
	program    = b''.join(t.code for t in prog.tokens)
	name_bytes = prog.name.upper().encode('ascii')[:8].ljust(8, b'\x00')
	locked     = _LOCKED if prog.locked else _UNLOCKED
	flag       = 0x80 if prog.archived else 0x00
	data_len   = len(program) + 2  # +2 for the prog_len prefix inside var data

	var_entry = (
		b'\x0d\x00'                        # entry header type: 0x000D (includes version + flag)
		+ data_len.to_bytes(2, 'little')   # length of var data
		+ bytes([locked])                  # 0x05 = program, 0x06 = edit-locked
		+ name_bytes                       # variable name, null-padded to 8 bytes
		+ b'\x01'                          # version
		+ bytes([flag])                    # 0x80 = archived in flash, 0x00 = RAM
		+ data_len.to_bytes(2, 'little')   # length of var data (repeated)
		+ len(program).to_bytes(2, 'little')  # length of program body
		+ program
	)

	comment_bytes = prog.comment.encode('ascii')[:42].ljust(42, b'\x00')
	header    = _SIGNATURE + comment_bytes + len(var_entry).to_bytes(2, 'little')
	checksum  = sum(var_entry) & 0xFFFF
	Path(path).write_bytes(header + var_entry + checksum.to_bytes(2, 'little'))


if __name__ == '__main__':
	import sys

	for path in sys.argv[1:]:
		prog = read(path)
		print(f"PRGM:{prog.name} (locked={prog.locked}, archived={prog.archived})")
		if prog.comment:
			print(prog.comment)
		print(''.join(t.text for t in prog.tokens).encode(sys.stdout.encoding, errors='replace').decode(sys.stdout.encoding))

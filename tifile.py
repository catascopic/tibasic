"""Read and write TI-83/84 .8xp program files."""
from dataclasses import dataclass
from pathlib import Path

from tokens import Token, ALL_TOKENS, read_token


@dataclass
class TiProgram:
	name:     str
	tokens:   list[Token]
	comment:  str  = ''
	archived: bool = False
	locked:   bool = False

	def __repr__(self):
		return f"prgm{self.name}(tokens={len(self.tokens)};{'' if self.archived else 'un'}archived/{'' if self.locked else 'un'}locked)"
	
	def print(self):
		if self.comment:
			print(self.comment)
		print(f"PRGM:{self.name} ({'' if self.archived else 'un'}archived/{'' if self.locked else 'un'}locked)")
		print(''.join(t.text for t in self.tokens))

	@classmethod
	def read_from(cls, f):
		signature = f.read(8)
		if not signature.startswith(b'**TI8'):  # could be 82, 83, 83F
			raise ValueError(f"Invalid .8xp signature: {signature!r}")

		f.seek(3, 1)  # skip 1a 0a 00
		comment = f.read(42).rstrip(b'\x00 ').decode('ascii', errors='replace')
		f.seek(2, 1)  # skip meta/body length
		entry_type = int.from_bytes(f.read(2), 'little')  # 0x000B or 0x000D
		f.seek(2, 1)  # skip body/checksum length
		(file_type,) = f.read(1)
		name = f.read(8).rstrip(b'\x00').decode('ascii')
		_version, archived = f.read(2)  # TODO: field missing when entry_type != 0x000d???
		f.seek(2, 1)  # skip body/checksum length duplicate
		end = int.from_bytes(f.read(2), 'little') + f.tell()
		tokens = []
		while f.tell() < end:
			tokens.append(read_token(f))

		# TODO: checksum check
		return cls(
			name     = name,
			tokens   = tokens,
			comment  = comment,
			archived = bool(archived & 0x80),
			locked   = file_type == 0x06,
		)

	@classmethod
	def read(cls, file):
		with open(file, 'rb') as f:
			return cls.read_from(f)

	def write_to(self, f):
		program    = b''.join(t.code for t in self.tokens)
		name_bytes = self.name.upper().encode('ascii')[:8].ljust(8, b'\x00')
		locked     = 0x06 if self.locked else 0x05
		flag       = 0x80 if self.archived else 0x00
		data_len   = len(program) + 2  # +2 for the prog_len prefix inside var data

		var_entry = (
			b'\x0D\x00'                           # entry header type: 0x000D (includes version + flag)
			+ data_len.to_bytes(2, 'little')      # length of var data
			+ bytes([locked])                     # 0x05 = program, 0x06 = edit-locked
			+ name_bytes                          # variable name, null-padded to 8 bytes
			+ b'\x01'                             # version
			+ bytes([flag])                       # 0x80 = archived in flash, 0x00 = RAM
			+ data_len.to_bytes(2, 'little')      # length of var data (repeated)
			+ len(program).to_bytes(2, 'little')  # length of program body
			+ program
		)

		comment_bytes = self.comment.encode('ascii')[:42].ljust(42, b'\x00')
		checksum = sum(var_entry) & 0xFFFF
		f.write(b'**TI83F*')
		f.write(b'\x1a\x0a\x00')
		f.write(comment_bytes)
		f.write(len(var_entry).to_bytes(2, 'little'))
		f.write(var_entry)
		f.write(checksum.to_bytes(2, 'little'))

	def write(self, file):
		with open(file, 'wb') as f:
			self.write_to(f)


if __name__ == '__main__':
	import sys

	for path in sys.argv[1:]:
		read(path).print()

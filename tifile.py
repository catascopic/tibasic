"""Read and write TI-83/84 variable files (.8xp programs, .8xl real lists).

Every TI variable file shares one envelope — an 8-byte signature, a 42-byte
comment, a variable-entry header (type, name, archive flag), and a trailing
checksum — wrapping a type-specific body.  `_read_var_header` / `_write_var_file`
own that envelope; ProgramFile and ListFile each supply only the body: a
length-prefixed token stream for programs, a count-prefixed array of 9-byte TI
reals for lists.
"""
from dataclasses import dataclass
from pathlib import Path

from titoken import Token
from catalog import ALL_TOKENS, read_token


# 0x5D is TI's "list name" token: a named list's 8-byte name field is this byte
# followed by up to 5 ASCII characters (built-in L1-L6 use 0x5D + 0x00..0x05).
_LIST_NAME_TOKEN = 0x5D


def _read_var_header(f):
	"""Read the shared .8x* envelope and leave `f` at the start of the variable
	data (the body's own length/count prefix).

	Returns (file_type, name_bytes, archived, comment, version).  `name_bytes` is
	the raw 8-byte name field — each caller decodes it, since lists carry the 0x5D
	prefix token and programs don't.  `version` is the source file's var-version
	byte, captured so writers can reproduce it byte-for-byte.
	"""
	signature = f.read(8)
	if not signature.startswith(b'**TI8'):  # could be 82, 83, 83F
		raise ValueError(f"Invalid TI variable file signature: {signature!r}")

	f.seek(3, 1)  # skip 1a 0a 00
	comment = f.read(42).rstrip(b'\x00 ').decode('ascii', errors='replace')
	f.seek(2, 1)  # skip total var-entry length
	_entry_type = int.from_bytes(f.read(2), 'little')  # 0x000B or 0x000D
	f.seek(2, 1)  # skip data length
	(file_type,) = f.read(1)
	name_bytes = f.read(8)
	version, archived = f.read(2)  # TODO: field missing when entry_type != 0x000d???
	f.seek(2, 1)  # skip data length duplicate
	# TODO: checksum check
	return file_type, name_bytes, bool(archived & 0x80), comment, version


def _write_var_file(f, file_type, name_bytes, archived, comment, body, version):
	"""Write a complete .8x* file wrapping `body` (the variable data, including
	its own length/count prefix) in the standard envelope.

	`version` is the var-version byte.  It varies per file (e.g. real .8xp files
	carry 0x00 or 0x03, not the 0x01 TI-Connect emits for fresh programs), so
	callers pass through the value captured on read to round-trip byte-for-byte.
	"""
	data_len = len(body)
	flag     = 0x80 if archived else 0x00

	var_entry = (
		b'\x0D\x00'                       # entry header type: 0x000D (includes version + flag)
		+ data_len.to_bytes(2, 'little')  # length of var data
		+ bytes([file_type])              # 0x01 list, 0x05 program, 0x06 locked program
		+ name_bytes                      # variable name, null-padded to 8 bytes
		+ bytes([version])                # var version
		+ bytes([flag])                   # 0x80 = archived in flash, 0x00 = RAM
		+ data_len.to_bytes(2, 'little')  # length of var data (repeated)
		+ body
	)

	comment_bytes = comment.encode('ascii')[:42].ljust(42, b'\x00')
	checksum = sum(var_entry) & 0xFFFF
	f.write(b'**TI83F*')
	f.write(b'\x1a\x0a\x00')
	f.write(comment_bytes)
	f.write(len(var_entry).to_bytes(2, 'little'))
	f.write(var_entry)
	f.write(checksum.to_bytes(2, 'little'))


@dataclass
class ProgramFile:
	name:     str
	tokens:   list[Token]
	comment:  str  = ''
	archived: bool = False
	locked:   bool = False
	version:  int  = 0x01  # var-version byte; 0x01 is TI-Connect's default for fresh programs

	def __repr__(self):
		return f"prgm{self.name}(tokens={len(self.tokens)};{'' if self.archived else 'un'}archived/{'' if self.locked else 'un'}locked)"

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"PRGM:{self.name} ({'' if self.archived else 'un'}archived/{'' if self.locked else 'un'}locked)")
		print(''.join(t.text for t in self.tokens))

	@classmethod
	def read_from(cls, f):
		file_type, name_bytes, archived, comment, version = _read_var_header(f)
		name = name_bytes.rstrip(b'\x00').decode('ascii')
		end = int.from_bytes(f.read(2), 'little') + f.tell()
		tokens = []
		while f.tell() < end:
			tokens.append(read_token(f))

		return cls(
			name     = name,
			tokens   = tokens,
			comment  = comment,
			archived = archived,
			locked   = file_type == 0x06,
			version  = version,
		)

	@classmethod
	def load(cls, file):
		with open(file, 'rb') as f:
			return cls.read_from(f)

	def write_to(self, f):
		program    = b''.join(t.code_to_bytes() for t in self.tokens)
		name_bytes = self.name.upper().encode('ascii')[:8].ljust(8, b'\x00')
		file_type  = 0x06 if self.locked else 0x05
		body       = len(program).to_bytes(2, 'little') + program  # prog_len prefix + body
		_write_var_file(f, file_type, name_bytes, self.archived, self.comment, body, version=self.version)

	def write(self, file):
		with open(file, 'wb') as f:
			self.write_to(f)


# A TI real is 9 bytes: a flags byte (bit 7 = sign), an exponent biased by 0x80,
# and 7 BCD bytes holding 14 decimal digits as d.ddddddddddddd × 10^exponent.
def _decode_ti_real(b9: bytes) -> float:
	negative = bool(b9[0] & 0x80)
	exp      = b9[1] - 0x80
	digits   = b9[2:9].hex()  # 14 decimal digits (BCD nibbles are always 0-9)
	value    = float(f'{digits[0]}.{digits[1:]}e{exp}')
	return -value if negative else value


def _encode_ti_real(value: float) -> bytes:
	negative       = value < 0
	mant, exp_str  = f'{abs(value):.13e}'.split('e')  # 'd.ddddddddddddd', '±NN'
	digits         = mant.replace('.', '')            # 14 decimal digits
	flags          = 0x80 if negative else 0x00
	bcd            = bytes(int(digits[i:i+2], 16) for i in range(0, 14, 2))
	return bytes([flags, int(exp_str) + 0x80]) + bcd


def _format_real(v: float) -> str:
	return str(int(v)) if float(v).is_integer() else repr(v)


@dataclass
class ListFile:
	name:     str
	values:   list[float]
	comment:  str  = ''
	archived: bool = False
	version:  int  = 0x00  # var-version byte; real lists carry 0x00

	def __repr__(self):
		return f"list{self.name}(values={len(self.values)};{'' if self.archived else 'un'}archived)"

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"LIST:{self.name} ({'' if self.archived else 'un'}archived)")
		print('{' + ', '.join(_format_real(v) for v in self.values) + '}')

	@classmethod
	def read_from(cls, f):
		_file_type, name_bytes, archived, comment, version = _read_var_header(f)
		count  = int.from_bytes(f.read(2), 'little')
		values = [_decode_ti_real(f.read(9)) for _ in range(count)]

		return cls(
			name     = _decode_list_name(name_bytes),
			values   = values,
			comment  = comment,
			archived = archived,
			version  = version,
		)

	@classmethod
	def load(cls, file):
		with open(file, 'rb') as f:
			return cls.read_from(f)

	def write_to(self, f):
		name_bytes = _encode_list_name(self.name)
		reals      = b''.join(_encode_ti_real(v) for v in self.values)
		body       = len(self.values).to_bytes(2, 'little') + reals  # element count + reals
		_write_var_file(f, 0x01, name_bytes, self.archived, self.comment, body, version=self.version)

	def write(self, file):
		with open(file, 'wb') as f:
			self.write_to(f)


def _decode_list_name(name_bytes: bytes) -> str:
	if name_bytes[:1] != bytes([_LIST_NAME_TOKEN]):
		return name_bytes.rstrip(b'\x00').decode('ascii', errors='replace')
	# After the 0x5D token a built-in list L1-L6 stores a single index byte
	# 0x00..0x05 — note L1's index is 0x00, so it must not be rstripped as if it
	# were padding.  A user list instead stores ASCII, whose first byte is always
	# a letter (>= 0x41), so the 0x00..0x05 range cleanly tells the two apart.
	if name_bytes[1] <= 0x05:
		return str(name_bytes[1])
	return name_bytes[1:].rstrip(b'\x00').decode('ascii', errors='replace')


def _encode_list_name(name: str) -> bytes:
	# Built-in lists L1-L6 are named "1".."6" and store a single index byte after
	# the 0x5D token; user lists store their (uppercased) name as ASCII.  A name
	# starting with a digit is never a valid user list on the calculator, so
	# "1".."6" is an unambiguous marker for the built-ins.
	if name in ('1', '2', '3', '4', '5', '6'):
		body = bytes([int(name) - 1])
	else:
		body = name.upper().encode('ascii')[:5]
	return (bytes([_LIST_NAME_TOKEN]) + body).ljust(8, b'\x00')


if __name__ == '__main__':
	import sys

	for path in sys.argv[1:]:
		reader = ListFile if path.lower().endswith('.8xl') else ProgramFile
		reader.load(path).print()

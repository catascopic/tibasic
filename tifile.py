"""Read and write TI-83/84 variable files: programs (.8xp), lists (.8xl), number
variables (.8xn real / .8xc complex), matrices (.8xm), and pictures (.8xi).

Every TI variable file shares one envelope — an 8-byte signature, a 42-byte
comment, a variable-entry header (type, name, archive flag), and a trailing
checksum — wrapping a type-specific body.  `_read_var_header` / `_write_var_file`
own that envelope; each file class (all subclassing `TiFile`) supplies only the
body via its own read_from / write_to — a length-prefixed token stream for
programs, a count-prefixed array of 9-byte TI reals for lists, and so on.
"""
from dataclasses import dataclass
from pathlib import Path

from tokenbase import Token
from catalog import get_token, read_token
from bitmap import Bitmap, ROWS, COLS


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


class TiFile:
	"""Base for every TI variable file type.  Subclasses are dataclasses that
	implement read_from(f) / write_to(f) for their own body format; this base adds
	the shared path-level load() / write() wrappers around them."""

	@classmethod
	def load(cls, file):
		with open(file, 'rb') as f:
			return cls.read_from(f)

	def write(self, file):
		with open(file, 'wb') as f:
			self.write_to(f)


def _read_token_stream(f) -> list[Token]:
	"""Read a 2-byte length-prefixed run of tokens — the body shared by programs,
	strings, and equations — and return them as a list."""
	end = int.from_bytes(f.read(2), 'little') + f.tell()
	tokens = []
	while f.tell() < end:
		tokens.append(read_token(f))
	return tokens


def _token_stream_body(tokens: list[Token]) -> bytes:
	"""Encode tokens as a 2-byte length-prefixed byte run (program/string/equation body)."""
	data = b''.join(t.code_to_bytes() for t in tokens)
	return len(data).to_bytes(2, 'little') + data


@dataclass
class ProgramFile(TiFile):
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
		print(f"prgm{self.name} ({'' if self.archived else 'un'}archived/{'' if self.locked else 'un'}locked)")
		print(''.join(t.text for t in self.tokens))

	@classmethod
	def read_from(cls, f):
		file_type, name_bytes, archived, comment, version = _read_var_header(f)
		name = name_bytes.rstrip(b'\x00').decode('ascii')
		tokens = _read_token_stream(f)

		return cls(
			name     = name,
			tokens   = tokens,
			comment  = comment,
			archived = archived,
			locked   = file_type == 0x06,
			version  = version,
		)

	def write_to(self, f):
		name_bytes = self.name.upper().encode('ascii')[:8].ljust(8, b'\x00')
		file_type  = 0x06 if self.locked else 0x05
		_write_var_file(f, file_type, name_bytes, self.archived, self.comment,
		                _token_stream_body(self.tokens), version=self.version)


# A TI real is 9 bytes: a flags byte (bit 7 = sign, 0x0C = part of a complex
# number), an exponent biased by 0x80, and 7 BCD bytes holding 14 decimal digits
# as d.ddddddddddddd × 10^exponent.  A complex element is two of these back to
# back — real part then imaginary part — each carrying the 0x0C flag.
_COMPLEX_FLAG = 0x0C

def _decode_ti_real(b9: bytes) -> float:
	negative = bool(b9[0] & 0x80)
	exp      = b9[1] - 0x80
	digits   = b9[2:9].hex()  # 14 decimal digits (BCD nibbles are always 0-9)
	value    = float(f'{digits[0]}.{digits[1:]}e{exp}')
	return -value if negative else value


def _encode_ti_real(value: float, complex_flag: bool = False) -> bytes:
	negative       = value < 0
	mant, exp_str  = f'{abs(value):.13e}'.split('e')  # 'd.ddddddddddddd', '±NN'
	digits         = mant.replace('.', '')            # 14 decimal digits
	flags          = (_COMPLEX_FLAG if complex_flag else 0x00) | (0x80 if negative else 0x00)
	bcd            = bytes(int(digits[i:i+2], 16) for i in range(0, 14, 2))
	return bytes([flags, int(exp_str) + 0x80]) + bcd


def _format_real(v: float) -> str:
	return str(int(v)) if float(v).is_integer() else repr(v)


def _format_value(v: float | complex) -> str:
	if not isinstance(v, complex):
		return _format_real(v)
	sign = '-' if v.imag < 0 else '+'
	return f'{_format_real(v.real)}{sign}{_format_real(abs(v.imag))}i'


# A real list (.8xl) carries var type 0x01 and a 9-byte TI real per element; a
# complex list carries var type 0x0D and 18 bytes per element (real then imaginary
# TI real).  All elements of a complex list are 18 bytes — a value with no
# imaginary part is stored as x + 0i.
_LIST_TYPE_REAL    = 0x01
_LIST_TYPE_COMPLEX = 0x0D


@dataclass
class ListFile(TiFile):
	name:     str
	values:   list[float | complex]
	comment:  str  = ''
	archived: bool = False
	version:  int  = 0x00

	@property
	def is_complex(self) -> bool:
		return any(isinstance(v, complex) for v in self.values)

	def __repr__(self):
		kind = 'complex' if self.is_complex else 'values'
		return f"L{self.name}({kind}={len(self.values)};{'' if self.archived else 'un'}archived)"

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"LIST:{self.name} ({'' if self.archived else 'un'}archived)")
		print('{' + ', '.join(_format_value(v) for v in self.values) + '}')

	@classmethod
	def read_from(cls, f):
		file_type, name_bytes, archived, comment, version = _read_var_header(f)
		count = int.from_bytes(f.read(2), 'little')
		if file_type == _LIST_TYPE_COMPLEX:
			values = [complex(_decode_ti_real(f.read(9)), _decode_ti_real(f.read(9)))
			          for _ in range(count)]
		else:
			values = [_decode_ti_real(f.read(9)) for _ in range(count)]

		return cls(
			name     = _decode_list_name(name_bytes),
			values   = values,
			comment  = comment,
			archived = archived,
			version  = version,
		)

	def write_to(self, f):
		name_bytes = _encode_list_name(self.name)
		if self.is_complex:
			elements  = b''.join(_encode_ti_real(z.real, complex_flag=True)
			                     + _encode_ti_real(z.imag, complex_flag=True)
			                     for z in map(complex, self.values))
			file_type = _LIST_TYPE_COMPLEX
		else:
			elements  = b''.join(_encode_ti_real(v) for v in self.values)
			file_type = _LIST_TYPE_REAL
		body = len(self.values).to_bytes(2, 'little') + elements  # element count + data
		_write_var_file(f, file_type, name_bytes, self.archived, self.comment, body, version=self.version)


def _decode_list_name(name_bytes: bytes) -> str:
	if name_bytes[:1] != bytes([_LIST_NAME_TOKEN]):
		return name_bytes.rstrip(b'\x00').decode('ascii', errors='replace')
	# After the 0x5D token a built-in list L1-L6 stores a single index byte
	# 0x00..0x05 — note L1's index is 0x00, so it must not be rstripped as if it
	# were padding.  A user list instead stores ASCII, whose first byte is always
	# a letter (>= 0x41), so the 0x00..0x05 range cleanly tells the two apart.
	if name_bytes[1] <= 0x05:
		return str(name_bytes[1])  # 0x00→"0" (L₁), 0x01→"1" (L₂), ..., 0x05→"5" (L₆)
	return name_bytes[1:].rstrip(b'\x00').decode('ascii', errors='replace')


def _encode_list_name(name: str) -> bytes:
	# Built-in lists L₁–L₆ are named "0".."5" (matching the raw index byte);
	# they store that byte directly after the 0x5D prefix.  User lists store
	# their ASCII name.  "0".."5" is unambiguous because user list must start
	# with a letter
	if name.isdigit():
		body = bytes([int(name)])
	else:
		body = name.upper().encode('ascii')[:5]
	return (bytes([_LIST_NAME_TOKEN]) + body).ljust(8, b'\x00')


# ── Number variables (.8xn real / .8xc complex) ────────────────────────────────
# The letter variables A–Z, θ.  A real variable carries var type 0x00 and a single
# 9-byte TI real; a complex variable carries var type 0x0C and 18 bytes (real then
# imaginary TI real, each with the 0x0C flag).  Unlike lists, the body has no count
# prefix — it is just the one value.  (Compare list types: 0x01 real, 0x0D complex.)
_VAR_TYPE_REAL    = 0x00
_VAR_TYPE_COMPLEX = 0x0C

# θ is a number variable like A–Z, but its name byte is 0x5D's neighbour 0x5B —
# which is also ASCII '[', so it needs decoding to/from the real glyph.
_THETA           = 'θ'
_THETA_NAME_BYTE = 0x5B


def _decode_var_name(name_bytes: bytes) -> str:
	name = name_bytes.rstrip(b'\x00')
	if name == bytes([_THETA_NAME_BYTE]):
		return _THETA
	return name.decode('ascii', errors='replace')


def _encode_var_name(name: str) -> bytes:
	if name in (_THETA, _THETA.upper()):   # accept lower θ or upper Θ
		body = bytes([_THETA_NAME_BYTE])
	else:
		body = name.upper().encode('ascii')[:8]
	return body.ljust(8, b'\x00')


@dataclass
class VariableFile(TiFile):
	name:     str
	value:    float | complex
	comment:  str  = ''
	archived: bool = False
	version:  int  = 0x00

	@property
	def is_complex(self) -> bool:
		return isinstance(self.value, complex)

	def __repr__(self):
		return f"var{self.name}={_format_value(self.value)}({'' if self.archived else 'un'}archived)"

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({'' if self.archived else 'un'}archived)")
		print(_format_value(self.value))

	@classmethod
	def read_from(cls, f):
		file_type, name_bytes, archived, comment, version = _read_var_header(f)
		if file_type == _VAR_TYPE_COMPLEX:
			value = complex(_decode_ti_real(f.read(9)), _decode_ti_real(f.read(9)))
		else:
			value = _decode_ti_real(f.read(9))

		return cls(
			name     = _decode_var_name(name_bytes),
			value    = value,
			comment  = comment,
			archived = archived,
			version  = version,
		)

	def write_to(self, f):
		name_bytes = _encode_var_name(self.name)
		if self.is_complex:
			body      = (_encode_ti_real(self.value.real, complex_flag=True)
			             + _encode_ti_real(self.value.imag, complex_flag=True))
			file_type = _VAR_TYPE_COMPLEX
		else:
			body      = _encode_ti_real(self.value)
			file_type = _VAR_TYPE_REAL
		_write_var_file(f, file_type, name_bytes, self.archived, self.comment, body, version=self.version)


# ── Matrix files (.8xm) ─────────────────────────────────────────────────────────
# A matrix [A]–[J] holds only real numbers (var type 0x02).  The name is the 0x5C
# matrix-name token plus an index byte ([A]→0x00 … [J]→0x09).  The body is two
# dimension bytes — COLUMNS then ROWS, in that (TI) order — followed by rows×cols
# 9-byte TI reals in row-major order (row 1 left-to-right, then row 2, …).  (Note:
# the format spec claims elements may be complex, but matrices are real-only.)
_MATRIX_NAME_TOKEN = 0x5C


def _decode_matrix_name(name_bytes: bytes) -> str:
	return chr(ord('A') + name_bytes[1])  # 0x00→"A", …, 0x09→"J"


def _encode_matrix_name(name: str) -> bytes:
	index = ord(name.upper()) - ord('A')
	return bytes([_MATRIX_NAME_TOKEN, index]).ljust(8, b'\x00')


@dataclass
class MatrixFile(TiFile):
	name:     str
	values:   list[list[float]]   # row-major: values[row][col]
	comment:  str  = ''
	archived: bool = False
	version:  int  = 0x00

	@property
	def rows(self) -> int:
		return len(self.values)

	@property
	def cols(self) -> int:
		return len(self.values[0]) if self.values else 0

	def __repr__(self):
		return f"matrix[{self.name}]({self.rows}x{self.cols};{'' if self.archived else 'un'}archived)"

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"[{self.name}] ({self.rows}x{self.cols}, {'' if self.archived else 'un'}archived)")
		for row in self.values:
			print('[' + ' '.join(_format_real(v) for v in row) + ']')

	@classmethod
	def read_from(cls, f):
		_file_type, name_bytes, archived, comment, version = _read_var_header(f)
		cols   = f.read(1)[0]
		rows   = f.read(1)[0]
		values = [[_decode_ti_real(f.read(9)) for _ in range(cols)] for _ in range(rows)]

		return cls(
			name     = _decode_matrix_name(name_bytes),
			values   = values,
			comment  = comment,
			archived = archived,
			version  = version,
		)

	def write_to(self, f):
		name_bytes = _encode_matrix_name(self.name)
		reals      = b''.join(_encode_ti_real(v) for row in self.values for v in row)
		body       = bytes([self.cols, self.rows]) + reals  # cols, rows, then row-major reals
		_write_var_file(f, 0x02, name_bytes, self.archived, self.comment, body, version=self.version)


# ── Strings (.8xs) and equations (.8xy) ──────────────────────────────────────────
# Both store their body exactly like a program — a 2-byte length-prefixed token
# stream — and differ only in the var type byte (string 0x04, equation 0x03) and in
# how the 8-byte name field encodes the variable name.  In both cases the name is
# itself a token: string names are 0xAA + index (Str1..Str0); equation names are
# 0x5E + code (Y₁..Y₀, X₁ₜ.., r₁.., 𝑢/𝑣/𝑤).  So the catalog's own text for that
# token is the canonical name, and both directions are just a lookup.

# The 8-byte name field of a string or equation is a single 2-byte token: 0xAA +
# index for Str1..Str0, and 0x5E + code for Y₁.., X₁ₜ.., r₁.., 𝑢/𝑣/𝑤.  get_token
# turns the stored code straight into that name; _NAME_TO_CODE is the small reverse
# map (these specific codes only) used for writing.
_NAME_TOKEN_CODES = (
	*range(0xAA00, 0xAA0A),   # Str1..Str0
	*range(0x5E10, 0x5E1A),   # Y₁..Y₀
	*range(0x5E20, 0x5E2C),   # X₁ₜ/Y₁ₜ .. X₆ₜ/Y₆ₜ
	*range(0x5E40, 0x5E46),   # r₁..r₆
	*range(0x5E80, 0x5E83),   # 𝑢, 𝑣, 𝑤
)
_NAME_TO_CODE = {get_token(code).text: code for code in _NAME_TOKEN_CODES}


def _decode_token_name(name_bytes: bytes) -> str:
	return get_token(int.from_bytes(name_bytes[:2], 'big')).text


def _encode_token_name(name: str) -> bytes:
	return _NAME_TO_CODE[name].to_bytes(2, 'big').ljust(8, b'\x00')


@dataclass
class TokenVarFile(TiFile):
	"""Base for variables whose body is a length-prefixed token stream and whose
	name is a single token (strings and equations).  A subclass only sets _VAR_TYPE
	(the var type byte)."""
	name:     str
	tokens:   list[Token]
	comment:  str  = ''
	archived: bool = False
	version:  int  = 0x00

	@property
	def text(self) -> str:
		"""The token stream as text — a string's contents or an equation's formula."""
		return ''.join(t.text for t in self.tokens)

	def __repr__(self):
		return f"{type(self).__name__}({self.name!r}, {len(self.tokens)} tokens)"

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({'' if self.archived else 'un'}archived)")
		print(self.text)

	@classmethod
	def read_from(cls, f):
		_file_type, name_bytes, archived, comment, version = _read_var_header(f)
		tokens = _read_token_stream(f)
		return cls(_decode_token_name(name_bytes), tokens, comment, archived, version)

	def write_to(self, f):
		_write_var_file(f, self._VAR_TYPE, _encode_token_name(self.name),
		                self.archived, self.comment, _token_stream_body(self.tokens),
		                version=self.version)


class StringFile(TokenVarFile):
	_VAR_TYPE = 0x04


class EquationFile(TokenVarFile):
	_VAR_TYPE = 0x03


# ── Picture files (.8xi) ───────────────────────────────────────────────────────
# A picture is the 96-wide graph bitmap stored bit-packed MSB-first, 12 bytes per
# row, wrapped in the usual 2-byte length prefix.  Two heights occur in the wild:
# 63 rows (756 bytes, 0x02F4) stores just the graph-screen height, while 64 rows
# (768 bytes, 0x0300) also stores the extra bottom LCD row.  Which one a file uses
# is a per-picture property — both sizes turn up mixed within a single calculator's
# backup and the OS reads either — so the row count is captured (PictureFile.rows)
# and writes reproduce the source size.  Pictures are the ten fixed variables
# Pic1–Pic9, Pic0, named by a 0x60 token plus an index byte (Pic1→0x00, …,
# Pic9→0x08, Pic0→0x09).

_PIC_NAME_TOKEN = 0x60
_PIC_ROW_BYTES  = COLS // 8   # 12 bytes per row (96 columns, 8 pixels/byte)


def _read_picture_data(f, rows: int) -> Bitmap:
	"""Unpack `rows` bit-packed scanlines into a fresh 96×64 Bitmap (MSB-first).  A
	picture storing fewer than ROWS rows leaves the missing bottom rows off."""
	bmp = Bitmap()
	for row in range(rows):
		rowbytes = f.read(_PIC_ROW_BYTES)
		buf = bmp.buffer[row]
		for col in range(COLS):
			buf[col] = (rowbytes[col >> 3] >> (7 - (col & 7))) & 1
	return bmp


def _encode_picture_data(bmp: Bitmap, rows: int) -> bytes:
	"""Bit-pack the top `rows` scanlines of a Bitmap into the picture body."""
	out = bytearray()
	for row in range(rows):
		rowbytes = bytearray(_PIC_ROW_BYTES)
		buf = bmp.buffer[row]
		for col in range(COLS):
			if buf[col]:
				rowbytes[col >> 3] |= 0x80 >> (col & 7)
		out += rowbytes
	return bytes(out)


def _decode_pic_name(name_bytes: bytes) -> str:
	"""Picture index byte → picture number: 0x00→"1", …, 0x08→"9", 0x09→"0"."""
	return str((name_bytes[1] + 1) % 10)


def _encode_pic_name(name: str) -> bytes:
	"""Picture number → 0x60 token + index byte: "1"→0x00, …, "9"→0x08, "0"→0x09."""
	index = (int(name) - 1) % 10
	return bytes([_PIC_NAME_TOKEN, index]).ljust(8, b'\x00')


@dataclass
class PictureFile(TiFile):
	name:     str
	bitmap:   Bitmap
	comment:  str  = ''
	archived: bool = False
	version:  int  = 0x00
	rows:     int  = ROWS   # scanlines stored: 64 (full LCD) or 63 (graph screen only)

	def __repr__(self):
		return f"Pic{self.name}({COLS}x{self.rows};{'' if self.archived else 'un'}archived)"

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"Pic{self.name} ({COLS}x{self.rows}, {'' if self.archived else 'un'}archived)")
		self.bitmap.disp()

	@classmethod
	def read_from(cls, f):
		_file_type, name_bytes, archived, comment, version = _read_var_header(f)
		pixel_len = int.from_bytes(f.read(2), 'little')   # body's 2-byte length prefix
		rows      = pixel_len // _PIC_ROW_BYTES           # 64 (full LCD) or 63 (graph screen)
		bitmap    = _read_picture_data(f, rows)

		return cls(
			name     = _decode_pic_name(name_bytes),
			bitmap   = bitmap,
			comment  = comment,
			archived = archived,
			version  = version,
			rows     = rows,
		)

	def write_to(self, f):
		name_bytes = _encode_pic_name(self.name)
		pixels     = _encode_picture_data(self.bitmap, self.rows)
		body       = len(pixels).to_bytes(2, 'little') + pixels  # pixel-byte count + data
		_write_var_file(f, 0x07, name_bytes, self.archived, self.comment, body, version=self.version)


if __name__ == '__main__':
	import sys

	_READERS = {
		'.8xp': ProgramFile,
		'.8xl': ListFile,
		'.8xm': MatrixFile,
		'.8xn': VariableFile,
		'.8xc': VariableFile,
		'.8xs': StringFile,
		'.8xy': EquationFile,
		'.8xi': PictureFile,
	}

	for path in sys.argv[1:]:
		reader = _READERS.get(path[-4:].lower())
		if reader is None:
			print(f"tifile: unrecognized file type: {path}", file=sys.stderr)
			continue
		reader.load(path).print()

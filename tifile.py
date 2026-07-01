"""Read and write TI-83/84 variable files: programs (.8xp), lists (.8xl), number
variables (.8xn real / .8xc complex), matrices (.8xm), strings (.8xs), equations
(.8xy), and pictures (.8xi).

Every TI variable file shares one envelope — an 8-byte signature, a 42-byte
comment, a variable-entry header (type, name, archive flag), and a trailing
checksum — wrapping a type-specific body.  A file is modelled as:

  * an `accessor` — the variable's storage location (a catalog VariableToken for
    A–Z/[A]/Str1/Y1/L1, a `UserList`, a `ProgramAccessor`, or a `PicAccessor`).  It
    supplies the name and, via `_set`, where `store_to` installs the value.
  * a `value` — the runtime model (a number, TiList, TiMatrix, TiString,
    TiEquation, token list, or Bitmap).

The base `TiFile` owns the envelope (read_from/write_to), the name, load/write, and
store_to; each subclass supplies only its body via `_parse_body`/`_encode_body`/
`_var_type`.
"""
from io import BytesIO

from tokenbase import Token, Accessor
from catalog import get_token, read_token
from bitmap import Bitmap, ROWS, COLS
from core import TiList, TiMatrix, TiString, TiEquation
from environment import UserList


# 0x5D is TI's "list name" token: a named list's 8-byte name field is this byte
# followed by up to 5 ASCII characters (built-in L1-L6 use 0x5D + 0x00..0x05).
_LIST_NAME_TOKEN = 0x5D


# ── Accessors: a variable file's name *is* a storage location ───────────────────
# Most variables name themselves with a catalog token that is already an Accessor
# (a VariableToken — including pictures, see PicToken).  User lists carry a UserList;
# only programs (no token, not an expression value) need an accessor here.
# read_accessor/write_accessor are the file layer's name codec — translating the
# 8-byte name field to/from an accessor.

class ProgramAccessor(Accessor):
	"""A program prgmNAME — a slot in env.programs keyed by name.  Programs aren't
	expression values, so they have no token; a file load installs one via `_set`,
	wrapping its token list in a Program."""

	def __init__(self, name: str):
		self.name = name

	def _get(self, env):
		return env.programs.get(self.name)

	def _set(self, env, tokens):
		from program import Program
		env.programs[self.name] = Program(tokens, self.name)

	def is_invokable(self) -> bool:
		return False

	def __repr__(self):
		return f"ProgramAccessor({self.name!r})"


def read_accessor(name_bytes: bytes) -> Accessor:
	"""Decode an 8-byte name field (everything except a program) into its accessor."""
	if name_bytes[0] == _LIST_NAME_TOKEN:
		# 0x5D + index 0x00..0x05 is a built-in list (its catalog token is the
		# accessor); 0x5D + ASCII is a user list (whose name starts with a letter).
		if name_bytes[1] <= 0x05:
			return get_token(_LIST_NAME_TOKEN << 8 | name_bytes[1])
		return UserList(name_bytes[1:].rstrip(b'\x00').decode('ascii', errors='replace'))

	return read_token(BytesIO(name_bytes))   # a VariableToken (var/matrix/string/equation/picture)


def read_program_name(name_bytes: bytes) -> ProgramAccessor:
	return ProgramAccessor(name_bytes.rstrip(b'\x00').decode('ascii', errors='replace'))


def write_accessor(acc: Accessor) -> bytes:
	"""Encode an accessor back into the 8-byte, null-padded name field."""
	if isinstance(acc, UserList):
		return (bytes([_LIST_NAME_TOKEN]) + acc.name.upper().encode('ascii')[:5]).ljust(8, b'\x00')
	if isinstance(acc, ProgramAccessor):
		return acc.name.upper().encode('ascii')[:8].ljust(8, b'\x00')
	return acc.code_to_bytes().ljust(8, b'\x00')   # a Token: var / matrix / string / equation / picture / built-in list


# ── Shared envelope ─────────────────────────────────────────────────────────────

def _read_var_header(f):
	"""Read the shared .8x* envelope and leave `f` at the start of the variable
	data (the body's own length/count prefix).

	Returns (file_type, name_bytes, archived, comment, version).  `name_bytes` is
	the raw 8-byte name field.  `version` is the source file's var-version byte,
	captured so writers can reproduce it byte-for-byte.
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
	"""Base for every TI variable file.

	Holds an `accessor` (the variable's storage location / name) and a `value` (its
	runtime model), plus the shared envelope metadata.  This base owns the envelope
	read/write flow, the name, the load/write path wrappers, and store_to; a subclass
	supplies only its body via `_parse_body` (read), `_encode_body` (write), and
	`_var_type`.  Programs and pictures override read_from for their extra field
	(locked / rows).
	"""

	DEFAULT_VERSION = 0x00

	def __init__(self, accessor, value, comment='', archived=False, version=None):
		self.accessor = accessor
		self.value    = value
		self.comment  = comment
		self.archived = archived
		self.version  = self.DEFAULT_VERSION if version is None else version

	@property
	def name(self) -> str:
		acc = self.accessor
		return acc.text if isinstance(acc, Token) else acc.name

	def __repr__(self):
		return f"{type(self).__name__}({self.name!r})"

	# ── path wrappers ──────────────────────────────────────────────────────────
	@classmethod
	def load(cls, file):
		with open(file, 'rb') as f:
			return cls.read_from(f)

	def write(self, file):
		with open(file, 'wb') as f:
			self.write_to(f)

	# ── envelope read/write (subclasses fill in the body hooks) ────────────────
	@classmethod
	def read_from(cls, f):
		file_type, name_bytes, archived, comment, version = _read_var_header(f)
		return cls(cls._read_accessor(name_bytes), cls._parse_body(f, file_type),
		           comment, archived, version)

	def write_to(self, f):
		_write_var_file(f, self._var_type(), write_accessor(self.accessor),
		                self.archived, self.comment, self._encode_body(), version=self.version)

	def store_to(self, env):
		"""Install this variable into a running environment.  A file load isn't a
		Store command, so it uses the accessor's raw `_set` — bypassing store()'s
		validation/side-effects and working even for write-protected accessors
		(pictures)."""
		self.accessor._set(env, self.value)

	@staticmethod
	def _read_accessor(name_bytes):
		return read_accessor(name_bytes)


# ── Token-stream body (programs, strings, equations) ────────────────────────────

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


# ── Programs (.8xp) ─────────────────────────────────────────────────────────────

class ProgramFile(TiFile):
	DEFAULT_VERSION = 0x01   # 0x01 is TI-Connect's default for fresh programs

	def __init__(self, accessor, value, comment='', archived=False, locked=False, version=None):
		if isinstance(accessor, str):
			accessor = ProgramAccessor(accessor)
		super().__init__(accessor, value, comment, archived, version)
		self.locked = locked

	@property
	def tokens(self) -> list[Token]:
		return self.value

	@staticmethod
	def _read_accessor(name_bytes):
		return read_program_name(name_bytes)

	@classmethod
	def read_from(cls, f):
		file_type, name_bytes, archived, comment, version = _read_var_header(f)
		return cls(cls._read_accessor(name_bytes), _read_token_stream(f),
		           comment, archived, locked=(file_type == 0x06), version=version)

	def _var_type(self):
		return 0x06 if self.locked else 0x05

	def _encode_body(self):
		return _token_stream_body(self.value)

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({'' if self.archived else 'un'}archived/{'' if self.locked else 'un'}locked)")
		print(''.join(t.text for t in self.value))


# ── TI reals & numeric formatting ───────────────────────────────────────────────
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


def _decode_real_body(f, file_type, complex_type) -> float | complex:
	"""Read one element — a 9-byte real, or (when file_type marks complex) two
	9-byte reals as real+imaginary."""
	if file_type == complex_type:
		return complex(_decode_ti_real(f.read(9)), _decode_ti_real(f.read(9)))
	return _decode_ti_real(f.read(9))


def _encode_real_body(value: float | complex) -> bytes:
	"""Encode one element: a real as 9 bytes, a complex as 18 (each part flagged)."""
	if isinstance(value, complex):
		return (_encode_ti_real(value.real, complex_flag=True)
		        + _encode_ti_real(value.imag, complex_flag=True))
	return _encode_ti_real(value)


# ── Number variables (.8xn real / .8xc complex) ────────────────────────────────
# A real variable carries var type 0x00 and a single 9-byte TI real; a complex
# variable carries var type 0x0C and 18 bytes.  Unlike lists, the body has no count
# prefix — it is just the one value.
_VAR_TYPE_REAL    = 0x00
_VAR_TYPE_COMPLEX = 0x0C


class VariableFile(TiFile):
	@classmethod
	def _parse_body(cls, f, file_type):
		return _decode_real_body(f, file_type, _VAR_TYPE_COMPLEX)

	def _var_type(self):
		return _VAR_TYPE_COMPLEX if self.is_complex else _VAR_TYPE_REAL

	def _encode_body(self):
		return _encode_real_body(self.value)

	@property
	def is_complex(self) -> bool:
		return isinstance(self.value, complex)

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({'' if self.archived else 'un'}archived)")
		print(_format_value(self.value))


# ── Lists (.8xl real / complex) ─────────────────────────────────────────────────
# A real list carries var type 0x01 and a 9-byte TI real per element; a complex
# list carries var type 0x0D and 18 bytes per element (real then imaginary).  All
# elements of a complex list are 18 bytes — a value with no imaginary part is x+0i.
_LIST_TYPE_REAL    = 0x01
_LIST_TYPE_COMPLEX = 0x0D


class ListFile(TiFile):
	@classmethod
	def _parse_body(cls, f, file_type):
		count = int.from_bytes(f.read(2), 'little')
		return TiList([_decode_real_body(f, file_type, _LIST_TYPE_COMPLEX) for _ in range(count)])

	def _var_type(self):
		return _LIST_TYPE_COMPLEX if self.value.is_complex else _LIST_TYPE_REAL

	def _encode_body(self):
		data = b''.join(_encode_real_body(v) for v in self.value.data)
		return len(self.value).to_bytes(2, 'little') + data

	@property
	def is_complex(self) -> bool:
		return self.value.is_complex

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({'' if self.archived else 'un'}archived)")
		print('{' + ', '.join(_format_value(v) for v in self.value) + '}')


# ── Matrix files (.8xm) ─────────────────────────────────────────────────────────
# A matrix [A]–[J] holds only real numbers (var type 0x02).  The body is two
# dimension bytes — COLUMNS then ROWS, in that (TI) order — followed by rows×cols
# 9-byte TI reals in row-major order (row 1 left-to-right, then row 2, …).  (Note:
# the format spec claims elements may be complex, but matrices are real-only.)

class MatrixFile(TiFile):
	@classmethod
	def _parse_body(cls, f, file_type):
		cols = f.read(1)[0]
		rows = f.read(1)[0]
		return TiMatrix([[_decode_ti_real(f.read(9)) for _ in range(cols)] for _ in range(rows)])

	def _var_type(self):
		return 0x02

	def _encode_body(self):
		m = self.value
		reals = b''.join(_encode_ti_real(v) for row in m.data for v in row)
		return bytes([m.cols, m.rows]) + reals  # cols, rows, then row-major reals

	@property
	def rows(self) -> int:
		return self.value.rows

	@property
	def cols(self) -> int:
		return self.value.cols

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({self.rows}x{self.cols}, {'' if self.archived else 'un'}archived)")
		for row in self.value.data:
			print('[' + ' '.join(_format_real(v) for v in row) + ']')


# ── Strings (.8xs) and equations (.8xy) ──────────────────────────────────────────
# Both store their body exactly like a program — a 2-byte length-prefixed token
# stream — and differ only in the var type byte (string 0x04, equation 0x03).

class TokenVarFile(TiFile):
	"""Base for variables whose body is a length-prefixed token stream (strings and
	equations).  A subclass sets _VAR_TYPE (the var type byte) and _MODEL (the model
	wrapping the tokens)."""

	@classmethod
	def _parse_body(cls, f, file_type):
		return cls._MODEL(_read_token_stream(f))

	def _var_type(self):
		return self._VAR_TYPE

	def _encode_body(self):
		return _token_stream_body(self.value.tokens)

	@property
	def text(self) -> str:
		"""The token stream as text — a string's contents or an equation's formula."""
		return str(self.value)

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({'' if self.archived else 'un'}archived)")
		print(self.text)


class StringFile(TokenVarFile):
	_VAR_TYPE = 0x04
	_MODEL    = TiString


class EquationFile(TokenVarFile):
	_VAR_TYPE = 0x03
	_MODEL    = TiEquation


# ── Picture files (.8xi) ───────────────────────────────────────────────────────
# A picture is the 96-wide graph bitmap stored bit-packed MSB-first, 12 bytes per
# row, wrapped in the usual 2-byte length prefix.  Two heights occur in the wild:
# 63 rows (756 bytes, 0x02F4) stores just the graph-screen height, while 64 rows
# (768 bytes, 0x0300) also stores the extra bottom LCD row.  Which one a file uses
# is a per-picture property — both sizes turn up mixed within a single calculator's
# backup and the OS reads either — so the row count is captured (PictureFile.rows)
# and writes reproduce the source size.

_PIC_ROW_BYTES = COLS // 8   # 12 bytes per row (96 columns, 8 pixels/byte)


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


class PictureFile(TiFile):
	def __init__(self, accessor, value, comment='', archived=False, version=None, rows=ROWS):
		super().__init__(accessor, value, comment, archived, version)
		self.rows = rows   # scanlines stored: 64 (full LCD) or 63 (graph screen only)

	@property
	def bitmap(self) -> Bitmap:
		return self.value

	@classmethod
	def read_from(cls, f):
		_file_type, name_bytes, archived, comment, version = _read_var_header(f)
		pixel_len = int.from_bytes(f.read(2), 'little')   # body's 2-byte length prefix
		rows      = pixel_len // _PIC_ROW_BYTES           # 64 (full LCD) or 63 (graph screen)
		return cls(cls._read_accessor(name_bytes), _read_picture_data(f, rows),
		           comment, archived, version, rows=rows)

	def _var_type(self):
		return 0x07

	def _encode_body(self):
		pixels = _encode_picture_data(self.value, self.rows)
		return len(pixels).to_bytes(2, 'little') + pixels  # pixel-byte count + data

	def print(self):
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({COLS}x{self.rows}, {'' if self.archived else 'un'}archived)")
		self.value.disp()


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

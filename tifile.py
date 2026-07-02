"""Read and write TI-83/84 variable files: programs (.8xp), lists (.8xl), number
variables (.8xn real / .8xc complex), matrices (.8xm), strings (.8xs), equations
(.8xy), and pictures (.8xi).

Every TI variable file shares one envelope — an 8-byte signature, a 42-byte
comment, a variable-entry header (type, name, archive flag), and a trailing
checksum — wrapping a type-specific body.  A file is modelled as:

  * an `accessor` — the variable's storage location: a FileVar Accessor.  A catalog
    token for token-named variables (A–Z/[A]/Str1/Y1/L1/Pic1), a `UserList`, or a
    `ProgramAccessor`.  Supplies name_bytes for writing and set() for store_to.
    Constructors also take the variable's natural name — in its natural type, no
    alternate spellings (`_named_accessor`): StringFile(4) is Str4, MatrixFile('A')
    is [A], VariableFile('Z') is Z, ListFile(3) is L₃ while ListFile('CW') is ʟCW.
    Equations, whose names aren't typeable, take only an accessor
    (EquationFile(FuncEquationToken(0)) is Y₁ — tokens compare by code, so a fresh
    instance is as good as the catalog's).
  * a `value` — the runtime model (a number, TiList, TiMatrix, TiString,
    TiEquation, token list, or Bitmap).

The base `TiFile` owns the envelope (read_from/write_to), the name, load/write, and
store_to; each subclass supplies only its body via `_parse_body`/`_encode_body`/
`_var_type`.
"""
from io import BytesIO

from tokenbase import Token, Accessor, FileVar
from tokentypes import LetterToken, MatrixToken, ListToken, StringToken, PicToken
from catalog import get_token, read_token
from bitmap import Bitmap, ROWS, COLS
from core import TiList, TiMatrix, TiString, TiEquation, str_to_tokens
from environment import UserList
from program import Program


# 0x5D is TI's "list name" token: a named list's 8-byte name field is this byte
# followed by up to 5 ASCII characters (built-in L1-L6 use 0x5D + 0x00..0x05).
_LIST_NAME_TOKEN = 0x5D


# ── Accessors: a variable file's name *is* a storage location ───────────────────
# Most variables name themselves with a catalog token that is already a FileVar
# Accessor (a TokenFileVar — including pictures, see PicToken).  User lists carry a
# UserList; only programs (no token, not an expression value) need an accessor here.
# read_accessor/write_accessor are the file layer's name codec — translating the
# 8-byte name field to/from an accessor; the fixed-field padding lives entirely in
# this pair (ljust on write, rstrip on read), never in the accessors.

class ProgramAccessor(FileVar, Accessor):
	"""A program prgmNAME — a slot in env.programs keyed by name.  Programs aren't
	expression values, so they have no token; a file load installs one via `set`,
	wrapping its token list in a Program."""

	def __init__(self, name: str):
		if not 1 <= len(name) <= 8:
			# The calculator's own limit; checked at construction rather than
			# silently truncated at write time.
			raise ValueError(f"Program name must be 1-8 characters: {name!r}")
		self.name = name

	def set(self, env, tokens):
		env.programs[self.name] = Program(tokens, self.name)

	def name_tokens(self) -> list:
		return str_to_tokens(self.name)

	def name_bytes(self) -> bytes:
		return self.name.upper().encode('ascii')   # uppercased in the binary


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


def write_accessor(acc: FileVar) -> bytes:
	"""Encode an accessor into its 8-byte, null-padded name field.  Each accessor
	supplies its meaningful bytes (Token → code bytes, UserList → 0x5D + ASCII,
	program → ASCII); padding to the VAT's fixed 8-byte field is a fact about the
	file format, not the variable, so it happens here — the write-side half of the
	name codec, mirroring the rstrip on read."""
	field = acc.name_bytes()
	assert len(field) <= 8, f"name field too long: {field!r}"   # accessors validate names
	return field.ljust(8, b'\x00')


def _digit_slot(digit) -> int:
	"""A digit-named variable's display digit (1..9, 0 = the tenth) → its 0-based
	slot index (Str4 → 3, Pic0 → 9).  The one place this conversion lives — the
	token classes take slot indices, the TiFile API takes the visible digit."""
	if isinstance(digit, int) and 0 <= digit <= 9:
		return (digit - 1) % 10
	raise ValueError(f"Expected a variable digit 0-9, got {digit!r}")


# ── Shared envelope ─────────────────────────────────────────────────────────────

def _read_var_header(f):
	"""Read and verify the shared .8x* envelope, leaving `f` at the start of the
	variable data (the body's own length/count prefix).

	Returns (file_type, name_bytes, archived, comment, version).  `name_bytes` is
	the raw 8-byte name field.  `version` is the source file's var-version byte,
	captured so writers can reproduce it byte-for-byte.

	The whole var entry is read up front (they cap out around the calculator's 64K
	of RAM) so the trailing checksum can be verified before anything is parsed —
	corrupt and truncated files fail here, not with a confusing token error later —
	and `f` is then rewound to the body for the caller.

	The entry's leading word is the length of the sub-header that follows it:
	0x0D (TI-83+/84+: version and archive-flag bytes present) or 0x0B (original
	TI-83: both absent — no Flash, so nothing could be archived).
	"""
	signature = f.read(8)
	if not signature.startswith(b'**TI8'):  # could be 82, 83, 83F
		raise ValueError(f"Invalid TI variable file signature: {signature!r}")

	f.seek(3, 1)  # skip 1a 0a 00
	comment = f.read(42).rstrip(b'\x00 ').decode('ascii', errors='replace')
	entry_len = int.from_bytes(f.read(2), 'little')
	entry_start = f.tell()
	entry = f.read(entry_len)
	if len(entry) < entry_len:
		raise ValueError(f"Truncated file: var entry is {len(entry)} of {entry_len} bytes")
	stored = int.from_bytes(f.read(2), 'little')
	computed = sum(entry) & 0xFFFF
	if stored != computed:
		raise ValueError(f"Checksum mismatch: file has 0x{stored:04X}, contents sum to 0x{computed:04X}")

	header_len = int.from_bytes(entry[0:2], 'little')
	file_type  = entry[4]
	name_bytes = entry[5:13]
	if header_len == 0x0D:
		version, flag = entry[13:15]
	elif header_len == 0x0B:
		version, flag = 0x00, 0x00
	else:
		raise ValueError(f"Unrecognized var-entry header length: 0x{header_len:04X}")
	# entry = header word (2) + sub-header (header_len) + data length duplicate (2) + body
	f.seek(entry_start + header_len + 4)
	return file_type, name_bytes, bool(flag & 0x80), comment, version


class TiFile:
	"""Base for every TI variable file.

	Holds an `accessor` (the variable's storage location / name) and a `value` (its
	runtime model), plus the shared envelope metadata.  This base owns the envelope
	read/write flow, the name, the load/write path wrappers, and store_to; a subclass
	supplies only its body via `_parse_body` (read), `_encode_body` (write), and
	`_var_type`.  Programs and pictures override read_from for their extra field
	(locked / rows).

	`version` is the var-version byte.  It varies per file (e.g. real .8xp files
	carry 0x00 or 0x03, not the 0x01 TI-Connect emits for fresh programs), so it is
	captured on read and passed back through to round-trip byte-for-byte.
	"""

	DEFAULT_VERSION = 0x00

	def __init__(self, accessor, value, comment='', archived=False, version=None):
		if not isinstance(accessor, FileVar):
			accessor = self._named_accessor(accessor)
		self.accessor = accessor
		self.value    = value
		self.comment  = comment
		self.archived = archived
		self.version  = self.DEFAULT_VERSION if version is None else version

	@classmethod
	def _named_accessor(cls, spec) -> FileVar:
		"""The accessor for a friendly variable name — each subclass accepts its
		family's natural spelling (StringFile(1) → Str1, MatrixFile('A') → [A],
		ListFile('CW') → ʟCW, …).  Always a *lookup* of the catalog singleton,
		never a fresh token: tokens are identity-compared throughout."""
		raise TypeError(f"{cls.__name__} cannot name a variable from {spec!r}; pass a FileVar accessor")

	@property
	def name(self) -> str:
		return str(TiString(self.accessor.name_tokens()))

	def __repr__(self):
		return f"{type(self).__name__}({self.name!r})"

	def print(self):
		"""Human-readable dump: comment, a name + summary header, then the body.
		Subclasses supply `_summary` extras and the body."""
		if self.comment:
			print(self.comment)
		print(f"{self.name} ({self._summary()})")
		self._print_body()

	def _summary(self) -> str:
		"""The parenthetical after the name; subclasses prepend/append their extras."""
		return ('' if self.archived else 'un') + 'archived'

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
		"""Write the complete .8x* file.  The entry header carries the body's length
		(twice) and the trailing checksum covers header+body, so the body must be
		encoded to bytes first — this format cannot be streamed."""
		body = self._encode_body()
		header = (
			b'\x0D\x00'                            # entry header type: 0x000D (includes version + flag)
			+ len(body).to_bytes(2, 'little')      # length of var data
			+ bytes([self._var_type()])            # 0x01 list, 0x05 program, 0x06 locked program
			+ write_accessor(self.accessor)        # variable name, null-padded to 8 bytes
			+ bytes([self.version])                # var version (round-tripped from read)
			+ bytes([0x80 if self.archived else 0x00])   # 0x80 = archived in flash, 0x00 = RAM
			+ len(body).to_bytes(2, 'little')      # length of var data (repeated)
		)
		f.write(b'**TI83F*')
		f.write(b'\x1a\x0a\x00')
		f.write(self.comment.encode('ascii')[:42].ljust(42, b'\x00'))
		f.write((len(header) + len(body)).to_bytes(2, 'little'))   # total var-entry length
		f.write(header)
		f.write(body)
		f.write(((sum(header) + sum(body)) & 0xFFFF).to_bytes(2, 'little'))

	def store_to(self, env):
		"""Install this variable into a running environment.  A file load isn't a
		Store command, so it uses the accessor's raw `set` — bypassing store()'s
		validation/side-effects and working even for write-protected accessors
		(pictures)."""
		self.accessor.set(env, self.value)

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
		super().__init__(accessor, value, comment, archived, version)
		self.locked = locked

	@staticmethod
	def _named_accessor(spec) -> ProgramAccessor:
		return ProgramAccessor(spec)

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

	def _summary(self) -> str:
		return f"{super()._summary()}/{'' if self.locked else 'un'}locked"

	def _print_body(self):
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

	@staticmethod
	def _named_accessor(spec):
		"""A letter variable: 'A'..'Z', or 'θ' (also spelled 'theta')."""
		s = str(spec)
		if len(s) == 1 and 'A' <= s <= 'Z':
			return LetterToken(ord(s) - ord('A'))
		if s in ('θ', 'theta'):
			return LetterToken(26)
		raise ValueError(f"Expected a variable letter A-Z or θ, got {spec!r}")

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

	def _print_body(self):
		print(_format_value(self.value))


# ── Lists (.8xl real / complex) ─────────────────────────────────────────────────
# A real list carries var type 0x01 and a 9-byte TI real per element; a complex
# list carries var type 0x0D and 18 bytes per element (real then imaginary).  All
# elements of a complex list are 18 bytes — a value with no imaginary part is x+0i.
_LIST_TYPE_REAL    = 0x01
_LIST_TYPE_COMPLEX = 0x0D


class ListFile(TiFile):
	@staticmethod
	def _named_accessor(spec):
		"""An int 1-6 is a built-in list (ListFile(3) is L₃); a string is the user
		list of that name (ListFile('CW') is ʟCW).  The *type* disambiguates, so a
		user list named 'L1' can't collide with a built-in."""
		if isinstance(spec, int):
			if not 1 <= spec <= 6:
				raise ValueError(f"Built-in lists are L1-L6, got {spec!r}")
			return ListToken(spec - 1)
		if isinstance(spec, str):
			return UserList(spec)
		raise TypeError(f"Expected a list digit or user-list name, got {spec!r}")

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

	def _print_body(self):
		print('{' + ', '.join(_format_value(v) for v in self.value) + '}')


# ── Matrix files (.8xm) ─────────────────────────────────────────────────────────
# A matrix [A]–[J] holds only real numbers (var type 0x02).  The body is two
# dimension bytes — COLUMNS then ROWS, in that (TI) order — followed by rows×cols
# 9-byte TI reals in row-major order (row 1 left-to-right, then row 2, …).  (Note:
# the format spec claims elements may be complex, but matrices are real-only.)

class MatrixFile(TiFile):
	@staticmethod
	def _named_accessor(spec):
		"""A matrix letter 'A'..'J' — the letter alone is the name (no brackets:
		the [] is display dressing, and matrices have no number on the calculator)."""
		s = str(spec)
		if len(s) == 1 and 'A' <= s <= 'J':
			return MatrixToken(ord(s) - ord('A'))
		raise ValueError(f"Expected a matrix letter A-J, got {spec!r}")

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

	def _summary(self) -> str:
		return f"{self.rows}x{self.cols}, {super()._summary()}"

	def _print_body(self):
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

	def _print_body(self):
		print(self.text)


class StringFile(TokenVarFile):
	_VAR_TYPE = 0x04
	_MODEL    = TiString

	@staticmethod
	def _named_accessor(spec):
		"""A string variable by digit: StringFile(4) is Str4, StringFile(0) is Str0."""
		return StringToken(_digit_slot(spec))


class EquationFile(TokenVarFile):
	# No _named_accessor: equations have no typed name on the calculator (they're
	# picked from a menu) and span four families — pass the token itself, e.g.
	# EquationFile(FuncEquationToken(0), …) for Y₁.
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

	@staticmethod
	def _named_accessor(spec):
		"""A picture by digit: PictureFile(4) is Pic4, PictureFile(0) is Pic0."""
		return PicToken(_digit_slot(spec))

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

	def _summary(self) -> str:
		return f"{COLS}x{self.rows}, {super()._summary()}"

	def _print_body(self):
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

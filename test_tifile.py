"""Tests for the TI variable file formats (read_from / write_to)."""
import os
import pytest
from io import BytesIO

from tifile import (
	ProgramFile, ListFile, PictureFile, VariableFile, MatrixFile,
	StringFile, EquationFile,
	ProgramAccessor, read_accessor, write_accessor,
)
from bitmap import Bitmap, ROWS, COLS
from core import TiList, TiMatrix, TiString, TiEquation
from catalog import get_token
from environment import Environment, UserList
from test_tibasic import toks


# A file's name is an Accessor; production code reads it from disk or carries the
# token directly.  Tests construct files by friendly name, so this reverse map
# (token text → accessor) is a test-only convenience.  Every named variable —
# including pictures (PicToken) — is a catalog token, which is its own accessor.
_NAME_TOKENS = {
	get_token(c).text: get_token(c)
	for c in (*range(0x41, 0x5C),       # A..Z, θ          (number variables)
	          *range(0x5C00, 0x5C0A),   # [A]..[J]         (matrices)
	          *range(0x5D00, 0x5D06),   # L₁..L₆           (built-in lists)
	          *range(0x6000, 0x600A),   # Pic1..Pic0       (pictures)
	          *range(0xAA00, 0xAA0A),   # Str1..Str0       (strings)
	          *range(0x5E10, 0x5E1A), *range(0x5E20, 0x5E2C),
	          *range(0x5E40, 0x5E46), *range(0x5E80, 0x5E83))   # equations
}


def tok(name: str):
	"""The accessor a file with this name carries — its catalog token (every named
	variable, including pictures via PicToken, is its own accessor)."""
	return _NAME_TOKENS[name]


# ── Helpers ───────────────────────────────────────────────────────────────────

def roundtrip(prog: ProgramFile) -> ProgramFile:
	"""Write prog to a BytesIO buffer, seek back, and read it out again."""
	buf = BytesIO()
	prog.write_to(buf)
	buf.seek(0)
	return ProgramFile.read_from(buf)


def make_prog(**kwargs) -> ProgramFile:
	"""Construct a ProgramFile with sensible defaults."""
	name   = kwargs.pop('name', 'TEST')
	tokens = kwargs.pop('tokens', [])
	return ProgramFile(name, tokens, **kwargs)


# ── Roundtrip: metadata ───────────────────────────────────────────────────────

class TestRoundtripMetadata:
	def test_name(self):
		assert roundtrip(make_prog(name='HELLO')).name == 'HELLO'

	def test_name_lowercase_normalised(self):
		# write() uppercases the name in the binary; read() decodes it as-is
		prog = ProgramFile('hello', [])
		assert roundtrip(prog).name == 'HELLO'

	def test_name_short(self):
		assert roundtrip(make_prog(name='A')).name == 'A'

	def test_comment(self):
		assert roundtrip(make_prog(comment='my comment')).comment == 'my comment'

	def test_comment_empty(self):
		assert roundtrip(make_prog(comment='')).comment == ''

	def test_comment_truncated_to_42(self):
		long = 'X' * 100
		result = roundtrip(make_prog(comment=long))
		assert result.comment == 'X' * 42

	def test_archived_true(self):
		assert roundtrip(make_prog(archived=True)).archived is True

	def test_archived_false(self):
		assert roundtrip(make_prog(archived=False)).archived is False

	def test_locked_true(self):
		assert roundtrip(make_prog(locked=True)).locked is True

	def test_locked_false(self):
		assert roundtrip(make_prog(locked=False)).locked is False

	def test_archived_and_locked(self):
		result = roundtrip(make_prog(archived=True, locked=True))
		assert result.archived is True
		assert result.locked is True


# ── Roundtrip: tokens ─────────────────────────────────────────────────────────

class TestRoundtripTokens:
	def test_empty_program(self):
		assert roundtrip(make_prog(tokens=[])).tokens == []

	def test_single_digit(self):
		prog = make_prog(tokens=toks('3'))
		assert roundtrip(prog).tokens == toks('3')

	def test_single_letter(self):
		prog = make_prog(tokens=toks('A'))
		assert roundtrip(prog).tokens == toks('A')

	def test_arithmetic_expression(self):
		program = toks('1+2*3')
		result = roundtrip(make_prog(tokens=program))
		assert result.tokens == program

	def test_two_byte_token(self):
		# stdDev( is a two-byte token (0xBB0D)
		program = toks('stdDev(')
		result = roundtrip(make_prog(tokens=program))
		assert result.tokens == program

	def test_mixed_one_and_two_byte_tokens(self):
		# cumSum( is two-byte (0xBB29); the digits are one-byte
		program = toks('cumSum( 10')
		result = roundtrip(make_prog(tokens=program))
		assert result.tokens == program

	def test_multiline_program(self):
		program = toks('A@B\nB+1')  # A→B (newline) B+1
		result = roundtrip(make_prog(tokens=program))
		assert result.tokens == program

	def test_token_count_preserved(self):
		program = toks('0123456789') * 5  # 50 tokens
		assert len(roundtrip(make_prog(tokens=program)).tokens) == 50


# ── Binary format: write_to ───────────────────────────────────────────────────

class TestWriteFormat:
	def _write(self, **kwargs) -> bytes:
		buf = BytesIO()
		make_prog(**kwargs).write_to(buf)
		return buf.getvalue()

	def test_signature(self):
		assert self._write().startswith(b'**TI83F*')

	def test_file_header_bytes(self):
		data = self._write()
		assert data[8:11] == b'\x1a\x0a\x00'

	def test_comment_field_is_42_bytes(self):
		data = self._write(comment='hi')
		# comment starts at byte 11
		assert len(data[11:53]) == 42
		assert data[11:13] == b'hi'
		assert data[13:53] == b'\x00' * 40

	def test_checksum_correct(self):
		buf = BytesIO()
		make_prog(tokens=toks('5')).write_to(buf)
		data = buf.getvalue()
		# var_entry starts at byte 55 (8 sig + 3 header + 42 comment + 2 length)
		var_entry = data[55:-2]
		expected = sum(var_entry) & 0xFFFF
		actual = int.from_bytes(data[-2:], 'little')
		assert actual == expected

	def test_archived_flag_byte(self):
		archived = self._write(archived=True)
		not_archived = self._write(archived=False)
		# flag byte is at a fixed offset within var_entry; just check they differ
		assert archived != not_archived

	def test_locked_file_type_byte(self):
		locked = self._write(locked=True)
		not_locked = self._write(locked=False)
		assert locked != not_locked


# ── read_from: error handling ─────────────────────────────────────────────────

class TestReadErrors:
	def test_bad_signature_raises(self):
		buf = BytesIO(b'BADSIG!!' + b'\x00' * 100)
		with pytest.raises(ValueError, match='signature'):
			ProgramFile.read_from(buf)

	def test_wrong_signature_prefix(self):
		# Starts with **TI but not **TI8x
		buf = BytesIO(b'**TIXX**' + b'\x00' * 100)
		with pytest.raises(ValueError):
			ProgramFile.read_from(buf)


# ── ListFile helpers ──────────────────────────────────────────────────────────

def roundtrip_list(lst: ListFile) -> ListFile:
	buf = BytesIO()
	lst.write_to(buf)
	buf.seek(0)
	return ListFile.read_from(buf)


def make_list(**kwargs) -> ListFile:
	# a built-in list name ("L₁".."L₆") resolves to its token; anything else is a user list
	name  = kwargs.pop('name', 'CW')
	value = TiList(kwargs.pop('values', []))
	acc   = _NAME_TOKENS.get(name) or UserList(name)
	return ListFile(acc, value, **kwargs)


# ── List name coding via read_accessor / write_accessor ───────────────────────

class TestListNameCoding:
	def test_decode_builtin(self):
		# 0x5D + index 0x00..0x05 → the built-in list's token (L₁..L₆)
		assert read_accessor(b'\x5d\x00' + b'\x00' * 6).text == 'L₁'
		assert read_accessor(b'\x5d\x05' + b'\x00' * 6).text == 'L₆'

	def test_decode_user_list(self):
		acc = read_accessor(b'\x5d\x43\x57' + b'\x00' * 5)
		assert isinstance(acc, UserList) and acc.name == 'CW'

	def test_decode_user_list_named_l1(self):
		# a user list literally named "L1" (0x5D + 'L','1'), distinct from built-in L₁
		acc = read_accessor(b'\x5d\x4c\x31' + b'\x00' * 5)
		assert isinstance(acc, UserList) and acc.name == 'L1'

	def test_encode_builtin(self):
		assert write_accessor(_NAME_TOKENS['L₁']) == b'\x5d\x00' + b'\x00' * 6
		assert write_accessor(_NAME_TOKENS['L₆']) == b'\x5d\x05' + b'\x00' * 6

	def test_encode_user_list(self):
		assert write_accessor(UserList('CW')) == b'\x5d\x43\x57' + b'\x00' * 5

	def test_encode_user_list_lowercase_normalised(self):
		assert write_accessor(UserList('cw')) == b'\x5d\x43\x57' + b'\x00' * 5

	# roundtrip consistency for all built-ins
	def test_all_builtins_roundtrip(self):
		for i in range(6):
			t = _NAME_TOKENS[f'L{chr(0x2080 + i + 1)}']   # L₁..L₆
			assert read_accessor(write_accessor(t)) is t


# ── ListFile roundtrip ────────────────────────────────────────────────────────

class TestListFileRoundtrip:
	def test_user_list_name(self):
		assert roundtrip_list(make_list(name='CW')).name == 'CW'

	def test_builtin_list_name_l1(self):
		assert roundtrip_list(make_list(name='L₁')).name == 'L₁'

	def test_builtin_list_name_l6(self):
		assert roundtrip_list(make_list(name='L₆')).name == 'L₆'

	def test_values_preserved(self):
		result = roundtrip_list(make_list(values=[1.0, 2.5, -3.0]))
		assert list(result.value) == pytest.approx([1.0, 2.5, -3.0])

	def test_empty_values(self):
		assert list(roundtrip_list(make_list(values=[])).value) == []

	def test_archived(self):
		assert roundtrip_list(make_list(archived=True)).archived is True

	def test_comment(self):
		assert roundtrip_list(make_list(comment='my list')).comment == 'my list'

	def test_version_preserved(self):
		assert roundtrip_list(make_list(version=0x24)).version == 0x24


# ── ListFile complex lists ────────────────────────────────────────────────────

class TestComplexListRoundtrip:
	def test_is_complex_flag(self):
		assert make_list(values=[1 + 2j]).value.is_complex is True
		assert make_list(values=[1.0, 2.0]).value.is_complex is False

	def test_complex_values_preserved(self):
		vals = [1 + 0j, 1j, 1 + 1j, -2 - 3j]
		result = roundtrip_list(make_list(values=vals))
		assert result.value.is_complex is True
		assert list(result.value) == pytest.approx(vals)

	def test_complex_negative_parts(self):
		# exercises the sign bit combined with the 0x0C complex flag
		result = roundtrip_list(make_list(values=[-1.5 - 2.5j]))
		assert list(result.value) == pytest.approx([-1.5 - 2.5j])

	def test_mixed_real_and_complex_promotes_whole_list(self):
		# a real value in a complex list is stored as x + 0i and reads back complex
		result = roundtrip_list(make_list(values=[3.0, 4j]))
		assert result.value.is_complex is True
		assert list(result.value) == pytest.approx([3 + 0j, 4j])

	def test_real_list_stays_real(self):
		result = roundtrip_list(make_list(values=[1.0, 2.0, 3.0]))
		assert result.value.is_complex is False
		assert all(not isinstance(v, complex) for v in result.value)


# ── ListFile real-file byte-exact roundtrip ───────────────────────────────────

_CPX_LIST_FILE = r'C:\Users\Max\Documents\MyTiData\Backups\TI84PlusSilverEdition_12\CPX.8xl'


def test_complex_list_from_file():
	if not os.path.exists(_CPX_LIST_FILE):
		pytest.skip('real file not found')
	lst = ListFile.load(_CPX_LIST_FILE)
	assert lst.name == 'CPX'
	assert lst.is_complex is True
	assert list(lst.value) == pytest.approx([1, 1j, 1 + 1j, 2 ** 0.5, 7 ** 0.5 * 1j, 0.69314718055994 + 3.1415926535898j])


def test_complex_list_byte_exact_roundtrip():
	if not os.path.exists(_CPX_LIST_FILE):
		pytest.skip('real file not found')
	orig = open(_CPX_LIST_FILE, 'rb').read()
	buf = BytesIO()
	ListFile.load(_CPX_LIST_FILE).write_to(buf)
	assert buf.getvalue() == orig

_BUILTIN_LIST_FILES = [
	(f'L{chr(0x2080 + i + 1)}', rf'C:\Users\Max\Documents\MyTiData\Backups\TI84Plus_1\L_{i+1}_.8xl')
	for i in range(6)   # L₁..L₆
]


@pytest.mark.parametrize('expected_name,path', _BUILTIN_LIST_FILES)
def test_builtin_list_name_from_file(expected_name, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	assert ListFile.load(path).name == expected_name


@pytest.mark.parametrize('expected_name,path', _BUILTIN_LIST_FILES)
def test_builtin_list_byte_exact_roundtrip(expected_name, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	orig = open(path, 'rb').read()
	lst = ListFile.load(path)
	buf = BytesIO()
	lst.write_to(buf)
	assert buf.getvalue() == orig


# ── PictureFile helpers ───────────────────────────────────────────────────────

def roundtrip_pic(pic: PictureFile) -> PictureFile:
	buf = BytesIO()
	pic.write_to(buf)
	buf.seek(0)
	return PictureFile.read_from(buf)


def make_bitmap(pixels=()) -> Bitmap:
	"""A Bitmap with the given (row, col) pixels turned on."""
	bmp = Bitmap()
	for row, col in pixels:
		bmp.set(row, col)
	return bmp


def make_pic(**kwargs) -> PictureFile:
	name  = kwargs.pop('name', 'Pic1')
	value = kwargs.pop('bitmap', None)
	if value is None:
		value = make_bitmap()
	return PictureFile(tok(name), value, **kwargs)


def pixels_of(bmp: Bitmap) -> set:
	return {(r, c) for r in range(ROWS) for c in range(COLS) if bmp.get(r, c)}


# ── PictureFile roundtrip ─────────────────────────────────────────────────────

class TestPictureFileRoundtrip:
	def test_name(self):
		assert roundtrip_pic(make_pic(name='Pic9')).name == 'Pic9'

	def test_pixels_corners_and_edges(self):
		# all four corners of the full screen (default 64-row format), plus interior
		pix = {(0, 0), (0, COLS - 1), (ROWS - 1, 0), (ROWS - 1, COLS - 1), (31, 47)}
		result = roundtrip_pic(make_pic(bitmap=make_bitmap(pix)))
		assert pixels_of(result.value) == pix

	def test_empty_bitmap(self):
		assert pixels_of(roundtrip_pic(make_pic()).value) == set()

	def test_full_height_keeps_bottom_row(self):
		# The 64-row format stores the bottom LCD row (row 63).
		result = roundtrip_pic(make_pic(rows=64, bitmap=make_bitmap({(63, 0), (5, 5)})))
		assert pixels_of(result.value) == {(63, 0), (5, 5)}

	def test_graph_screen_height_drops_bottom_row(self):
		# The 63-row format omits row 63, so it never survives a roundtrip.
		result = roundtrip_pic(make_pic(rows=63, bitmap=make_bitmap({(63, 0), (5, 5)})))
		assert pixels_of(result.value) == {(5, 5)}

	def test_rows_preserved(self):
		assert roundtrip_pic(make_pic(rows=63)).rows == 63
		assert roundtrip_pic(make_pic(rows=64)).rows == 64

	def test_archived(self):
		assert roundtrip_pic(make_pic(archived=True)).archived is True

	def test_comment(self):
		assert roundtrip_pic(make_pic(comment='my pic')).comment == 'my pic'

	def test_version_preserved(self):
		assert roundtrip_pic(make_pic(version=0x0A)).version == 0x0A


# ── PictureFile real-file byte-exact roundtrip ────────────────────────────────
# Pic9 is the 63-row size (756 bytes, graph screen only); Pic3 is the 64-row size
# (768 bytes, full LCD).  Both sizes occur mixed within a single TI-84+ backup.

_PIC_DIR = r'C:\Users\Max\Documents\MyTiData\Backups\TI84PlusSilverEdition_11'
_PIC_FILES = [
	('Pic9', 63, rf'{_PIC_DIR}\Pic9.8xi'),
	('Pic3', 64, rf'{_PIC_DIR}\Pic3.8xi'),
]


@pytest.mark.parametrize('name,rows,path', _PIC_FILES)
def test_pic_metadata_from_file(name, rows, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	pic = PictureFile.load(path)
	assert pic.name == name
	assert pic.rows == rows


@pytest.mark.parametrize('name,rows,path', _PIC_FILES)
def test_pic_byte_exact_roundtrip(name, rows, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	orig = open(path, 'rb').read()
	buf = BytesIO()
	PictureFile.load(path).write_to(buf)
	assert buf.getvalue() == orig


# ── VariableFile (.8xn real / .8xc complex) ───────────────────────────────────

THETA = 'θ'


def roundtrip_var(var: VariableFile) -> VariableFile:
	buf = BytesIO()
	var.write_to(buf)
	buf.seek(0)
	return VariableFile.read_from(buf)


def make_var(**kwargs) -> VariableFile:
	kwargs.setdefault('value', 0.0)
	return VariableFile(tok(kwargs.pop('name', 'A')), **kwargs)


class TestNameToken:
	# the 8-byte name field decodes to an accessor; read_accessor reads it and
	# write_accessor writes it back (1-byte letters, 2-byte matrix/pic/string/equation)
	def test_decode_letter(self):
		assert read_accessor(b'A' + b'\x00' * 7).text == 'A'

	def test_decode_theta(self):
		assert read_accessor(b'\x5b' + b'\x00' * 7).text == THETA

	def test_decode_two_byte(self):
		assert read_accessor(b'\xaa\x00' + b'\x00' * 6).text == 'Str1'      # Str1
		assert read_accessor(b'\x5c\x00' + b'\x00' * 6).text == '[A]'       # matrix [A]
		assert read_accessor(b'\x5e\x10' + b'\x00' * 6).text == 'Y₁'        # Y₁
		assert read_accessor(b'\x60\x00' + b'\x00' * 6).text == 'Pic1'      # Pic1 (PicToken)

	def test_name_field_round_trips(self):
		for name in ('A', THETA, 'Str1', '[A]', 'Pic1', 'Y₁', '\U0001d462'):
			field = write_accessor(tok(name))
			assert len(field) == 8
			assert read_accessor(field).text == name


class TestVariableFileRoundtrip:
	def test_real_value(self):
		assert roundtrip_var(make_var(value=8.0)).value == pytest.approx(8.0)

	def test_real_negative(self):
		assert roundtrip_var(make_var(value=-3.5)).value == pytest.approx(-3.5)

	def test_is_complex_flag(self):
		assert make_var(value=1 + 2j).is_complex is True
		assert make_var(value=4.0).is_complex is False

	def test_complex_value(self):
		result = roundtrip_var(make_var(value=1 + 2j))
		assert result.is_complex is True
		assert result.value == pytest.approx(1 + 2j)

	def test_complex_negative_parts(self):
		assert roundtrip_var(make_var(value=-1.5 - 2.5j)).value == pytest.approx(-1.5 - 2.5j)

	def test_real_value_stays_real(self):
		result = roundtrip_var(make_var(value=2.0))
		assert result.is_complex is False
		assert not isinstance(result.value, complex)

	def test_name_preserved(self):
		assert roundtrip_var(make_var(name='Z')).name == 'Z'

	def test_theta_name_preserved(self):
		assert roundtrip_var(make_var(name=THETA, value=5.0)).name == THETA

	def test_archived(self):
		assert roundtrip_var(make_var(archived=True)).archived is True

	def test_comment(self):
		assert roundtrip_var(make_var(comment='my var')).comment == 'my var'


# ── VariableFile real-file byte-exact roundtrip ───────────────────────────────

_VAR_FILES = [
	('A', 8.0, r'C:\Users\Max\Documents\MyTiData\Backups\TI84PlusSilverEdition_11\A.8xn'),
	('D', 2 ** 0.5, r'C:\Users\Max\Documents\MyTiData\Backups\TI84PlusSilverEdition_12\D.8xc'),
]


@pytest.mark.parametrize('name,value,path', _VAR_FILES)
def test_var_metadata_from_file(name, value, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	var = VariableFile.load(path)
	assert var.name == name
	assert var.value == pytest.approx(value)


@pytest.mark.parametrize('name,value,path', _VAR_FILES)
def test_var_byte_exact_roundtrip(name, value, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	orig = open(path, 'rb').read()
	buf = BytesIO()
	VariableFile.load(path).write_to(buf)
	assert buf.getvalue() == orig


# ── MatrixFile (.8xm) ─────────────────────────────────────────────────────────

def roundtrip_matrix(mat: MatrixFile) -> MatrixFile:
	buf = BytesIO()
	mat.write_to(buf)
	buf.seek(0)
	return MatrixFile.read_from(buf)


def make_matrix(**kwargs) -> MatrixFile:
	name = kwargs.pop('name', '[A]')
	kwargs['value'] = TiMatrix(kwargs.pop('values', [[1.0, 2.0], [3.0, 4.0]]))
	return MatrixFile(tok(name), **kwargs)


class TestMatrixFileRoundtrip:
	def test_dims_property(self):
		mat = make_matrix(values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
		assert (mat.rows, mat.cols) == (2, 3)

	def test_square(self):
		vals = [[1.0, 2.0], [3.0, 4.0]]
		assert roundtrip_matrix(make_matrix(values=vals)).value.data == vals

	def test_rectangular_preserves_shape(self):
		# 2x5 — guards against a row/column transpose
		vals = [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]]
		result = roundtrip_matrix(make_matrix(values=vals))
		assert (result.rows, result.cols) == (2, 5)
		assert result.value.data == vals

	def test_row_vector(self):
		vals = [[1.0, 2.0, 3.0]]
		result = roundtrip_matrix(make_matrix(values=vals))
		assert (result.rows, result.cols) == (1, 3)
		assert result.value.data == vals

	def test_column_vector(self):
		vals = [[1.0], [2.0], [3.0]]
		result = roundtrip_matrix(make_matrix(values=vals))
		assert (result.rows, result.cols) == (3, 1)
		assert result.value.data == vals

	def test_negative_values(self):
		vals = [[-1.5, 2.5], [3.0, -4.0]]
		assert roundtrip_matrix(make_matrix(values=vals)).value.data == vals

	def test_name_preserved(self):
		assert roundtrip_matrix(make_matrix(name='[J]')).name == '[J]'

	def test_archived(self):
		assert roundtrip_matrix(make_matrix(archived=True)).archived is True

	def test_comment(self):
		assert roundtrip_matrix(make_matrix(comment='my matrix')).comment == 'my matrix'


# ── MatrixFile real-file byte-exact roundtrip ─────────────────────────────────

_MATRIX_DIR = r'C:\Users\Max\Documents\MyTiData\Backups\TI84PlusSilverEdition_13'
_MATRIX_FILES = [
	('[A]', 1, 1, rf'{_MATRIX_DIR}\A.8xm'),
	('[B]', 2, 2, rf'{_MATRIX_DIR}\B.8xm'),
	('[C]', 3, 3, rf'{_MATRIX_DIR}\C.8xm'),
	('[D]', 50, 1, rf'{_MATRIX_DIR}\D.8xm'),
]


@pytest.mark.parametrize('name,rows,cols,path', _MATRIX_FILES)
def test_matrix_metadata_from_file(name, rows, cols, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	mat = MatrixFile.load(path)
	assert mat.name == name
	assert (mat.rows, mat.cols) == (rows, cols)


@pytest.mark.parametrize('name,rows,cols,path', _MATRIX_FILES)
def test_matrix_byte_exact_roundtrip(name, rows, cols, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	orig = open(path, 'rb').read()
	buf = BytesIO()
	MatrixFile.load(path).write_to(buf)
	assert buf.getvalue() == orig


# ── StringFile (.8xs) and EquationFile (.8xy) ─────────────────────────────────

def roundtrip_tokenvar(obj):
	buf = BytesIO()
	obj.write_to(buf)
	buf.seek(0)
	return type(obj).read_from(buf)


class TestStringFileRoundtrip:
	def test_contents(self):
		result = roundtrip_tokenvar(StringFile(tok('Str1'), TiString(toks('ABC'))))
		assert result.name == 'Str1'
		assert result.value.tokens == toks('ABC')

	def test_empty_string(self):
		result = roundtrip_tokenvar(StringFile(tok('Str3'), TiString([])))
		assert result.value.tokens == []

	def test_archived_and_comment(self):
		result = roundtrip_tokenvar(StringFile(tok('Str9'), TiString(toks('X')), comment='hi', archived=True))
		assert result.comment == 'hi'
		assert result.archived is True

	def test_var_type_byte(self):
		buf = BytesIO()
		StringFile(tok('Str1'), TiString(toks('A'))).write_to(buf)
		assert buf.getvalue()[59] == 0x04


class TestEquationFileRoundtrip:
	def test_contents(self):
		result = roundtrip_tokenvar(EquationFile(tok('Y₁'), TiEquation(toks('X'))))
		assert result.name == 'Y₁'
		assert result.value.tokens == toks('X')

	def test_sequence_var_name(self):
		result = roundtrip_tokenvar(EquationFile(tok('𝑢'), TiEquation(toks('X'))))  # 𝑢
		assert result.name == '𝑢'

	def test_var_type_byte(self):
		buf = BytesIO()
		EquationFile(tok('Y₁'), TiEquation(toks('X'))).write_to(buf)
		assert buf.getvalue()[59] == 0x03


# ── StringFile / EquationFile real-file byte-exact roundtrip ──────────────────

_TOKENVAR_DIR = r'C:\Users\Max\Documents\MyTiData\Backups\14'
_TOKENVAR_FILES = [
	(StringFile,   'Str1',        rf'{_TOKENVAR_DIR}\Str1.8xs'),
	(StringFile,   'Str2',        rf'{_TOKENVAR_DIR}\Str2.8xs'),
	(StringFile,   'Str3',        rf'{_TOKENVAR_DIR}\Str3.8xs'),   # empty string
	(EquationFile, 'Y₁',     rf'{_TOKENVAR_DIR}\Y_1_.8xy'),
	(EquationFile, 'Y₂',     rf'{_TOKENVAR_DIR}\Y_2_.8xy'),
	(EquationFile, '\U0001d462',  rf'{_TOKENVAR_DIR}\u.8xy'),       # 𝑢 (sequence)
]


@pytest.mark.parametrize('cls,name,path', _TOKENVAR_FILES)
def test_tokenvar_name_from_file(cls, name, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	assert cls.load(path).name == name


@pytest.mark.parametrize('cls,name,path', _TOKENVAR_FILES)
def test_tokenvar_byte_exact_roundtrip(cls, name, path):
	if not os.path.exists(path):
		pytest.skip('real file not found')
	orig = open(path, 'rb').read()
	buf = BytesIO()
	cls.load(path).write_to(buf)
	assert buf.getvalue() == orig


# ── Envelope validation: checksum, truncation, 0x0B header ────────────────────

def _ti83_file(name: bytes, body: bytes, file_type: int = 0x05) -> bytes:
	"""Hand-build an original-TI-83 file: **TI83** signature and the 0x0B entry
	layout — an 11-byte sub-header with no version/flag bytes."""
	entry = (b'\x0B\x00' + len(body).to_bytes(2, 'little') + bytes([file_type])
	         + name.ljust(8, b'\x00') + len(body).to_bytes(2, 'little') + body)
	return (b'**TI83**\x1a\x0a\x00' + b'\x00' * 42
	        + len(entry).to_bytes(2, 'little') + entry
	        + (sum(entry) & 0xFFFF).to_bytes(2, 'little'))


def _prog_body(tokens) -> bytes:
	"""A program body: the 2-byte-length-prefixed token stream."""
	data = b''.join(t.code_to_bytes() for t in tokens)
	return len(data).to_bytes(2, 'little') + data


class TestEnvelopeValidation:
	def test_reads_ti83_0x0b_header(self):
		# The 0x0B layout has no version/flag bytes; the body follows the repeated
		# data length directly.  A TI-BASIC program saved by original-TI-83 link
		# software reads like any other (version defaults to 0, nothing archived —
		# the TI-83 had no Flash).
		data = _ti83_file(b'ABC', _prog_body(toks('Disp 42')))
		prog = ProgramFile.read_from(BytesIO(data))
		assert prog.name == 'ABC'
		assert prog.tokens == toks('Disp 42')
		assert prog.version == 0
		assert prog.archived is False
		assert prog.locked is False

	def test_0x0b_protected_program_is_locked(self):
		data = _ti83_file(b'ASM', _prog_body([]), file_type=0x06)
		assert ProgramFile.read_from(BytesIO(data)).locked is True

	def test_checksum_mismatch_rejected(self):
		buf = BytesIO()
		make_prog(tokens=toks('Disp 42')).write_to(buf)
		corrupt = bytearray(buf.getvalue())
		corrupt[-4] ^= 0xFF                    # flip a body byte, keep the stored checksum
		with pytest.raises(ValueError, match='Checksum'):
			ProgramFile.read_from(BytesIO(bytes(corrupt)))

	def test_truncated_file_rejected(self):
		buf = BytesIO()
		make_prog(tokens=toks('Disp 42')).write_to(buf)
		with pytest.raises(ValueError, match='Truncated'):
			ProgramFile.read_from(BytesIO(buf.getvalue()[:-10]))

	def test_unknown_header_length_rejected(self):
		# A well-checksummed entry whose leading word is neither 0x0B nor 0x0D.
		body = _prog_body([])
		entry = (b'\x0C\x00' + len(body).to_bytes(2, 'little') + b'\x05'
		         + b'X'.ljust(8, b'\x00') + len(body).to_bytes(2, 'little') + body)
		data = (b'**TI83F*\x1a\x0a\x00' + b'\x00' * 42
		        + len(entry).to_bytes(2, 'little') + entry
		        + (sum(entry) & 0xFFFF).to_bytes(2, 'little'))
		with pytest.raises(ValueError, match='header length'):
			ProgramFile.read_from(BytesIO(data))


_TI83_ASM_FILE = r'C:\Users\Max\Documents\LegacyTiData\Backups\Bomberk.8xp'


def test_real_ti83_header_fields():
	"""Bomberk.8xp is a real original-TI-83 file (0x0B layout).  Its body is Z80
	assembly, not tokens, so only the envelope/header is checked here."""
	if not os.path.exists(_TI83_ASM_FILE):
		pytest.skip('real file not found')
	from tifile import _read_var_header
	with open(_TI83_ASM_FILE, 'rb') as f:
		file_type, name_bytes, archived, comment, version = _read_var_header(f)
		assert file_type == 0x06                     # protected program
		assert name_bytes == b'BOMBERK\x00'
		assert version == 0 and archived is False    # fields absent in the 0x0B layout
		inner = int.from_bytes(f.read(2), 'little')
		assert len(f.read()) - 2 == inner            # f left at body: prefix matches content (+2 checksum)


# ── store_to(env): install a loaded file into a running environment ────────────

class TestStoreTo:
	def test_number_variable(self):
		env = Environment()
		VariableFile(tok('A'), 7.0).store_to(env)
		assert tok('A').resolve(env) == 7.0

	def test_complex_variable(self):
		env = Environment()
		VariableFile(tok('B'), 3 + 4j).store_to(env)
		assert tok('B').resolve(env) == 3 + 4j

	def test_builtin_list(self):
		env = Environment()
		ListFile(tok('L₁'), TiList([1.0, 2.0, 3.0])).store_to(env)
		assert env.lists[0].data == [1.0, 2.0, 3.0]

	def test_user_list(self):
		env = Environment()
		ListFile(UserList('CW'), TiList([5.0, 6.0])).store_to(env)
		assert env.user_lists['CW'].data == [5.0, 6.0]

	def test_matrix(self):
		env = Environment()
		MatrixFile(tok('[A]'), TiMatrix([[1.0, 2.0], [3.0, 4.0]])).store_to(env)
		assert env.matrices[0].data == [[1.0, 2.0], [3.0, 4.0]]

	def test_string(self):
		env = Environment()
		StringFile(tok('Str1'), TiString(toks('HI'))).store_to(env)
		assert str(env.strings[0]) == 'HI'

	def test_picture(self):
		env = Environment()
		PictureFile(tok('Pic1'), make_bitmap({(5, 5)})).store_to(env)
		assert pixels_of(env.pics[0]) == {(5, 5)}

	def test_program(self):
		env = Environment()
		ProgramFile('TEST', toks('5')).store_to(env)
		assert env.programs['TEST'].tokens == toks('5')

	def test_equation(self):
		env = Environment()
		EquationFile(tok('Y₁'), TiEquation(toks('X'))).store_to(env)
		assert tok('Y₁').get(env).tokens == toks('X')

	def test_store_then_round_trip_through_file_and_env(self):
		# load a real list file and install it; the env slot matches the file value
		path = _CPX_LIST_FILE
		if not os.path.exists(path):
			pytest.skip('real file not found')
		env = Environment()
		lst = ListFile.load(path)          # name 'CPX' (user list)
		lst.store_to(env)
		assert list(env.user_lists['CPX']) == pytest.approx(list(lst.value))

"""Tests for ProgramFile and ListFile binary file format (read_from / write_to)."""
import os
import pytest
from io import BytesIO

from tifile import (
	ProgramFile, ListFile, PictureFile,
	_decode_list_name, _encode_list_name,
	_decode_pic_name, _encode_pic_name,
)
from bitmap import Bitmap, ROWS, COLS
from test_tibasic import toks


# ── Helpers ───────────────────────────────────────────────────────────────────

def roundtrip(prog: ProgramFile) -> ProgramFile:
	"""Write prog to a BytesIO buffer, seek back, and read it out again."""
	buf = BytesIO()
	prog.write_to(buf)
	buf.seek(0)
	return ProgramFile.read_from(buf)


def make_prog(**kwargs) -> ProgramFile:
	"""Construct a ProgramFile with sensible defaults."""
	kwargs.setdefault('name', 'TEST')
	kwargs.setdefault('tokens', [])
	return ProgramFile(**kwargs)


# ── Roundtrip: metadata ───────────────────────────────────────────────────────

class TestRoundtripMetadata:
	def test_name(self):
		assert roundtrip(make_prog(name='HELLO')).name == 'HELLO'

	def test_name_lowercase_normalised(self):
		# write() uppercases the name in the binary; read() decodes it as-is
		prog = ProgramFile(name='hello', tokens=[])
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
	kwargs.setdefault('name', 'TEST')
	kwargs.setdefault('values', [])
	return ListFile(**kwargs)


# ── List name coding ──────────────────────────────────────────────────────────

class TestListNameCoding:
	# built-in decode: L₁ is "0", L₂ is "1", ..., L₆ is "5" (raw index byte)
	def test_decode_l1(self):
		assert _decode_list_name(b'\x5d\x00' + b'\x00' * 6) == '0'

	def test_decode_l2(self):
		assert _decode_list_name(b'\x5d\x01' + b'\x00' * 6) == '1'

	def test_decode_l6(self):
		assert _decode_list_name(b'\x5d\x05' + b'\x00' * 6) == '5'

	# user-list decode
	def test_decode_user_list(self):
		assert _decode_list_name(b'\x5d\x43\x57' + b'\x00' * 5) == 'CW'

	def test_decode_user_list_named_l1(self):
		# A user list literally named "L1" (distinct from built-in L₁)
		assert _decode_list_name(b'\x5d\x4c\x31' + b'\x00' * 5) == 'L1'

	# built-in encode
	def test_encode_l1(self):
		assert _encode_list_name('0') == b'\x5d\x00' + b'\x00' * 6

	def test_encode_l6(self):
		assert _encode_list_name('5') == b'\x5d\x05' + b'\x00' * 6

	# user-list encode
	def test_encode_user_list(self):
		assert _encode_list_name('CW') == b'\x5d\x43\x57' + b'\x00' * 5

	def test_encode_user_list_lowercase_normalised(self):
		assert _encode_list_name('cw') == b'\x5d\x43\x57' + b'\x00' * 5

	def test_encode_user_list_named_l1(self):
		assert _encode_list_name('L1') == b'\x5d\x4c\x31' + b'\x00' * 5

	# roundtrip consistency for all built-ins
	def test_all_builtins_roundtrip(self):
		for i in range(6):
			name = str(i)
			assert _decode_list_name(_encode_list_name(name)) == name


# ── ListFile roundtrip ────────────────────────────────────────────────────────

class TestListFileRoundtrip:
	def test_user_list_name(self):
		assert roundtrip_list(make_list(name='CW')).name == 'CW'

	def test_builtin_list_name_l1(self):
		assert roundtrip_list(make_list(name='0')).name == '0'

	def test_builtin_list_name_l6(self):
		assert roundtrip_list(make_list(name='5')).name == '5'

	def test_values_preserved(self):
		result = roundtrip_list(make_list(values=[1.0, 2.5, -3.0]))
		assert result.values == pytest.approx([1.0, 2.5, -3.0])

	def test_empty_values(self):
		assert roundtrip_list(make_list(values=[])).values == []

	def test_archived(self):
		assert roundtrip_list(make_list(archived=True)).archived is True

	def test_comment(self):
		assert roundtrip_list(make_list(comment='my list')).comment == 'my list'

	def test_version_preserved(self):
		assert roundtrip_list(make_list(version=0x24)).version == 0x24


# ── ListFile real-file byte-exact roundtrip ───────────────────────────────────

_BUILTIN_LIST_FILES = [
	(str(i), rf'C:\Users\Max\Documents\MyTiData\Backups\TI84Plus_1\L_{i+1}_.8xl')
	for i in range(6)
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
	kwargs.setdefault('name', '1')
	kwargs.setdefault('bitmap', make_bitmap())
	return PictureFile(**kwargs)


def pixels_of(bmp: Bitmap) -> set:
	return {(r, c) for r in range(ROWS) for c in range(COLS) if bmp.get(r, c)}


# ── Picture name coding ───────────────────────────────────────────────────────

class TestPicNameCoding:
	# decode: index byte → picture number (Pic1→0x00, …, Pic9→0x08, Pic0→0x09)
	def test_decode_pic1(self):
		assert _decode_pic_name(b'\x60\x00' + b'\x00' * 6) == '1'

	def test_decode_pic9(self):
		assert _decode_pic_name(b'\x60\x08' + b'\x00' * 6) == '9'

	def test_decode_pic0(self):
		assert _decode_pic_name(b'\x60\x09' + b'\x00' * 6) == '0'

	# encode: picture number → 0x60 token + index byte
	def test_encode_pic1(self):
		assert _encode_pic_name('1') == b'\x60\x00' + b'\x00' * 6

	def test_encode_pic9(self):
		assert _encode_pic_name('9') == b'\x60\x08' + b'\x00' * 6

	def test_encode_pic0(self):
		assert _encode_pic_name('0') == b'\x60\x09' + b'\x00' * 6

	def test_all_pics_roundtrip(self):
		for n in '1234567890':
			assert _decode_pic_name(_encode_pic_name(n)) == n


# ── PictureFile roundtrip ─────────────────────────────────────────────────────

class TestPictureFileRoundtrip:
	def test_name(self):
		assert roundtrip_pic(make_pic(name='9')).name == '9'

	def test_pixels_corners_and_edges(self):
		# all four corners of the full screen (default 64-row format), plus interior
		pix = {(0, 0), (0, COLS - 1), (ROWS - 1, 0), (ROWS - 1, COLS - 1), (31, 47)}
		result = roundtrip_pic(make_pic(bitmap=make_bitmap(pix)))
		assert pixels_of(result.bitmap) == pix

	def test_empty_bitmap(self):
		assert pixels_of(roundtrip_pic(make_pic()).bitmap) == set()

	def test_full_height_keeps_bottom_row(self):
		# The 64-row format stores the bottom LCD row (row 63).
		result = roundtrip_pic(make_pic(rows=64, bitmap=make_bitmap({(63, 0), (5, 5)})))
		assert pixels_of(result.bitmap) == {(63, 0), (5, 5)}

	def test_graph_screen_height_drops_bottom_row(self):
		# The 63-row format omits row 63, so it never survives a roundtrip.
		result = roundtrip_pic(make_pic(rows=63, bitmap=make_bitmap({(63, 0), (5, 5)})))
		assert pixels_of(result.bitmap) == {(5, 5)}

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
	('9', 63, rf'{_PIC_DIR}\Pic9.8xi'),
	('3', 64, rf'{_PIC_DIR}\Pic3.8xi'),
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

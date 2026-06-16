"""Tests for ProgramFile binary file format (read_from / write_to)."""
import pytest
from io import BytesIO

from tifile import ProgramFile
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

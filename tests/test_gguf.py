from __future__ import annotations

import struct

from larsm.findings import Severity
from larsm.scanner import scan_file


def _u32(v: int) -> bytes:
    return struct.pack("<I", v)


def _u64(v: int) -> bytes:
    return struct.pack("<Q", v)


def _gstr(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return _u64(len(encoded)) + encoded


def _kv(key: str, value: str) -> bytes:
    return _gstr(key) + _u32(8) + _gstr(value)


def _header(tensors: int, kvs: int, version: int = 3) -> bytes:
    return b"GGUF" + _u32(version) + _u64(tensors) + _u64(kvs)


def _write(tmp_path, name, data):
    target = tmp_path / name
    target.write_bytes(data)
    return target


def _checks(result):
    return {finding.check for finding in result.findings}


def test_clean_gguf_parses_without_high_findings(tmp_path):
    data = _header(0, 1) + _kv("general.architecture", "llama")
    result = scan_file(_write(tmp_path, "clean.gguf", data))
    assert "gguf.parsed" in _checks(result)
    assert result.max_severity < Severity.HIGH


def test_traversal_in_metadata_is_flagged(tmp_path):
    data = _header(0, 1) + _kv("general.name", "../../../etc/passwd")
    result = scan_file(_write(tmp_path, "traversal.gguf", data))
    assert "gguf.metadata_traversal" in _checks(result)
    assert result.max_severity >= Severity.HIGH


def test_implausible_metadata_count_is_rejected(tmp_path):
    result = scan_file(_write(tmp_path, "bomb.gguf", _header(0, 4_000_000_000)))
    assert "gguf.malformed_header" in _checks(result)


def test_oversized_string_length_is_rejected(tmp_path):
    data = _header(0, 1) + _u64(2**40) + b"general.name"
    result = scan_file(_write(tmp_path, "oversized.gguf", data))
    assert "gguf.malformed_header" in _checks(result)


def test_chat_template_escape_is_flagged(tmp_path):
    template = "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    data = _header(0, 1) + _kv("tokenizer.chat_template", template)
    result = scan_file(_write(tmp_path, "ssti.gguf", data))
    assert "gguf.template_injection" in _checks(result)


def test_gguf_v1_is_reported_as_unparsed(tmp_path):
    result = scan_file(_write(tmp_path, "old.gguf", _header(0, 0, version=1)))
    assert "gguf.unsupported_version" in _checks(result)

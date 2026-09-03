from __future__ import annotations

import json
import struct

from larsm.findings import Severity
from larsm.scanner import scan_file


def _build(header: dict, data_bytes: int) -> bytes:
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + b"\x00" * data_bytes


def _write(tmp_path, name, data):
    target = tmp_path / name
    target.write_bytes(data)
    return target


def _checks(result):
    return {finding.check for finding in result.findings}


def test_clean_safetensors_parses_cleanly(tmp_path):
    data = _build({"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}, 16)
    result = scan_file(_write(tmp_path, "clean.safetensors", data))
    assert "safetensors.parsed" in _checks(result)
    assert result.max_severity < Severity.HIGH


def test_offsets_beyond_buffer_are_flagged(tmp_path):
    data = _build({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 999_999]}}, 16)
    result = scan_file(_write(tmp_path, "oob.safetensors", data))
    assert "safetensors.offsets_out_of_bounds" in _checks(result)
    assert result.max_severity >= Severity.HIGH


def test_oversized_header_length_is_rejected(tmp_path):
    data = struct.pack("<Q", 1_073_741_824) + b'{"w":{}}'
    result = scan_file(_write(tmp_path, "huge.safetensors", data))
    assert "safetensors.malformed_header" in _checks(result)


def test_overlapping_regions_are_flagged(tmp_path):
    header = {
        "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
    }
    result = scan_file(_write(tmp_path, "overlap.safetensors", _build(header, 24)))
    assert "safetensors.tensor_overlap" in _checks(result)


def test_shape_size_mismatch_is_flagged(tmp_path):
    header = {"w": {"dtype": "F32", "shape": [100], "data_offsets": [0, 16]}}
    result = scan_file(_write(tmp_path, "mismatch.safetensors", _build(header, 16)))
    assert "safetensors.shape_size_mismatch" in _checks(result)


def test_traversal_in_metadata_is_flagged(tmp_path):
    header = {
        "__metadata__": {"src": "../../../etc/shadow"},
        "w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
    }
    result = scan_file(_write(tmp_path, "meta.safetensors", _build(header, 16)))
    assert "safetensors.metadata_traversal" in _checks(result)


def test_non_json_header_is_rejected(tmp_path):
    body = b"not json at all"
    data = struct.pack("<Q", len(body)) + body
    result = scan_file(_write(tmp_path, "bad.safetensors", data))
    assert "safetensors.malformed_header" in _checks(result)

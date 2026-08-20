from __future__ import annotations

import io
import json
import os
import pickle
import struct
import zipfile

import pytest

from larsm.findings import Severity
from larsm.formats import ModelFormat, detect_format
from larsm.scanner import scan_file


class _CommandPayload:
    def __reduce__(self):
        return (os.system, ("echo LARSM_TEST",))


def _write(tmp_path, name, data):
    target = tmp_path / name
    target.write_bytes(data)
    return target


def _clean_safetensors() -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}},
        separators=(",", ":"),
    ).encode()
    return struct.pack("<Q", len(header)) + header + b"\x00" * 16


def _clean_gguf() -> bytes:
    return b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0) + struct.pack("<Q", 0)


def _check_ids(result):
    return {finding.check for finding in result.findings}


def test_detects_gguf_and_safetensors(tmp_path):
    assert detect_format(_write(tmp_path, "a.gguf", _clean_gguf())) is ModelFormat.GGUF
    assert detect_format(
        _write(tmp_path, "b.safetensors", _clean_safetensors())
    ) is ModelFormat.SAFETENSORS


def test_clean_safetensors_has_no_high_findings(tmp_path):
    result = scan_file(_write(tmp_path, "clean.safetensors", _clean_safetensors()))
    assert result.max_severity < Severity.HIGH


@pytest.mark.parametrize("protocol", [2, 4])
def test_raw_pickle_with_os_system_is_critical(tmp_path, protocol):
    payload = pickle.dumps(_CommandPayload(), protocol=protocol)
    result = scan_file(_write(tmp_path, f"evil{protocol}.pkl", payload))
    assert result.file_format == ModelFormat.RAW_PICKLE.value
    assert result.max_severity is Severity.CRITICAL
    assert "pickle.dangerous_import" in _check_ids(result)


def test_pickle_disguised_as_safetensors_is_flagged(tmp_path):
    payload = pickle.dumps(_CommandPayload(), protocol=2)
    result = scan_file(_write(tmp_path, "disguised.safetensors", payload))
    assert "format.extension_mismatch" in _check_ids(result)
    assert result.max_severity is Severity.CRITICAL


def test_zip_slip_member_is_detected(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"ok": 1}, protocol=2))
        archive.writestr("../../escaped.pkl", b"x")
    result = scan_file(_write(tmp_path, "slip.pt", buffer.getvalue()))
    assert "zip.path_traversal" in _check_ids(result)


def test_truncated_pickle_reports_malformed(tmp_path):
    payload = pickle.dumps(_CommandPayload(), protocol=2)
    result = scan_file(_write(tmp_path, "truncated.pkl", payload[: len(payload) // 2]))
    assert "pickle.malformed" in _check_ids(result)

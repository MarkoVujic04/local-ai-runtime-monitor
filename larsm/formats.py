from __future__ import annotations

import struct
import zipfile
from enum import Enum
from pathlib import Path

_MAX_PLAUSIBLE_HEADER = 512 * 1024 * 1024


class ModelFormat(str, Enum):
    GGUF = "gguf"
    GGML_LEGACY = "ggml-legacy"
    SAFETENSORS = "safetensors"
    ZIP_PICKLE = "zip-pickle"
    ZIP_ARCHIVE = "zip-archive"
    RAW_PICKLE = "raw-pickle"
    ONNX = "onnx"
    EMPTY = "empty"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


EXECUTABLE_FORMATS = frozenset({ModelFormat.ZIP_PICKLE, ModelFormat.RAW_PICKLE})

PERCEIVED_SAFE_EXTENSIONS = frozenset({".gguf", ".ggml", ".safetensors", ".onnx"})

EXTENSION_EXPECTATIONS: dict[str, frozenset[ModelFormat]] = {
    ".gguf": frozenset({ModelFormat.GGUF, ModelFormat.GGML_LEGACY}),
    ".ggml": frozenset({ModelFormat.GGML_LEGACY, ModelFormat.GGUF}),
    ".safetensors": frozenset({ModelFormat.SAFETENSORS}),
    ".onnx": frozenset({ModelFormat.ONNX, ModelFormat.UNKNOWN}),
    ".pt": frozenset({ModelFormat.ZIP_PICKLE, ModelFormat.ZIP_ARCHIVE, ModelFormat.RAW_PICKLE}),
    ".pth": frozenset({ModelFormat.ZIP_PICKLE, ModelFormat.ZIP_ARCHIVE, ModelFormat.RAW_PICKLE}),
    ".ckpt": frozenset({ModelFormat.ZIP_PICKLE, ModelFormat.ZIP_ARCHIVE, ModelFormat.RAW_PICKLE}),
    ".pkl": frozenset({ModelFormat.RAW_PICKLE, ModelFormat.ZIP_PICKLE}),
    ".pickle": frozenset({ModelFormat.RAW_PICKLE, ModelFormat.ZIP_PICKLE}),
    ".npz": frozenset({ModelFormat.ZIP_ARCHIVE, ModelFormat.ZIP_PICKLE}),

    ".bin": frozenset(
        {
            ModelFormat.ZIP_PICKLE,
            ModelFormat.ZIP_ARCHIVE,
            ModelFormat.RAW_PICKLE,
            ModelFormat.SAFETENSORS,
            ModelFormat.GGUF,
            ModelFormat.GGML_LEGACY,
            ModelFormat.UNKNOWN,
        }
    ),
}

MODEL_EXTENSIONS = frozenset(EXTENSION_EXPECTATIONS)


def _zip_subtype(path: Path) -> ModelFormat:
    """Distinguish a torch-style pickle container from any other ZIP."""
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                lowered = name.lower()
                if lowered.endswith((".pkl", ".pickle")) or lowered.endswith("/data.pkl"):
                    return ModelFormat.ZIP_PICKLE
    except (zipfile.BadZipFile, OSError):
        return ModelFormat.ZIP_ARCHIVE
    return ModelFormat.ZIP_ARCHIVE


def _looks_like_safetensors(head: bytes, size: int) -> bool:
    """Heuristic: u64 little-endian header length, then a JSON object."""
    if len(head) < 9:
        return False
    (header_len,) = struct.unpack("<Q", head[:8])
    if not 2 <= header_len <= _MAX_PLAUSIBLE_HEADER:
        return False
    if header_len + 8 > size:
        return False
    return head[8:9] == b"{"


def detect_format(path: Path) -> ModelFormat:
    """Return the format implied by the file's bytes, ignoring its name."""
    size = path.stat().st_size
    if size == 0:
        return ModelFormat.EMPTY

    with path.open("rb") as handle:
        head = handle.read(32)

    if head.startswith(b"GGUF"):
        return ModelFormat.GGUF
    if head[:4] in (b"ggml", b"ggmf", b"ggjt", b"ggla", b"lmgg"):
        return ModelFormat.GGML_LEGACY
    if head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
        return _zip_subtype(path)

    if len(head) >= 2 and head[0] == 0x80 and 2 <= head[1] <= 5:
        return ModelFormat.RAW_PICKLE
    if head[:1] in (b"(", b"]", b"}", b"c", b"\x28"):
        return ModelFormat.RAW_PICKLE

    if _looks_like_safetensors(head, size):
        return ModelFormat.SAFETENSORS

    if head[:1] == b"\x08" and path.suffix.lower() == ".onnx":
        return ModelFormat.ONNX

    return ModelFormat.UNKNOWN

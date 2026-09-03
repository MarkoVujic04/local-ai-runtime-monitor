from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import struct
import zipfile
from pathlib import Path

MARKER = "echo LARSM_BENIGN_TEST_MARKER"


class _HarmlessReduce:
    def __reduce__(self):
        return (print, ("LARSM benign reduce sample",))


class _SimulatedCommandExecution:
    """Exercises the real attack shape: GLOBAL os.system + REDUCE.

    Serialised only. The command is `echo`, so even accidental execution
    is inert, but the structure is the genuine article.
    """

    def __reduce__(self):
        return (os.system, (MARKER,))


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _u64(value: int) -> bytes:
    return struct.pack("<Q", value)


def _gguf_string(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return _u64(len(encoded)) + encoded


def build_clean_gguf() -> bytes:
    """Minimal structurally valid GGUF v3 header with two string metadata keys."""
    pairs = [("general.architecture", "llama"), ("general.name", "larsm-clean-sample")]
    body = b""
    for key, value in pairs:
        body += _gguf_string(key) + _u32(8) + _gguf_string(value)
    header = b"GGUF" + _u32(3) + _u64(0) + _u64(len(pairs))
    return header + body


def build_clean_safetensors() -> bytes:
    tensor_data = b"\x00" * 16
    header = {
        "__metadata__": {"format": "pt", "producer": "larsm-testkit"},
        "weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _u64(len(encoded)) + encoded + tensor_data


def _gguf_kv_string(key: str, value: str) -> bytes:
    return _gguf_string(key) + _u32(8) + _gguf_string(value)


def _gguf_header(tensor_count: int, kv_count: int, version: int = 3) -> bytes:
    return b"GGUF" + _u32(version) + _u64(tensor_count) + _u64(kv_count)


def build_traversal_gguf() -> bytes:
    """Metadata value carrying a path-traversal payload."""
    pairs = [
        _gguf_kv_string("general.architecture", "llama"),
        _gguf_kv_string("general.name", "../../../../etc/cron.d/larsm-demo"),
        _gguf_kv_string("tokenizer.ggml.model", "gpt2"),
    ]
    return _gguf_header(0, len(pairs)) + b"".join(pairs)


def build_template_injection_gguf() -> bytes:
    """Chat template attempting a Jinja sandbox escape."""
    template = (
        "{% for m in messages %}{{ m.content }}{% endfor %}"
        "{{ ''.__class__.__mro__[1].__subclasses__() }}"
    )
    pairs = [
        _gguf_kv_string("general.architecture", "llama"),
        _gguf_kv_string("tokenizer.chat_template", template),
    ]
    return _gguf_header(0, len(pairs)) + b"".join(pairs)


def build_implausible_count_gguf() -> bytes:
    """Header claiming four billion metadata entries in a 24-byte file."""
    return _gguf_header(0, 4_000_000_000)


def build_oversized_string_gguf() -> bytes:
    """A metadata key whose declared length runs past the end of the file."""
    return _gguf_header(0, 1) + _u64(2**40) + b"general.name"


def build_oob_safetensors() -> bytes:
    """Tensor offsets pointing far beyond the data section."""
    header = {
        "weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]},
        "hidden": {"dtype": "F32", "shape": [1024, 1024], "data_offsets": [64, 4_194_368]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _u64(len(encoded)) + encoded + b"\x00" * 64


def build_overlapping_safetensors() -> bytes:
    """Two tensors claiming the same bytes, plus a traversal in metadata."""
    header = {
        "__metadata__": {"format": "pt", "source": "..\\..\\windows\\system32\\larsm.txt"},
        "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return _u64(len(encoded)) + encoded + b"\x00" * 24


def build_huge_header_safetensors() -> bytes:
    """Header length field claiming a gigabyte in a 40-byte file."""
    return _u64(1_073_741_824) + b'{"weight":{"dtype":"F32"}}'


def build_torch_style_zip(pickle_bytes: bytes, extra_member: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", pickle_bytes)
        archive.writestr("archive/version", "3\n")
        archive.writestr("archive/data/0", b"\x00" * 16)
        if extra_member:
            archive.writestr(extra_member, pickle_bytes)
    return buffer.getvalue()


def build_samples() -> dict[str, bytes]:
    harmless_pickle = pickle.dumps(_HarmlessReduce(), protocol=2)
    command_pickle = pickle.dumps(_SimulatedCommandExecution(), protocol=2)
    command_pickle_p4 = pickle.dumps(_SimulatedCommandExecution(), protocol=4)

    return {
        "clean_model.gguf": build_clean_gguf(),
        "clean_model.safetensors": build_clean_safetensors(),
        "benign_reduce.pkl": harmless_pickle,
        "simulated_rce_proto2.pkl": command_pickle,
        "simulated_rce_proto4.pkl": command_pickle_p4,
        "simulated_rce_torch_style.bin": build_torch_style_zip(command_pickle),
        "format_confusion.safetensors": command_pickle,
        "zip_slip.pt": build_torch_style_zip(
            harmless_pickle, extra_member="../../larsm_escaped.pkl"
        ),
        "truncated.pkl": command_pickle[: len(command_pickle) // 2],

        "traversal_metadata.gguf": build_traversal_gguf(),
        "template_injection.gguf": build_template_injection_gguf(),
        "implausible_count.gguf": build_implausible_count_gguf(),
        "oversized_string.gguf": build_oversized_string_gguf(),

        "oob_offsets.safetensors": build_oob_safetensors(),
        "overlapping.safetensors": build_overlapping_safetensors(),
        "huge_header.safetensors": build_huge_header_safetensors(),
    }


WARNING_TEXT = """\
LARSM test artefacts — DO NOT LOAD
==================================

Files in this directory are generated by tools/make_test_models.py for the
sole purpose of testing the LARSM scanner.

Several of them contain genuine pickle code-execution structures. They are
safe to *scan* (LARSM only parses opcodes; it never deserialises) and safe to
*store*. They are NOT safe to load.

Never run pickle.load(), pickle.loads(), or torch.load() on these files, and
never place them in a directory a model runtime watches.

The embedded command is a harmless `echo`, but treat that as a courtesy, not
a guarantee. Delete this directory when you are done.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate benign LARSM test model files.")
    parser.add_argument("--outdir", type=Path, default=Path("samples"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "DO-NOT-LOAD.txt").write_text(WARNING_TEXT, encoding="utf-8")

    for name, data in build_samples().items():
        target = args.outdir / name
        target.write_bytes(data)
        print(f"wrote {target}  ({len(data)} bytes)")

    print(
        f"\nGenerated in {args.outdir.resolve()}."
        f"\nos.system serialises as module {'nt' if os.name == 'nt' else 'posix'} on this OS."
        "\nThese files are for scanning only. Do not load them."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

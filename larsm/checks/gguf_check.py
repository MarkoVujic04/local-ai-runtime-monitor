from __future__ import annotations

from pathlib import Path
from typing import Any

from ..findings import Finding, Severity
from ..pathsafety import looks_like_traversal
from ..reader import BoundedReader, ImplausibleLength, ParseError, TruncatedFile

MAX_REPORTED_ITEMS = 10

_T_UINT8, _T_INT8, _T_UINT16, _T_INT16 = 0, 1, 2, 3
_T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL = 4, 5, 6, 7
_T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_WIDTHS = {
    _T_UINT8: 1, _T_INT8: 1, _T_BOOL: 1,
    _T_UINT16: 2, _T_INT16: 2,
    _T_UINT32: 4, _T_INT32: 4, _T_FLOAT32: 4,
    _T_UINT64: 8, _T_INT64: 8, _T_FLOAT64: 8,
}

_GGML_TYPE_SIZES: dict[int, tuple[int, int]] = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 36),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    24: (1, 1),
    25: (1, 2),
    26: (1, 4),
    27: (1, 8),
    28: (1, 8),
    30: (1, 2),
}

_TEMPLATE_ESCAPE_MARKERS = (
    "__class__", "__subclasses__", "__globals__", "__builtins__",
    "__import__", "__mro__", "__bases__", "__init__.__",
    "os.popen", "os.system", "subprocess", "self.__", "cycler.__",
    "lipsum", "request.application", "config.__",
)


def _read_gguf_string(reader: BoundedReader, what: str) -> str:
    length = reader.checked_length(reader.u64(f"{what} length"), what)
    return reader.read_exact(length, what).decode("utf-8", "replace")


def _skip_value(reader: BoundedReader, type_tag: int, what: str) -> Any:
    if type_tag in _SCALAR_WIDTHS:
        reader.read_exact(_SCALAR_WIDTHS[type_tag], what)
        return None
    if type_tag == _T_STRING:
        return _read_gguf_string(reader, what)
    if type_tag == _T_ARRAY:
        element_type = reader.u32(f"{what} element type")
        count = reader.u64(f"{what} count")
        if element_type in _SCALAR_WIDTHS:
            width = _SCALAR_WIDTHS[element_type]
            reader.checked_count(count, width, f"{what} array")
            reader.read_exact(count * width, f"{what} array data")
            return None
        if element_type == _T_STRING:
            reader.checked_count(count, 8, f"{what} string array")
            return [_read_gguf_string(reader, f"{what}[{i}]") for i in range(count)]
        raise ImplausibleLength(f"{what}: unsupported array element type {element_type}")
    raise ImplausibleLength(f"{what}: unknown value type {type_tag}")


def _tensor_byte_size(type_tag: int, dims: list[int]) -> int | None:
    entry = _GGML_TYPE_SIZES.get(type_tag)
    if entry is None:
        return None
    block_size, bytes_per_block = entry
    elements = 1
    for dim in dims:
        elements *= dim
    if elements % block_size:
        return None
    return (elements // block_size) * bytes_per_block


def _template_findings(key: str, value: str) -> list[Finding]:
    if "template" not in key.lower():
        return []
    lowered = value.lower()
    hits = [marker for marker in _TEMPLATE_ESCAPE_MARKERS if marker in lowered]
    if not hits:
        return []
    return [
        Finding(
            check="gguf.template_injection",
            severity=Severity.HIGH,
            title="Chat template contains sandbox-escape constructs",
            detail=(
                f"Metadata key {key!r} holds a template referencing "
                + ", ".join(sorted(set(hits))[:5])
                + ". Runtimes render this template with a Jinja-style engine; attribute "
                "traversal of this shape is the standard escape from such a sandbox, "
                "which makes it a code-execution path in a format usually called inert."
            ),
            remediation=(
                "Do not load. Inspect the template manually and obtain the model from "
                "the publisher if the template is not expected."
            ),
            evidence={"key": key, "markers": sorted(set(hits))[:MAX_REPORTED_ITEMS]},
        )
    ]


def analyze_gguf(path: Path, size: int) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    errors: list[str] = []

    traversal_hits: list[dict[str, str]] = []
    oob_tensors: list[dict[str, Any]] = []
    unsized_types: set[int] = set()
    metadata_count = 0
    tensor_count = 0

    try:
        with path.open("rb") as handle:
            reader = BoundedReader(handle, size)

            magic = reader.read_exact(4, "magic")
            version = reader.u32("version")

            if version == 1:
                return (
                    [
                        Finding(
                            check="gguf.unsupported_version",
                            severity=Severity.LOW,
                            title="GGUF version 1 is not parsed",
                            detail=(
                                "Version 1 uses 32-bit length fields rather than 64-bit. "
                                "LARSM does not parse it, so this file was identified but "
                                "not inspected."
                            ),
                            remediation="Convert to GGUF v2 or later before relying on a scan.",
                            evidence={"magic": magic.decode("ascii", "replace"), "version": 1},
                        )
                    ],
                    [],
                )
            if version > 3:
                findings.append(
                    Finding(
                        check="gguf.unknown_version",
                        severity=Severity.LOW,
                        title="GGUF version is newer than this scanner understands",
                        detail=(
                            f"File declares version {version}. Parsing continues on the v2/v3 "
                            "layout, so results may be incomplete."
                        ),
                        remediation="Update LARSM, or verify the file another way.",
                        evidence={"version": version},
                    )
                )

            tensor_count = reader.checked_count(reader.u64("tensor count"), 24, "tensor table")
            metadata_count = reader.checked_count(reader.u64("metadata count"), 12, "metadata")

            for index in range(metadata_count):
                key = _read_gguf_string(reader, f"metadata[{index}] key")
                type_tag = reader.u32(f"metadata[{index}] type")
                value = _skip_value(reader, type_tag, f"metadata[{index}] value")

                if looks_like_traversal(key):
                    traversal_hits.append({"where": "key", "value": key[:200]})

                strings = [value] if isinstance(value, str) else (value or [])
                for item in strings:
                    if looks_like_traversal(item):
                        traversal_hits.append({"where": f"value of {key}", "value": item[:200]})
                if isinstance(value, str):
                    findings.extend(_template_findings(key, value))

            tensor_data_start = reader.pos
            regions: list[tuple[int, int, str]] = []

            for index in range(tensor_count):
                name = _read_gguf_string(reader, f"tensor[{index}] name")
                n_dims = reader.u32(f"tensor[{index}] ndims")
                if n_dims > 8:
                    raise ImplausibleLength(f"tensor[{index}] declares {n_dims} dimensions")
                dims = [reader.u64(f"tensor[{index}] dim") for _ in range(n_dims)]
                type_tag = reader.u32(f"tensor[{index}] type")
                offset = reader.u64(f"tensor[{index}] offset")

                if looks_like_traversal(name):
                    traversal_hits.append({"where": "tensor name", "value": name[:200]})

                byte_size = _tensor_byte_size(type_tag, dims)
                if byte_size is None:
                    unsized_types.add(type_tag)
                    continue
                if offset + byte_size > size:
                    oob_tensors.append(
                        {
                            "name": name[:120],
                            "offset": offset,
                            "bytes": byte_size,
                            "file_size": size,
                        }
                    )
                else:
                    regions.append((offset, offset + byte_size, name[:120]))

            regions.sort()
            overlaps = [
                {"first": regions[i][2], "second": regions[i + 1][2]}
                for i in range(len(regions) - 1)
                if regions[i][1] > regions[i + 1][0]
            ]

    except (TruncatedFile, ImplausibleLength) as exc:
        findings.append(
            Finding(
                check="gguf.malformed_header",
                severity=Severity.HIGH,
                title="GGUF structure makes claims the file cannot satisfy",
                detail=(
                    f"{exc} A parser that trusted this field without a bounds check would read "
                    "or allocate past the end of the file — the out-of-bounds-read class."
                ),
                remediation="Do not load. Re-download from the publisher and verify the hash.",
                evidence={"error": str(exc), "file_size": size},
            )
        )
        errors.append(f"{path.name}: {exc}")
        return findings, errors
    except (OSError, ParseError) as exc:
        errors.append(f"{path.name}: {exc}")
        return findings, errors

    if traversal_hits:
        findings.append(
            Finding(
                check="gguf.metadata_traversal",
                severity=Severity.HIGH,
                title="GGUF metadata embeds path-traversal-shaped strings",
                detail=(
                    "Found "
                    + "; ".join(f"{h['where']}: {h['value']!r}" for h in traversal_hits[:5])
                    + ". Tooling that derives filenames from model metadata can be steered "
                    "outside its intended directory — the CVE-2024-37032 primitive."
                ),
                remediation="Do not load. Report the artefact to the model publisher.",
                evidence={"hits": traversal_hits[:MAX_REPORTED_ITEMS]},
            )
        )

    if oob_tensors:
        findings.append(
            Finding(
                check="gguf.tensor_out_of_bounds",
                severity=Severity.HIGH,
                title="Tensor descriptor points past the end of the file",
                detail=(
                    f"{len(oob_tensors)} tensor(s) declare data extending beyond the file. "
                    "A loader that memory-maps at these offsets reads unmapped memory."
                ),
                remediation="Do not load. The file is truncated or deliberately malformed.",
                evidence={"tensors": oob_tensors[:MAX_REPORTED_ITEMS]},
            )
        )

    if overlaps:
        findings.append(
            Finding(
                check="gguf.tensor_overlap",
                severity=Severity.MEDIUM,
                title="Tensor data regions overlap",
                detail=(
                    f"{len(overlaps)} overlapping pair(s). Overlap is not valid in a normal "
                    "GGUF file and can be used to make one buffer alias another."
                ),
                remediation="Treat as malformed; verify against the publisher's hash.",
                evidence={"pairs": overlaps[:MAX_REPORTED_ITEMS]},
            )
        )

    if unsized_types:
        findings.append(
            Finding(
                check="gguf.unsized_tensor_type",
                severity=Severity.INFO,
                title="Some tensor types were not bounds-checked",
                detail=(
                    f"Tensor type id(s) {sorted(unsized_types)} are not in LARSM's size table, "
                    "so their offsets could not be validated. This is a gap in the scanner, "
                    "not a property of the file."
                ),
                remediation="",
                evidence={"type_ids": sorted(unsized_types)[:MAX_REPORTED_ITEMS]},
            )
        )

    findings.append(
        Finding(
            check="gguf.parsed",
            severity=Severity.INFO,
            title="GGUF header and tensor table parsed",
            detail=(
                f"Read {metadata_count} metadata entries and {tensor_count} tensor "
                f"descriptors; all length fields were within file bounds."
            ),
            remediation="",
            evidence={
                "metadata_entries": metadata_count,
                "tensors": tensor_count,
                "data_section_start": tensor_data_start,
            },
        )
    )

    return findings, errors

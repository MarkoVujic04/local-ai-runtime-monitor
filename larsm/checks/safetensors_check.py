from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..findings import Finding, Severity
from ..pathsafety import looks_like_traversal
from ..reader import BoundedReader, ImplausibleLength, ParseError, TruncatedFile

MAX_REPORTED_ITEMS = 10

MAX_HEADER_BYTES = 100 * 1024 * 1024

_DTYPE_WIDTHS = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def _malformed(detail: str, size: int, error: str) -> Finding:
    return Finding(
        check="safetensors.malformed_header",
        severity=Severity.HIGH,
        title="Safetensors header makes claims the file cannot satisfy",
        detail=(
            f"{detail} A loader that trusted this field without bounds-checking it would "
            "read past the end of the mapped region."
        ),
        remediation="Do not load. Re-download and verify against the publisher's hash.",
        evidence={"error": error, "file_size": size},
    )


def analyze_safetensors(path: Path, size: int) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []
    errors: list[str] = []

    try:
        with path.open("rb") as handle:
            reader = BoundedReader(handle, size)
            header_len = reader.u64("header length")

            if header_len > MAX_HEADER_BYTES:
                return (
                    [
                        _malformed(
                            f"Header declares {header_len} bytes, above the "
                            f"{MAX_HEADER_BYTES}-byte ceiling.",
                            size,
                            "oversized header",
                        )
                    ],
                    [],
                )

            reader.checked_length(header_len, "JSON header")
            raw_header = reader.read_exact(header_len, "JSON header")
            data_start = reader.pos
            data_size = size - data_start
    except (TruncatedFile, ImplausibleLength) as exc:
        return [_malformed(str(exc), size, str(exc))], [f"{path.name}: {exc}"]
    except (OSError, ParseError) as exc:
        return [], [f"{path.name}: {exc}"]

    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return (
            [_malformed(f"Header is not valid UTF-8 JSON: {exc}.", size, str(exc))],
            [f"{path.name}: {exc}"],
        )

    if not isinstance(header, dict):
        return (
            [_malformed("Header JSON is not an object.", size, "header not an object")],
            [],
        )

    traversal_hits: list[dict[str, str]] = []
    oob: list[dict[str, Any]] = []
    mismatched: list[dict[str, Any]] = []
    unknown_dtypes: set[str] = set()
    regions: list[tuple[int, int, str]] = []
    malformed_entries: list[str] = []

    metadata = header.get("__metadata__")
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            for text in (str(key), str(value)):
                if looks_like_traversal(text):
                    traversal_hits.append({"where": f"__metadata__.{key}", "value": text[:200]})

    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if looks_like_traversal(name):
            traversal_hits.append({"where": "tensor name", "value": name[:200]})

        if not isinstance(spec, dict):
            malformed_entries.append(name[:120])
            continue

        offsets = spec.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) for v in offsets)
        ):
            malformed_entries.append(name[:120])
            continue

        begin, end = offsets
        if begin < 0 or end < begin or end > data_size:
            oob.append(
                {"name": name[:120], "begin": begin, "end": end, "data_bytes": data_size}
            )
            continue

        regions.append((begin, end, name[:120]))

        dtype = spec.get("dtype")
        shape = spec.get("shape")
        width = _DTYPE_WIDTHS.get(dtype) if isinstance(dtype, str) else None
        if width is None:
            if isinstance(dtype, str):
                unknown_dtypes.add(dtype[:32])
            continue
        if not isinstance(shape, list) or not all(
            isinstance(d, int) and d >= 0 for d in shape
        ):
            malformed_entries.append(name[:120])
            continue

        elements = 1
        for dim in shape:
            elements *= dim
        expected = elements * width
        if expected != end - begin:
            mismatched.append(
                {
                    "name": name[:120],
                    "declared_bytes": end - begin,
                    "shape_implies": expected,
                    "dtype": dtype,
                }
            )

    regions.sort()
    overlaps = [
        {"first": regions[i][2], "second": regions[i + 1][2]}
        for i in range(len(regions) - 1)
        if regions[i][1] > regions[i + 1][0]
    ]

    if traversal_hits:
        findings.append(
            Finding(
                check="safetensors.metadata_traversal",
                severity=Severity.HIGH,
                title="Safetensors header embeds path-traversal-shaped strings",
                detail=(
                    "Found "
                    + "; ".join(f"{h['where']}: {h['value']!r}" for h in traversal_hits[:5])
                    + ". Tooling that writes files named from tensor or metadata keys can be "
                    "steered outside its intended directory."
                ),
                remediation="Do not load. Report the artefact to the model publisher.",
                evidence={"hits": traversal_hits[:MAX_REPORTED_ITEMS]},
            )
        )

    if oob:
        findings.append(
            Finding(
                check="safetensors.offsets_out_of_bounds",
                severity=Severity.HIGH,
                title="Tensor offsets fall outside the data buffer",
                detail=(
                    f"{len(oob)} tensor(s) declare a byte range outside the "
                    f"{data_size}-byte data section. Reading at these offsets is an "
                    "out-of-bounds read."
                ),
                remediation="Do not load. The file is truncated or deliberately malformed.",
                evidence={"tensors": oob[:MAX_REPORTED_ITEMS]},
            )
        )

    if mismatched:
        findings.append(
            Finding(
                check="safetensors.shape_size_mismatch",
                severity=Severity.MEDIUM,
                title="Declared shape does not match the declared byte range",
                detail=(
                    f"{len(mismatched)} tensor(s) declare a shape and dtype implying a "
                    "different size than their data_offsets span. A loader that sizes its "
                    "read from shape while bounding it by offsets can walk off the buffer."
                ),
                remediation="Treat as malformed; verify against the publisher's hash.",
                evidence={"tensors": mismatched[:MAX_REPORTED_ITEMS]},
            )
        )

    if overlaps:
        findings.append(
            Finding(
                check="safetensors.tensor_overlap",
                severity=Severity.MEDIUM,
                title="Tensor data regions overlap",
                detail=(
                    f"{len(overlaps)} overlapping pair(s). Valid safetensors files have "
                    "disjoint regions; overlap makes one tensor alias another."
                ),
                remediation="Treat as malformed; verify against the publisher's hash.",
                evidence={"pairs": overlaps[:MAX_REPORTED_ITEMS]},
            )
        )

    if malformed_entries:
        findings.append(
            Finding(
                check="safetensors.malformed_entry",
                severity=Severity.MEDIUM,
                title="Header contains entries that do not match the schema",
                detail=(
                    f"{len(malformed_entries)} entry/entries lack a well-formed dtype, shape "
                    "or data_offsets field."
                ),
                remediation="Do not load; the header does not conform to the format.",
                evidence={"entries": malformed_entries[:MAX_REPORTED_ITEMS]},
            )
        )

    if unknown_dtypes:
        findings.append(
            Finding(
                check="safetensors.unknown_dtype",
                severity=Severity.LOW,
                title="Header declares dtypes LARSM does not recognise",
                detail=(
                    f"Unrecognised dtype(s): {', '.join(sorted(unknown_dtypes))}. Size "
                    "consistency could not be checked for those tensors."
                ),
                remediation="Verify the file was produced by a current safetensors writer.",
                evidence={"dtypes": sorted(unknown_dtypes)[:MAX_REPORTED_ITEMS]},
            )
        )

    tensor_total = len([k for k in header if k != "__metadata__"])
    findings.append(
        Finding(
            check="safetensors.parsed",
            severity=Severity.INFO,
            title="Safetensors header parsed and bounds-checked",
            detail=(
                f"Header is {header_len} bytes describing {tensor_total} tensor(s) over a "
                f"{data_size}-byte data section."
            ),
            remediation="",
            evidence={
                "header_bytes": header_len,
                "tensors": tensor_total,
                "data_bytes": data_size,
            },
        )
    )

    return findings, errors

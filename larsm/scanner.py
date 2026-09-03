from __future__ import annotations

from pathlib import Path
from typing import Iterator

from .checks.pickle_check import analyze_pickle_stream, analyze_zip_container
from .findings import Finding, ScanResult, Severity
from .formats import (
    EXECUTABLE_FORMATS,
    EXTENSION_EXPECTATIONS,
    MODEL_EXTENSIONS,
    PERCEIVED_SAFE_EXTENSIONS,
    ModelFormat,
    detect_format,
)

from .checks.gguf_check import analyze_gguf
from .checks.safetensors_check import analyze_safetensors


def _format_findings(path: Path, detected: ModelFormat) -> list[Finding]:
    findings: list[Finding] = []
    extension = path.suffix.lower()

    if detected in EXECUTABLE_FORMATS:
        findings.append(
            Finding(
                check="format.pickle_based",
                severity=Severity.HIGH,
                title="File uses a pickle-based format that executes code on load",
                detail=(
                    f"Detected {detected}. Python's pickle protocol is a small program: "
                    "loading it can import arbitrary modules and call arbitrary callables. "
                    "This risk exists regardless of whether a malicious payload is present."
                ),
                remediation=(
                    "Prefer safetensors or GGUF. If you must load a pickle checkpoint, use "
                    "torch.load(..., weights_only=True) and load only from sources you trust."
                ),
                evidence={"format": str(detected)},
            )
        )

    expected = EXTENSION_EXPECTATIONS.get(extension)
    if expected and detected not in expected and detected is not ModelFormat.EMPTY:
        masquerading = extension in PERCEIVED_SAFE_EXTENSIONS and detected in EXECUTABLE_FORMATS
        findings.append(
            Finding(
                check="format.extension_mismatch",
                severity=Severity.CRITICAL if masquerading else Severity.HIGH,
                title=(
                    "Executable format disguised with a safe-looking extension"
                    if masquerading
                    else "File extension does not match file contents"
                ),
                detail=(
                    f"Extension {extension!r} implies "
                    f"{', '.join(sorted(str(f) for f in expected))}, but the bytes are "
                    f"{detected}."
                    + (
                        " A pickle renamed to a data-only extension is a deliberate bypass of "
                        "extension-based allowlists."
                        if masquerading
                        else " This may be mislabelling rather than an attack."
                    )
                ),
                remediation="Do not load. Verify the file's origin and integrity.",
                evidence={"extension": extension, "detected": str(detected)},
            )
        )

    if detected is ModelFormat.EMPTY:
        findings.append(
            Finding(
                check="format.empty_file",
                severity=Severity.LOW,
                title="File is empty",
                detail="Zero-byte file; likely a truncated or failed download.",
                remediation="Re-download and re-verify.",
                evidence={},
            )
        )
    elif detected is ModelFormat.UNKNOWN:
        findings.append(
            Finding(
                check="format.unknown",
                severity=Severity.LOW,
                title="File format not recognised",
                detail=(
                    "No known model-format signature matched. LARSM could not apply any "
                    "content-specific check, so absence of findings means nothing here."
                ),
                remediation="Identify the format before loading.",
                evidence={},
            )
        )
    elif detected in (ModelFormat.GGUF, ModelFormat.GGML_LEGACY, ModelFormat.SAFETENSORS):
        findings.append(
            Finding(
                check="format.data_only",
                severity=Severity.INFO,
                title="Format does not execute code on load",
                detail=(
                    f"Detected {detected}. These formats are data containers, which removes "
                    "the deserialisation-RCE class. Header and offset validation is applied "
                    "separately; this finding covers the format choice only."
                ),
                remediation="",
                evidence={"format": str(detected)},
            )
    )

    return findings


def scan_file(path: Path) -> ScanResult:
    """Scan one file. Never imports, executes, or deserialises its contents."""
    path = Path(path)
    result = ScanResult(path=str(path), file_format=str(ModelFormat.UNKNOWN), size_bytes=0)

    if path.is_symlink():
        result.findings.append(
            Finding(
                check="fs.symlink_skipped",
                severity=Severity.LOW,
                title="Path is a symbolic link and was not followed",
                detail=(
                    "LARSM does not follow links, so a link cannot redirect the scanner to a "
                    "file outside the scanned tree. Scan the link target directly if needed."
                ),
                remediation="Resolve and scan the real path explicitly.",
                evidence={"path": str(path)},
            )
        )
        return result

    try:
        result.size_bytes = path.stat().st_size
        detected = detect_format(path)
    except OSError as exc:
        result.errors.append(f"{path}: cannot read file: {exc}")
        return result

    result.file_format = str(detected)
    result.findings.extend(_format_findings(path, detected))

    if detected is ModelFormat.RAW_PICKLE:
        try:
            with path.open("rb") as handle:
                findings, errors = analyze_pickle_stream(handle, path.name)
            result.findings.extend(findings)
            result.errors.extend(errors)
        except OSError as exc:
            result.errors.append(f"{path}: {exc}")
    elif detected in (ModelFormat.ZIP_PICKLE, ModelFormat.ZIP_ARCHIVE):
        findings, errors = analyze_zip_container(path)
        result.findings.extend(findings)
        result.errors.extend(errors)
    elif detected is ModelFormat.SAFETENSORS:
        findings, errors = analyze_safetensors(path, result.size_bytes)
        result.findings.extend(findings)
        result.errors.extend(errors)
    elif detected is ModelFormat.GGUF:
        findings, errors = analyze_gguf(path, result.size_bytes)
        result.findings.extend(findings)
        result.errors.extend(errors)

    return result


def iter_candidate_files(
    target: Path, recursive: bool = True, all_files: bool = False
) -> Iterator[Path]:
    target = Path(target)
    if target.is_file():
        yield target
        return

    pattern = "**/*" if recursive else "*"
    for candidate in sorted(target.glob(pattern)):
        if not candidate.is_file():
            continue
        if all_files or candidate.suffix.lower() in MODEL_EXTENSIONS:
            yield candidate


def scan_path(target: Path, recursive: bool = True, all_files: bool = False) -> list[ScanResult]:
    return [scan_file(p) for p in iter_candidate_files(target, recursive, all_files)]

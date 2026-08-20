from __future__ import annotations

import pickletools
import zipfile
from collections import deque
from pathlib import Path
from typing import BinaryIO, Iterable

from ..findings import Finding, Severity
from ..pathsafety import looks_like_traversal

MAX_OPCODES = 1_000_000
MAX_ZIP_MEMBERS = 500
MAX_REPORTED_ITEMS = 12

EXECUTION_OPCODES = frozenset(
    {"REDUCE", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX", "BUILD", "EXT1", "EXT2", "EXT4"}
)

IMPORT_OPCODES = frozenset({"GLOBAL", "STACK_GLOBAL", "INST"})

STRING_OPCODES = frozenset(
    {
        "STRING", "BINSTRING", "SHORT_BINSTRING",
        "UNICODE", "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8",
    }
)

DANGEROUS_CALLABLES = frozenset(
    {
        "os.system", "nt.system", "posix.system",
        "os.popen", "nt.popen", "posix.popen",
        "os.execv", "os.execve", "os.spawnv", "os.spawnl",
        "os.remove", "os.unlink", "os.rename", "os.chmod",
        "subprocess.Popen", "subprocess.run", "subprocess.call",
        "subprocess.check_call", "subprocess.check_output", "subprocess.getoutput",
        "builtins.eval", "builtins.exec", "builtins.compile",
        "builtins.__import__", "builtins.getattr", "builtins.open",
        "__builtin__.eval", "__builtin__.exec", "__builtin__.execfile",
        "importlib.import_module", "runpy.run_path", "runpy._run_code",
        "pty.spawn", "shutil.rmtree", "shutil.move",
        "ctypes.CDLL", "ctypes.WinDLL", "ctypes.windll",
        "pickle.loads", "_pickle.loads", "torch.load", "numpy.load",
        "webbrowser.open", "socket.socket", "timeit.timeit",
    }
)

DANGEROUS_MODULES = frozenset(
    {
        "os", "nt", "posix", "subprocess", "sys", "socket", "shutil",
        "ctypes", "importlib", "runpy", "pty", "builtins", "__builtin__",
        "commands", "popen2", "multiprocessing", "asyncio", "webbrowser",
        "urllib", "urllib.request", "http.client", "requests", "ftplib",
        "pickle", "_pickle", "dill", "joblib", "code", "codeop", "timeit",
    }
)

GADGET_MODULES = frozenset({"codecs", "_codecs", "base64", "binascii", "operator", "functools"})

ALLOWED_MODULE_PREFIXES = ("torch", "torchvision", "numpy", "collections", "__torch__")


def _is_allowed_module(module: str) -> bool:
    return any(module == p or module.startswith(p + ".") for p in ALLOWED_MODULE_PREFIXES)


def _as_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", "replace")
    return ""


def analyze_pickle_stream(stream: BinaryIO, source: str) -> tuple[list[Finding], list[str]]:
    """Walk one pickle stream and return (findings, errors).
    `source` labels where the stream came from, e.g. "archive/data.pkl".
    """
    findings: list[Finding] = []
    errors: list[str] = []

    imports: list[tuple[str, str, int | None]] = []
    exec_ops: list[tuple[str, int | None]] = []
    traversal_strings: list[tuple[str, int | None]] = []
    recent_strings: deque[str] = deque(maxlen=2)
    opcode_count = 0
    truncated = False

    try:
        for opcode, arg, pos in pickletools.genops(stream):
            opcode_count += 1
            if opcode_count > MAX_OPCODES:
                truncated = True
                break

            name = opcode.name

            if name in STRING_OPCODES:
                text = _as_text(arg)
                recent_strings.append(text)
                if looks_like_traversal(text):
                    traversal_strings.append((text, pos))

            if name in ("GLOBAL", "INST"):
                module, _, attribute = _as_text(arg).partition(" ")
                imports.append((module, attribute, pos))
            elif name == "STACK_GLOBAL":
                if len(recent_strings) == 2:
                    imports.append((recent_strings[0], recent_strings[1], pos))
                else:
                    imports.append(("?", "?", pos))

            if name in EXECUTION_OPCODES:
                exec_ops.append((name, pos))

    except Exception as exc:
        errors.append(f"{source}: pickle stream could not be fully parsed: {exc}")
        findings.append(
            Finding(
                check="pickle.malformed",
                severity=Severity.HIGH,
                title="Pickle stream is malformed or truncated",
                detail=(
                    f"{source}: parsing stopped after {opcode_count} opcode(s). A stream that "
                    "a parser rejects may still be partially processed by a permissive loader, "
                    "and corruption here is a common fuzzing/exploitation artefact."
                ),
                remediation="Do not load this file. Re-download from a trusted source.",
                evidence={"source": source, "opcodes_parsed": opcode_count, "error": str(exc)},
            )
        )

    if truncated:
        errors.append(f"{source}: opcode limit ({MAX_OPCODES}) reached; analysis is partial.")

    critical, high, gadget, unknown = [], [], [], []
    for module, attribute, pos in imports:
        qualified = f"{module}.{attribute}"
        entry = {"import": qualified, "offset": pos}
        if qualified in DANGEROUS_CALLABLES:
            critical.append(entry)
        elif module in DANGEROUS_MODULES:
            high.append(entry)
        elif module in GADGET_MODULES:
            gadget.append(entry)
        elif not _is_allowed_module(module):
            unknown.append(entry)

    if critical:
        findings.append(
            Finding(
                check="pickle.dangerous_import",
                severity=Severity.CRITICAL,
                title="Pickle imports a known code-execution primitive",
                detail=(
                    f"{source}: the stream imports "
                    + ", ".join(e["import"] for e in critical[:MAX_REPORTED_ITEMS])
                    + ". Combined with a REDUCE-family opcode this executes on load."
                ),
                remediation=(
                    "Treat this file as hostile. Delete it and obtain the model in "
                    "safetensors or GGUF form from the publisher."
                ),
                evidence={"source": source, "imports": critical[:MAX_REPORTED_ITEMS]},
            )
        )

    if high:
        findings.append(
            Finding(
                check="pickle.suspicious_import",
                severity=Severity.HIGH,
                title="Pickle imports from a module with no place in a model file",
                detail=(
                    f"{source}: imports from process/network/interpreter modules: "
                    + ", ".join(e["import"] for e in high[:MAX_REPORTED_ITEMS])
                ),
                remediation="Do not load. Prefer a safetensors build of the same model.",
                evidence={"source": source, "imports": high[:MAX_REPORTED_ITEMS]},
            )
        )

    if gadget:
        findings.append(
            Finding(
                check="pickle.gadget_import",
                severity=Severity.MEDIUM,
                title="Pickle imports a known gadget-capable helper",
                detail=(
                    f"{source}: imports "
                    + ", ".join(e["import"] for e in gadget[:MAX_REPORTED_ITEMS])
                    + ". These appear in legitimate numpy/torch pickles but are also used to "
                    "stage encoded payloads, so presence alone is not proof of malice."
                ),
                remediation="Review manually, or convert the model to safetensors.",
                evidence={"source": source, "imports": gadget[:MAX_REPORTED_ITEMS]},
            )
        )

    if unknown:
        findings.append(
            Finding(
                check="pickle.unknown_import",
                severity=Severity.MEDIUM,
                title="Pickle imports symbols outside the expected ML allowlist",
                detail=(
                    f"{source}: unrecognised imports: "
                    + ", ".join(e["import"] for e in unknown[:MAX_REPORTED_ITEMS])
                    + f" ({len(unknown)} total)."
                ),
                remediation="Verify each import is expected before loading this checkpoint.",
                evidence={"source": source, "imports": unknown[:MAX_REPORTED_ITEMS]},
            )
        )

    if exec_ops:
        opcode_names = sorted({name for name, _ in exec_ops})
        findings.append(
            Finding(
                check="pickle.execution_opcodes",
                severity=Severity.MEDIUM,
                title="Pickle contains callable-invoking opcodes",
                detail=(
                    f"{source}: found {len(exec_ops)} opcode(s) of type "
                    f"{', '.join(opcode_names)}. Every legitimate PyTorch checkpoint also "
                    "contains these, so this is context, not a verdict on its own."
                ),
                remediation=(
                    "Use torch.load(..., weights_only=True) where loading is unavoidable, "
                    "or migrate to safetensors."
                ),
                evidence={"source": source, "opcodes": opcode_names, "count": len(exec_ops)},
            )
        )

    if traversal_strings:
        findings.append(
            Finding(
                check="pickle.traversal_string",
                severity=Severity.MEDIUM,
                title="Pickle embeds path-traversal-shaped strings",
                detail=(
                    f"{source}: strings such as "
                    + ", ".join(repr(s[:80]) for s, _ in traversal_strings[:5])
                    + " could steer a permissive loader outside its model directory."
                ),
                remediation="Do not load. Report the artefact to the model publisher.",
                evidence={
                    "source": source,
                    "strings": [{"value": s[:200], "offset": p}
                                for s, p in traversal_strings[:MAX_REPORTED_ITEMS]],
                },
            )
        )

    return findings, errors


def _zip_member_findings(info: zipfile.ZipInfo) -> Iterable[Finding]:
    name = info.filename
    normalised = name.replace("\\", "/")

    if normalised.startswith("/") or ".." in normalised.split("/") or looks_like_traversal(name):
        yield Finding(
            check="zip.path_traversal",
            severity=Severity.CRITICAL,
            title="Archive member escapes the extraction directory (zip-slip)",
            detail=(
                f"Member {name!r} contains an absolute or parent-relative path. An extractor "
                "that joins this onto a base directory writes outside it — the same primitive "
                "as CVE-2024-37032 in Ollama."
            ),
            remediation="Delete this archive. Do not extract or load it.",
            evidence={"member": name},
        )

    if normalised.lower().endswith((".py", ".pyc", ".so", ".dll", ".dylib", ".sh", ".bat", ".ps1")):
        yield Finding(
            check="zip.executable_member",
            severity=Severity.HIGH,
            title="Archive contains code or native library members",
            detail=f"Member {name!r} is code, not tensor data.",
            remediation="Do not load. A weights archive should contain only weights and metadata.",
            evidence={"member": name},
        )

    if info.compress_size > 0 and info.file_size > 100 * 1024 * 1024:
        ratio = info.file_size / info.compress_size
        if ratio > 200:
            yield Finding(
                check="zip.decompression_ratio",
                severity=Severity.MEDIUM,
                title="Archive member has an extreme decompression ratio",
                detail=(
                    f"Member {name!r} expands {ratio:.0f}x to {info.file_size} bytes — "
                    "consistent with a decompression bomb."
                ),
                remediation="Do not extract without a size-capped extractor.",
                evidence={"member": name, "ratio": round(ratio, 1), "file_size": info.file_size},
            )


def analyze_zip_container(path: Path) -> tuple[list[Finding], list[str]]:
    """Inspect a ZIP-based checkpoint (PyTorch .pt/.bin) and its pickle members."""
    findings: list[Finding] = []
    errors: list[str] = []

    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_ZIP_MEMBERS:
                errors.append(
                    f"archive has {len(infos)} members; inspecting first {MAX_ZIP_MEMBERS}."
                )
                infos = infos[:MAX_ZIP_MEMBERS]

            pickle_members = 0
            for info in infos:
                findings.extend(_zip_member_findings(info))

                lowered = info.filename.lower()
                if lowered.endswith((".pkl", ".pickle")) or lowered.endswith("/data.pkl"):
                    pickle_members += 1
                    try:
                        with archive.open(info) as member:
                            member_findings, member_errors = analyze_pickle_stream(
                                member, info.filename
                            )
                        findings.extend(member_findings)
                        errors.extend(member_errors)
                    except (zipfile.BadZipFile, OSError, RuntimeError) as exc:
                        errors.append(f"{info.filename}: could not read member: {exc}")

            if pickle_members == 0:
                findings.append(
                    Finding(
                        check="zip.no_pickle_member",
                        severity=Severity.INFO,
                        title="ZIP archive contains no pickle member",
                        detail="No .pkl member found; the deserialisation check did not apply.",
                        remediation="",
                        evidence={"members": len(infos)},
                    )
                )
    except (zipfile.BadZipFile, OSError) as exc:
        errors.append(f"{path}: not a readable ZIP archive: {exc}")

    return findings, errors

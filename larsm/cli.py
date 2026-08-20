from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .findings import ScanResult, Severity
from .scanner import scan_path

EXIT_CLEAN = 0
EXIT_FINDINGS = 1
EXIT_ERROR = 2


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _render_text(results: list[ScanResult], threshold: Severity) -> None:
    print(f"LARSM {__version__} — model file scan")
    print(f"Scanned {len(results)} file(s)\n")

    counts: dict[str, int] = {}
    for result in results:
        counts[result.max_severity.name] = counts.get(result.max_severity.name, 0) + 1

        marker = "!" if result.max_severity >= threshold else " "
        print(
            f"{marker} [{result.max_severity.name:<8}] {result.path}  "
            f"({result.file_format}, {_human_size(result.size_bytes)})"
        )
        for finding in result.sorted_findings():
            print(f"      - [{finding.severity.name}] {finding.check}: {finding.title}")
            print(f"        {finding.detail}")
            if finding.remediation:
                print(f"        fix: {finding.remediation}")
        for error in result.errors:
            print(f"      ! error: {error}")
        print()

    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing scanned")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="larsm",
        description="Local AI Runtime Security Monitor — model file scanner.",
    )
    parser.add_argument("--version", action="version", version=f"larsm {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="scan a model file or directory")
    scan.add_argument("target", type=Path, help="file or directory to scan")
    scan.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    scan.add_argument("--no-recurse", action="store_true", help="do not descend into subfolders")
    scan.add_argument(
        "--all-files",
        action="store_true",
        help="scan every file, not just known model extensions",
    )
    scan.add_argument(
        "--fail-on",
        default="HIGH",
        help="minimum severity that sets exit code 1 (default: HIGH)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        threshold = Severity.parse(args.fail_on)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if not args.target.exists():
        print(f"error: no such path: {args.target}", file=sys.stderr)
        return EXIT_ERROR

    results = scan_path(args.target, recursive=not args.no_recurse, all_files=args.all_files)

    if args.json:
        payload = {
            "tool": "larsm",
            "version": __version__,
            "target": str(args.target),
            "fail_on": threshold.name,
            "results": [r.to_dict() for r in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        _render_text(results, threshold)

    if any(r.max_severity >= threshold for r in results):
        return EXIT_FINDINGS
    return EXIT_CLEAN


if __name__ == "__main__":
    raise SystemExit(main())

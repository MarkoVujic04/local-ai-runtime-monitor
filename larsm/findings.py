from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

class Severity(IntEnum):
    INFO = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

    def __str__(self) -> str:
        return self.name

    @classmethod
    def parse(cls, text: str) -> "Severity":
        try:
            return cls[text.strip().upper()]
        except KeyError as exc:
            valid = ", ".join(level.name for level in cls)
            raise ValueError(f"unknown severity {text!r}; expected one of: {valid}") from exc


@dataclass
class Finding:
    check: str
    severity: Severity
    title: str
    detail: str
    remediation: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.name,
            "title": self.title,
            "detail": self.detail,
            "remediation": self.remediation,
            "evidence": self.evidence,
        }


@dataclass
class ScanResult:
    path: str
    file_format: str
    size_bytes: int
    findings: list[Finding] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def max_severity(self) -> Severity:
        if not self.findings:
            return Severity.INFO
        return max(finding.severity for finding in self.findings)

    def sorted_findings(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: (-int(f.severity), f.check))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "format": self.file_format,
            "size_bytes": self.size_bytes,
            "max_severity": self.max_severity.name,
            "findings": [f.to_dict() for f in self.sorted_findings()],
            "errors": self.errors,
        }

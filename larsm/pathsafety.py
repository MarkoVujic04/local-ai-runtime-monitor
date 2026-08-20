from __future__ import annotations

MAX_INSPECTED_LENGTH = 4096

_TRAVERSAL_MARKERS = (
    "../",
    "..\\",
    "%2e%2e",
    "..%2f",
    "..%5c",
    "....//",
    "..;/",
)

_ABSOLUTE_PREFIXES = (
    "/etc/",
    "/root/",
    "/proc/",
    "/var/",
    "/usr/",
    "\\\\",
    "%windir%",
    "%systemroot%",
    "%userprofile%",
    "$home",
)


def looks_like_traversal(text: str) -> bool:
    if not text or len(text) > MAX_INSPECTED_LENGTH:
        return False

    lowered = text.lower()

    if any(marker in lowered for marker in _TRAVERSAL_MARKERS):
        return True
    if lowered.startswith(_ABSOLUTE_PREFIXES):
        return True
    return len(lowered) > 3 and lowered[1:3] == ":\\" and lowered[0].isalpha()

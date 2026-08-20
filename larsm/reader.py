from __future__ import annotations

import struct
from typing import BinaryIO

MAX_STRING_BYTES = 64 * 1024 * 1024

MAX_COLLECTION_ITEMS = 10_000_000


class ParseError(Exception):
    """Base class for structural problems found while parsing."""


class TruncatedFile(ParseError):
    """A field claims more bytes than the file contains."""


class ImplausibleLength(ParseError):
    """A length or count is arithmetically impossible for this file size."""


class BoundedReader:
    """A read cursor that refuses to move past a known file size.
    Wraps a seekable binary handle. Every read is checked before it is
    performed, so a hostile length field raises rather than allocating.
    """

    def __init__(self, handle: BinaryIO, size: int) -> None:
        self._handle = handle
        self.size = size

    @property
    def pos(self) -> int:
        return self._handle.tell()

    @property
    def remaining(self) -> int:
        return max(0, self.size - self.pos)

    def seek(self, offset: int) -> None:
        if offset < 0 or offset > self.size:
            raise ImplausibleLength(f"seek to {offset} outside file of {self.size} bytes")
        self._handle.seek(offset)

    def read_exact(self, count: int, what: str = "field") -> bytes:
        if count < 0:
            raise ImplausibleLength(f"negative length {count} for {what}")
        if count > self.remaining:
            raise TruncatedFile(
                f"{what} claims {count} bytes but only {self.remaining} remain "
                f"(file is {self.size} bytes)"
            )
        data = self._handle.read(count)
        if len(data) != count:
            raise TruncatedFile(f"{what}: short read, got {len(data)} of {count} bytes")
        return data

    def _unpack(self, fmt: str, width: int, what: str) -> int:
        (value,) = struct.unpack(fmt, self.read_exact(width, what))
        return value

    def u32(self, what: str = "u32") -> int:
        return self._unpack("<I", 4, what)

    def u64(self, what: str = "u64") -> int:
        return self._unpack("<Q", 8, what)

    def i32(self, what: str = "i32") -> int:
        return self._unpack("<i", 4, what)

    def i64(self, what: str = "i64") -> int:
        return self._unpack("<q", 8, what)

    def checked_length(self, count: int, what: str) -> int:
        """Validate a byte-length field before it is used to read."""
        if count < 0:
            raise ImplausibleLength(f"negative length {count} for {what}")
        if count > MAX_STRING_BYTES:
            raise ImplausibleLength(
                f"{what} declares {count} bytes, above the {MAX_STRING_BYTES}-byte ceiling"
            )
        if count > self.remaining:
            raise TruncatedFile(
                f"{what} declares {count} bytes but only {self.remaining} remain"
            )
        return count

    def checked_count(self, count: int, min_bytes_each: int, what: str) -> int:
        """Validate a declared item count against what the file could hold.
        This is the allocation guard, a file claiming ten billion entries
        in 200 bytes is probably lying, and we can show that
        before allocating a single list slot.
        """
        if count < 0:
            raise ImplausibleLength(f"negative count {count} for {what}")
        if count > MAX_COLLECTION_ITEMS:
            raise ImplausibleLength(
                f"{what} declares {count} items, above the {MAX_COLLECTION_ITEMS} ceiling"
            )
        if min_bytes_each > 0 and count * min_bytes_each > self.remaining:
            raise ImplausibleLength(
                f"{what} declares {count} items needing at least "
                f"{count * min_bytes_each} bytes, but only {self.remaining} remain"
            )
        return count

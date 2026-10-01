#!/usr/bin/env python3
"""Verify the HHR source manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "HHR_MANIFEST.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    expected = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        recorded, relative_text = line.split("  ", 1)
        relative = PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"unsafe manifest path: {relative_text}")
        expected[relative.as_posix()] = recorded

    actual = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise SystemExit(f"manifest file-set mismatch; missing={missing}, extra={extra}")

    failures = [name for name, path in actual.items() if digest(path) != expected[name]]
    if failures:
        raise SystemExit("manifest hash mismatch: " + ", ".join(sorted(failures)))
    print(f"HHR manifest verified: {len(actual)} files")


if __name__ == "__main__":
    main()

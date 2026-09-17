"""Confirm the provided runner/scoring section of hw1.py is byte-identical to
the upstream starter.

The expected SHA-256 lives in tools/_runner_section.sha256 and was computed from
the pristine upstream file, so this check needs no network and no second copy of
the starter in the repository.

Exit code 0 means untouched; 1 means the runner section differs.
"""
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MARKER = "# Everything below is provided runner/scoring code. No edits are needed."
EXPECTED_FILE = Path(__file__).resolve().parent / "_runner_section.sha256"

source = (ROOT / "hw1.py").read_text(encoding="utf-8")
if MARKER not in source:
    print(f"FAIL: runner marker not found in {ROOT / 'hw1.py'}")
    sys.exit(1)

section = source[source.index(MARKER):]
digest = hashlib.sha256(section.encode("utf-8")).hexdigest()
expected = EXPECTED_FILE.read_text(encoding="utf-8").strip()

print(f"runner section: {len(section)} chars")
print(f"sha256        : {digest}")
print(f"expected      : {expected}")

if digest == expected:
    print("\nIDENTICAL -> the provided runner/scoring code is untouched.")
    sys.exit(0)

print("\nMODIFIED -> the grader-provided code was edited. Revert it.")
sys.exit(1)

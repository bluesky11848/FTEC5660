"""End-to-end dry run of hw1.py with the vision model stubbed out.

Proves the REAL entry point works -- argparse, build_chain/answer_queries
wiring, and the provided write_results() -- without needing an API key. The
stub returns the ground-truth per-receipt figures so the only thing being
tested is the plumbing, not the model.

Usage:
    python tools/mock_cli_run.py            # uses public_test/
    python tools/mock_cli_run.py <folder>
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hw1  # noqa: E402


class _ScriptedChain:
    """Returns one canned extraction per call, in order."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def _next(self):
        self.calls += 1
        return self.script[(self.calls - 1) % len(self.script)]

    def invoke(self, payload):
        return self._next()

    def batch(self, payloads, config=None, return_exceptions=False):
        out = []
        for _ in payloads:
            try:
                out.append(self._next())
            except Exception as exc:  # noqa: BLE001
                if not return_exceptions:
                    raise
                out.append(exc)
        return out


def build_script(folder):
    """One canned extraction per image, in the order hw1.py will read them."""
    truth = json.loads((folder / "ground_truth.json").read_text(encoding="utf-8"))
    per_receipt = truth["receipts"]
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in hw1.IMAGE_EXTENSIONS
    )
    script = []
    for image in images:
        figures = per_receipt.get(image.name)
        if figures is None:
            script.append({})
            continue
        subtotal = Decimal(str(figures["subtotal_after_discounts_before_rounding"]))
        paid = Decimal(str(figures["amount_paid_after_rounding"]))
        discount = Decimal(str(figures["discount_total"]))
        script.append({
            "receipt_type": "MOCK",
            "amount_paid_after_rounding": f"{paid:.2f}",
            "subtotal_after_discounts_before_rounding": f"{subtotal:.2f}",
            "discount_total": f"{discount:.2f}",
            "rounding_adjustment": f"{paid - subtotal:.2f}",
            "line_items": [],
            "discount_lines": [f"{discount:.2f}"],
        })
    return script


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "public_test"
    if not (folder / "ground_truth.json").is_file():
        print(f"no ground_truth.json in {folder}; cannot build a mock")
        return 2

    script = build_script(folder)

    # Swap in the stub, then run the untouched main().
    original = hw1.build_chain
    hw1.build_chain = lambda: _ScriptedChain(script)
    try:
        sys.argv = ["hw1.py", "--image-folder", str(folder)]
        rc = hw1.main()
    finally:
        hw1.build_chain = original

    print(f"\nmain() returned {rc}")
    csv_path = Path("results.csv")
    if csv_path.is_file():
        print("\n--- results.csv ---")
        print(csv_path.read_text(encoding="utf-8").strip())
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

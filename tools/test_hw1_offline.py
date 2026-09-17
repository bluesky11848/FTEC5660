"""Offline validation for hw1.py -- exercises the deterministic half of the chain.

Stage 1 (the vision model) is replaced by a fake chain returning scripted
transcriptions, so aggregation, reconciliation and rendering can be tested
without an API key. Real image files are generated so the full
answer_queries -> observe -> normalise -> aggregate -> render path is covered.

Run:  python _tools/test_offline.py
"""
import base64
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hw1  # noqa: E402

PASSED = 0
FAILED = []


def check(name, got, want):
    global PASSED
    if got == want:
        PASSED += 1
        print(f"  PASS  {name}")
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}: got {got!r} want {want!r}")


# A tiny but genuinely valid 1x1 PNG so image_data_url() reads real bytes.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
TMP = Path(tempfile.mkdtemp(prefix="hw1_offline_"))


def make_image(name):
    path = TMP / name
    path.write_bytes(PNG)
    return path


# Per-receipt figures transcribed from public_test/ground_truth.json exactly as
# a vision model would report them after reading each receipt image.
GT = {
    "receipt1.jpg": ("394.70", "394.72", "85.48", "480.20"),
    "receipt2.jpg": ("316.10", "316.11", "76.09", "392.20"),
    "receipt3.jpg": ("140.80", "140.88", "19.22", "160.10"),
    "receipt4.jpg": ("514.00", "514.09", "76.71", "590.80"),
    "receipt5.jpg": ("102.30", "102.31", "5.39", "107.70"),
    "receipt6.jpg": ("190.80", "190.89", "30.31", "221.20"),
    "receipt7.jpg": ("315.60", "315.64", "80.36", "396.00"),
}

IMAGES = [make_image(n) for n in GT]


def scripted(name, **overrides):
    paid, subtotal, discount, _ = GT[name]
    # ROUNDING = paid - subtotal, which is negative on every public receipt.
    rounding = Decimal(paid) - Decimal(subtotal)
    data = {
        "receipt_type": "PARKNSHOP",
        "amount_paid_after_rounding": paid,
        "subtotal_after_discounts_before_rounding": subtotal,
        "discount_total": discount,
        "rounding_adjustment": f"{rounding:.2f}",
        "line_items": [],
        "discount_lines": [discount],
    }
    data.update(overrides)
    return data


class FakeChain:
    """Returns scripted model output, cycling if exhausted."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def _next(self):
        self.calls += 1
        return self.script[(self.calls - 1) % len(self.script)]

    def invoke(self, payload):
        assert "image" in payload, "prompt variable missing"
        return self._next()

    def batch(self, payloads, config=None, return_exceptions=False):
        # Mirror LangChain: one result per input, in order.
        out = []
        for payload in payloads:
            try:
                out.append(self.invoke(payload))
            except Exception as exc:  # noqa: BLE001
                if not return_exceptions:
                    raise
                out.append(exc)
        return out


class BoomChain:
    def invoke(self, payload):
        raise RuntimeError("simulated API failure")

    def batch(self, payloads, config=None, return_exceptions=False):
        out = []
        for _ in payloads:
            try:
                out.append(self.invoke({}))
            except Exception as exc:  # noqa: BLE001
                if not return_exceptions:
                    raise
                out.append(exc)
        return out


class NoBatchChain:
    """A chain exposing only invoke(), to exercise the sequential fallback."""

    def __init__(self, script):
        self.inner = FakeChain(script)

    def invoke(self, payload):
        return self.inner.invoke(payload)


# --------------------------------------------------------------------------- #
print("\n=== T1: full public_test folder (7 receipts) via answer_queries ===")
resp = hw1.answer_queries(FakeChain([scripted(n) for n in GT]), IMAGES)
check("Q1 response", resp[hw1.QUERY_1], "HK$1974.30")
check("Q2 response", resp[hw1.QUERY_2], "HK$2348.20")
check("Q1 parses as one number", hw1.parse_single_amount(resp[hw1.QUERY_1]), Decimal("1974.30"))
check("Q2 parses as one number", hw1.parse_single_amount(resp[hw1.QUERY_2]), Decimal("2348.20"))
check("exactly the two query keys", set(resp), set(hw1.QUERIES))
check("values are plain strings", all(isinstance(v, str) for v in resp.values()), True)

print("\n=== T2: PDF worked example (receipt5), verbose JSON-in-prose output ===")
noisy = (
    "Here is the receipt:\n```json\n"
    '{"receipt_type":"PARKNSHOP","line_items":["10.00","36.90","60.80"],'
    '"discount_lines":["5.39"],"subtotal_after_discounts_before_rounding":"102.31",'
    '"rounding_adjustment":"-0.01","amount_paid_after_rounding":"102.30"}'
    "\n```\nLet me know if you need anything else."
)
resp = hw1.answer_queries(FakeChain([noisy]), [make_image("r5.jpg")])
check("Q1 (receipt5)", resp[hw1.QUERY_1], "HK$102.30")
check("Q2 (receipt5)", resp[hw1.QUERY_2], "HK$107.70")

print("\n=== T3: discount summary absent, only itemised discount lines ===")
resp = hw1.answer_queries(
    FakeChain([{
        "line_items": ["10.00", "36.90", "60.80"],
        "discount_lines": ["5.39"],
        "subtotal_after_discounts_before_rounding": "102.31",
        "rounding_adjustment": "-0.01",
        "amount_paid_after_rounding": "102.30",
    }]),
    [make_image("r3.jpg")],
)
check("Q1 (derived discount)", resp[hw1.QUERY_1], "HK$102.30")
check("Q2 (derived discount)", resp[hw1.QUERY_2], "HK$107.70")

print("\n=== T4: model omits paid amount; rounding lets us reconstruct it ===")
resp = hw1.answer_queries(
    FakeChain([{
        "subtotal_after_discounts_before_rounding": "102.31",
        "discount_total": "5.39",
        "rounding_adjustment": "-0.01",
        "line_items": ["10.00", "36.90", "60.80"],
    }]),
    [make_image("r4.jpg")],
)
check("Q1 via subtotal+rounding", resp[hw1.QUERY_1], "HK$102.30")
check("Q2 via subtotal+discount", resp[hw1.QUERY_2], "HK$107.70")

print("\n=== T5: receipt with no discounts at all ===")
resp = hw1.answer_queries(
    FakeChain([{
        "subtotal_after_discounts_before_rounding": "50.00",
        "discount_total": "0",
        "rounding_adjustment": "0.00",
        "amount_paid_after_rounding": "50.00",
        "line_items": ["20.00", "30.00"],
    }]),
    [make_image("r5b.jpg")],
)
check("Q1 (no discount)", resp[hw1.QUERY_1], "HK$50.00")
check("Q2 (no discount)", resp[hw1.QUERY_2], "HK$50.00")

print("\n=== T6: model quotes the printed negative discount ===")
rec = hw1._normalise(scripted("receipt5.jpg", discount_total="-5.39"))
check("discount normalised to +5.39", rec["discount"], Decimal("5.39"))
check("Q2 base", rec["without_discount"], Decimal("107.70"))

print("\n=== T7: reconciliation between SUBTOTAL, ROUNDING and the paid total ===")
# (a) Model omits the ROUNDING line; the paid figure still reveals the adjustment.
rec = hw1._normalise({
    "subtotal_after_discounts_before_rounding": "102.31",
    "discount_total": "5.39",
    "amount_paid_after_rounding": "102.30",
})
check("rounding recovered from paid", rec["rounding"], Decimal("-0.01"))
check("paid preserved", rec["paid"], Decimal("102.30"))
check("Q2 base unaffected by rounding", rec["without_discount"], Decimal("107.70"))

# (b) Model reports the pre-discount total as "paid" but also gives ROUNDING.
# The printed ROUNDING line is authoritative, so Q1 must come out at 102.30.
rec = hw1._normalise(scripted("receipt5.jpg", amount_paid_after_rounding="107.70"))
check("paid rebuilt from subtotal+rounding", rec["paid"], Decimal("102.30"))
check("rounding kept", rec["rounding"], Decimal("-0.01"))

# (c) A wildly implausible paid total (> 0.50 away from subtotal+rounding) is
# discarded on the same principle.
rec = hw1._normalise({
    "subtotal_after_discounts_before_rounding": "102.31",
    "discount_total": "5.39",
    "rounding_adjustment": "-0.01",
    "amount_paid_after_rounding": "9999.90",
})
check("absurd paid discarded", rec["paid"], Decimal("102.30"))

# (d) All three sources agreeing must be left exactly as they are.
rec = hw1._normalise(scripted("receipt3.jpg"))
check("receipt3 paid untouched", rec["paid"], Decimal("140.80"))
check("receipt3 rounding", rec["rounding"], Decimal("-0.08"))
check("receipt3 Q2 base", rec["without_discount"], Decimal("160.10"))

print("\n=== T8: printed 'without discount' total is used when plausible ===")
rec = hw1._normalise(
    scripted("receipt5.jpg", printed_amount_without_discounts="107.70")
)
check("Q2 base honours printed total", rec["without_discount"], Decimal("107.70"))
rec = hw1._normalise(
    scripted("receipt5.jpg", printed_amount_without_discounts="999.00")
)
check("implausible printed total ignored", rec["without_discount"], Decimal("107.70"))

print("\n=== T9: garbled output is skipped, never fatal ===")
resp = hw1.answer_queries(FakeChain(["I cannot read this image."]), [make_image("g.jpg")])
check("Q1 survives garbage", resp[hw1.QUERY_1], "HK$0.00")
check("Q2 survives garbage", resp[hw1.QUERY_2], "HK$0.00")

print("\n=== T10: every API call failing is still not fatal ===")
resp = hw1.answer_queries(BoomChain(), [make_image("b.jpg")])
check("Q1 survives API failure", resp[hw1.QUERY_1], "HK$0.00")
check("Q2 survives API failure", resp[hw1.QUERY_2], "HK$0.00")

print("\n=== T11: missing image files are skipped, not fatal ===")
resp = hw1.answer_queries(FakeChain([scripted("receipt5.jpg")]),
                          [Path("does_not_exist.jpg")])
check("Q1 with unreadable path", resp[hw1.QUERY_1], "HK$0.00")

print("\n=== T12: renderer always yields exactly one parseable number ===")
for amount in ("1974.30", "0.00", "2348.20", "102.30", "1234567.89", "5.39"):
    text = hw1._render(Decimal(amount))
    check(f"render {amount} -> {text}",
          (len(hw1._MONEY_RE.findall(text)), hw1.parse_single_amount(text)),
          (1, Decimal(amount)))

print("\n=== T13: the grader REJECTS these naive answer styles (why we render) ===")
for bad in ("Total HK$1974.30 across 7 receipts",
            "HK$1974.30 (was HK$2348.20)",
            "After 5% off: HK$1974.30",
            "Sum of 7 receipts = HK$1974.30",
            "HK$1974.30.",              # trailing period -> ZERO matches
            "2024-09-29 HK$1974.30",    # a date contributes extra numbers
            "HK$1974.30 and HK$1974.30"):
    check(f"rejected: {bad[:34]!r}", hw1.parse_single_amount(bad), None)

print("\n=== T13b: renderer must never emit a trailing period or lost sign ===")
# The graded regex forbids a trailing '.', which makes "HK$1974.30." match
# nothing at all; and "-HK$5.39" silently loses its sign. The renderer emits
# neither shape, so assert the contract explicitly.
check("no trailing period", hw1._render(Decimal("1974.30")).endswith("."), False)
check("negative renders with sign kept",
      hw1.parse_single_amount(hw1._render(Decimal("-5.39"))), Decimal("-5.39"))
check("trailing period really is fatal",
      hw1.parse_single_amount("HK$1974.30."), None)
check("leading-minus form drops the sign (grader quirk)",
      hw1.parse_single_amount("-HK$5.39"), Decimal("5.39"))

print("\n=== T14: each single receipt in isolation ===")
for name, (paid, subtotal, discount, without) in GT.items():
    resp = hw1.answer_queries(FakeChain([scripted(name)]), [make_image(f"s_{name}")])
    check(f"{name} Q1", resp[hw1.QUERY_1], f"HK${Decimal(paid):.2f}")
    check(f"{name} Q2", resp[hw1.QUERY_2], f"HK${Decimal(without):.2f}")

print("\n=== T15: determinism over 5 runs ===")
runs = {
    tuple(sorted(hw1.answer_queries(
        FakeChain([scripted(n) for n in GT]), IMAGES).items()))
    for _ in range(5)
}
check("5 runs byte-identical", len(runs), 1)

print("\n=== T16: results.csv through the provided writer ===")
truth = {hw1.QUERY_1: Decimal("1974.30"), hw1.QUERY_2: Decimal("2348.20")}
csv_path = hw1.write_results(
    hw1.answer_queries(FakeChain([scripted(n) for n in GT]), IMAGES), truth
)
lines = Path(csv_path).read_text(encoding="utf-8").strip().splitlines()
print("    " + "\n    ".join(lines))
check("header", lines[0], "query,model_response,correctness")
check("row1 correct", lines[1].endswith("correct"), True)
check("row2 correct", lines[2].endswith("correct"), True)
Path(csv_path).unlink()

print("\n" + "=" * 64)
if FAILED:
    print(f"RESULT: {len(FAILED)} FAILED / {PASSED} passed")
    for name in FAILED:
        print(f"   - {name}")
    sys.exit(1)
print(f"RESULT: ALL {PASSED} OFFLINE CHECKS PASSED")

"""Debug harness: run the REAL vision chain on the public receipts and compare
each extracted field against public_test/ground_truth.json.

This is the only script that needs an API key. It prints per-receipt diffs so
prompt changes can be judged quickly, and it also verifies the two aggregate
answers through the same deterministic aggregation used by hw1.py.

Usage:
    python tools/debug_real.py                 # all 7 receipts
    python tools/debug_real.py receipt5.jpg    # one receipt
"""
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hw1  # noqa: E402

PUBLIC = ROOT / "public_test"
TRUTH = json.loads((PUBLIC / "ground_truth.json").read_text(encoding="utf-8"))
ANSWERS = TRUTH["answers"]
RECEIPTS = TRUTH["receipts"]


def cent(value):
    return None if value is None else Decimal(str(value)).quantize(Decimal("0.01"))


def main():
    wanted = sys.argv[1:]
    if wanted:
        names = [n for n in RECEIPTS if n in wanted or Path(n).name in wanted]
    else:
        names = sorted(RECEIPTS, key=lambda n: int("".join(c for c in n if c.isdigit())))

    images = [PUBLIC / n for n in names]
    hw1.load_env_file(ROOT / ".env")
    chain = hw1.build_chain()

    print(f"model: {__import__('os').environ.get('HW1_MODEL', 'deepseek-v4-flash-vision-exp')}")
    print(f"receipts: {len(images)}\n")

    ok = 0
    raw_records = []
    for name, path in zip(names, images):
        truth = RECEIPTS[name]
        print(f"--- {name} ---")
        try:
            raw = hw1._extract_once(chain, hw1.image_data_url(path))
        except Exception as exc:  # noqa: BLE001
            print(f"    ERROR: {type(exc).__name__}: {exc}\n")
            continue
        raw_records.append(raw)

        record = hw1._normalise(raw)
        exp_sub = cent(truth["subtotal_after_discounts_before_rounding"])
        exp_paid = cent(truth["amount_paid_after_rounding"])
        exp_without = cent(truth["amount_without_discounts"])
        exp_disc = cent(truth["discount_total"])

        if record is None:
            print(f"    FAILED TO PARSE. raw = {raw}\n")
            continue

        got_sub = record["subtotal"]
        got_paid = record["paid"]
        got_without = record["without_discount"]
        got_disc = record["discount"]

        def line(label, got, exp):
            flag = "ok  " if got == exp else "DIFF"
            print(f"    [{flag}] {label:<22} got {got:>10}   expected {exp:>10}")

        line("subtotal", got_sub, exp_sub)
        line("discount_total", got_disc, exp_disc)
        line("paid(Q1)", got_paid, exp_paid)
        line("without_disc(Q2 base)", got_without, exp_without)
        print(f"    self_consistent={record['self_consistent']}")

        if (got_sub, got_paid, got_without) == (exp_sub, exp_paid, exp_without):
            ok += 1
        else:
            print(f"    raw model output: {json.dumps(raw, ensure_ascii=False)[:600]}")
        print()

    print("=" * 64)
    print(f"receipts fully matching ground truth: {ok}/{len(images)}")

    records = [r for r in (hw1._normalise(x) for x in raw_records) if r]
    totals = hw1._aggregate(records)
    print(f"\nQ1 aggregated = {totals[hw1.QUERY_1]}  (expected {cent(ANSWERS[hw1.QUERY_1])})")
    print(f"Q2 aggregated = {totals[hw1.QUERY_2]}  (expected {cent(ANSWERS[hw1.QUERY_2])})")
    q1_ok = totals[hw1.QUERY_1] == cent(ANSWERS[hw1.QUERY_1])
    q2_ok = totals[hw1.QUERY_2] == cent(ANSWERS[hw1.QUERY_2])
    print(f"\nQ1 {'CORRECT' if q1_ok else 'WRONG'} | Q2 {'CORRECT' if q2_ok else 'WRONG'}")
    return 0 if (q1_ok and q2_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())

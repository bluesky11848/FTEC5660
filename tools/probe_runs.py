"""Run the real homework command N times and report which runs differ.

The assignment grades across three independent runs, so any run-to-run
variation matters. This prints the exact model_response of every run and, when
a run disagrees with ground truth, re-reads that receipt to show the raw
extraction.

Usage: python tools/probe_runs.py [n]
"""
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def main():
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    csv_path = ROOT / "results.csv"
    outcomes = []

    for i in range(1, runs + 1):
        if csv_path.is_file():
            csv_path.unlink()
        proc = subprocess.run(
            [sys.executable, "hw1.py", "--image-folder", "public_test"],
            cwd=ROOT, capture_output=True, text=True,
        )
        rows = read_csv(csv_path) if csv_path.is_file() else []
        responses = [r[1] for r in rows[1:]] if len(rows) > 1 else []
        verdicts = [r[2] for r in rows[1:]] if len(rows) > 1 else []
        ok = len(verdicts) == 2 and all(v.strip() == "correct" for v in verdicts)
        outcomes.append((tuple(responses), ok))
        print(f"run {i}: rc={proc.returncode} "
              f"{'OK  ' if ok else 'FAIL'} {responses}")
        if not ok:
            for row in rows[1:]:
                if row[2].strip() != "correct":
                    print(f"        -> {row[0][:45]}... | {row[2]}")
            if proc.stderr.strip():
                print("        stderr: " + proc.stderr.strip()[-400:])

            # Re-read every receipt and report which one disagrees, plus the
            # raw figures, so the cause is visible instead of inferred.
            print("        per-receipt diagnosis:")
            import json as _json

            import hw1  # noqa: E402

            hw1.load_env_file(ROOT / ".env")
            chain = hw1.build_chain()
            truth = _json.loads(
                (ROOT / "public_test" / "ground_truth.json").read_text("utf-8")
            )["receipts"]
            folder = ROOT / "public_test"
            images = sorted(
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in hw1.IMAGE_EXTENSIONS
            )
            for image in images:
                expected = truth.get(image.name)
                if expected is None:
                    continue
                raw = chain.invoke({"image": hw1.image_data_url(image)})
                rec = hw1._normalise(raw) if raw else None
                if rec is None:
                    print(f"          {image.name}: UNPARSEABLE")
                    continue
                exp_base = hw1._to_decimal(expected["amount_without_discounts"])
                flag = "ok " if rec["without_discount"] == exp_base else "DIFF"
                print(f"          [{flag}] {image.name}: q2base={rec['without_discount']} "
                      f"(exp {exp_base}) sub={rec['subtotal']} disc={rec['discount']} "
                      f"items_sum={rec['items_sum']}")

    print()
    distinct = {o[0] for o in outcomes}
    all_ok = all(o[1] for o in outcomes)
    print(f"distinct response sets: {len(distinct)}")
    print(f"all runs correct      : {all_ok}")
    if len(distinct) > 1:
        print("\nThe model is NOT run-to-run deterministic; each distinct set:")
        for response_set in distinct:
            print(f"   {response_set}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

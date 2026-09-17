#!/usr/bin/env python3
"""One-command test runner for FTEC5660 HW1.

Runs the three checks in increasing order of cost:

  1. offline  -- deterministic half of the chain, stubbed model, no API key
  2. mock     -- the real hw1.py entry point end to end, stubbed model
  3. real     -- the actual assignment command against the live model

Usage:
    python tools/run_tests.py            # checks 1 and 2 (no API key needed)
    python tools/run_tests.py --real     # also run check 3
    python tools/run_tests.py --real --runs 3   # and repeat the real run

Run it with the project's own interpreter, e.g. on Windows:
    .venv\\Scripts\\python.exe tools\\run_tests.py --real
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def banner(text):
    print("\n" + "=" * 68)
    print(text)
    print("=" * 68)


def run(label, cmd, keep_output=False):
    print(f"\n--- {label} ---")
    print("$ " + " ".join(str(c) for c in cmd))
    started = time.time()
    result = subprocess.run(cmd, cwd=ROOT, capture_output=keep_output, text=True)
    elapsed = time.time() - started
    if keep_output and result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        tail = result.stderr.strip().splitlines()[-8:]
        print("\n".join(tail))
    ok = result.returncode == 0
    print(f"[{'PASS' if ok else 'FAIL'}] {label}  ({elapsed:.1f}s)")
    return ok


def csv_all_correct(path):
    """True when results.csv has exactly the two expected rows, both 'correct'.

    Counting substrings is wrong here: "incorrect" also contains "correct".
    """
    import csv as _csv

    if not path.is_file():
        return False
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(_csv.reader(handle))
    if len(rows) != 3 or rows[0] != ["query", "model_response", "correctness"]:
        return False
    return all(row[2].strip() == "correct" for row in rows[1:])


def main():
    parser = argparse.ArgumentParser(description="Run the HW1 test suite")
    parser.add_argument("--real", action="store_true",
                        help="also run the live-model check (needs DEEPSEEK_API_KEY)")
    parser.add_argument("--runs", type=int, default=1,
                        help="how many times to repeat the live-model check")
    args = parser.parse_args()

    python = sys.executable
    results = {}

    banner("1/3  OFFLINE  (deterministic chain, stubbed model, no API key)")
    results["offline"] = run("offline self-check", [python, "tools/test_hw1_offline.py"])

    banner("2/3  MOCK     (real hw1.py entry point, stubbed model, no API key)")
    results["mock"] = run("mock end-to-end", [python, "tools/mock_cli_run.py"],
                          keep_output=True)
    csv_path = ROOT / "results.csv"
    results["mock_csv"] = csv_all_correct(csv_path)
    print(f"[{'PASS' if results['mock_csv'] else 'FAIL'}] "
          f"mock results.csv has two 'correct' rows")

    if not args.real:
        banner("3/3  REAL     (skipped -- pass --real to run it)")
    else:
        # .env is loaded by hw1.py itself; just make sure it exists.
        if not (ROOT / ".env").is_file():
            print("\nNo .env found. Create one containing:")
            print("    DEEPSEEK_API_KEY=your_key_here")
            return 2
        outputs = []
        for i in range(1, args.runs + 1):
            banner(f"3/3  REAL     (live model, run {i}/{args.runs})")
            results[f"real_{i}"] = run(
                f"python hw1.py --image-folder public_test  (run {i})",
                [python, "hw1.py", "--image-folder", "public_test"],
                keep_output=True,
            )
            if csv_path.is_file():
                outputs.append(csv_path.read_text(encoding="utf-8").strip())
                print(outputs[-1])
            results[f"real_{i}_correct"] = csv_all_correct(csv_path)
        if len(outputs) > 1:
            same = len(set(outputs)) == 1
            print(f"\n[{'PASS' if same else 'FAIL'}] {len(outputs)} runs are "
                  f"byte-identical (determinism)")
            results["determinism"] = same

    banner("SUMMARY")
    for key, value in results.items():
        print(f"  {'PASS' if value else 'FAIL'}  {key}")
    failed = [k for k, v in results.items() if not v]
    print()
    if failed:
        print(f"{len(failed)} CHECK(S) FAILED: {failed}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

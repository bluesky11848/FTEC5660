"""Audit every hard requirement stated in HW1.pdf against the submitted repo.

This is a checklist, not a test of model accuracy. Each item prints PASS/FAIL
with the concrete evidence.

Usage: python tools/audit_requirements.py
"""
import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_MARKER = "# Everything below is provided runner/scoring code. No edits are needed."

results = []


def check(label, ok, evidence=""):
    results.append((label, bool(ok), evidence))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if evidence:
        print(f"         {evidence}")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")


print("=" * 72)
print("HW1.pdf requirement audit")
print("=" * 72)

# --- 1. repo contents -------------------------------------------------------
print("\n[1] Required files present")
for name in ("hw1.py", "requirements.txt", "README.md"):
    check(f"{name} exists", (ROOT / name).is_file())

# --- 2. only two functions edited -------------------------------------------
print("\n[2] hw1.py scope: only the two ### YOUR CODE HERE functions")
up = git("show", "origin/main:hw1.py").stdout


def funcs(src):
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def toplevel(src):
    tree = ast.parse(src)
    return [ast.get_source_segment(src, n) for n in tree.body
            if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


cur_src = (ROOT / "hw1.py").read_text(encoding="utf-8")
uf, cf = funcs(up), funcs(cur_src)
changed = sorted(n for n in set(uf) & set(cf) if uf[n] != cf[n])
check("only build_chain + answer_queries changed", changed == ["answer_queries", "build_chain"],
      f"changed: {changed}")
check("no upstream function removed", not (set(uf) - set(cf)),
      f"removed: {sorted(set(uf) - set(cf)) or 'none'}")
check("module-level statements untouched", toplevel(up) == toplevel(cur_src))
check("no duplicate definitions shadowing grader code",
      len(cf) == len(set(cf)) and not ({n for n in cf if n in uf and n not in
                                        ("build_chain", "answer_queries")} - set(uf)))

# --- 3. provided runner untouched -------------------------------------------
print("\n[3] Provided runner/scoring code untouched")
tail_cur = cur_src[cur_src.index(UPSTREAM_MARKER):]
tail_up = up[up.index(UPSTREAM_MARKER):]
check("runner section byte-identical", tail_cur == tail_up,
      f"{len(tail_cur)} chars compared")

# --- 4. requirements.txt untouched -----------------------------------------
print("\n[4] requirements.txt")
up_req = git("show", "origin/main:requirements.txt").stdout
check("byte-identical to starter", (ROOT / "requirements.txt").read_text(encoding="utf-8").replace("\r\n", "\n") == up_req.replace("\r\n", "\n"))

# --- 5. .env never committed ------------------------------------------------
print("\n[5] .env must never be committed")
gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
check(".gitignore lists .env", any(l.strip() == ".env" for l in gi.splitlines()))
tracked = git("ls-files").stdout.splitlines()
check(".env not tracked", ".env" not in tracked)
remote_files = git("ls-tree", "-r", "--name-only", "fork/main").stdout.splitlines()
check(".env absent from remote", ".env" not in remote_files)
check("results.csv not tracked", "results.csv" not in tracked)

key = None
env = ROOT / ".env"
if env.is_file():
    m = re.search(r"DEEPSEEK_API_KEY=(\S+)", env.read_text(encoding="utf-8", errors="replace"))
    key = m.group(1) if m else None
if key:
    leaked = []
    for name in tracked:
        p = ROOT / name
        if p.is_file():
            try:
                if key in p.read_text(encoding="utf-8", errors="ignore"):
                    leaked.append(name)
            except OSError:
                pass
    check("API key not present in any tracked file", not leaked, f"leaks: {leaked or 'none'}")

# --- 6. model + framework ---------------------------------------------------
print("\n[6] Backbone model and framework")
check("ChatDeepSeek (LangChain) used", "from langchain_deepseek import ChatDeepSeek" in cur_src)
check("mandated model name is the default",
      'os.environ.get("HW1_MODEL", "deepseek-v4-flash-vision-exp")' in cur_src)
check("LCEL chain assembled (prompt | model | parser)",
      re.search(r"return prompt \| model \| RunnableLambda", cur_src) is not None)
check("images placed in the human message, not system",
      '{"type": "image_url", "image_url": {"url": "{image}"}}' in cur_src)

# --- 7. exact query strings -------------------------------------------------
print("\n[7] Exact query strings")
sys.path.insert(0, str(ROOT))
import hw1  # noqa: E402

check("QUERY_1 exact", hw1.QUERY_1 == "How much money did I spend in total for these bills?")
check("QUERY_2 exact", hw1.QUERY_2 == "How much would I have had to pay without the discount?")
check("QUERIES tuple unchanged", hw1.QUERIES == (hw1.QUERY_1, hw1.QUERY_2))

# --- 8. results.csv shape ---------------------------------------------------
print("\n[8] results.csv (present only if the command has been run)")
csv_path = ROOT / "results.csv"
if csv_path.is_file():
    rows = list(csv.reader(csv_path.open(encoding="utf-8", newline="")))
    check("header is query,model_response,correctness",
          rows and rows[0] == ["query", "model_response", "correctness"])
    check("two rows, one per query", len(rows) == 3)
    check("each response holds exactly one number",
          all(len(hw1._MONEY_RE.findall(r[1])) == 1 for r in rows[1:]) if len(rows) == 3 else False)
    check("both graded correct",
          all(r[2].strip() == "correct" for r in rows[1:]) if len(rows) == 3 else False,
          "; ".join(r[1] for r in rows[1:]) if len(rows) == 3 else "")
else:
    print("  [SKIP] results.csv not present in the working tree (it is gitignored)")

# --- 9. README requirements -------------------------------------------------
print("\n[9] README: 'Homework 1 solution' section")
readme = (ROOT / "README.md").read_text(encoding="utf-8")
check("section heading present", "Homework 1 solution" in readme)
solution = readme[readme.find("Homework 1 solution"):]
check("includes a chain-design visualisation",
      "STAGE 1" in solution and "```" in solution)
check("includes a solution paragraph", "### Description" in solution)
check("Task 2 reflection section present", "Task 2: Reflection" in readme)

# --- 10. .gitignore safety --------------------------------------------------
print("\n[10] .gitignore still protects secrets")
for pattern in (".env", ".venv/", "results.csv"):
    check(f"ignores {pattern}", any(l.strip() == pattern for l in gi.splitlines()))
check("upstream .gitignore lines not deleted (solution.py)",
      "solution.py" in gi or "solution.py" in git("show", "origin/main:.gitignore").stdout)

print("\n" + "=" * 72)
failed = [r for r in results if not r[1]]
print(f"TOTAL: {len(results) - len(failed)}/{len(results)} checks passed")
if failed:
    print("\nFAILED:")
    for label, _ok, ev in failed:
        print(f"  - {label}  {ev}")
    sys.exit(1)
print("ALL REQUIREMENT CHECKS PASSED")

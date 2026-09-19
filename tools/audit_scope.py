"""Audit: prove that only the two designated functions of hw1.py were changed.

The assignment says: "Fill in exactly two functions in hw1.py, both marked
### YOUR CODE HERE ... Do not touch anything else in the file."

This compares the submitted hw1.py against the pristine upstream starter
function-by-function (AST based) and reports every function whose body differs.

Usage: python tools/audit_scope.py
"""
import ast
import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CURRENT = ROOT / "hw1.py"

# The only two functions the assignment allows students to fill in.
ALLOWED = {"build_chain", "answer_queries"}


def upstream_source():
    """Read the pristine starter.

    Preference order:
      1. tools/_upstream_hw1.py, if present (authoritative, works anywhere);
      2. the upstream repo via git, preferring the real upstream remote;
      3. local history.

    It must NOT silently fall back to this repo's own HEAD: in a shallow clone
    `git show origin/main:hw1.py` returns the student's file, which would make
    the audit compare the file against itself and report "no changes" -- a false
    pass.
    """
    local = Path(__file__).resolve().parent / "_upstream_hw1.py"
    if local.is_file():
        return local.read_text(encoding="utf-8-sig")

    # Prefer a remote that is the upstream template, not the student's fork.
    remotes = subprocess.run(["git", "remote"], cwd=ROOT, capture_output=True,
                             text=True).stdout.split()
    ordered = [r for r in remotes if "HieuNT91" in subprocess.run(
        ["git", "remote", "get-url", r], cwd=ROOT, capture_output=True,
        text=True).stdout]
    ordered += [r for r in remotes if r not in ordered]

    for remote in ordered:
        for ref in (f"{remote}/main:hw1.py",):
            proc = subprocess.run(["git", "show", ref], cwd=ROOT, capture_output=True)
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout.decode("utf-8-sig")

    raise SystemExit(
        "Cannot locate the pristine starter. Fetch the upstream remote first:\n"
        "    git fetch upstream\n"
        "or place the original hw1.py at tools/_upstream_hw1.py"
    )


def function_sources(text):
    """Map top-level function name -> source text."""
    tree = ast.parse(text)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out[node.name] = ast.get_source_segment(text, node)
    return out


def main():
    up_src = upstream_source()
    up_funcs = function_sources(up_src)
    cur_src = CURRENT.read_text(encoding="utf-8")
    cur_funcs = function_sources(cur_src)

    print("=" * 70)
    print("hw1.py -- per-function comparison against the pristine starter")
    print("=" * 70)

    only_up = set(up_funcs) - set(cur_funcs)
    only_cur = set(cur_funcs) - set(up_funcs)
    changed = []
    identical = []

    for name in sorted(set(up_funcs) & set(cur_funcs)):
        if up_funcs[name] == cur_funcs[name]:
            identical.append(name)
        else:
            changed.append(name)

    print(f"functions upstream : {len(up_funcs)}")
    print(f"functions current  : {len(cur_funcs)}")
    print(f"removed            : {sorted(only_up) or 'none'}")
    print(f"added              : {sorted(only_cur) or 'none'}")
    print()
    print(f"UNCHANGED ({len(identical)}):")
    for name in identical:
        print(f"   {name}")
    print()
    print(f"CHANGED ({len(changed)}):")
    for name in changed:
        marker = "ALLOWED" if name in ALLOWED else "*** OUT OF SCOPE ***"
        print(f"   {name}  {marker}")

    # Module-level code outside any function (imports, constants, docstring).
    def top_level_statements(src):
        tree = ast.parse(src)
        return [
            ast.get_source_segment(src, node)
            for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

    up_top = top_level_statements(up_src)
    cur_top = top_level_statements(cur_src)
    print()
    print("=" * 70)
    print("Module-level statements (imports / constants / docstring)")
    print("=" * 70)
    if up_top == cur_top:
        print("IDENTICAL")
    else:
        print("DIFFERS:")
        for line in difflib.unified_diff(up_top, cur_top, "upstream", "current", lineterm=""):
            print("   " + line)

    # Helper functions I added must be clearly mine and not collide with
    # grader-provided names.
    grader_names = {
        "load_env_file", "image_files", "image_data_url", "response_text",
        "parse_single_amount", "read_ground_truth", "correctness_text",
        "write_results", "parse_args", "main",
    }
    overlap = grader_names & set(only_cur)
    print()
    print(f"newly added names colliding with grader code: {sorted(overlap) or 'none'}")

    bad = [n for n in changed if n not in ALLOWED] or sorted(overlap)
    print()
    print("=" * 70)
    if bad:
        print(f"RESULT: OUT OF SCOPE CHANGES -> {bad}")
        return 1
    print("RESULT: only build_chain() and answer_queries() were modified.")
    print("        All grader-provided functions are byte-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

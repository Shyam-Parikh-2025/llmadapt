"""Run every offline test file in tests/ and report a single pass/fail summary.

Keeps the project's plain-assert style: each test file is just a script that
raises on failure. This runner shells out to each one in its own subprocess so
a crash in one file can't take down the rest of the suite.

Usage (from the repo root):
    PYTHONPATH=src python3 tests/run_all.py

Skipped files (not automated tests):
  - test_trial_own.py   : an interactive REPL scratchpad; blocks on input().
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Files that live in tests/ but are not automated tests.
SKIP = {"run_all.py", "test_trial_own.py"}


def discover():
    names = []
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".py") or name in SKIP:
            continue
        if not (name.startswith("test_") or name.startswith("test ")):
            continue
        names.append(name)
    return names


def main():
    env = dict(os.environ)
    src = os.path.join(ROOT, "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")

    failures = []
    total_pass = 0
    for name in discover():
        path = os.path.join(HERE, name)
        proc = subprocess.run([sys.executable, path], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=900)
        out = (proc.stdout or "") + (proc.stderr or "")
        n_pass = sum(1 for line in out.splitlines() if line.strip().startswith("PASS"))
        total_pass += n_pass
        if proc.returncode == 0:
            print(f"  ok   {name:<28} ({n_pass} checks)")
        else:
            print(f"  FAIL {name:<28} (exit {proc.returncode})")
            failures.append((name, out))

    print()
    if failures:
        for name, out in failures:
            print(f"===== {name} =====")
            print("\n".join(out.splitlines()[-25:]))
            print()
        print(f"SUITE FAILED: {len(failures)} file(s) failed, {total_pass} checks passed")
        return 1

    print(f"SUITE PASSED: {len(discover())} files, {total_pass} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())

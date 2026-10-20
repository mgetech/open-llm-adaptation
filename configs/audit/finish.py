"""The last step: check the three jobs and put the results into results.zip."""

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CHECKS = [
    ("A_write", ["records_check.py"]),
    ("B_label", ["audit_check.py", "check"]),
    ("C_rewrite", ["rewrite_check.py", "check"]),
]


def main() -> int:
    failed = False
    for folder, command in CHECKS:
        print(f"== {folder}")
        run = subprocess.run(
            [sys.executable, str(ROOT / "tools" / command[0]), *command[1:]],
            cwd=ROOT / folder,
            capture_output=True,
            text=True,
        )
        print(run.stdout.strip() or run.stderr.strip())
        if run.returncode != 0:
            failed = True
        if folder == "C_rewrite" and "waiting 0," not in run.stdout:
            print("Job C is not finished: run  python ../tools/rewrite_check.py next  again.")
            failed = True
    if failed:
        print("Fix the problems above (jobs A and B must be complete), then run finish.py again.")
        return 1
    target = ROOT / "results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(ROOT / "A_write" / "new_records.jsonl", "A_write/new_records.jsonl")
        archive.write(ROOT / "B_label" / "answers.jsonl", "B_label/answers.jsonl")
        for name in ("accepted.jsonl", "report.txt"):
            archive.write(ROOT / "C_rewrite" / name, f"C_rewrite/{name}")
        for path in sorted((ROOT / "C_rewrite" / "out").glob("*.jsonl")):
            archive.write(path, f"C_rewrite/out/{path.name}")
    print(f"Everything is done. Send this file: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

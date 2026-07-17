"""Fast root-level entry point for all manuscript regression checks."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def main() -> None:
    for name in ("verify_key_results.py", "verify_table_ii.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "Analysis" / name)],
            cwd=ROOT,
            check=True,
        )
    print("All root-level regression checks passed.")


if __name__ == "__main__":
    main()

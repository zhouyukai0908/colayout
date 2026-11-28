"""One-command offline reproduction of all paper statistics, tables, macros, and figures from runs_v2/."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce all CoLayout paper tables, LaTeX number macros, Fig. 2, and Fig. 3 offline from runs_v2/."
    )
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs_v2")
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper")
    parser.add_argument("--skip-analysis", action="store_true", help="Reuse existing results_summary.json.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running pytest before export.")
    args = parser.parse_args()

    py = sys.executable
    if not args.skip_tests:
        _run([py, "-m", "pytest", "-q"])
    if not args.skip_analysis:
        _run([py, "scripts/analyze_runs_v2.py", "--runs-dir", str(args.runs_dir)])
    _run([
        py,
        "scripts/export_paper_v2.py",
        "--runs-dir",
        str(args.runs_dir),
        "--paper-dir",
        str(args.paper_dir),
    ])
    print(f"Done. Reproduced all tables, macros, and figures into {args.paper_dir}")


if __name__ == "__main__":
    main()

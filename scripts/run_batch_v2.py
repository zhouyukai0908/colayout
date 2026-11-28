"""Durable 8-method x 36-brief x 3-seed batch orchestrator (Plan B + 20% stop rule).

Writes strictly under runs_v2/seed_{1,2,3}/ (never touching legacy results/).
Executes up to 3 whole-run attempts per pair (attempt_index = 0, 1, 2) using
derived API seeds (effective_seed = seed + 1000 * attempt_index). Halts
immediately if any method's whole-run failure rate exceeds 20% in a seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_experiments import METHODS, run_experiments

MAX_WHOLE_ATTEMPTS = 3
MAX_FAILURE_RATE = 0.20


def check_failure_guard(seed_dir: Path, seed: int, n_briefs: int = 36) -> dict[str, dict]:
    raw_dir = seed_dir / "raw"
    fail_dir = seed_dir / "failures"
    stats = {}
    breaches = []
    for m in METHODS:
        n_ok = len(list(raw_dir.glob(f"{m}__*.json"))) if raw_dir.is_dir() else 0
        n_fail_attempts = len(list(fail_dir.glob(f"{m}__*.json"))) if fail_dir.is_dir() else 0
        n_residual = max(0, n_briefs - n_ok)
        attempt_fail_rate = n_fail_attempts / (n_ok + n_fail_attempts) if (n_ok + n_fail_attempts) else 0.0
        residual_rate = n_residual / n_briefs
        stats[m] = {
            "completed": n_ok,
            "failed_attempts": n_fail_attempts,
            "residual_failed": n_residual,
            "attempt_fail_rate": attempt_fail_rate,
            "residual_rate": residual_rate,
        }
        if attempt_fail_rate > MAX_FAILURE_RATE or residual_rate > MAX_FAILURE_RATE:
            breaches.append(
                f"{m}: completed={n_ok}/{n_briefs}, failed_attempts={n_fail_attempts} "
                f"(attempt_fail_rate={attempt_fail_rate*100:.1f}%, residual={residual_rate*100:.1f}%)"
            )
    if breaches:
        raise RuntimeError(f"STOP RULE (>20% failure rate in seed_{seed}): " + "; ".join(breaches))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full 8 x 36 x 3 batch into runs_v2/.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs_v2")
    parser.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-rounds", type=int, default=5)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    methods = list(METHODS)
    t_all = time.perf_counter()

    for seed in seeds:
        seed_dir = args.out_dir / f"seed_{seed}"
        t_seed = time.perf_counter()
        print(f"\n================ SEED {seed} ({args.model}) ================", flush=True)
        for attempt_idx in range(MAX_WHOLE_ATTEMPTS):
            eff_seed = seed + 1000 * attempt_idx
            print(f"[seed_{seed}] Attempt {attempt_idx + 1}/{MAX_WHOLE_ATTEMPTS} (effective_seed={eff_seed})...", flush=True)
            try:
                run_experiments(
                    methods,
                    None,
                    args.max_rounds,
                    args.out_dir,
                    workers=args.workers,
                    continue_on_invalid=True,
                    seed=seed,
                    model=args.model,
                    attempt_index=attempt_idx,
                )
                break
            except ValueError as exc:
                if "Incomplete benchmark:" not in str(exc):
                    raise
                print(f"[seed_{seed}] Attempt {attempt_idx + 1} finished with residual failures: {exc}", flush=True)
                # Early guard right after attempt 0 if attempt failure rate already exceeds 20%
                raw_dir = seed_dir / "raw"
                fail_dir = seed_dir / "failures"
                for m in methods:
                    n_ok = len(list(raw_dir.glob(f"{m}__*.json"))) if raw_dir.is_dir() else 0
                    n_f = len(list(fail_dir.glob(f"{m}__*.json"))) if fail_dir.is_dir() else 0
                    if (n_ok + n_f) >= 36 and (n_f / (n_ok + n_f)) > MAX_FAILURE_RATE:
                        raise RuntimeError(
                            f"STOP RULE (>20% whole-run failure rate in seed_{seed} after attempt {attempt_idx+1}): "
                            f"{m} ok={n_ok}, failed_attempts={n_f} ({n_f/(n_ok+n_f)*100:.1f}%)"
                        ) from exc

        stats = check_failure_guard(seed_dir, seed)
        tot_ok = sum(s["completed"] for s in stats.values())
        tot_fail = sum(s["failed_attempts"] for s in stats.values())
        tot_res = sum(s["residual_failed"] for s in stats.values())
        print(
            f"[seed_{seed}] COMPLETE in {time.perf_counter() - t_seed:.1f}s "
            f"(completed={tot_ok}/{len(methods)*36}, failed_attempts={tot_fail}, residual_failed={tot_res})",
            flush=True,
        )
        for m, st in stats.items():
            print(
                f"  {m:<26} ok={st['completed']:2d}/36  fails={st['failed_attempts']:2d} "
                f"({st['attempt_fail_rate']*100:4.1f}%)  residual={st['residual_failed']}",
                flush=True,
            )

    print(f"\nALL SEEDS {seeds} COMPLETE in {(time.perf_counter() - t_all)/60:.1f} min.", flush=True)


if __name__ == "__main__":
    main()

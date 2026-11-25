# CoLayout: Coordinating Layout Design Agents Through a Shared Design State

Official repository for the paper:

> **CoLayout: Coordinating Layout Design Agents Through a Shared Design State**  
> Yukai Zhou  
> Faculty of Design, University of Macau, Macau SAR, China

This repository provides the structured 36-brief layout benchmark, source code for CoLayout and all five baseline/ablation methods, deterministic conflict detector, priority arbiter, automated unit test suite, and complete experiment logs (`216` completed runs + `57` logged validation-failure attempts) to support full offline verification and reproduction of every table, figure, and statistical test in the manuscript.

## Repository Contents

- `data/briefs.json`: All 36 structured layout design briefs (`3` domains × `3` difficulty tiers × `4` briefs).
- `src/`: Complete implementation of the shared blackboard (`src/methods/blackboard.py`), deterministic constraint detector (`src/detector.py`), priority arbiter (`src/methods/arbiter.py`), headless renderer (`src/renderer.py`), and all six evaluated methods (`src/methods/`):
  - `single_agent.py` (Single-Agent one-shot)
  - `self_refine.py` (Self-Refine)
  - `naive_multi_agent.py` (Naive Multi-Agent)
  - `single_agent_detector.py` (Single-Agent + Det.)
  - `ours_no_detector.py` (Ours w/o Detector)
  - `ours_full.py` (CoLayout Full)
- `results/raw/`: All 216 completed run checkpoints (`6` methods × `36` briefs, model `gpt-4o-mini`, correction budget `K = 5`).
- `results/failures/`: All 57 logged validation-failure attempts with provider-reported token usage.
- `results/summary.csv` & `results/PROVENANCE.json`: Aggregated metrics and model provenance metadata.
- `scripts/verify_paper_stats.py`: Standalone offline verification script reproducing Table I & Table II metrics, exact pre-arbiter CSR, exact two-sided McNemar tests, 95% paired bootstrap confidence intervals, text-fit rates, and total token audits.
- `tests/`: Automated unit test suite (`79` tests).

## Quick Verification (Offline, No API Key Required)

Requires Python 3.10+. Install the package with test and paper-export dependencies:

```bash
python -m pip install -e ".[test,paper]"
```

Run the unit tests, verify all quantitative claims and statistical tests offline, and regenerate the paper tables and figures from the released raw logs:

```bash
pytest -q
python scripts/verify_paper_stats.py
python src/export_results.py --case-task-id editorial_hero_hard_04 --paper-dir exported_paper
```

## Re-Running LLM Experiments (Optional)

To re-run the full benchmark against the OpenAI API, set `OPENAI_API_KEY` in your environment (default model: `gpt-4o-mini`), then execute:

```bash
python src/run_experiments.py --methods all --max-rounds 5 --out-dir results_rerun --workers 4 --continue-on-invalid
```

## License

This project is released under the [MIT License](LICENSE).

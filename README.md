# CoLayout: Coordinating Layout Design Agents Through a Shared Design State

Official repository for the paper:

> **CoLayout: Coordinating Layout Design Agents Through a Shared Design State**  
> Yukai Zhou  
> Department of Industrial Design, Faculty of Design, University of Macau, Macao SAR, China

This repository provides the structured 36-brief layout benchmark, source code for CoLayout and all five baseline/ablation methods, deterministic conflict detector, priority arbiter, automated unit test suite, and complete experiment logs (`216` completed runs + `57` logged validation-failure attempts) to support full offline verification and reproduction of every table, figure, diagnostic metric, and statistical test in the manuscript.

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
- `scripts/verify_paper_stats.py`: Standalone offline verification script reproducing Table I & Table II metrics, exact pre-arbiter CSR, exact two-sided McNemar tests, 20,000-sample paired percentile bootstrap CIs, repair displacement, black/white foreground color shifts, hard-brief zone adherence, strict raster text-fit rates, and total token audits.
- `tests/`: Automated unit test suite (`79` tests).

## Runtime Configuration & Diagnostic Definitions

- **LLM API Configuration** (`src/llm_client.py`): OpenAI Chat Completions API with model identifier `gpt-4o-mini`, `response_format={"type": "json_object"}`, API default sampling parameters (no `temperature`/`top_p` override), no explicit `max_tokens` cap, and up to two within-round corrective retries on JSON/schema errors. Runs were executed on November 24, 2026 (`single_agent`, `self_refine`, `naive_multi_agent`, `ours_no_detector`, `ours_full`) and November 25, 2026 (`single_agent_detector`).
- **Statistical Tests** (`scripts/verify_paper_stats.py`): Exact two-sided McNemar tests on paired binary pass/fail outcomes; 20,000-sample paired percentile bootstrap (`seed = 42`) resampling over briefs (`n = 36` overall; `n = 12` hard) for 95% confidence intervals on token and round differences.
- **Offline Diagnostics** (`scripts/verify_paper_stats.py`):
  - *Mean center displacement*: Euclidean center shift (px) from Round-0 to terminal state, averaged across all elements per brief, then averaged across 36 briefs.
  - *Net black/white foreground shifts*: Count of elements per brief whose `fg_color` changed from Round-0 to `#000000` or `#FFFFFF` in the terminal state (counted once per element; reported both overall across all edits and restricted to Priority Arbiter overrides).
  - *Hard-brief zone adherence*: Fraction of all elements per hard brief lying inside their brief-specified vertical zone (top-64 px header, bottom-56 px footer, or middle `[64, H-56]`), averaged across the 12 hard briefs; header and dark-card enclosure rates are likewise brief-averaged.
  - *Strict raster text-fit rate*: Fraction of text boxes whose Pillow `arial.ttf` greedy-wrapped glyph bounding box (`1.2×` line height) stays within `[0, w] × [0, h]`, brief-averaged across 36 briefs (with micro-averaged counts over all `216` text boxes also reported).

## Quick Verification (Offline, No API Key Required)

Requires Python 3.10+. Install the package with test and paper-export dependencies:

```bash
python -m pip install -e ".[test,paper]"
```

Run the unit tests, verify all quantitative claims, diagnostics, and statistical tests offline, and regenerate the paper tables and figures from the released raw logs:

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

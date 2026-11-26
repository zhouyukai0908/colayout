# CoLayout: Coordinating Layout Design Agents Through a Shared Design State

Official repository for the paper:

> **CoLayout: Coordinating Layout Design Agents Through a Shared Design State**  
> Yukai Zhou  
> Department of Industrial Design, Faculty of Design, University of Macau, Macao SAR, China

This repository provides the structured 36-brief layout benchmark, source code for CoLayout and all six baseline/ablation methods (`7` methods total), deterministic conflict detector, priority arbiter, automated unit test suite (`80` tests), and complete experiment logs (`252` completed runs totaling `2,519,275` tokens + `57` logged validation-failure attempts totaling `278,064` tokens; **`2,797,339` tokens grand total**) to support full offline verification and reproduction of every table, figure, diagnostic metric, cross-attribute regression audit, and statistical test in the manuscript.

## Repository Contents

- `data/briefs.json`: All 36 structured layout design briefs (`3` domains × `3` difficulty tiers × `4` briefs).
- `src/`: Complete implementation of the shared blackboard (`src/methods/blackboard.py`), deterministic constraint detector (`src/detector.py`), priority arbiter (`src/methods/arbiter.py`), headless renderer (`src/renderer.py`), and all seven evaluated methods (`src/methods/`):
  - `single_agent.py` (Single-Agent one-shot)
  - `self_refine.py` (Self-Refine)
  - `naive_multi_agent.py` (Naive Multi-Agent)
  - `single_agent_detector.py` (Single-Agent + Det.)
  - `single_agent_bb_detector.py` (Single-Agent + BB + Det.: unlocked single-agent blackboard ablation with partial patches, programmatic copy injection, and union of both roles' rules and conflict IDs)
  - `ours_no_detector.py` (Ours w/o Detector)
  - `ours_full.py` (CoLayout Full)
- `results/raw/`: All 252 completed run checkpoints (`7` methods × `36` briefs, model `gpt-4o-mini`, correction budget `K = 5`).
- `results/failures/`: All 57 logged validation-failure attempts with provider-reported token usage (`0` failures for `single_agent_bb_detector`).
- `results/summary.csv` & `results/PROVENANCE.json`: Aggregated 7-method metrics and model provenance metadata.
- `scripts/verify_paper_stats.py`: Standalone offline verification script reproducing Table I & Table II metrics, Round-0 and pre-arbiter CSR, exact two-sided McNemar tests, 20,000-sample paired percentile bootstrap 95% CIs (completed & amortized tokens, rounds, displacement, black/white shifts, text-fit rate), repair fidelity, hard-brief zone adherence, fine-grained cross-attribute regressions, strict raster text-fit rates, and total token audits (`2,797,339` tokens).
- `scripts/audit_bb_ablation.py`: Detailed 7-method blackboard-vs-role-lock decomposition audit including per-difficulty summaries, round convergence distributions, and example traces for cross-attribute regressions.
- `tests/`: Automated unit test suite (`80` tests).

## Runtime Configuration & Diagnostic Definitions

- **LLM API Configuration** (`src/llm_client.py`): OpenAI Chat Completions API with model identifier `gpt-4o-mini`, `response_format={"type": "json_object"}`, API default sampling parameters (no `temperature`/`top_p` override), no explicit `max_tokens` cap, and up to two within-round corrective retries on JSON/schema errors. Runs were executed on November 24, 2026 (`single_agent`, `self_refine`, `naive_multi_agent`, `ours_no_detector`, `ours_full`), November 25, 2026 (`single_agent_detector`), and November 26, 2026 (`single_agent_bb_detector`).
- **Statistical Tests** (`scripts/verify_paper_stats.py`): Exact two-sided McNemar tests on paired binary pass/fail outcomes; 20,000-sample paired percentile bootstrap (`seed = 42`) resampling over briefs (`n = 36` overall; `n = 12` hard) for 95% confidence intervals on token, round, displacement, color-override, and text-fit differences.
- **Offline Diagnostics** (`scripts/verify_paper_stats.py`):
  - *Mean center displacement*: Euclidean center shift (px) from Round-0 to terminal state, averaged across all elements per brief, then averaged across 36 briefs.
  - *Net black/white foreground shifts*: Count of elements per brief whose `fg_color` changed from Round-0 to `#000000` or `#FFFFFF` in the terminal state (counted once per element; reported both overall across all edits and restricted to Priority Arbiter overrides).
  - *Hard-brief zone adherence*: Fraction of all elements per hard brief lying inside their brief-specified vertical zone (top-64 px header, bottom-56 px footer, or middle `[64, H-56]`), averaged across the 12 hard briefs; header, middle, footer, and dark-card enclosure rates are likewise brief-averaged.
  - *Cross-attribute regressions*: Tracked across correction rounds from exact state replay—(a) geometry edits and new geometry conflicts introduced on contrast-only rounds or contrast-only flagged elements, and (b) geometry edits inducing pure secondary contrast regressions without style changes.
  - *Strict raster text-fit rate*: Fraction of text boxes whose Windows `arial.ttf` greedy-wrapped glyph bounding box (`1.2×` line height) stays within `[0, w] × [0, h]`, brief-averaged across 36 layouts (`216` text boxes per method, with micro-averaged rates also reported). To prevent silent cross-platform font drift, `scripts/verify_paper_stats.py` requires `arial.ttf` by default (pass `--font-path /path/to/arial.ttf` on Linux/macOS to reproduce the paper's exact glyph metrics, or `--allow-font-fallback` to allow Pillow's built-in default font).

## Quick Verification (Offline, No API Key Required)

Requires Python 3.10+. Install the package with test and paper-export dependencies:

```bash
python -m pip install -e ".[test,paper]"
```

Run all 80 unit tests, verify every quantitative claim, diagnostic, cross-attribute regression, and statistical test offline, and regenerate the 7-method paper tables and figures from the released raw logs:

```bash
pytest -q
python scripts/verify_paper_stats.py
python scripts/audit_bb_ablation.py
python src/export_results.py --case-task-id editorial_hero_hard_04 --paper-dir exported_paper
```

## Re-Running LLM Experiments (Optional)

To re-run the full 7-method benchmark against the OpenAI API, set `OPENAI_API_KEY` in your environment (default model: `gpt-4o-mini`), then execute:

```bash
python src/run_experiments.py --methods all --max-rounds 5 --out-dir results_rerun --workers 4 --continue-on-invalid
```

## License

This project is released under the [MIT License](LICENSE).

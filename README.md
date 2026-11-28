# CoLayout: Coordinating Layout Design Agents Through a Shared Design State

Official artifact for the IEEE CSCWD 2027 manuscript:

> **CoLayout: Coordinating Layout Design Agents Through a Shared Design State**  
> Yukai Zhou  
> Department of Industrial Design, Faculty of Design, University of Macau, Macao SAR, China

This repository contains the 36-brief layout benchmark, source code for CoLayout and all seven controlled baselines/ablations (`8` methods total with verbatim-aligned rule prompts), the deterministic constraint detector, priority arbiter, automated unit test suite (`93` tests), complete multi-seed experiment logs (`runs_v2/`: `864` completed runs across `3` seeds on pinned `gpt-4o-mini-2024-07-18`), and scripts to reproduce every table, inline LaTeX number macro, and figure in the manuscript offline without an API key.

---

## One-Command Offline Reproduction (No API Key Required)

Requires Python 3.10+ (Windows `arial.ttf` is used for strict raster text-fit checks; pass `--font-path` to `scripts/analyze_runs_v2.py` on Linux/macOS if Arial is installed elsewhere).

```bash
python -m pip install -e ".[test,paper]"
python scripts/reproduce_paper.py
```

This single command:
1. Runs all `93` unit tests (`pytest -q`), including prompt-alignment, strict zero-silent-drop patch parsing, and schema checks;
2. Recomputes all primary and diagnostic metrics, strict blackboard trace replays (`432/432`), and brief-clustered percentile bootstrap 95% CIs (`B = 20,000`, `seed = 42`) from `runs_v2/seed_{1,2,3}/` into `runs_v2/results_summary.{json,md}`;
3. Regenerates every paper artifact directly from logs into `paper/`:
   - `paper/tables/table_main.tex` (Table I: 8-method comparison including `Pre&Fit`)
   - `paper/tables/table_contrasts.tex` (Table II: 4 controlled architectural contrasts with 95% CIs)
   - `paper/tables/table_diagnostics.tex` (Table III: displacement, arbiter triggers, cross-attribute regressions)
   - `paper/generated_numbers.tex` (all inline LaTeX number macros—zero hand-copied numbers)
   - `paper/figures/fig2_performance.{pdf,png}` (8-method CSR and amortized tokens by difficulty)
   - `paper/figures/fig3_case_study.{pdf,png}` (illustrative hard brief `dashboard_card_hard_04`, seed 1)

To verify reproduction into an isolated directory without overwriting `paper/`:

```bash
python scripts/reproduce_paper.py --paper-dir exported_paper
```

---

## Repository Structure

- `data/briefs.json`: 36 structured layout design briefs (`3` domains × `3` difficulties × `4` briefs).
- `src/`:
  - `schema.py`, `detector.py`, `renderer.py`: Shared versioned `LayoutState`, deterministic overlap/overflow/WCAG contrast evaluator, and HTML/PNG renderer.
  - `methods/rules.py`: Shared verbatim `GEOMETRY_RULES`, `STYLE_RULES`, `REVISION_RULES`, and concrete-placeholder output format templates.
  - `methods/`: All 8 evaluated methods—`single_agent.py`, `self_refine.py`, `naive_multi_agent.py`, `single_agent_detector.py`, `single_agent_bb_detector.py`, `sa_twostage_bb_det.py`, `ours_no_detector.py`, `ours_full.py`, plus `blackboard.py` and `arbiter.py`.
  - `run_experiments.py` & `llm_client.py`: Multi-seed runner recording pinned model snapshot, `system_fingerprint`, UTC timestamps, and derived retry seeds.
- `runs_v2/`: Complete 8-method × 36-brief × 3-seed (`864` completed runs + `28` retried failure logs, `0` residual failures; `8,478,768` total tokens) batch executed on `gpt-4o-mini-2024-07-18`, together with the analysis plan (`ANALYSIS_PLAN.md`) and summary outputs (`results_summary.{json,md}`).
- `scripts/`:
  - `reproduce_paper.py`: One-command offline verification and paper artifact generator.
  - `analyze_runs_v2.py`: Full statistical and diagnostic analysis over `runs_v2/`.
  - `export_paper_v2.py`: Generates all LaTeX tables, number macros, Fig. 2, and Fig. 3.
  - `run_batch_v2.py`: Multi-seed batch runner used to produce `runs_v2/`.
- `paper/`: Compiled 6-page IEEE manuscript (`CoLayout_CSCWD2027.pdf`) and full LaTeX sources (`main.tex`, `related.tex`, `method.tex`, `refs.bib`, `generated_numbers.tex`, `tables/`, `figures/`). Toggle `\anonymousfalse` / `\anonymoustrue` in `paper/main.tex` to build single-blind named or double-blind anonymous PDFs.
- `tests/`: Automated unit test suite (`93` tests).
- `results/`: Preserved read-only historical archive from earlier single-seed runs (not mixed with `runs_v2/`).

---

## Re-Running Live LLM Experiments (Optional)

Set `OPENAI_API_KEY` in your environment (or a local `.env.local` file, excluded by `.gitignore`) and run:

```bash
python scripts/run_batch_v2.py --out-root runs_rerun --model gpt-4o-mini-2024-07-18 --seeds 1,2,3 --workers 6
```

## Metric Conventions

| Metric | Definition |
| --- | --- |
| `OR` | Sum of pairwise foreground intersection areas divided by total foreground area (`z_index >= 1`, excluding cards). |
| `BOR` | Fraction of all elements extending outside `[0, W] × [0, H]`. |
| `CPR` | Fraction of text elements meeting WCAG 2.1 contrast (`>= 4.5`, or `>= 3.0` for `font_size >= 24`). |
| `Pre-CSR` | Binary pass indicator (`OR = 0`, `BOR = 0`, `CPR = 1`) scored *before* Priority Arbiter fallback. |
| `Pre&Fit` | Layout-level joint pass rate requiring both `Pre-CSR = 1` and 100% pre-arbitration raster text-fit. |
| `Hard Fit` | Terminal `arial.ttf` (`1.2×` line height) raster text-fit rate on the 12 hard briefs. |

## License

MIT License.

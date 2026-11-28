# Pre-Registered Analysis Plan (`runs_v2/`)

**Registered UTC**: `2026-11-27T02:15:00+00:00` (locked **before** any smoke test or full experiment execution in `runs_v2/`).  
**Rule**: Once full experimental data are collected, **neither the primary hypotheses nor the primary statistical test may be altered**. All outcomes—including null or unfavorable results—will be reported verbatim.

---

## 1. Motivation & Experimental Controls
Stage 1 offline auditing of the legacy 2026-11-24/25/26 runs (`results/`) revealed two confounds that prevent attributing Round-0 or Pre-CSR gains solely to multi-agent role locks:
1. **Prompt asymmetry on `bg_color`**: Visual R0 constrained `bg_color` to `"containers and images"`, whereas single-agent controls wrote `"optional hex bg_color (or null)"`, causing `75.3%` (`64/85`) of `SA+BB+Det.` Round-0 contrast failures via low-contrast button/tag fills.
2. **Non-simultaneous single-run execution**: Detector-guided methods ran on three different days using unpinned `gpt-4o-mini` without `seed` or `system_fingerprint`.

To resolve both confounds cleanly:
- **Verbatim Rule Alignment (`src/methods/rules.py`)**: `GEOMETRY_RULES`, `STYLE_RULES`, and `REVISION_RULES` are extracted verbatim from CoLayout's existing Architect and Visual prompts (CoLayout prompts are byte-identical to prior runs). `single_agent_detector`, `single_agent_bb_detector`, and the new `sa_twostage_bb_det` concatenate the exact same `GEOMETRY_RULES` and `STYLE_RULES`, differing only in output format. No rule wording is tuned on the 36 benchmark briefs.
- **Same-Batch Multi-Seed Protocol**: All **8 methods** (`single_agent`, `self_refine`, `naive_multi_agent`, `single_agent_detector`, `single_agent_bb_detector`, `sa_twostage_bb_det`, `ours_no_detector`, `ours_full`) × **36 briefs** × **3 seeds** (`seed_1`, `seed_2`, `seed_3`) run in one contiguous batch with a pinned dated snapshot (`gpt-4o-mini-2024-07-18`; if unavailable from the endpoint, fall back to `gpt-4o-mini` and verify `system_fingerprint` consistency across all checkpoints). Each checkpoint logs `model`, `resolved_models`, `system_fingerprint`, `seed`, and `timestamp_utc`.
- **Legacy Preservation**: `results/` remains untouched as historical archive only and is never mixed with `runs_v2/`.

---

## 2. Primary Hypotheses & Primary Endpoints
**Primary Outcome**: Pre-arbitration hard constraint satisfaction rate (**Pre-CSR**; binary per run, pooled across `3` seeds per brief → brief-level mean in `{0, 1/3, 2/3, 1}` over `n = 36` briefs, and stratified on the `12` hard briefs as a pre-specified subgroup).

**Primary Three-Way Decomposition (Core Ladder)**:
1. **H1 (Sequential Decomposition Effect)**: `sa_twostage_bb_det` vs. `single_agent_bb_detector` (`SA+BB+Det.`) on Pre-CSR — isolates two-call Round-0 separation (geometry then style) with unlocked single-agent revision.
2. **H2 (Field-Locked Role Revision Effect)**: `ours_full` (`CoLayout`) vs. `sa_twostage_bb_det` on Pre-CSR — isolates role-scoped field locks during correction rounds when both methods share sequential two-stage Round-0 initialization.
3. **H3 (Combined Two-Role + Field-Lock Effect)**: `ours_full` (`CoLayout`) vs. `single_agent_bb_detector` (`SA+BB+Det.`) on Pre-CSR — overall gain of CoLayout over the prompt-aligned unlocked blackboard single agent.

*(Blackboard patching isolation `SA+BB+Det.` vs. `SA+Det.` and detector isolation `CoLayout` vs. `Ours w/o Detector` are reported in the full 8-method comparison table using the same statistical procedure.)*

---

## 3. Primary Statistical Test (Locked)
- **Pooling & Clustering Unit**: Pool all `3` seeds (`108` runs per method = `36` briefs × `3` seeds). Treat each **brief** (`n = 36` overall; `n = 12` hard) as the independent cluster so repeated seeds on the same brief are never treated as independent samples.
- **Cluster-Bootstrap Estimator**: For each contrast `A − B`, compute the paired brief-level difference `d_i = mean_s(Y_{A,i,s}) − mean_s(Y_{B,i,s})` for `i = 1..36`, then run a **brief-clustered paired percentile bootstrap** (`B = 20,000` resamples with replacement over briefs, fixed RNG `seed = 42`).
- **Reported Quantities**: Point estimate (mean paired difference across briefs), **95% percentile CI**, and seed-to-seed standard deviation across `{seed_1, seed_2, seed_3}`.
- **Multiplicity Note**: For the three primary Pre-CSR contrasts (`H1`, `H2`, `H3`), any contrast whose 95% CI includes `0` (or does not survive multiplicity adjustment when evaluated alongside the family) will be reported honestly as exploratory / directionally consistent or null—no cherry-picking of single seeds or subsets.

---

## 4. Secondary Metrics
Evaluated under the exact same 3-seed brief-clustered aggregation:
1. **Round-0 Compliance (R0-CSR)** and Round-0 conflict breakdown (`contrast`, `overlap`, `overflow`, plus button/tag explicit `bg_color` fill rate).
2. **Revision Conversion Rate**: `P(Pre-CSR = 1 | R0-CSR = 0)`.
3. **Efficiency**: Mean `rounds_used`, raw LLM call count, completed `total_tokens`, failure-attempt count, and amortized `total_tokens` (including `failures/`).
4. **Repair Fidelity & Structural Safety**: Round-0→Final mean center displacement (`px`), net `#000000`/`#FFFFFF` shifts (overall and arbiter-only), arbiter trigger rate, and cross-attribute regressions (`C→G` and `G→C`).
5. **Unmonitored Raster Text-Fit Rate**: Strict glyph-bounds fit rate (`Windows arial.ttf`, `1.2×` line height, `216` text boxes per seed × `3` seeds = `648` boxes/method), overall and hard tier.

---

## 5. Pre-Committed Decision Rules for Stage 4 Revision
- **Rule A (`sa_twostage_bb_det ≈ CoLayout` on Pre-CSR)**: Reframe the main convergence contribution as *sequential decomposition (geometry-before-style initialization)*, while attributing *structural safety* (`C→G = 0` by schema construction) to field locks; tone down "multi-agent collaboration" claims accordingly.
- **Rule B (`CoLayout` clearly outperforms `sa_twostage_bb_det` on Pre-CSR)**: Report both sequential decomposition (`H1`) and the independent revision-stage contribution of role-scoped field locks (`H2`).
- **Rule C (`CoLayout ≈ sa_twostage_bb_det ≈ SA+BB+Det.` after prompt alignment)**: Report honestly that once `bg_color` prompt rules are aligned, convergence differences narrow/disappear, and center the paper's validated empirical claims on deterministic detection, blackboard token savings, and structural `C→G` safety.
- **Execution Stop Rule**: During Stage 3, if any method's schema-validation failure rate exceeds `2×` its historical rate in `results/failures/`, halt immediately and report to the user before continuing.

---

## 6. Pre-Experiment Amendments (Logged Before Full Execution)

### Amendment 1 — `2026-11-27T02:38:00+00:00` (Pre-Full-Run Format, Strict Parser & Card-Coloring Fixes)
Triggered by the 2-brief Stage-2 smoke test (`runs_v2/smoke/seed_1/`), which exposed two parser/schema defects prior to full execution: (i) the placeholder `"field":value` caused `gpt-4o-mini-2024-07-18` to nest updates under a literal `"field"` key that `_updates()` silently dropped as a no-op (also contaminating `15/36` legacy `ours_full` R0 runs and `11/36` legacy `SA+BB+Det.` revision runs in `results/raw/`), and (ii) `single_agent_detector` repeatedly assigned `z_index=0` to `visual` (`type='image'`).

**Approved changes committed before the 6-brief smoke gate and full `runs_v2/` run**:
1. **Zero-silent-drop patch parser & non-ambiguous placeholder templates (`src/methods/rules.py`, `src/methods/blackboard.py`, `src/methods/single_agent_bb_detector.py`, `src/methods/sa_twostage_bb_det.py`)**:
   - Replaced `"field":value` across all shared patch templates with concrete field names and neutral placeholder tokens (`<element_id>`, `<number>`, `<int>`, `"<hex>"`, `"<hex_or_null>"`; no concrete hex colors).
   - `_updates()` (with default `strict_keys=True`) now raises `ValueError` on unexpected top-level keys, literal `"field"` keys, any nested `dict`, unknown/non-writable fields, out-of-scope IDs, or empty update rows—triggering standard in-round retry so the parser never silently drops model output.
   - Removed pre-filled passing default colors/font sizes (`#000000`, `16`) from Round-0 geometry seeding; Round-0 styling for `ours_full`, `ours_no_detector`, `sa_twostage_bb_det`, and `single_agent_bb_detector` strictly requires explicit `fg_color` and `font_size` for every text element and non-null `bg_color` for every `card` element.
   - Added per-call and per-run `format_rejections` tracking in `src/llm_client.py`, `src/methods/common.py`, and `src/run_experiments.py` as an explicit reliability diagnostic.
2. **Shared `z_index` schema clarification (`src/methods/rules.py`, `src/methods/common.py`)**:
   - Appended `"type 'image' and all text types are foreground (z_index >= 1)"` to the shared `GEOMETRY_RULES` constant (applied uniformly to all aligned methods) and included `type='<type>'` in `validate_layout()`'s `z_index` error message.
3. **Additional lightweight diagnostic (non-hard)**:
   - Added **card coloring compliance** on Medium/Hard briefs (fraction of card elements with non-null `bg_color`, dark luminance `< 0.2`, and exact brief-requested hex color). Reported as a secondary diagnostic only; hard CSR definition is unchanged.
4. **Invariance of Primary Analysis**:
   - **Primary hypotheses (`H1`, `H2`, `H3`) and primary statistical test (3-seed pooled brief-clustered paired percentile bootstrap on Pre-CSR, `B = 20,000`, `seed = 42`) remain strictly unchanged.**

### Amendment 2 — `2026-11-27T02:56:00+00:00` (Plan C: Verbatim-Error Diagnostic, Derived Retry Seeds, ITT vs. Complete-Case Dual Reporting & 20% Stop Rule)
Triggered by the 6-brief × 8-method smoke gate (`runs_v2/smoke_v2/seed_1/`), where 7/8 methods achieved `6/6` with `0` silent drops, `0` empty visual inits, and `100.0%` hard-card coloring, while `single_agent_detector` repeated verbatim-copy errors on 2 hard briefs under identical `seed=1` retries.

**Approved changes committed before launching the full 8 × 36 × 3 batch**:
1. **Plan A (shared `validate_layout()` verbatim-copy error message)**: Added `expected <required>, got <actual>` to `validate_layout()`'s verbatim-copy `ValueError` in `src/methods/common.py`. Applied uniformly across all methods (conservative change strengthening full-JSON baselines).
2. **Plan B (derived whole-run retry seeds, max 3 whole-run attempts)**: On whole-run retry `attempt_index ∈ {0, 1, 2}` (maximum `3` whole-run attempts per `(method, brief, seed)`), API calls use `effective_seed = seed + 1000 * attempt_index`. Every attempt's `attempt_index`, `effective_seed`, `format_rejections`, and failure `reason` are logged in full.
3. **Dual Failure-Accounting Protocol (pre-registered)**:
   - **Primary (Intention-to-Treat / ITT)**: Any `(method, brief, seed)` pair still uncompleted after `3` whole-run attempts is scored as `R0-CSR = 0`, `Pre-CSR = 0`, `CSR = 0`, with tokens equal to the sum across all logged attempts for that pair.
   - **Sensitivity (Complete-Case)**: Re-computes all method summaries and paired brief-clustered bootstrap contrasts excluding residual failed runs.
   - **Mandatory dual reporting**: Both ITT and Complete-Case tables are reported side by side; selecting only one favorable convention is prohibited.
4. **Gate 3 Stop Rule Clarification**: Halt and report if any method's whole-run failure rate exceeds **`20%`** (`failed_whole_attempts / total_whole_attempts > 0.20` or `residual_failed_briefs / 36 > 0.20`) within a seed; isolated failures are logged without halting.
5. **No smoke-driven prompt/method tuning**: No rule prompts or method designs were adjusted based on inter-method smoke outcomes.



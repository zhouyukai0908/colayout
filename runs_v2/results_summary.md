# CoLayout `runs_v2/` Pre-Registered Multi-Seed Results & Diagnostics

- **Model snapshot**: `['gpt-4o-mini-2024-07-18']` | **Seeds**: `[1, 2, 3]` | **Scheduled**: `864` | **Completed**: `864` | **Residual failed**: `0` | **Failed whole-run attempts**: `28`
- **Total token spend**: `8,270,403` completed + `208,365` failed = **`8,478,768`** tokens
- **UTC window**: `2026-11-27T03:04:43.822711+00:00` → `2026-11-27T03:54:06.224325+00:00`
- **Integrity**: `metric_mismatches = 0`, `strict_bb_replays_verified = 432` (zero silent drops), observed `system_fingerprints` count = `19`

## 1. Main 8-Method Results — Primary ITT vs. Complete-Case Sensitivity

| Method | Done / 108 | Whole-Run Fail Attempts (Rate) | Format Rej | **ITT R0-CSR% (±SD)** | **ITT Pre-CSR% (±SD)** | **ITT Amort Tok (±SD)** | CC Pre-CSR% (±SD) | CC Rounds / Calls | CC Tokens (±SD) | CC TFR% (Hard%) | Hard Card (Col/Dark/Exact) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `single_agent` | 108/108 | 4 (3.6%) | 73 | 0.0 (±0.0) | **0.0** (±0.0) | 2,341 (±513) | 0.0 (±0.0) | 1.00 / 1.00 | 2,124 (±399) | 97.0 (93.9) | 100/100/100% |
| `self_refine` | 108/108 | 5 (4.4%) | 76 | 1.9 (±3.2) | **5.6** (±2.8) | 9,351 (±382) | 5.6 (±2.8) | 4.93 / 4.93 | 9,087 (±154) | 96.5 (92.9) | 100/100/100% |
| `naive_multi_agent` | 108/108 | 4 (3.6%) | 37 | 5.6 (±2.8) | **11.1** (±2.8) | 19,893 (±1,184) | 11.1 (±2.8) | 4.30 / 12.89 | 19,587 (±951) | 97.6 (95.1) | 100/100/100% |
| `single_agent_detector` | 108/108 | 8 (6.9%) | 96 | 3.7 (±1.6) | **60.2** (±4.2) | 9,467 (±656) | 60.2 (±4.2) | 4.02 / 4.02 | 8,979 (±329) | 97.3 (95.5) | 100/100/100% |
| `single_agent_bb_detector` | 108/108 | 3 (2.7%) | 48 | 4.6 (±3.2) | **77.8** (±2.8) | 5,331 (±219) | 77.8 (±2.8) | 3.39 / 3.39 | 5,183 (±153) | 96.5 (94.2) | 100/100/100% |
| `sa_twostage_bb_det` | 108/108 | 0 (0.0%) | 30 | 6.5 (±4.2) | **88.0** (±8.5) | 5,548 (±131) | 88.0 (±8.5) | 2.98 / 3.98 | 5,548 (±131) | 93.8 (85.8) | 100/100/100% |
| `ours_no_detector` | 108/108 | 4 (3.6%) | 83 | 6.5 (±1.6) | **16.7** (±5.6) | 21,247 (±700) | 16.7 (±5.6) | 6.00 / 16.09 | 20,741 (±311) | 94.1 (84.2) | 100/100/100% |
| `ours_full` | 108/108 | 0 (0.0%) | 19 | 3.7 (±4.2) | **87.0** (±1.6) | 5,329 (±356) | 87.0 (±1.6) | 2.93 / 4.10 | 5,329 (±356) | 94.6 (85.8) | 100/100/100% |

## 2A. Primary Pre-Registered Contrasts — Intention-to-Treat (Brief-Clustered Bootstrap, B=20,000, seed=42)

| Contrast | Scope | Δ ITT Pre-CSR (pp) [95% CI] | Δ ITT R0-CSR (pp) [95% CI] | Δ ITT Amort Tok [95% CI] |
|---|---|---:|---:|---:|
| **H1 (Sequential Decomp.)** (`sa_twostage_bb_det` − `single_agent_bb_detector`) | All 36 (n=36) | +10.2 [-0.9, +22.2] | +1.9 [-3.7, +7.4] | +216 [-663, +983] |
| **H1 (Sequential Decomp.)** (`sa_twostage_bb_det` − `single_agent_bb_detector`) | Hard 12 (n=12) | +19.4 [-8.3, +47.2] | -2.8 [-8.3, +0.0] | -113 [-2,204, +1,652] |
| **H2 (Field-Locked Revision)** (`ours_full` − `sa_twostage_bb_det`) | All 36 (n=36) | -0.9 [-10.2, +7.4] | -2.8 [-7.4, +1.9] | -219 [-739, +315] |
| **H2 (Field-Locked Revision)** (`ours_full` − `sa_twostage_bb_det`) | Hard 12 (n=12) | -5.6 [-30.6, +16.7] | +0.0 [+0.0, +0.0] | -364 [-1,535, +877] |
| **H3 (Two-Role + Field Locks)** (`ours_full` − `single_agent_bb_detector`) | All 36 (n=36) | +9.3 [+0.0, +18.5] | -0.9 [-7.4, +4.6] | -3 [-753, +714] |
| **H3 (Two-Role + Field Locks)** (`ours_full` − `single_agent_bb_detector`) | Hard 12 (n=12) | +13.9 [-8.3, +36.1] | -2.8 [-8.3, +0.0] | -477 [-2,155, +1,092] |
| **Ladder: BB Patching** (`single_agent_bb_detector` − `single_agent_detector`) | All 36 (n=36) | +17.6 [+4.6, +31.5]* | +0.9 [-6.5, +6.5] | -4,136 [-5,374, -2,952]* |
| **Ladder: BB Patching** (`single_agent_bb_detector` − `single_agent_detector`) | Hard 12 (n=12) | +52.8 [+27.8, +75.0]* | +2.8 [+0.0, +8.3] | -8,033 [-9,553, -6,617]* |
| **Ladder: Full vs SA+Det** (`ours_full` − `single_agent_detector`) | All 36 (n=36) | +26.9 [+14.8, +39.8]* | +0.0 [-7.4, +5.6] | -4,139 [-5,483, -2,921]* |
| **Ladder: Full vs SA+Det** (`ours_full` − `single_agent_detector`) | Hard 12 (n=12) | +66.7 [+44.4, +86.1]* | +0.0 [+0.0, +0.0] | -8,510 [-10,437, -6,764]* |
| **Ladder: Det vs Critic** (`ours_full` − `ours_no_detector`) | All 36 (n=36) | +70.4 [+60.2, +79.6]* | -2.8 [-7.4, +0.0] | -15,918 [-17,699, -14,208]* |
| **Ladder: Det vs Critic** (`ours_full` − `ours_no_detector`) | Hard 12 (n=12) | +72.2 [+52.8, +88.9]* | +0.0 [+0.0, +0.0] | -21,264 [-23,256, -19,557]* |

## 2B. Sensitivity Contrasts — Complete-Case Only (Brief-Clustered Bootstrap, B=20,000, seed=42)

| Contrast | Scope | Δ CC Pre-CSR (pp) [95% CI] | Δ CC R0-CSR (pp) [95% CI] | Δ Rounds [95% CI] | Δ Calls [95% CI] | Δ CC Tokens [95% CI] | Δ TFR (pp) [95% CI] |
|---|---|---:|---:|---:|---:|---:|---:|
| **H1 (Sequential Decomp.)** (`sa_twostage_bb_det` − `single_agent_bb_detector`) | All 36 (n=36) | +10.2 [-0.9, +22.2] | +1.9 [-3.7, +7.4] | -0.41 [-0.87, +0.04] | +0.59 [+0.13, +1.04]* | +364 [-413, +1,062] | -2.7 [-5.1, -0.7]* |
| **H1 (Sequential Decomp.)** (`sa_twostage_bb_det` − `single_agent_bb_detector`) | Hard 12 (n=12) | +19.4 [-8.3, +47.2] | -2.8 [-8.3, +0.0] | -0.67 [-1.56, +0.19] | +0.33 [-0.56, +1.19] | +140 [-1,551, +1,665] | -8.4 [-13.1, -3.9]* |
| **H2 (Field-Locked Revision)** (`ours_full` − `sa_twostage_bb_det`) | All 36 (n=36) | -0.9 [-10.2, +7.4] | -2.8 [-7.4, +1.9] | -0.06 [-0.40, +0.30] | +0.12 [-0.23, +0.48] | -219 [-739, +315] | +0.8 [-0.2, +1.9] |
| **H2 (Field-Locked Revision)** (`ours_full` − `sa_twostage_bb_det`) | Hard 12 (n=12) | -5.6 [-30.6, +16.7] | +0.0 [+0.0, +0.0] | -0.11 [-0.89, +0.69] | +0.22 [-0.56, +1.03] | -364 [-1,535, +877] | +0.0 [-2.2, +2.2] |
| **H3 (Two-Role + Field Locks)** (`ours_full` − `single_agent_bb_detector`) | All 36 (n=36) | +9.3 [+0.0, +18.5] | -0.9 [-7.4, +4.6] | -0.46 [-0.84, -0.09]* | +0.71 [+0.31, +1.12]* | +145 [-491, +791] | -1.9 [-4.2, +0.2] |
| **H3 (Two-Role + Field Locks)** (`ours_full` − `single_agent_bb_detector`) | Hard 12 (n=12) | +13.9 [-8.3, +36.1] | -2.8 [-8.3, +0.0] | -0.78 [-1.50, -0.03]* | +0.56 [-0.17, +1.33] | -224 [-1,478, +1,140] | -8.4 [-12.5, -4.3]* |
| **Ladder: BB Patching** (`single_agent_bb_detector` − `single_agent_detector`) | All 36 (n=36) | +17.6 [+4.6, +31.5]* | +0.9 [-6.5, +6.5] | -0.63 [-1.05, -0.20]* | -0.63 [-1.05, -0.20]* | -3,795 [-4,882, -2,749]* | -0.8 [-2.0, +0.4] |
| **Ladder: BB Patching** (`single_agent_bb_detector` − `single_agent_detector`) | Hard 12 (n=12) | +52.8 [+27.8, +75.0]* | +2.8 [+0.0, +8.3] | -1.42 [-2.06, -0.81]* | -1.42 [-2.06, -0.81]* | -7,056 [-8,301, -5,811]* | -1.3 [-3.4, +0.7] |
| **Ladder: Full vs SA+Det** (`ours_full` − `single_agent_detector`) | All 36 (n=36) | +26.9 [+14.8, +39.8]* | +0.0 [-7.4, +5.6] | -1.09 [-1.54, -0.67]* | +0.08 [-0.35, +0.51] | -3,650 [-4,748, -2,601]* | -2.7 [-5.0, -0.6]* |
| **Ladder: Full vs SA+Det** (`ours_full` − `single_agent_detector`) | Hard 12 (n=12) | +66.7 [+44.4, +86.1]* | +0.0 [+0.0, +0.0] | -2.19 [-2.94, -1.39]* | -0.86 [-1.61, -0.08]* | -7,280 [-8,599, -5,925]* | -9.7 [-13.5, -5.8]* |
| **Ladder: Det vs Critic** (`ours_full` − `ours_no_detector`) | All 36 (n=36) | +70.4 [+60.2, +79.6]* | -2.8 [-7.4, +0.0] | -3.07 [-3.42, -2.70]* | -11.99 [-12.34, -11.65]* | -15,412 [-16,931, -13,909]* | +0.5 [-0.4, +1.5] |
| **Ladder: Det vs Critic** (`ours_full` − `ours_no_detector`) | Hard 12 (n=12) | +72.2 [+52.8, +88.9]* | +0.0 [+0.0, +0.0] | -2.31 [-3.00, -1.61]* | -12.50 [-13.19, -11.83]* | -20,287 [-21,536, -19,019]* | +1.6 [-0.4, +3.5] |

*Asterisk `*` marks 95% cluster-bootstrap CIs excluding zero.*

## 3. Stratified by Difficulty (36 Scheduled Runs per Method × Difficulty)

| Method | Easy ITT R0 / ITT Pre / CC Pre / Amort | Medium ITT R0 / ITT Pre / CC Pre / Amort | Hard ITT R0 / ITT Pre / CC Pre / Amort |
|---|---:|---:|---:|
| `single_agent` | 0.0% / 0.0% / 0.0% / 2,188 | 0.0% / 0.0% / 0.0% / 2,127 | 0.0% / 0.0% / 0.0% / 2,709 |
| `self_refine` | 2.8% / 8.3% / 8.3% / 6,825 | 0.0% / 0.0% / 0.0% / 9,266 | 2.8% / 8.3% / 8.3% / 11,962 |
| `naive_multi_agent` | 8.3% / 19.4% / 19.4% / 11,193 | 0.0% / 11.1% / 11.1% / 20,461 | 8.3% / 2.8% / 2.8% / 28,024 |
| `single_agent_detector` | 11.1% / 94.4% / 94.4% / 3,831 | 0.0% / 80.6% / 80.6% / 7,999 | 0.0% / 5.6% / 5.6% / 16,572 |
| `single_agent_bb_detector` | 11.1% / 94.4% / 94.4% / 2,742 | 0.0% / 80.6% / 80.6% / 4,712 | 2.8% / 58.3% / 58.3% / 8,539 |
| `sa_twostage_bb_det` | 8.3% / 97.2% / 97.2% / 3,337 | 11.1% / 88.9% / 88.9% / 4,880 | 0.0% / 77.8% / 77.8% / 8,426 |
| `ours_no_detector` | 8.3% / 41.7% / 41.7% / 12,845 | 11.1% / 8.3% / 8.3% / 21,570 | 0.0% / 0.0% / 0.0% / 29,326 |
| `ours_full` | 2.8% / 100.0% / 100.0% / 2,858 | 8.3% / 88.9% / 88.9% / 5,066 | 0.0% / 72.2% / 72.2% / 8,062 |

## 4. Diagnostics Across Detector-Guided Methods (Complete-Case)

| Metric | `single_agent_detector` | `single_agent_bb_detector` | `sa_twostage_bb_det` | `ours_full` |
|---|---:|---:|---:|---:|
| Completed / Scheduled | 108/108 | 108/108 | 108/108 | 108/108 |
| OR / BOR / CPR / AS (%) | 0.0 / 0.0 / 100.0 / 82.4 | 0.0 / 0.0 / 100.0 / 83.3 | 0.0 / 0.0 / 100.0 / 83.1 | 0.0 / 0.0 / 100.0 / 84.0 |
| Revision conversion (CC / ITT) | 61/104 (58.7%) / 61/104 (58.7%) | 79/103 (76.7%) / 79/103 (76.7%) | 88/101 (87.1%) / 88/101 (87.1%) | 90/104 (86.5%) / 90/104 (86.5%) |
| Mean center shift (px) | 5.07 | 5.00 | 1.66 | 1.20 |
| Net B/W shifts (arbiter-only) | 1.41 (0.91) | 1.14 (0.44) | 1.26 (0.19) | 1.07 (0.26) |
| Arbiter triggers (geom/crop/shift/recolor) | 43/108 (3g/2c/4s/98r) | 24/108 (11g/8c/23s/48r) | 13/108 (1g/2c/0s/21r) | 14/108 (1g/0c/1s/28r) |
| Hard zone adherence (all / hdr / ftr / encl %) | 94.4 / 79.2 / 91.7 / 97.2 | 86.2 / 73.6 / 73.6 / 98.1 | 95.4 / 93.1 / 84.7 / 99.4 | 96.5 / 94.4 / 88.9 / 100.0 |
| Cross-attribute regressions (C→G / G→C) | 0 / 4 | 5 / 3 | 0 / 0 | 0 / 0 |
| Raster text-fit: all / hard / micro (%) | 97.3 / 95.5 / 96.8 | 96.5 / 94.2 / 95.8 | 93.8 / 85.8 / 92.0 | 94.6 / 85.8 / 92.7 |
| R0 btn/tag bg_color filled (contrast fails) | 124/306 (110) | 172/306 (135) | 295/306 (196) | 258/306 (168) |
| R0 conflicts (contrast / overlap / overflow) | 275 / 40 / 1 | 253 / 39 / 16 | 278 / 18 / 3 | 247 / 14 / 2 |
| Hard card colored / dark / exact brief hex (%) | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 | 100.0 / 100.0 / 100.0 |
| Medium card colored (%) | 36.1 | 100.0 | 100.0 | 100.0 |
| Format rejections / failed whole-run attempts | 96 / 8 (6.9%) | 48 / 3 (2.7%) | 30 / 0 (0.0%) | 19 / 0 (0.0%) |

## 5. Seed-by-Seed Breakdown (Zero Seed Selection)

| Method | Seed 1 (Done / ITT-R0 / ITT-Pre / HardPre / Amort / Fails / Rej) | Seed 2 | Seed 3 |
|---|---|---|---|
| `single_agent` | 36/36 · 0.0% / 0.0% / 0.0% · 2,205 · 2f / 22r | 36/36 · 0.0% / 0.0% / 0.0% · 1,910 · 0f / 17r | 36/36 · 0.0% / 0.0% / 0.0% · 2,909 · 2f / 34r |
| `self_refine` | 36/36 · 0.0% / 2.8% / 0.0% · 9,069 · 1f / 21r | 36/36 · 5.6% / 5.6% / 8.3% · 9,199 · 0f / 22r | 36/36 · 0.0% / 8.3% / 16.7% · 9,785 · 4f / 33r |
| `naive_multi_agent` | 36/36 · 2.8% / 13.9% / 0.0% · 20,457 · 2f / 11r | 36/36 · 5.6% / 8.3% / 8.3% · 18,532 · 0f / 9r | 36/36 · 8.3% / 11.1% / 0.0% · 20,688 · 2f / 17r |
| `single_agent_detector` | 36/36 · 2.8% / 63.9% / 8.3% · 8,893 · 1f / 27r | 36/36 · 2.8% / 55.6% / 0.0% · 10,183 · 5f / 40r | 36/36 · 5.6% / 61.1% / 8.3% · 9,326 · 2f / 29r |
| `single_agent_bb_detector` | 36/36 · 2.8% / 75.0% / 50.0% · 5,411 · 1f / 18r | 36/36 · 8.3% / 77.8% / 58.3% · 5,083 · 1f / 15r | 36/36 · 2.8% / 80.6% / 66.7% · 5,500 · 1f / 15r |
| `sa_twostage_bb_det` | 36/36 · 11.1% / 86.1% / 66.7% · 5,491 · 0f / 14r | 36/36 · 5.6% / 80.6% / 75.0% · 5,698 · 0f / 7r | 36/36 · 2.8% / 97.2% / 91.7% · 5,455 · 0f / 9r |
| `ours_no_detector` | 36/36 · 8.3% / 16.7% / 0.0% · 20,921 · 1f / 28r | 36/36 · 5.6% / 11.1% / 0.0% · 22,051 · 2f / 36r | 36/36 · 5.6% / 22.2% / 0.0% · 20,769 · 1f / 19r |
| `ours_full` | 36/36 · 8.3% / 86.1% / 58.3% · 4,995 · 0f / 7r | 36/36 · 0.0% / 86.1% / 75.0% · 5,288 · 0f / 6r | 36/36 · 2.8% / 88.9% / 83.3% · 5,703 · 0f / 6r |

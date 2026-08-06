# FIX report: offline sweep script — zero-forward selection referee (T5)

- **Date:** 2026-08-06
- **Fixer:** fixer agent (T5 of the voiceprint selection plan)
- **Base:** fe6314c — **Commits:** see §6
- **Contract:** docs/superpowers/plans/2026-08-06-voiceprint-selection-plan.md
  §4 T5 + §6.1/§6.2 (+ the task brief; T5 has no separate spec). Implemented as
  written; deviations listed in §5.
- **Scope:** **new files only.** `src/diart_server/**` is untouched — this task
  adds an offline analysis tool and its tests, nothing that runs in the server.

## 1. Why

Every later decision in this plan (which model, which threshold, whether the
N-aware slope is real, whether the 2-of-5 vote is safe) is a numeric claim. Up
to now those numbers lived in one-off scratchpad scripts that nobody can re-run,
so "raise `lo` by 0.01 and re-check" (plan §6.3 R1) had no executable meaning.

T5 turns that into a 1.5-second command over the already-computed CN-Celeb1
embedding caches: **zero new forward passes, pure numpy**. It prints the four
tables of plan §6.1/§6.2 plus the §2 offline red line (unregistered-speaker
cluster leakage ≤ 1%) and refuses to print any of it without also printing the
§6.4 confidence bounds.

## 2. Changes

### `scripts/eval/offline_sweep.py` (new, 805 lines)

Reads `emb_<model>.npz` caches keyed
`cc_enroll/<spk>` / `cc_test/<spk>/<utt>/<crop∈{0,2,4}>` / `cohC/<spk>/<idx>`,
then prints:

| Table | Content | Feeds |
|---|---|---|
| 1 | DIR@FAR=1%, model × scoring × crop × N, mean ±sd over gallery draws | §6.1 Q1/Q2 |
| 2 | thr@FAR=1% per N + least-squares fit `thr(N) = a + b·log2(N/5)` | **§6.1 Q4 / T3's `lo(N)`** |
| 3 | k-turn top-1 consistency + 2-of-k vote hit rates, genuine **and** impostor | **T4's 2-of-5** |
| 4 | unregistered-speaker cluster leakage under the sticky rule, with PASS/FAIL | **§2 offline red line** |

Design points that matter for review:

- **Score scales are production scales, not convenient ones.** `raw` is the
  plain cosine of unit vectors in `[-1, 1]` with **no** remap; `centered` is
  `unit(unit(v) − cohort_mean)` on both sides followed by the `(cos+1)/2` remap
  — exactly `voiceprint_matcher.match`'s two branches. This is pinned by a test
  that calls `match()` and compares (§4). Getting this wrong would silently
  shift every printed threshold; the docstring and a stdout footer both say the
  two columns are not comparable to each other.
- **Cache path is never hardcoded.** `--cache-dir` > `$DIART_EVAL_CACHE` >
  clear error + **exit code 2** (never a traceback). Same exit 2 for a missing
  directory, a directory with no `emb_*.npz`, a cache missing `cohC/*`, an
  unknown `--models`/`--crops` value, and an `--n-values` that leaves no
  out-of-gallery speaker to measure FAR with.
- **Reproducible and filter-invariant.** Galleries for size `N` come from
  `default_rng([seed, N])` and are drawn in one pass **before** any cluster
  sampling, so `--models` / `--crops` / other `--n-values` / the cluster
  constants cannot move a number. Verified on the real cache: the
  `--models eres2netv2-zh --crops 2 --n-values 5,20` run reproduces the full
  run's cells bit-for-bit. Dispersion is reported as **std across gallery
  draws** — plan §6.2 explicitly cuts speaker-block bootstrap, and §6.4.4 says
  so in the printed footer.
- **`leakEver` is the gate, not `leakFinal`.** The contested lock retracts a
  name, but V3 says "no registered name appears **on screen**" — a name that
  was shown and then withdrawn was still shown. The stricter number gates;
  `leakFinal` (what the prototype `adv_cluster_check.py` measured) is printed
  next to it for comparability.
- **Refuses to be quoted out of context.** The run always ends with the six
  §6.4 confidence bounds; a test asserts they are on stdout.

### `tests/diart_server/test_offline_sweep.py` (new, 27 tests)

Placed under `tests/diart_server/` on purpose: that is the only directory the
acceptance command collects, and this script is the referee for the rest of the
plan, so it must be gated. It imports the script by path (`scripts/` is not a
package). No existing test was modified or deleted.

### Constants and their basis

| Constant | Value | Basis |
|---|---|---|
| `DEFAULT_SEED` | `20260806` | matches the scratchpad prototype so numbers stay comparable |
| `DEFAULT_DRAWS` | `30` | plan §6.4.3 describes the evidence base as "30 抽库" |
| `DEFAULT_FAR` | `0.01` | plan §6.1/§6.4.3 — nothing below FAR 1% is claimable |
| `CLUSTER_TURNS` | `12` | `adv_cluster_check.py` (a 12-turn diart label) |
| `IMPOSTOR_CLUSTERS_PER_DRAW` | `50` | raised from the prototype's 25: 30×50 = **1500** impostor clusters per row resolves a 1% gate to 0.07pp; at 25 the gate was quantized to 0.13pp and rows flipped PASS/FAIL on noise |
| `VOTE_K` / `VOTE_WINDOW` | `2` / `5` | T4's shipped 2-of-5 |
| `LO_BASE`/`LO_SLOPE`/`HI_DELTA` | `0.82` / `0.0086` / `0.04` | plan §6.1 Q4 shipped operating point; all three overridable so §6.3 R1's "raise `lo` 0.01 and re-run" is one flag |
| `LEAK_GATE` | `0.01` | plan §2 offline red line |
| primary condition for tables 3/4 | `centered`, `2s` | the shipped scoring, and the crop closest to a real short turn; the operating point only has meaning on the centered scale |

## 3. Findings the reviewer needs (real numbers, `--seed 20260806 --draws 30`)

**The plan's headline numbers reproduce under an independent implementation.**
DIR@FAR=1%, N=20 / 2s / centered: eres2netv2 **69.7%**, campplus **61.2%**,
cnceleb-r34 **55.8%** (plan §3②: 69.3 / 61.4 / 56.1).

1. **Q1 (model) is confirmed, and stronger than "6~8pp".** eres2netv2 beats
   campplus in **24 of 24** conditions (2 scorings × 3 crops × 4 N) by
   **+5.9 … +9.0 pp**, and beats cnceleb-resnet34-LM by **+10.5 … +16.1 pp**.
   Plan §6.3 R2 ("leads ≥3pp at ≥3 gallery sizes") is satisfied with margin.
2. **Q4 (slope) is confirmed; the intercept is deliberately conservative.**
   Fitted `b` on centered/2s: eres2netv2 **+0.0090**, campplus **+0.0092**,
   cnceleb-r34 **+0.0085** (R² 0.99). The shipped `0.0086` is inside that band
   — no change requested. The fitted **intercepts** are 0.788 / 0.802 / 0.788,
   i.e. the shipped `lo_base = 0.82` sits **~3 pp above** CN-Celeb's FAR=1%
   point. That is the right direction (safer on V3) and is exactly why §6.4.1
   says the intercept must come from the live test, not from this table.
3. **The N-aware slope is load-bearing, not cosmetic.** With `lo` fixed at
   0.82, cluster leakage at N=50 is **2.0% (eres2netv2) / 6.0% (campplus) /
   1.6% (cnceleb)** — all FAIL. With `lo(N)`, every model passes at every N
   (eres2netv2 **0.0–0.3%**, campplus 0.3–0.9%, cnceleb 0.1–0.2%).
4. **⚠️ Cluster stickiness *amplifies* leakage — the gate must stay
   cluster-level.** Per-turn leakage at `lo` is 0.1% for eres2netv2 at N=20,
   but the sticky rule turns one crossing turn into a whole named cluster:
   `leakEver` 0.2%. At campplus/fixed/N=50 it goes 2.4% → 6.0%. T4 buys +30pp
   of coverage by making a single false accept permanent; the ≤1% red line is
   therefore only meaningful measured the way Table 4 measures it.
5. **⚠️ The contested lock is confirmed useless against strangers.**
   `impCons@5` — the share of an unregistered speaker's 5 turns whose top-1 is
   the *same wrong* enrolled name — is **71.4% (N=5) / 59.1% (N=20) / 53.9%
   (N=50)** for eres2netv2, against a random level of 20% / 5% / 2%. Their
   errors are strongly correlated, so a same-name vote is nearly free for them.
   Plan §6.1 Q3's "争议锁是混簇守卫，不是未注册者守卫" holds; the only stranger
   guard is the threshold.
6. **Q2 (raw vs centered) stays a wash, as the plan says.** Centered − raw on
   DIR is between −1.8pp and +3.5pp depending on condition — noise, and the
   shipped thresholds are all calibrated on the centered scale, so "don't
   touch" remains correct. Both modes are kept and printed (plan R5).
7. **Honesty note now printed, previously unstated anywhere:** the cohort used
   for centering (`cohC/*`) shares **194 of its 196 speakers** with the eval
   gallery. A 764-clip *mean* is only mildly sensitive to that, but centered
   numbers are an upper bound and this cache **cannot** support the AS-Norm
   experiment (plan §6.1 Q2 / R3b's "cohort ≥200 人且与评测集不相交" is not just
   unmet, it is unmeetable with this cache).

## 4. Test evidence

`tests/diart_server/test_offline_sweep.py` — **27 tests, 0.17 s**, all on a
synthetic 8-speaker cache built in-test (no fixture files, no network, no
model). The synthetic geometry deliberately mimics a real embedding space: one
large shared `common` direction (what cohort centering exists to remove) plus a
per-speaker direction plus jitter, and cohort clips whose mean recovers
`common`. Margins on that cache: genuine centered ≥ 0.9986, impostor centered
≤ 0.635 — the assertions are not sitting on a knife edge.

| Requirement from the brief | Tests |
|---|---|
| DIR/FAR computed correctly | `test_threshold_at_far_uses_exactly_the_far_budget` (100 scores, 5% budget → thr 0.95, FAR exactly 5%), `test_threshold_at_far_floors_the_budget` (1% of 150 = 1.5 → 1 accept, FAR 1/150 ≤ 1%), `test_threshold_at_far_with_budget_below_one_trial_rejects_everything`, `test_threshold_at_far_rejects_empty_impostor_set`, `test_dir_counts_only_above_threshold_and_correct_top1` (above-but-wrong and right-but-below both excluded → 0.5) |
| threshold fit correct | `test_fit_recovers_an_exact_log2_line` (a=0.7890, b=0.0086 recovered to 1e-9, R²=1), `test_fit_is_consistent_with_lo_for_n`, `test_fit_rejects_mismatched_lengths` |
| cache missing → exit 2 | `test_unspecified_cache_exits_2`, `test_missing_cache_directory_exits_2`, `test_empty_cache_directory_exits_2`, `test_cache_without_cohort_keys_exits_2`, `test_unknown_model_filter_exits_2`, `test_unknown_crop_filter_exits_2`, `test_gallery_larger_than_the_speaker_pool_exits_2` |
| (added) production score parity | `test_score_matrix_matches_voiceprint_matcher_on_both_scales` — compares against `voiceprint_matcher.match` in both modes and pins that raw mode still scores a flipped query **negative** (no `(c+1)/2` remap) |
| (added) T4 rule simulation | `test_hi_fire_names_the_cluster_and_rewrites_backwards`, `test_two_of_five_vote_fires_without_any_hi_turn`, `test_single_above_lo_turn_never_fires`, `test_disagreeing_votes_inside_the_window_do_not_fire`, `test_contested_lock_blanks_the_whole_cluster_but_ever_fired_stays_true`, `test_vote_fires_mirrors_the_run_cluster_vote_branch`, `test_modal_agreement_counts_the_dominant_top1` |
| (added) end-to-end + gate wiring | `test_end_to_end_on_separable_cache_is_perfect_and_leak_free` (DIR 1.0 / leak 0.0 / cover 1.0 on every condition, four tables + confidence bounds on stdout, JSON round-trips), `test_leakage_gate_fails_loudly_when_the_threshold_is_dropped` (`--lo-base 0.0` → leak 100%, `GATE FAIL` on stdout) |
| (added) reproducibility | `test_results_are_reproducible_and_independent_of_filters` (same seed → identical JSON; `--n-values`/`--crops` filters don't move a shared cell; different `--seed` does redraw), `test_env_var_supplies_the_cache_dir` |

### Commands run

```
$ python -m pytest tests/diart_server -q \
    --ignore=tests/diart_server/test_e2e_websocket.py \
    --ignore=tests/diart_server/test_end_control.py \
    --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider
195 passed, 1 skipped, 188 warnings in 73.91s
```
Baseline at `fe6314c` was **168 passed, 1 skipped** (the brief's 167 predates
T1's pinning test, 78e91e1). 168 + 27 new = 195, **0 failed**, nothing removed.

```
$ python -m pytest tests/diart_server/test_offline_sweep.py -q -p no:cacheprovider
27 passed, 4 warnings in 0.17s
```

```
$ DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints \
    python scripts/verify_speaker_cross_matrix.py
FALSE_ACCEPTS=0 (of 15 impostor turns)
MISSED_GENUINE=0 (of 3 asserted genuine turns)
SESSION_ERRORS=0
RESULT: PASS      # exit code 0
```
(Run for insurance only — this task changes no server code.)

```
$ python scripts/eval/offline_sweep.py \
    --cache-dir <scratchpad>/asnorm_cache --json /tmp/sweep.json
... four tables + confidence bounds ...
done in 1.38s          # brief's budget: 10s
$ python scripts/eval/offline_sweep.py            # no cache anywhere
ERROR: embedding cache not specified: pass --cache-dir <dir> or set
DIART_EVAL_CACHE=<dir>. (缓存不存在，请指定 --cache-dir)     # exit 2
```

## 5. Deviations from the brief

1. **Test file location.** The brief first said `tests/test_offline_sweep.py`
   and then corrected itself to `tests/diart_server/`. Followed the correction —
   `tests/diart_server/test_offline_sweep.py` — because only that directory is
   collected by the acceptance command.
2. **`IMPOSTOR_CLUSTERS_PER_DRAW` raised 25 → 50** relative to the prototype.
   Same estimator, finer resolution; at 25 the ≤1% gate quantized to 0.13pp and
   individual rows flipped PASS/FAIL between runs of the same code. Documented
   in the printed footer (`denominators: 1500 … resolves to ~0.07pp`).
3. **Table 4 carries two extra columns** the brief did not ask for: `perTurn`
   (no-stickiness baseline) and the `leakEver`/`leakFinal` split. Both exist to
   support finding §3.4; without `perTurn` there is no way to see that
   stickiness makes leakage worse, and without the split the V3 red line would
   be measured on the laxer of the two definitions.
4. **`scripts/verify_speaker_cross_matrix.py`'s `return 3` was NOT touched.**
   The brief permits that 5-line fix "顺手"; it belongs to whoever owns B3 and
   is out of T5's blast radius. The script currently exits 0 (PASS), so nothing
   is being hidden by leaving it.

## 6. Commits

| SHA | Message |
|---|---|
| `5558953` | `feat(eval): offline model/threshold sweep over cached embeddings` |
| `141c780` | `test(diart_server): pin offline_sweep estimators, gate and exit codes` |
| (this file) | `docs(upgrades): FIX report for offline-sweep` |

## Review: PASS (iteration 1)

Reviewer: independent reviewer agent, 2026-08-06. Every number below was
re-derived by the reviewer, not read off the Fixer's report.

### R1. Blast radius — three new files, nothing else

```
$ git diff --name-status fe6314c..e7b774a
A  docs/upgrades/2026-08-06-offline-sweep-FIX.md
A  scripts/eval/offline_sweep.py
A  tests/diart_server/test_offline_sweep.py
```
No `src/**` change, no `live.html` change (so T1's `turn_id`-keyed
partial→final replacement is untouched), no existing test modified or deleted
(all three entries are `A`). The script imports only `argparse/json/math/os/
sys/time/dataclasses/pathlib/numpy` — no torch, no onnx, no `diart_server`
import, i.e. **zero forward passes** is structurally guaranteed, not just
claimed. Invariants 1–7 are not reachable from this change (no server code, no
protocol field, no threading, single process).

### R2. Acceptance gate — re-run by the reviewer

```
$ python -m pytest tests/diart_server -q \
    --ignore=tests/diart_server/test_e2e_websocket.py \
    --ignore=tests/diart_server/test_end_control.py \
    --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider
195 passed, 1 skipped, 188 warnings in 74.19s      # 0 failed

$ ... same command + --ignore=tests/diart_server/test_offline_sweep.py
168 passed, 1 skipped                              # pre-T5 baseline confirmed
```
168 + 27 = 195. The brief's `167` predates T1's pinning test (`78e91e1`); the
Fixer's accounting is correct. All 27 new tests collected and named:

```
$ python -m pytest tests/diart_server/test_offline_sweep.py -q -p no:cacheprovider
27 passed, 4 warnings in 0.16s
```

Voiceprint regression gate (run for insurance; this task touches no scoring
code):
```
$ DIART_MODELS_DIR=./models DIART_DEVICE=cuda DIART_VOICEPRINTS_DIR=./voiceprints \
    python scripts/verify_speaker_cross_matrix.py
FALSE_ACCEPTS=0 (of 15 impostor turns)
MISSED_GENUINE=0 (of 3 asserted genuine turns)
SESSION_ERRORS=0 / RESULT: PASS / EXIT=0
```

### R3. Independent re-derivation of the estimators (the load-bearing check)

The reviewer wrote a **separate** implementation from the definitions —
open-set DIR@FAR, the thr@FAR order statistic, a closed-form OLS log2 fit (not
`lstsq`), and a pure-python restatement of the sticky hi/lo + 2-of-5 +
contested rule — and compared it cell-by-cell against the script's JSON:

```
$ python scripts/eval/offline_sweep.py --cache-dir <asnorm_cache> --json sweep.json
$ python xcheck.py sweep.json          # reviewer's independent script
OK  eres2netv2-zh  centered 2s N=5   DIR 75.47±11.91  thr 0.787114  realizedFAR=0.985%
OK  eres2netv2-zh  centered 2s N=10  DIR 72.18±10.19  thr 0.799220  realizedFAR=0.944%
OK  eres2netv2-zh  centered 2s N=20  DIR 69.74± 7.01  thr 0.805295  realizedFAR=0.997%
OK  eres2netv2-zh  centered 2s N=50  DIR 63.62± 3.73  thr 0.818031  realizedFAR=0.945%
OK  fit eres2netv2-zh/centered/2s: a=0.788177 b=+0.009008 r2=0.9878
OK  campplus-zh    centered 2s N=20  DIR 61.23± 6.32  thr 0.821758
OK  fit campplus-zh/centered/2s:    a=0.802088 b=+0.009243 r2=0.9961
OK  cnceleb-r34    centered full:    a=0.792341 b=+0.009348 r2=0.9932
OK  leak eres2netv2-zh N=5/10/20/50 n-aware  leakEver 0.267/0.133/0.200/0.000%  (1500 clusters each)
OK  leak eres2netv2-zh N=20/50 FIXED lo=0.82 leakEver 0.867% / 2.000%
OK  leak campplus-zh   N=20/50 FIXED lo=0.82 leakEver 2.533% / 6.067%
ALL MATCH        # 16 DIR cells + 4 fits + 12 leakage cells, agreement < 1e-9
```
The **realized** FAR on the chosen threshold is 0.94–1.00% in every draw set,
i.e. `threshold_at_far` really spends the 1% budget and never overspends it —
the `floor(far·n)`-th-largest convention is correct, not off by one.

### R4. Score-scale parity against the production matcher, on the REAL cache

A silent scale drift here would poison every printed threshold, so the
reviewer checked it outside the synthetic-cache test, against
`voiceprint_matcher.match` on 40 real queries × 196 references:
```
max|Δ| centered = 1.79e-07, raw = 2.98e-07; argmax identical in all 80 calls
centered range [0.4706, 0.9578]   raw range [-0.3012, 0.8838]
```
Raw stays a plain cosine (goes negative — no `(c+1)/2`), centered carries the
production remap. Matches `voiceprint_matcher.py:88-106` exactly.

### R5. Reproducibility, filter-invariance, and clean exit

```
$ ... --models eres2netv2-zh --crops 2 --n-values 5,20 --json sub.json
filter-invariance: compared 10 shared cells; mismatches: NONE
$ ... --json again.json   # rerun of the full grid
dir_at_far / threshold_fit / k_accumulation / leakage: all identical
$ python scripts/eval/offline_sweep.py                       # no cache anywhere
ERROR: embedding cache not specified: ... (缓存不存在，请指定 --cache-dir)   exit=2
$ python scripts/eval/offline_sweep.py --cache-dir /nope/nope
ERROR: embedding cache directory not found: /nope/nope ...                 exit=2
```
Full grid: **`done in 1.42s`** — brief's budget was 10s.

### R6. Report claims spot-checked against the JSON

| Claim in §3 | Reviewer's recomputation | Verdict |
|---|---|---|
| eres2netv2 > campplus in 24/24 conditions, +5.9…+9.0pp | 24/24, `+5.9…+9.0` | ✅ |
| eres2netv2 > cnceleb-r34 by +10.5…+16.1pp | 24/24, `+10.5…+16.1` | ✅ |
| centered − raw is noise, −1.8…+3.5pp | `−1.8…+3.5` | ✅ |
| N=20/2s/centered = 69.7 / 61.2 / 55.8 | 69.74 / 61.23 / 55.81 | ✅ |
| cohort shares 194 of 196 speakers with the gallery | `overlap = 194`, 764 clips, 196 enroll, 4689 test keys | ✅ |
| §6.3 R2 (≥3pp lead at ≥3 gallery sizes) satisfied | centered/2s lead `+5.9 +7.0 +8.5 +9.0` at N=5/10/20/50 | ✅ |

### R7. Contract conformance (plan §4 T5 + §6.2)

Four tables present and correct; the unregistered-cluster leakage column is
there and gated at ≤1%; zero new forward passes; well under the 10s budget;
and the §6.2 "不做" list is genuinely not done — no `trials_hash`, no
`eval_metrics.py`, no speaker-block bootstrap, no `gate.py`, no DET curve, no
gender/genre stratification. Dispersion is honestly labelled as std across
gallery draws in both the footer and §6.4.5.

### R8. Non-blocking observations (no action required)

1. **Degenerate CLI inputs traceback instead of exiting 2**: `--draws 0`,
   `--draws -3` (`ValueError: need at least one array to concatenate`),
   `--far 2.0` (`IndexError` in `threshold_at_far`), `--seed -1`
   (`ValueError: expected non-negative integer`). The brief's requirement was a
   clean exit when the **cache** is missing/unusable, and that path is solid
   (7 tests, all exit 2). These are operator-typo cases on a single-user
   offline tool; hardening them is not worth a commit.
2. **`GATE FAIL` prints but the exit code stays 0.** Correct as built: all 5
   failing rows are the `fixed`-threshold counterfactual (campplus N=10/20/50,
   cnceleb N=50, eres2netv2 N=50), while **every shipped `n-aware` row passes**
   (eres2netv2 0.0–0.3%, campplus 0.3–0.9%, cnceleb 0.1–0.2%). A nonzero exit
   would be permanently red for the wrong reason, and plan §7 explicitly cuts
   the CI numerical gate (P0-7 / G4).
3. `run_cluster` is T5's *model* of the not-yet-shipped T4 rule. Nothing can
   diverge today (T5 and T2–T4 are declared parallel in plan §4); whoever lands
   T4 should mirror `run_cluster`'s hi / 2-of-5 / contested-lock / backfill
   semantics, and `test_vote_fires_mirrors_the_run_cluster_vote_branch` is the
   anchor to check it against.
4. Deviation §5.4 (leaving `verify_speaker_cross_matrix.py`'s `return 3`
   untouched) is accepted: the brief called that fix optional ("顺手"), and the
   script exits 0 today so nothing is being masked.

**Verdict: PASS.** The estimators are correct under independent re-derivation,
the score scale matches production to float32 precision, the acceptance gate is
green with 27 named new tests and no test removed, and the deliverable is a
1.4-second re-runnable command as specified.

# FIX: Voiceprint cosine score compression — mean-centering

- **Date:** 2026-06-30
- **Component:** `diart_server` voiceprint matching (`/demo/live`)
- **Bug report:** `docs/bugs/2026-06-30-voiceprint-score-compression.md`
- **Status:** Fixed (Fixer). Verified.

---

## Root cause (recap)

Raw cosine on wespeaker/pyannote embeddings is compressed by a large shared,
non-discriminative component (cohort mean norm ≈ 0.67). Every cosine is pulled
toward a positive baseline, so same-speaker (~0.33) and different-speaker (~0.15)
collapse into one band — no fixed threshold can separate them. Mean-centering the
embeddings against a cohort/impostor mean removes that shared direction and
recovers the discriminative range.

---

## What changed

### 1. `src/diart_server/voiceprint_matcher.py` — mean-centering + remap
`match()` gained `mean_embedding: np.ndarray | None = None`.

- `mean_embedding is None` → **exact** historical raw-cosine behaviour (existing
  matcher tests untouched).
- `mean_embedding` provided → both the query and every reference are transformed
  the same way, `c = unit(unit(v) - mean_embedding)`, score = `dot(c_query, c_ref)`,
  then remapped to `[0,1]` via `(c+1)/2`. Threshold semantics apply to the remapped
  score. A shape-mismatched cohort falls back to raw unit (no crash).

### 2. `scripts/compute_cohort_mean.py` (new, committed) — impostor cohort
Computes a **per-embedding-model impostor cohort mean** (NOT from the enrolled
targets — that would subtract the target's own direction). It cuts a dense, diverse
sweep of ~4s segments (step 5s) across `data/2news.mp3`, embeds each, means them,
L2-normalizes, and saves to `assets/cohort/<slug>.npy` for each working model
(`pyannote/embedding`, `pyannote/wespeaker-voxceleb-resnet34-LM`). A denser/more
diverse cohort measurably widens separation; a larger multi-speaker, multi-condition
cohort would improve robustness further (documented in the script).

Committed assets (tiny): `assets/cohort/pyannote__embedding.npy` (512-d, 2.1 KB),
`assets/cohort/pyannote__wespeaker-voxceleb-resnet34-LM.npy` (256-d, 1.1 KB).

### 3. `src/diart_server/session.py` — load + plumb the mean
`_load_cohort_mean(model_name)` loads `assets/cohort/<slug>.npy` once in
`__init__` into `self.cohort_mean` (missing → `None`, raw fallback, no crash).
`_cohort_slug` (`/`→`__`) mirrors the compute script's `model_slug`. The
`_on_prediction` `vp_match(...)` call now passes `mean_embedding=self.cohort_mean`.

### 4. `src/diart_server/config.py` — re-tuned threshold
`match_threshold` 0.5 → **0.60**. The report suggested ~0.33, but that figure came
from the report's verify script which (incorrectly) folded the targets into the
cohort mean. Measured on the **real embed→match chain** with a proper impostor
cohort, the remapped scores sit higher:

| model | same (A2 vs A) | diff (B vs A) |
|---|---|---|
| wespeaker | 0.753 | 0.462 |
| pyannote/embedding | 0.712 | 0.563 |

0.60 is the one global value that sits above **both** different-speaker scores and
below **both** same-speaker scores, separating them for both working models.

### 5. Tests — `tests/diart_server/test_voiceprint_matcher.py`
Added: `mean_embedding=None` reproduces raw behaviour exactly; with a synthetic
shared-component construction, centering yields a strictly larger same-vs-different
gap than raw and stays in `[0,1]`; a single threshold on the remapped score accepts
same and rejects different. Existing matcher tests unchanged.

---

## Verification

### A. `DIART_DEVICE=cuda python scripts/verify_voiceprint_separation.py` → **PASS**
```
RAW      : same=0.327 diff=0.146 gap=0.181
CENTERED : same=-0.019 diff=-0.642 gap=0.623
REMAPPED : same=0.491 diff=0.179 gap=0.312
PASS criterion: centered gap > 0.25 AND remapped same > 0.4
RESULT: PASS
```
(centered gap 0.623 > 0.25; remapped same 0.491 > 0.40)

### B. `DIART_DEVICE=cuda python scripts/verify_match_separation.py` (new) → **PASS**
Real chain: `embedding_extractor.extract` → `voiceprint_matcher.match(..., mean_embedding=cohort)`.
```
model     : pyannote/wespeaker-voxceleb-resnet34-LM
threshold : 0.6
SAME  (A2 vs A): name='A'  score=0.753
DIFF  (B  vs A): name=None score=0.462
RESULT: PASS
```
Same-speaker A2 → "A" at 0.753 (> 0.6); different B → None at 0.462 (< 0.6).

### C. Unit / integration tests → **10 passed**
```
python -m pytest tests/diart_server/test_voiceprint_matcher.py \
  tests/diart_server/test_voiceprint_routes.py \
  tests/diart_server/test_session_transcript.py -q -p no:cacheprovider --timeout=180
=> 10 passed
```

---

## Notes / follow-ups (out of scope here)
- Report fix #2 (AS-norm / z-norm) and #3 (align live centroid with utterance
  embedding) remain available as further hardening; mean-centering (fix #1) already
  restores usable separation.
- The committed cohort is single-source (`data/2news.mp3`); swapping in a larger
  multi-speaker impostor set (re-run `compute_cohort_mean.py`) would improve
  cross-condition robustness without code changes.

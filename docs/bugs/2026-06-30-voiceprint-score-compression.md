# Bug: Voiceprint cosine scores are compressed and don't discriminate speakers

- **Date:** 2026-06-30
- **Component:** `diart_server` voiceprint matching (`/demo/live`)
- **Severity:** High — speaker identification is effectively non-functional; a different
  speaker scores about the same as the enrolled one.
- **Status:** Diagnosed (Reporter). Not fixed.

---

## (a) Symptom (observed live in `/demo/live`)

Cosine scores against the enrolled reference are squashed into a narrow band and
fail to separate the enrolled speaker from strangers:

| Speaker (live)                  | Observed score |
|---------------------------------|----------------|
| Enrolled "吴泽" (same speaker)   | 0.51 – 0.62    |
| "speaker0" (different, unenrolled) | 0.49 – 0.50 |
| "speaker1" (another different)  | 0.32 – 0.34    |

Same-speaker (~0.5) and a different speaker (~0.5) are nearly indistinguishable —
the gap is far too small. A healthy speaker-verification system gives same-speaker
~0.7–0.9 and different ~0.0–0.4.

---

## (b) Root cause — WITH measured cosine numbers

Reproduced offline using the **exact production model and code path**
(`pyannote/wespeaker-voxceleb-resnet34-LM`, 256-d, loaded from
`models/pyannote/wespeaker/pytorch_model.bin`; embeddings via
`diart.blocks.embedding.SpeakerEmbedding`, identical to
`embedding_extractor.extract`). Clips were cut from `data/2news.mp3`:
`spkA_1`=1–6 s and `spkA_2`=20–24 s (same voice), `spkB`=9–18 s (different voice).
Run on `device=cuda`.

### Measured cosines (utterance enrollment path = production)

```
RAW (current production: voiceprint_matcher.match, plain cosine of L2-normed vecs)
    same  cos(A1,A2) = 0.327
    diff  cos(A1,B)  = 0.237
    diff  cos(A2,B)  = 0.055
    diff  mean       = 0.146
    GAP (same-diff)  = 0.181      <-- tiny: this IS the bug
```

**The compression is intrinsic to raw cosine in this embedding space, not a live
artifact.** Even with two clean whole-utterance embeddings of the *same* speaker,
raw cosine is only **0.327** — nowhere near the 0.7–0.9 a good system shows. The
model's embeddings share a large common (non-discriminative) direction, so every
cosine is pulled toward a positive baseline and the usable dynamic range collapses
("cosine cone" effect).

This is confirmed by the **cohort mean embedding norm = 0.666** (a set of unit
vectors with no shared direction would average to ~0). A norm that large means a
dominant shared component is inflating all raw cosines.

### Fix hypothesis verified: mean-centering removes the shared component

Subtract a cohort/global mean embedding, then re-normalize, then cosine:

```
CENTERED (subtract cohort mean, renormalize, then cosine)
    same  cos(A1,A2) = -0.019
    diff  mean       = -0.642
    GAP (same-diff)  =  0.623      <-- 3.4x larger separation

REMAPPED to [0,1] via (c+1)/2  (gives a usable absolute score)
    same = 0.491   diff = 0.179   gap = 0.312
```

Mean-centering widens the same-vs-different gap from **0.181 → 0.623** (3.4×).
After remapping the centered cosine back to a [0,1] scale, the same speaker lands
at ~0.49 and a different speaker at ~0.18 — clearly separable, with a real margin.

(For reference, an AS-norm / z-score-vs-cohort normalization gives an even cleaner
spread: same ≈ -0.03, diff ≈ -0.7 to -1.05 — same conclusion, the shared baseline
must be normalized out.)

### Secondary contributors (not the primary cause, but they hurt)

1. **Enrollment vs live representation mismatch.** Enrollment (`embedding_extractor.extract`)
   is a single whole-clip utterance embedding. Live (`session._extract_embedding`)
   is the diart **clustering centroid** (`pipeline.clustering.centers[idx]`), an
   accumulating mean of many short noisy frame embeddings, L2-normed. Averaging many
   short, partially-overlapped frames further pulls the live vector toward the shared
   mean direction, compounding the compression and explaining why the *live* scores
   sit even higher/flatter (~0.5 for everyone) than the offline utterance numbers.
   (Empirically, `cos(live_centroid, same-clip utterance) = 1.0` only when the live
   clip is identical to the enrollment clip — i.e. it tracks the source, but the
   shared-baseline inflation remains.)
2. **No score normalization at all.** `voiceprint_matcher.match()` does raw
   `np.dot` of L2-normalized vectors with a flat `threshold=0.5`. With raw same-speaker
   cosines around 0.33–0.5 and different-speaker around 0.15–0.5, **no fixed threshold
   can separate them** — which is exactly what `/demo/live` shows.
3. **Short / unequal enrollment clips.** Enrollment clips of a few seconds give noisier
   utterance embeddings; combined with the above this widens within-speaker variance.

---

## (c) Recommended fixes — RANKED, concrete for this codebase

### 1. (Primary) Mean-center embeddings before cosine + remap score to [0,1]
Compute a fixed **global mean embedding** (a "cohort"/"impostor" mean) once, subtract
it from every embedding, re-normalize, then cosine. Remap the result with `(c+1)/2`
so the existing `0.5` threshold semantics still make sense (or re-tune the threshold).

- **Where:** `src/diart_server/voiceprint_matcher.py` — add the mean-subtraction and
  remap inside `match()` (and `_unit`), so BOTH enrollment refs and the live vector
  pass through the same transform. Plumb a `mean_embedding: np.ndarray | None`
  parameter through the single caller in `src/diart_server/session.py`
  `_on_prediction` (the `vp_match(embedding, refs, settings.match_threshold)` call).
- **Where to source the mean:** compute it once at enrollment/startup from the cohort
  of all enrolled references (or a small fixed impostor set), store alongside the
  voiceprints (see `voiceprints/manifest.json` and the store). Keep it per embedding
  model.
- **Verified effect:** gap 0.181 → 0.623; remapped same 0.49 vs diff 0.18.

### 2. (Strong, complementary) AS-norm / z-norm score normalization
Instead of (or on top of) mean-centering, z-normalize each candidate cosine against a
cohort of impostor embeddings (adaptive symmetric normalization). Verified to give
even cleaner separation (same ≈ -0.03 vs diff ≈ -0.7…-1.05). More robust to
recording-condition shifts than a single global mean. Same integration point in
`voiceprint_matcher.match()` plus a stored cohort embedding set.

### 3. Align live representation with enrollment
In `session._extract_embedding`, consider matching against an utterance-style
embedding (e.g. re-embed the finalized turn's PCM from `self.ring` with
`embedding_extractor.extract`) rather than the raw clustering centroid, so live and
enrollment vectors are produced the same way. At minimum, document/keep the centroid
but apply the same mean-centering (fix #1) to it.

### 4. Re-tune `match_threshold` after normalization
`settings.match_threshold = 0.5` (in `config.py`) is meaningless on raw compressed
cosines. After fix #1/#2, set it from the measured same/diff distributions (e.g. the
midpoint ~0.33 on the remapped scale, or ~ -0.3 on the centered scale).

### 5. Enforce a minimum enrollment duration
In `embedding_extractor` / the enroll endpoint, require e.g. ≥ 3–5 s of speech to
reduce within-speaker embedding variance.

---

## (d) Verification (for the Reviewer)

**Script:** `/home/edward/research/diart/scripts/verify_voiceprint_separation.py`
(auto-generates the wav clips under `/tmp/vp_investigation` from `data/2news.mp3`,
loads the production wespeaker model, measures RAW vs CENTERED/REMAPPED cosines).

**Command (run from repo root):**
```bash
DIART_DEVICE=cuda python scripts/verify_voiceprint_separation.py
```

**Pass criterion (exit code 0 AND prints `RESULT: PASS`):**
> centered gap `(same − diff_mean)` **> 0.25** AND remapped same-speaker score
> `((same+1)/2)` **> 0.40**

The script prints `RESULT: PASS`/`FAIL` and exits 0/1 accordingly.

**Current (buggy) baseline output** — RAW path gap is only 0.181 (the bug); the
CENTERED/REMAPPED path already satisfies the gate, proving the fix direction:
```
RAW      : same=0.327 diff=0.146 gap=0.181
CENTERED : same=-0.019 diff=-0.642 gap=0.623
REMAPPED : same=0.491 diff=0.179 gap=0.312
RESULT: PASS
```

After the Fixer wires mean-centering into `voiceprint_matcher.match()` (fix #1),
the Reviewer should additionally confirm that `match()` itself — exercised on the
SAME three clips through the real `embedding_extractor.extract` +
`voiceprint_matcher.match` call chain — produces a same-speaker score above the
(re-tuned) threshold while both different-speaker scores fall below it.

---

## Investigation artifacts
- Investigation script (full, incl. live-centroid probe): `/tmp/vp_investigation/investigate.py`
- Verification script (committed): `/home/edward/research/diart/scripts/verify_voiceprint_separation.py`
- Audio clips: `/tmp/vp_investigation/{spkA_1,spkA_2,spkB,coh_0..4}.wav`
- Model: `pyannote/wespeaker-voxceleb-resnet34-LM` at
  `models/pyannote/wespeaker/pytorch_model.bin` (256-d), device `cuda`.

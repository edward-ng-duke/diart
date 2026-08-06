# FIX report: drop the ≈candidate hint and cross-turn guessing (T1)

- **Date:** 2026-08-06
- **Fixer:** fixer agent (T1 of the voiceprint selection plan)
- **Base:** 47e0ecd — **Commits:** 7fbca19 (live.html + reversed assertions),
  78e91e1 (pinning test), this report
- **Contract:** docs/superpowers/plans/2026-08-06-voiceprint-selection-plan.md
  §4 T1 (+ the task brief; T1 has no separate spec). Implemented as written;
  no deviations.
- **Scope:** frontend only. **Server code: zero changes** — `cand_name` /
  `cand_score` remain in `protocol.py` and are still emitted by
  `session.py:744-745`; T3 is what makes them permanently `None`.

## 1. Why

Plan §3 ⑤: the `≈某某 0.79?` hint appears on **100%** of unregistered-speaker
turns and is **0%** correct. It costs no recall, but it is the only
*structural* failure source for acceptance criterion V3 (plan §2, hard red
line): "an unregistered speaker talks for 2 minutes and no registered name
appears on screen". As long as the page renders a best-match name below the
confirm gate, V3 fails by construction, regardless of model or threshold.

The second leak in the same branch is quieter: below the confirm gate the
server sends `name=None` **with a real `score`** (`session.py:721`
`name, score = None, cand_score`), and the old page rendered that score next
to the neutral label. A bare number next to a nameless line is exactly the
"invites the reader to fill in the name" failure the plan's protocol
invariant 2 warns about, so it goes too.

## 2. Changes

### src/diart_server/static/live.html (the whole change)

- **Removed** `const lastVerified = new Map()` and its only writer, the
  `ws.onmessage` branch
  `else if(m.type==='prediction'){ if(m.name!=null) lastVerified.set(...) }`.
  Cross-turn guessing is gone: nothing on the page now carries an identity
  from one turn to another. (T4 will re-introduce a `prediction` consumer for
  cluster-level stickiness — that is a *retroactive rewrite of confirmed
  names*, a different mechanism; nothing is left behind for it here.)
- **Removed** the whole candidate branch of `lineHTML`, i.e. both
  `≈${m.cand_name} ${m.cand_score.toFixed(2)}?` and
  `≈${lastVerified.get(m.speaker).name}?`, **and** the
  `if(m.score!=null) sc = …` that lived in the nameless branch.
- **Added** `neutralLabel(speaker)` (4 lines) next to `turnLines`; the
  nameless branch is now one line:
  `who = \`<span class="unk">${neutralLabel(m.speaker)}</span>\``.
- **Removed** the now-unused `.hint{…}` CSS rule (it existed only to dim the
  hint text).

**Unchanged, verified by the new pinning test:** the confirmed-name render
(`<span class="name">` + `(score)`), the `turn_id`-keyed partial→final
in-place replacement including the stale-partial guard, the `⑂重叠` badge,
the enrollment path, and the `voiceprint_ready` dropdown filter.

### Constants / choices and their basis

| Choice | Value | Basis |
|---|---|---|
| Neutral label text | `说话人 N` | Plan §4 T1 verbatim ("未确认一律渲染中性「说话人 N」"), §2 ("宁可显示『说话人 2』") |
| `N` for diart label `speakerK` | `K + 1` | Human-facing 1-based counting; `说话人 0` reads as a bug to a test subject. The digits are read off the trailing `\d+` of the diart label, so numbering is stable per session and identical to the label ordering |
| Unknown-shaped label | bare `说话人` | Fail-safe: if diart ever emits a non-`speakerN` label the page still shows a neutral string rather than echoing the raw label |
| Score on unconfirmed lines | never rendered | Plan §5 protocol invariant 2 ("`unknown` 与 `insufficient` 对用户渲染成同一个中性标签"); the server still ships one (`session.py:721`) |
| No new constants, no new env vars | — | T1 is 0.25 person-day display surgery |

### tests/diart_server/test_live_route.py

- `test_live_page_identity_continuity_hint_static_asserts` → **replaced by**
  `test_live_page_no_similarity_hint`. The four overturned assertions,
  recorded in the new test's docstring per the file's existing convention:

  | Overturned assertion | Why it is now false |
  |---|---|
  | `assert "≈" in r.text` | the hint character exists nowhere on the page (nor in a comment — deliberately, so this page-wide assertion is airtight) |
  | `assert "lastVerified" in r.text` | cross-turn identity carry-over deleted |
  | `assert "m.cand_name" in r.text` | per-line candidate name never rendered |
  | `assert "m.cand_score.toFixed(2)" in r.text` | per-line candidate score never rendered |

  (The same test also asserted `".hint{" in r.text` and
  `"type==='prediction'" in r.text`; the first is dropped with the CSS rule,
  the second is dropped because the branch has no consumer left. Neither is
  pinned in reverse — `prediction` consumption is legitimate again in T4.)
  **Basis for the reversal:** plan §3 ⑤ (100% appearance / 0% correctness on
  unregistered speakers) + §2 V3 as a hard red line. The old test was written
  when the hint was believed to be a continuity aid on *registered* speakers;
  the open-set measurement (plan §3 ③: pyannote@0.82 per-turn false-name rate
  1.96% at N=5, 6.93% at N=20) says the page cannot tell which world it is in.

- New `test_live_page_no_similarity_hint` asserts, positively:
  `"≈" not in r.text`, `"lastVerified" not in r.text`,
  `"cand_name" not in r.text`, `"cand_score" not in r.text`,
  `"说话人" in r.text`, and — by slicing the `else` branch of `lineHTML` out
  of the served page and stripping `//` comments — that the branch mentions
  neither `name` nor `score`.
- New `test_live_page_keeps_turn_replacement_and_overlap_badge` pins the
  behaviours T1 must not disturb (`turnLines`, `turnLines.set(tid, div)`,
  the stale-partial guard line, `⑂重叠`, `m.overlapped`).
- **No existing test deleted.** The three other tests in the file and all
  server-side `cand_name`/`cand_score` assertions
  (`test_identity_overlap_purity.py:194/217/255`,
  `test_session_transcript.py:357/391/534`) are untouched — the wire contract
  did not move.

## 3. Verification (all run on this machine, post-fix, at 78e91e1)

### 3.1 Unit gate (the ONLY acceptance command) — PASS

`python -m pytest tests/diart_server -q --ignore=tests/diart_server/test_e2e_websocket.py --ignore=tests/diart_server/test_end_control.py --ignore=tests/diart_server/test_failure_modes.py -p no:cacheprovider`

```
168 passed, 1 skipped, 188 warnings in 74.00s (0:01:13)
```

Baseline at 47e0ecd is `167 passed, 1 skipped`. Delta `+1` = the new pinning
test; the replaced test is 1-for-1. **0 failed.** Named:

```
test_live_route.py::test_live_page_served                                PASSED
test_live_route.py::test_live_page_enrollment_channel_static_asserts     PASSED
test_live_route.py::test_live_page_no_similarity_hint                    PASSED
test_live_route.py::test_live_page_keeps_turn_replacement_and_overlap_badge PASSED
test_live_route.py::test_live_page_dropdown_references_voiceprint_ready   PASSED
```

### 3.2 Render proof on the real page source — 0 leaks / 4 unconfirmed shapes

Static assertions prove the strings are gone; this proves the *rendering* is.
A throwaway script (out of tree, rerun with `node <path>`;
`/tmp/claude-1000/-home-edward-research-diart/6fa6a523-2deb-44f0-9b8b-14031b757d63/scratchpad/render_check.js`)
slices `neutralLabel`+`lineHTML` straight out of
`src/diart_server/static/live.html`, `new Function`s them, and feeds the five
`TranscriptMessage` shapes the server can actually produce:

```
 ok  | confirmed name + score           | <span class="t">1.0–2.4s</span> <span class="name">吴泽</span> (0.87): 你好
 ok  | below-gate candidate (impostor)  | <span class="t">1.0–2.4s</span> <span class="unk">说话人 1</span>: 你好
 ok  | candidate-only run (short turn)  | <span class="t">1.0–2.4s</span> <span class="unk">说话人 1</span>: 你好
 ok  | no identity at all               | <span class="t">1.0–2.4s</span> <span class="unk">说话人 1</span>: 你好
 ok  | second label, overlapped         | <span class="t">1.0–2.4s</span> <span class="unk">说话人 2</span>: 你好 <span class="badge">⑂重叠</span>

name/score leaks on unconfirmed lines: 0/4
```

Row 2 is the exact V3 failure case (`name=null, score=0.79,
cand_name='吴泽', cand_score=0.79`): pre-fix it rendered
`speaker0 ≈吴泽 0.79?`, post-fix `说话人 1`. Row 5 also shows the `⑂重叠`
badge and the second label numbering surviving. `node --check` on the page's
extracted `<script>` is clean (`JS_SYNTAX_OK`).

### 3.3 Voiceprint cross-matrix gate — NOT RUN, and correctly so

`git show --stat 7fbca19 78e91e1` touches exactly
`src/diart_server/static/live.html` and
`tests/diart_server/test_live_route.py`. T1 does not touch the scoring or
identity chain (`session.py`, `voiceprint_matcher.py`,
`embedding_extractor.py`, `config.py`, cohort loading are all byte-identical),
so `scripts/verify_speaker_cross_matrix.py` cannot change value and the
brief's condition ("触碰打分/身份链路时") is not met. **T2 is the next task
that must run it** (and must first fix its `return 3` SKIPPED-as-FAIL, plan
§7 P0-1).

## 4. Invariants — status

1. Identity only from this turn's own pure re-embedded audio; the server never
   stamps an unverified name: **untouched** (server diff is empty). The page
   now additionally refuses to *display* anything the server did not confirm —
   the last remaining place where a name could reach the screen without
   turn-level verification.
2. Fail-closed identification, never blocking ASR: **untouched**.
3. Majority-overlap turns not identified and not cached: **untouched**; and
   the client-side cache that used to accumulate names per label
   (`lastVerified`) is gone entirely.
4. Ring clock ≡ diart clock: **untouched**.
5. No cohort ⇒ identification disabled: **untouched** (the `voiceprint_ready`
   dropdown filter is still pinned by its own test).
6. Additive wire protocol: **no protocol change at all**; a page from this
   commit and one from before it both parse the same messages.
7. Thread safety: no server-side state touched. On the client, the only
   removed state (`lastVerified`) was single-threaded JS.

## 5. Known limits of this change

- V3 is only *structurally* fixed here: no unconfirmed line can show a name.
  A **confirmed** wrong name is still possible and is untouched by T1 — that
  is what T3's model swap + N-aware threshold and T4's dispute lock address.
  Per plan §3 ③, pyannote@0.82 still false-confirms 1.96% of turns at N=5.
- Neutral labels are not stable identities: they are diart cluster labels
  +1. If diart splits a speaker across two labels the page shows
  `说话人 1` / `说话人 2` for one person. Unchanged from before (the old page
  showed `speaker0`/`speaker1`), and out of T1's scope.
- The static assertions are string-level; they cannot prove a future edit
  reintroduces a leak through a differently-named variable. §3.2's render
  check is the behavioural backstop, but it is a scratchpad script, not part
  of the pytest gate (it needs node).

## Review

pending

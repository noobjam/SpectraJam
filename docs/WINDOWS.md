# Arbitrary-window embeddings

## Decision

SpectraJam treats a calendar year as an acquisition/storage shard, not as the
semantic unit of an embedding. The training and evaluation contract covers
causal windows from 7 to 365 days, with explicit 7-day and 14-day anchors.

This supports three views of the same observation history:

| Mode | Interval | Meaning |
|---|---|---|
| Arbitrary | `[start, end)` | Any requested historical interval |
| Rolling | `[end - width, end)` | Only the most recent `width` days |
| Coverage prefix | `[coverage_start, cutoff)` | Everything in the declared coverage interval so far |

All boundaries are UTC calendar-day indices and all intervals are half-open.
An observation at `end` is future information and is excluded. Absolute
day-of-year preserves phenological phase; absolute epoch day preserves order
across New Year; elapsed day from the window start supplies relative timing.

Core slicing supports cross-year windows when the caller supplies merged
timelines. The future materialization loader must join only adjacent shards in
the same temporal split; a train window must never enter validation or test.
It must never wrap day-of-year and then sort on day-of-year alone.

## Training distribution

The committed configuration samples a continuum rather than seven isolated
tasks:

- 70% of batches select uniformly from 7, 14, 30, 60, 90, 180, and 365 days.
- 30% draw a log-uniform integer duration from 7 through 365 days.
- 20% begin at the declared coverage origin to train cumulative prefixes; the rest
  use a random causal start and therefore cover arbitrary/rolling intervals.
- Two SSL views independently drop observations with probability 0.2 while
  sharing the *same exact interval* and preserving at least one observation
  from each non-empty modality.
- A short sequence is padded and masked. Observations are never duplicated to
  make a 7-day window look well observed.

The 70/30 and 20% values are experimental defaults, not published optima. They
must remain in the run configuration and enter the experiment fingerprint.

### Teacher-student

The compact student receives absolute day-of-year, relative elapsed day,
window duration, valid-observation counts, and missing-modality indicators. A
frozen, mask-aware TESSERA teacher encodes the same selected window. An annual
teacher target is forbidden during short-window training because it contains
future evidence that the student did not receive.

TESSERA was pretrained primarily on annual sequences, so its short-window
target is itself out of distribution. The default scales teacher alignment by
`min(1, window_days / 90) × min(1, max(valid_S1, valid_S2) / 10)` while the
regional two-view objective remains fully active. Ninety days and ten
observations are explicit ablation values, not evidence-backed thresholds.

### LoRA

`WindowedTesseraEncoder` is an external mask-aware runtime around the untouched
checkpoint graph. It compacts each modality before calling the official-shape
branch, groups equal sequence lengths, and preserves exact dense-input output
at step zero. A tiny zero-initialized residual conditioner supplies duration,
counts, and modality-presence metadata alongside LoRA; it is stored in the
adapter artifact and does not alter the frozen checkpoint. LoRA is trained
against an immutable base encoder on the same window.

TESSERA input mixup is used only when every mixed view has a full aligned draw.
Sparse short windows keep Barlow/base-anchor terms and skip mixup rather than
fabricating acquisition dates.

### Missing observations

A 7-day or 14-day interval can legitimately contain no cloud-free Sentinel-2
observation. The student handles either modality as missing and the TESSERA
wrapper gives an empty modality a zero branch. When both modalities are empty,
the public incremental builder returns `empty_window=true` and no embedding.
Every downstream record must retain duration and S1/S2 valid counts. Empty
windows are reported or explicitly abstained, not silently dropped, resampled,
or converted into an apparently valid vector.

## Exact incremental builder

The current encoders are bidirectional Transformers followed by temporal
pooling. Adding or evicting an observation can change every hidden token, so a
pooled vector cannot be safely updated by adding or subtracting one
contribution.

`ExactIncrementalEmbeddingBuilder` therefore means:

1. Upsert newly arrived observations by stable sensor-item ID.
2. Select the exact requested window.
3. Hash the model, adapter, preprocessing policy, provider profile, runtime
   policy, bounds, IDs, masks, timestamps, and normalized tensors.
4. Return a cached vector only for an identical hash; otherwise recompute the
   complete selected window.

This is exact with respect to a fresh full-window forward pass and naturally
invalidates affected historical windows after a late or reprocessed scene.
Unchanged windows stay cache hits. The implemented cache is in-process; a
durable SQLite/object-store backend can implement the same small cache
protocol without changing semantics.

A genuinely stateful append-only encoder is a separate model experiment. A
causal retention/state-space architecture could accelerate prefixes, but a
strict rolling window still needs a model whose state supports deletion or a
replay of retained observations. It must be compared against exact
recomputation before being called equivalent.

## Evaluation matrix

Evaluate both countries and every frozen/student/LoRA model at:

- durations: 7, 14, 30, 60, 90, 180, and 365 days;
- modes: rolling, cumulative coverage prefix, and arbitrary interval;
- valid S2 counts: 0, 1, 2, 3–5, 6–10, and more than 10;
- S1 states: available, missing, ascending-only, and descending-only;
- spatial, temporal, combined, and cross-country holdouts.

For early crop classification, create a prediction at each cutoff using only
scenes with sensing time before the cutoff. An operational replay should also
enforce scene availability time, because a product sensed before the cutoff
but published later was not actually available then. Report accuracy/F1 as a
curve over cutoff date and window length, plus coverage and abstention rate;
do not report only the best post-hoc cutoff.

# SpectraJam design

## 1. Decision summary

Use TESSERA v1.1 with the Microsoft Planetary Computer profile for the first
controlled experiment. Never mix MPC imagery, AWS normalization, and another
checkpoint. Run AWS/OPERA only as a separately named provider-profile ablation.

Build both requested adaptations, plus the frozen baseline:

1. Frozen global TESSERA v1.1.
2. Rwanda LoRA and Israel LoRA, with a shared-adapter control.
3. Rwanda student, Israel student, and shared Rwanda+Israel student.
4. Optional adapted-LoRA teacher to standalone student.

Each regional model is evaluated in both countries. Otherwise generic gains
from extra self-supervision can be mistaken for regional specialization.

## 2. Sampling frame

The preferred frame follows TESSERA's own spatial thinning: one anchor on a
200 m lattice. Rwanda and Israel together should yield roughly 1.1–1.3 million
land anchors after the boundary and land masks are applied.

The candidate-frame and future batch-balancing hierarchy is:

1. Country.
2. Primary stratum: RESOLVE ecoregion × ESA WorldCover 2021 class.
3. Balancing variables: elevation, slope, temperature, precipitation,
   precipitation seasonality, and valid-observation count quantiles.
4. Whole 20 km spatial blocks assigned to train, validation, or test.

Do not materialize the Cartesian product of all covariates. The current sampler
allocates proportional to the square root of the single persisted primary
`stratum`, with a floor of 200 where candidates exist. It makes a deterministic
hash-random draw from a separately constructed 200 m candidate lattice and
persists inclusion probabilities. Continuous elevation/climate/observation
balancing belongs to candidate-frame construction or the future batch sampler;
it is not yet implemented by `select_candidates`. A true maximin draw remains
a sampling ablation. Evaluation metrics remain area-weighted rather than
reflecting the oversampled training distribution.

The full-lattice distributed runner must keep every anchor, sample strata with
probability proportional to `sqrt(N_h)`, select Rwanda and Israel equally for a
joint model, and mix spatial blocks globally. That runner is a next milestone.

## 3. Years and leakage controls

The common window is 2019–2025. Global operational Sentinel-2 L2A production
began only in December 2018, and 2026 is incomplete.

- Train: 2019–2023.
- Temporal validation: 2024.
- Temporal test: 2025.
- Pilot train subset: 2019, 2021, 2023.

Spatial and temporal roles are separate columns. Training consumes only
`spatial_split=train AND year_split=train`. Report spatial-only,
temporal-only, and combined spatial-temporal holdouts. Adjacent years are not
positive pairs because crop rotation and land change are real signals.

Calendar years are storage/acquisition shards only. Training samples causal
half-open intervals from 7 through 365 days, with explicit 7/14/30/60/90/180/365
anchors plus continuous log-uniform durations. Rolling, cumulative-prefix, and
arbitrary windows share one contract. Cross-year windows join neighboring
shards only when every observation remains in the same temporal split, then
sort by absolute observation day. In particular, no train window may cross
from 2023 into the 2024 validation split. See [Windows](WINDOWS.md).

## 4. Boundary policy

Boundary choice is part of the experiment, not an invisible library default.
Use World Bank Official Boundaries v2 with a recorded publication date,
download URL, SHA-256, CRS, license, and policy. Rwanda is cross-checked against
the MININFRA Geoportal. For Israel, exclude Non-Determined Legal Status Areas in
the default run and name any alternate operational extent as a separate run.
This is a reproducibility policy, not a legal-status judgment.

## 5. STAC acquisition and completeness

Query STAC once per 10–20 km work tile and month/year, not once per point. The
client consumes all pagination links. Query snapshots are immutable and
resumable, while repeated raw items are content-addressed once. It does not
impose `eo:cloud_cover < 20`: that property summarizes a whole Sentinel tile
and is not point validity. Per-pixel SCL and upstream-compatible preprocessing
determine validity.

The durable ledger declares every expected `(sample_id, modality)` before work
starts. The implemented generic worker loop uses leases and full-jitter retry.
The point-store writer publishes immutable Parquet/Zstd shards with exact S2
`uint16` and S1 scaled-dB `int16` units, returns their SHA-256, and the reader
can enforce that digest. The future sparse imagery reader must commit those
digests to the ledger, add just-in-time MPC signing, and use asset-major COG
block reads. It must retry 408, 425, 429, 5xx, connection resets, and truncated
reads while honoring `Retry-After`; schema errors and malformed requests remain
terminal.

Every anchor-year eventually receives one terminal data outcome:

- complete
- insufficient valid observations
- no source observation
- terminal data error

The current ledger is bound to manifest/config hashes, rejects duplicate IDs,
revalidates the country/year universe, streams inserts, implements
success/failure and unresolved-state gating, and requires parity plus completion
before trainer construction. The
next materialization milestone adds the three explicit scientific terminal
reasons and per-sensor observation counts. The invariant is that expected rows
equal the sum of all terminal outcomes and unresolved rows equal zero.

## 6. Cloud and observation quality

Run two named profiles:

- Compatibility mask: exact upstream TESSERA v1.1 MPC preprocessing.
- Enhanced-mask ablation: no-data, defective, dark, shadow, medium/high cloud,
  cirrus, and snow invalid; SCL 7 uncertain; optional 40–60 m dilation.

SCL uses nearest-neighbor resampling. Never apply a blanket Sentinel-2 `-1000`
offset across providers; processing-baseline harmonization differs by source.

Keep sparse points and label their quality:

- green: at least 20 valid S2 observations/year
- amber: 10–19
- red: fewer than 10

Report every embedding metric by quality tier.

## 7. Teacher-student track

The default student has two S1/S2 temporal branches, width 256, two Transformer
blocks per branch, four heads, FFN width 1024, and a 128-dimensional fused
output. The medium ablation uses width 384, three blocks, six heads, and FFN
width 2048.

The v1.1 teacher sees the complete compatible sequence inside the *same sampled
window* as the student. Annual targets are never used for short windows.
Student views use at most 20, 30, or 40 real observations from that interval;
sparse windows are padded and masked without observation duplication.

The implemented core objective combines:

- cosine alignment to the frozen teacher
- pairwise Gram-geometry preservation
- Barlow two-view invariance
- variance and covariance collapse prevention

Add branch-level S1/S2 alignment after the upstream parity dataset exposes
intermediate features. Whole-modality dropout and contiguous-season masking are
ablations, not silent defaults.

The student conditions on absolute day-of-year, elapsed day from window start,
window duration, observation counts, and missing-modality flags. Teacher
alignment ramps with both configured duration and observation-count thresholds
so sparse short-window regional SSL is not dominated by an annual-pretrained
teacher.

## 8. LoRA track

The base encoder is immutable. Separate country adapters target packed Q/V and
attention output weights at ranks 4, 8, and 16. Rank 8 then expands to FFN
`linear1`/`linear2` and the fusion reducer. Input-projection adaptation is a
separate sensor-shift ablation.

All B matrices start at zero, so step-zero outputs must exactly equal the base.
The external mask-aware runtime compacts padded sequences and is exactly
base-equivalent for dense inputs; it does not alter checkpoint topology.
The implemented objective is Barlow two-view SSL, TESSERA-style mixup
consistency, and a sweepable frozen-base cosine/geometry anchor. Until the v1.1
training projector is reproduced, the 128-D objective is explicitly an ablation
rather than a claim of paper-equivalent pretraining.

## 9. Evaluation and stop gates

Report at least three seeds and these metrics:

- linear probe and kNN on multiple independent labels
- cross-country, spatial, temporal, and combined holdouts
- RankMe/effective rank and anisotropy
- teacher distance-matrix Spearman correlation
- neighborhood agreement at k=10 and k=100
- consistency at 10/20/30/40 valid observations
- rolling/prefix/arbitrary curves at 7/14/30/60/90/180/365 days
- metrics conditioned on 0/1/2/3–5/6–10/>10 valid S2 observations
- S1-missing and S2-sparse robustness
- trainable parameters, peak memory, throughput, and storage

Success gates:

- Student: at least 5× faster and below 20% of teacher parameters while staying
  within one point of frozen TESSERA on held-out probes.
- LoRA: at least one-point improvement on two independent regional probes with
  no more than one-point loss on spatial or temporal holdouts.
- If neither gate passes, publish the negative result and keep frozen TESSERA.

# Music Recommendation System

Music recommendations from implicit listening data. The project set out to test whether ALS with
**explicit negative feedback** beats the usual positives-only formulation. On the evaluation used
here it does not — see [Results](#results) — and the shipped model is positives-only.

## Data

| | |
|---|---|
| Interactions | 9,711,301 |
| Users | 962,037 |
| Tracks | 30,459 |
| Catalog (Music Info) | 50,683 tracks |

After dropping users and tracks with fewer than 5 interactions: **534,735 users x 27,049 tracks**.

The matrix is 0.033% dense, and every distribution in it is long-tailed. Playcount has a mean of
2.63 against a median of 1, and interactions per track have a mean of 319 against a median of 57 —
in both cases a thin tail is dragging the mean far above the typical row:

| | playcount | interactions per user | interactions per track |
|---|---|---|---|
| mean | 2.63 | 10.09 | 318.83 |
| std | 5.71 | 14.56 | 1,105.02 |
| median | 1 | 5 | 57 |
| p90 | 5 | 24 | 738 |
| p99 | 21 | 71 | 3,836 |
| max | 2,948 | 784 | 80,656 |

`playcount == 1` is 60.8% of all rows and `<= 2` is 76.4%, which is the whole reason the
negative-feedback formulation below is worth trying at all.

To reproduce these numbers and the full distribution:

```bash
uv run python -m src.data.explore
```

## Approach

Playcount is turned into a signed confidence. `implicit` reads a positive value as
"preference 1 with this confidence" and a negative value as "preference 0 with confidence
`abs(value)`", which is what makes explicit negatives possible:

```
playcount >= threshold  ->  +(1 + alpha * log1p(playcount))   positive
playcount <  threshold  ->  -beta                             explicit negative
```

The intuition for the negative: the user was exposed to the track and did not come back.
This matters because **playcount=1 is 61% of all rows**, so how it is treated largely defines
the model. At `threshold=1` nothing is negative and the formulation reduces to the standard
positives-only one, which makes it a built-in control rather than a separate baseline.

All three of `threshold`, `alpha` and `beta` are swept — `threshold` in {1, 2, 4}, `alpha` in
{10, 20, 40, 80}, `beta` in {2, 5, 10, 20}.

Two constraints worth knowing:

- **`beta` must be > 1.** The solver weights a preference-0 entry by `(confidence - 1)`, and
  unobserved pairs already carry confidence 1. At `beta = 1` an explicit negative would be
  exactly as informative as never having heard the track, so the signal would silently vanish.
- **`alpha` is pinned to 1.0 in `train_als`.** `implicit` rescales the matrix by its own alpha,
  which would scale the negatives along with the positives and undo the balance set in
  preprocessing. The tuning knobs live in `preprocess.confidence()` instead.

## Evaluation setup

20% of interactions are held out **before** any confidence transform is applied. This matters: an
earlier version split the confidence matrix instead, holding out only entries that were positive
*under the config being tested*. That gave each threshold a different test set — `threshold=4` was
graded only on heavy-play tracks, which are far easier to predict — and made the scores
incomparable across thresholds.

Ground truth is therefore fixed and threshold-independent: **a held-out interaction is relevant if
the user listened to the track at all**. Every config below is scored against that same test set —
1,735,296 held-out interactions, covering the 466,061 of 534,735 users who received at least one
(the rest have too few interactions for the 20% draw to land on any).

The 36-config sweep runs on a deterministic 40% sample of users at 15 ALS iterations; the winner is
then retrained on all users at 20 iterations. `factors=64`, `regularization=0.05` throughout.

`beta` is omitted at `threshold=1`, where no interaction can be negative (minimum playcount is 1),
so the 4x4x3 grid collapses from 48 configs to 36.

## Results

**Negative feedback did not help. Every configuration that used it scored worse than every
configuration that did not**, and the penalty grew monotonically with both the threshold and the
weight `beta`.

**threshold=1** — no negatives, so `beta` does not apply. ndcg@10 / map@10:

| alpha | 10 | 20 | 40 | 80 |
|---|---|---|---|---|
| ndcg@10 | 0.1473 | 0.1512 | **0.1525** | 0.1503 |
| map@10 | 0.0973 | 0.1000 | **0.1010** | 0.0998 |

**threshold=2 and 4** — ndcg@10, best `alpha` per row shown in full below:

| | beta=2 | beta=5 | beta=10 | beta=20 |
|---|---|---|---|---|
| threshold=2, alpha=80 | **0.1181** | 0.1151 | 0.1102 | 0.0993 |
| threshold=2, alpha=40 | 0.1177 | 0.1149 | 0.1096 | 0.0983 |
| threshold=2, alpha=20 | 0.1145 | 0.1121 | 0.1066 | 0.0958 |
| threshold=2, alpha=10 | 0.1097 | 0.1072 | 0.1016 | 0.0917 |
| threshold=4, alpha=40 | **0.0669** | 0.0628 | 0.0570 | 0.0466 |
| threshold=4, alpha=80 | 0.0661 | 0.0617 | 0.0562 | 0.0465 |
| threshold=4, alpha=20 | 0.0659 | 0.0630 | 0.0557 | 0.0459 |
| threshold=4, alpha=10 | 0.0643 | 0.0615 | 0.0538 | 0.0449 |

Three things fall out of this:

- **The best negative-feedback config loses badly.** `threshold=2, alpha=80, beta=2` reaches 0.1181,
  still **23% below** the worst positives-only config (0.1473) and 23% below the best (0.1525).
  There is no overlap between the two groups at all.
- **More negative weight is monotonically worse.** Raising `beta` from 2 to 20 costs ~16% of ndcg at
  threshold=2 and ~30% at threshold=4, without a single exception across any alpha.
- **`alpha` barely matters.** Across its whole 10-to-80 range it moves ndcg by a few percent, and it
  peaks in the middle (40) rather than at either end — so the log1p confidence scaling is doing
  something, but it is a second-order knob compared to how playcount=1 is treated.

Retrained on all users at `threshold=1, alpha=40` (still `factors=64`): ndcg@10 = 0.1518,
map@10 = 0.1003. The small drop from the sweep's 0.1525 is expected — the sweep scored on 40% of
users, and the full set includes the longer tail of sparse users.

`threshold=1, alpha=40` is held fixed from here on, and depth is tuned on top of it below.

### Why there is no baseline comparison

The winning config marks nothing negative, so "drop the negatives" is a no-op on it — a
positives-only baseline is *the same model*. The run prints identical numbers for both for exactly
this reason. The meaningful comparison is not that row but the sweep itself, where positives-only
(threshold=1) beat every negative-feedback config outright.

### Caveat: the evaluation favours threshold=1

This result is **confounded with the choice of ground truth** and should not be read as "negative
feedback is useless".

Relevance is defined as *any* listen, and `playcount == 1` is 61% of all rows — so 61% of the test
set consists of exactly the interactions that `threshold=2` and `threshold=4` are trained to push
*down*. Those configs are being asked to rank highly the very items they were told to treat as
negatives. Some of the gap is structural, not a genuine quality difference.

What the sweep does establish is narrower but still useful: **if the goal is predicting whether a
user will play a track at all, treating a single play as negative evidence is counterproductive.**

Testing the original hypothesis properly needs a relevance definition that does not move with the
training threshold — e.g. holding out only `playcount >= 2` interactions as relevant, fixed across
all configs. That is the obvious next experiment.

## Embedding depth

`factors` is the width of the user and track vectors — how many numbers the model gets to describe
each one. Too few and unrelated tastes are forced onto the same axis; too many and the model starts
fitting noise instead of signal. Everything else is pinned at the winning confidence config so depth
is the only thing moving.

```bash
uv run python -m src.depth_sweep
```

Same protocol as above: the sweep runs on a 40% user sample at 15 iterations, then the winner is
retrained on all users at 20 iterations. Scores are on the same held-out 20%, so a deeper model that
were merely memorising would score *worse* here, not better.

| factors | ndcg@10 | map@10 | precision@10 | auc | gain vs. previous |
|---|---|---|---|---|---|
| 16 | 0.0879 | 0.0544 | 0.1083 | 0.5593 | — |
| 32 | 0.1172 | 0.0752 | 0.1416 | 0.5773 | +33% |
| 64 | 0.1525 | 0.1010 | 0.1819 | 0.5980 | +30% |
| 128 | 0.1887 | 0.1289 | 0.2242 | 0.6177 | +24% |
| 256 | **0.2189** | 0.1536 | 0.2596 | 0.6323 | +16% |

**Depth dominates every other knob in this project.** The whole 36-config confidence sweep moved
ndcg between 0.145 and 0.152. Depth alone moved it from 0.088 to 0.219.

**No overfitting knee was found.** Held-out accuracy was still climbing at 256 — the gains decay
(33 → 30 → 24 → 16%) but never turn over. 256 is therefore a *stopping point chosen on cost*, not an
optimum the data settled on. A deeper model may well score higher.

Two caveats. `regularization` is fixed at 0.05 at every depth, and higher-capacity models normally
want more of it — so 256 is plausibly a little under-regularised rather than genuinely optimal, and
the depth curve is mildly confounded with it. And cost grows fast: a 256-factor fit is roughly 16x a
64-factor one.

### Final model

`threshold=1, alpha=40, factors=256`, trained on all users, saved to `data/models/als.npz`:

| metric | value |
|---|---|
| ndcg@10 | **0.2163** |
| map@10 | 0.1508 |
| precision@10 | 0.2575 |
| auc | 0.6321 |

A 42% improvement in ndcg@10 over the 64-factor model.

## Running

```bash
uv sync

uv run python -m src.data.explore
uv run python -m src.main
uv run python -m src.depth_sweep
```

Or with Docker. `data/` is not baked into the image (see `.dockerignore`), so it is mounted at
run time:

```bash
docker build -t music-rec .

docker run --rm -v "$PWD/data:/app/data" music-rec
docker run --rm -v "$PWD/data:/app/data" music-rec python -m src.depth_sweep
docker run --rm -v "$PWD/data:/app/data" music-rec python -m src.data.explore
```

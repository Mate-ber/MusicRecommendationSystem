# Music Recommendation System

Music recommendations from implicit listening data, using ALS with **explicit positive and
negative feedback** rather than the usual positives-only formulation.

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
playcount >= 2  ->  +(1 + alpha * log1p(playcount))   positive
playcount == 1  ->  -beta                             explicit negative
```

The intuition for the negative: the user was exposed to the track and did not come back.
This matters because **playcount=1 is 61% of all rows**, so how it is treated largely defines
the model. The split is 3,377,421 positives against 5,299,062 negatives.

Two constraints worth knowing:

- **`beta` must be > 1.** The solver weights a preference-0 entry by `(confidence - 1)`, and
  unobserved pairs already carry confidence 1. At `beta = 1` an explicit negative would be
  exactly as informative as never having heard the track, so the signal would silently vanish.
- **`alpha` is pinned to 1.0 in `train_als`.** `implicit` rescales the matrix by its own alpha,
  which would scale the negatives along with the positives and undo the balance set in
  preprocessing. The tuning knobs live in `preprocess.confidence()` instead.

## Results

Both models are evaluated against the same held-out positives (20% per user, 675,943 entries)
and filtered against the same train matrix, so neither gets an unfair filtering advantage.

| metric | positives-only (baseline) | with negatives |
|---|---|---|
| precision@10 | **0.2221** | 0.2116 |
| map@10 | **0.1192** | 0.1159 |
| ndcg@10 | **0.1618** | 0.1564 |
| auc | **0.6100** | 0.6057 |

**The negatives currently hurt.** Plain positives-only ALS wins on all four metrics, with
precision@10 about 5% higher. At `beta=10` and `positive_threshold=2`, the negative signal costs
accuracy rather than adding to it.

## Running

```bash
uv sync
uv run python -m src.main
```

Or with Docker. `data/` is not baked into the image (see `.dockerignore`), so it is mounted at
run time:

```bash
docker build -t music-rec .

docker run --rm -v "$PWD/data:/app/data" music-rec

docker run --rm -v "$PWD/data:/app/data" music-rec python -m src.data.explore
```

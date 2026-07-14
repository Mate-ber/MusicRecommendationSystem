from pathlib import Path

import numpy as np
from implicit.evaluation import ranking_metrics_at_k

from src.data.load import load_user_history
from src.data.preprocess import build_matrix
from src.data.preprocess import confidence
from src.data.preprocess import encode_interactions
from src.data.preprocess import filter_interactions
from src.model.als import save_model
from src.model.als import train_als
from src.model.evaluate import sample_users
from src.model.evaluate import split_interactions

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "data" / "models"

K = 10

THRESHOLD = 1
ALPHA = 40
BETA = 0.0

FACTORS = [16, 32, 64, 128, 256]

SWEEP_USER_FRAC = 0.4
SWEEP_ITERATIONS = 15
FINAL_ITERATIONS = 20

METRICS = ["ndcg", "map", "precision", "auc"]


def report(scores):
    return "  ".join(f"{m}@{K}={scores[m]:.4f}" for m in METRICS)

def matrices(rows, is_train, is_test, user_codes, track_codes, playcounts, shape):
    train = rows & is_train
    test = rows & is_test
    return (
        build_matrix(
            user_codes[train],
            track_codes[train],
            confidence(playcounts[train], THRESHOLD, ALPHA, BETA),
            shape,
        ),
        build_matrix(
            user_codes[test],
            track_codes[test],
            np.ones(test.sum(), dtype=np.float32),
            shape,
        ),
    )

def main():
    history = load_user_history(DATA_DIR / "User Listening History.csv")
    history = filter_interactions(history)

    user_codes, track_codes, playcounts, user_ids, track_ids = encode_interactions(history)
    shape = (len(user_ids), len(track_ids))

    is_test = split_interactions(len(playcounts))
    is_train = ~is_test

    in_sweep = sample_users(user_codes, len(user_ids), SWEEP_USER_FRAC)
    sweep_train, sweep_test = matrices(in_sweep, is_train, is_test, user_codes, track_codes, playcounts, shape)

    print(f"threshold={THRESHOLD}, alpha={ALPHA}")
    print(f"{len(FACTORS)} depths on {SWEEP_USER_FRAC:.0%} of users\n")

    best_score = float("-inf")
    best_factors = None

    for factors in FACTORS:
        print(f"factors={factors}")

        model = train_als(sweep_train, factors=factors, iterations=SWEEP_ITERATIONS)
        scores = ranking_metrics_at_k(model, sweep_train, sweep_test, K=K)

        print(f"    {report(scores)}")

        if scores["ndcg"] > best_score:
            best_score = scores["ndcg"]
            best_factors = factors

    print(f"\nBest depth: factors={best_factors}")
    print("Retraining on all users\n")

    train, test = matrices(np.ones(len(playcounts), dtype=bool), is_train, is_test, user_codes, track_codes, playcounts, shape)

    model = train_als(train, factors=best_factors, iterations=FINAL_ITERATIONS)
    scores = ranking_metrics_at_k(model, train, test, K=K)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, user_ids, track_ids, MODEL_DIR / "als.npz")

    print("\nFinal model (full data)")
    print(f"threshold={THRESHOLD}, alpha={ALPHA}, factors={best_factors}")
    print(f"    {report(scores)}")


if __name__ == "__main__":
    main()

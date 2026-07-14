from pathlib import Path

import numpy as np
from implicit.evaluation import ranking_metrics_at_k

from src.data.load import load_user_history
from src.data.preprocess import build_matrix
from src.data.preprocess import confidence
from src.data.preprocess import confidence_grid
from src.data.preprocess import encode_interactions
from src.data.preprocess import filter_interactions
from src.model.als import save_model
from src.model.als import train_als
from src.model.evaluate import drop_negatives
from src.model.evaluate import sample_users
from src.model.evaluate import split_interactions

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "data" / "models"

K = 10

THRESHOLDS = [1, 2, 4]
ALPHAS = [10, 20, 40, 80]
BETAS = [2, 5, 10, 20]

SWEEP_USER_FRAC = 0.4
SWEEP_ITERATIONS = 15
FINAL_ITERATIONS = 20


def main():
    history = load_user_history(DATA_DIR / "User Listening History.csv")
    history = filter_interactions(history)

    user_codes, track_codes, playcounts, user_ids, track_ids = encode_interactions(history)
    shape = (len(user_ids), len(track_ids))

    is_test = split_interactions(len(playcounts))
    is_train = ~is_test

    test_matrix = build_matrix(
        user_codes[is_test],
        track_codes[is_test],
        np.ones(is_test.sum(), dtype=np.float32),
        shape,
    )

    in_sweep = sample_users(user_codes, len(user_ids), SWEEP_USER_FRAC)
    sweep_train = is_train & in_sweep
    sweep_test = is_test & in_sweep

    sweep_test_matrix = build_matrix(
        user_codes[sweep_test],
        track_codes[sweep_test],
        np.ones(sweep_test.sum(), dtype=np.float32),
        shape,
    )

    grid = confidence_grid(playcounts.min(), THRESHOLDS, ALPHAS, BETAS)
    print(f"{len(grid)} configs on {SWEEP_USER_FRAC:.0%} of users\n")

    best_score = float("-inf")
    best_params = None

    for i, params in enumerate(grid, start=1):
        threshold, alpha, beta = params["threshold"], params["alpha"], params["beta"]
        print(f"[{i}/{len(grid)}] threshold={threshold}, alpha={alpha}, beta={beta}")

        values = confidence(playcounts[sweep_train], threshold, alpha, beta or 0.0)
        train_matrix = build_matrix(
            user_codes[sweep_train], track_codes[sweep_train], values, shape
        )

        model = train_als(train_matrix, iterations=SWEEP_ITERATIONS)
        scores = ranking_metrics_at_k(model, train_matrix, sweep_test_matrix, K=K)

        print(f"    ndcg@{K}={scores['ndcg']:.4f}  map@{K}={scores['map']:.4f}")

        if scores["ndcg"] > best_score:
            best_score = scores["ndcg"]
            best_params = params

    threshold, alpha, beta = best_params["threshold"], best_params["alpha"], best_params["beta"]
    print(f"\nBest sweep config: threshold={threshold}, alpha={alpha}, beta={beta}")
    print("Retraining on all users\n")

    values = confidence(playcounts[is_train], threshold, alpha, beta or 0.0)
    train_matrix = build_matrix(user_codes[is_train], track_codes[is_train], values, shape)

    model = train_als(train_matrix, iterations=FINAL_ITERATIONS)
    scores = ranking_metrics_at_k(model, train_matrix, test_matrix, K=K)

    positives_only = drop_negatives(train_matrix)
    baseline = train_als(positives_only, iterations=FINAL_ITERATIONS)
    baseline_scores = ranking_metrics_at_k(baseline, positives_only, test_matrix, K=K)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, user_ids, track_ids, MODEL_DIR / "als.npz")

    print("\nFinal model (full data)")
    print(f"threshold={threshold}, alpha={alpha}, beta={beta}")
    print(f"ndcg@{K}={scores['ndcg']:.4f}  map@{K}={scores['map']:.4f}")
    print("\nBaseline (same config, negatives dropped)")
    print(f"ndcg@{K}={baseline_scores['ndcg']:.4f}  map@{K}={baseline_scores['map']:.4f}")


if __name__ == "__main__":
    main()

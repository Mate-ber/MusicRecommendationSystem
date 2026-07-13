from pathlib import Path

from src.data.load import load_user_history
from src.data.preprocess import build_interactions
from src.data.preprocess import filter_interactions
from src.model.als import save_model
from src.model.als import train_als
from implicit.evaluation import ranking_metrics_at_k

from src.model.evaluate import drop_negatives
from src.model.evaluate import split_positives

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "data" / "models"

K = 10

def main():
    history = load_user_history(DATA_DIR / "User Listening History.csv")
    history = filter_interactions(history)

    matrix, user_ids, track_ids = build_interactions(history)
    print(f"matrix: {matrix.shape[0]:,} users x {matrix.shape[1]:,} tracks")

    train, test = split_positives(matrix)
    print(f"train: {train.nnz:,} entries  test: {test.nnz:,} held-out positives\n")

    model = train_als(train)
    baseline = train_als(drop_negatives(train))

    scores = ranking_metrics_at_k(model, train, test, K=K)
    baseline_scores = ranking_metrics_at_k(baseline, train, test, K=K)

    print(f"\n{'metric':<12}{'positives-only':>16}{'with negatives':>16}")
    for name in scores:
        print(f"{name:<12}{baseline_scores[name]:>16.4f}{scores[name]:>16.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_model(model, user_ids, track_ids, MODEL_DIR / "als.npz")
    print(f"\nsaved model to {MODEL_DIR / 'als.npz'}")


if __name__ == "__main__":
    main()

from pathlib import Path

from src.data.load import load_user_history
from src.data.preprocess import build_interactions
from src.data.preprocess import filter_interactions
from src.model.als import save_model
from src.model.als import train_als

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "data" / "models"


def main():
    history = load_user_history(DATA_DIR / "User Listening History.csv")
    history = filter_interactions(history)

    matrix, user_ids, track_ids = build_interactions(history)
    print(f"matrix: {matrix.shape[0]:,} users x {matrix.shape[1]:,} tracks")
    print(f"positives: {(matrix.data > 0).sum():,}  negatives: {(matrix.data < 0).sum():,}")

    model = train_als(matrix)

    MODEL_DIR.mkdir(exist_ok=True)
    save_model(model, user_ids, track_ids, MODEL_DIR / "als.npz")
    print(f"saved model to {MODEL_DIR / 'als.npz'}")


if __name__ == "__main__":
    main()

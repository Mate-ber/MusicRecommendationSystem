from pathlib import Path

from src.data.load import load_music_info
from src.data.load import load_user_history
from src.data.preprocess import build_interactions
from src.data.preprocess import filter_interactions

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    music = load_music_info(DATA_DIR / "Music Info.csv")
    history = load_user_history(DATA_DIR / "User Listening History.csv")

    history = filter_interactions(history)
    matrix, user_ids, track_ids = build_interactions(history)

    positives = (matrix.data > 0).sum()
    negatives = (matrix.data < 0).sum()

    print(f"tracks in catalog: {len(music):,}")
    print(f"matrix: {matrix.shape[0]:,} users x {matrix.shape[1]:,} tracks")
    print(f"positives: {positives:,}")
    print(f"negatives: {negatives:,}")


if __name__ == "__main__":
    main()

from pathlib import Path

from src.data.load import load_music_info
from src.data.load import load_user_history

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():

    music = load_music_info(DATA_DIR / "Music Info.csv")
    history = load_user_history(DATA_DIR / "User Listening History.csv")

    print(music.head())
    print(history.head())


if __name__ == "__main__":
    main()

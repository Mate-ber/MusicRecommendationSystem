import pandas as pd


def load_music_info(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_user_history(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

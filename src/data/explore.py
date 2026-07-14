from pathlib import Path

import numpy as np

from src.data.load import load_music_info
from src.data.load import load_user_history

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"

PERCENTILES = [50, 75, 90, 95, 99, 99.9]


def describe(name, values):
    values = np.asarray(values, dtype=np.float64)

    print(f"\n{name}")
    print(f"  count   {len(values):>12,}")
    print(f"  sum     {values.sum():>12,.0f}")
    print(f"  mean    {values.mean():>12.2f}")
    print(f"  std     {values.std():>12.2f}")
    print(f"  min     {values.min():>12,.0f}")
    print(f"  max     {values.max():>12,.0f}")

    for p in PERCENTILES:
        label = f"p{p:g}"
        print(f"  {label:<6}  {np.percentile(values, p):>12,.2f}")


def playcount_histogram(playcounts, cutoff=10):
    print(f"\nplaycount frequency (first {cutoff})")
    total = len(playcounts)
    counts = playcounts.value_counts().sort_index()

    for value, count in counts.head(cutoff).items():
        print(f"  {value:>4}  {count:>12,}  {count / total:>7.2%}")

    tail = counts[counts.index > cutoff].sum()
    print(f"  >{cutoff:<3}  {tail:>12,}  {tail / total:>7.2%}")


def explore_history(history):
    users = history["user_id"].nunique()
    tracks = history["track_id"].nunique()
    density = len(history) / (users * tracks)

    print("interactions")
    print(f"  rows    {len(history):>12,}")
    print(f"  users   {users:>12,}")
    print(f"  tracks  {tracks:>12,}")
    print(f"  density {density:>12.6%}")

    describe("playcount per interaction", history["playcount"])
    playcount_histogram(history["playcount"])

    describe("interactions per user", history.groupby("user_id").size())
    describe("interactions per track", history.groupby("track_id").size())
    describe("total plays per user", history.groupby("user_id")["playcount"].sum())
    describe("total plays per track", history.groupby("track_id")["playcount"].sum())


def explore_catalog(info):
    print("\ncatalog")
    print(f"  tracks  {len(info):>12,}")
    for column in info.columns:
        missing = info[column].isna().sum()
        print(f"  {column:<14} {info[column].nunique():>8,} unique  {missing:>8,} missing")


def main():
    history = load_user_history(DATA_DIR / "User Listening History.csv")
    explore_history(history)

    info = load_music_info(DATA_DIR / "Music Info.csv")
    explore_catalog(info)


if __name__ == "__main__":
    main()

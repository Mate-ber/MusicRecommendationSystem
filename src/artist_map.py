from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from src.data.load import load_music_info
from src.data.load import load_user_history
from src.model.als import load_model
from src.model.artists import artist_interactions
from src.model.artists import artist_vectors
from src.model.artists import hubness
from src.model.artists import neighbors
from src.model.artists import similarities
from src.model.artists import track_vectors
from src.model.artists import whiten
from src.model.project import denoise
from src.model.project import label_baseline
from src.model.project import neighborhood_agreement
from src.model.project import project_pca
from src.model.project import project_tsne
from src.model.project import structure_preserved

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "data" / "models"
PLOT_DIR = ROOT / "data" / "plots"

MIN_INTERACTIONS = 1000
K = 10
WHITEN_COMPONENTS = 50

# The catalog's genre column is 57% empty, so a modal genre scraped from two or
# three tagged tracks is noise. Only trust a label with real support behind it.
MIN_GENRE_TRACKS = 4
MIN_GENRE_SHARE = 0.5

QUERIES = [
    "Taylor Swift",
    "Drake",
    "Metallica",
    "Radiohead",
    "Daft Punk",
    "Johnny Cash",
]

PAIRS = [
    ("Taylor Swift", "Drake"),
    ("Taylor Swift", "Katy Perry"),
    ("Taylor Swift", "Metallica"),
    ("Metallica", "Megadeth"),
]


def artist_genres(catalog, artists):
    known = catalog.dropna(subset=["genre"])
    counts = known.groupby(["artist", "genre"]).size().unstack(fill_value=0)
    counts = counts.reindex(artists, fill_value=0)

    tally = counts.to_numpy()
    labelled = tally.sum(axis=1)

    ranked = np.sort(tally, axis=1)
    winner = ranked[:, -1]
    runner_up = ranked[:, -2] if tally.shape[1] > 1 else np.zeros(len(tally))

    share = np.divide(winner, labelled, out=np.zeros(len(tally), dtype=float), where=labelled > 0)

    # A label only counts if enough tracks carry one, the plurality is not a tie,
    # and it is not a bare plurality scraped from a handful of tagged tracks.
    trusted = (labelled >= MIN_GENRE_TRACKS) & (winner > runner_up) & (share >= MIN_GENRE_SHARE)

    return np.where(trusted, counts.columns.to_numpy()[tally.argmax(axis=1)], "")


def report_neighbors(artists, cosine, corrected):
    for query in QUERIES:
        table = neighbors(query, artists, cosine, corrected, k=5)
        if table is None:
            continue

        print(f"\n  {query}")
        for row in table.itertuples():
            print(f"    cos {row.cosine:+.3f}   {row.artist}")


def report_pairs(artists, cosine):
    index = {artist: i for i, artist in enumerate(artists)}

    print("\npairwise cosine")
    for left, right in PAIRS:
        if left in index and right in index:
            print(f"  {left:<14} .. {right:<12} {cosine[index[left], index[right]]:+.3f}")


def plot(frame, path, dims):
    figure = px.scatter(
        frame,
        x="x",
        y="y",
        color="genre",
        size="interactions",
        size_max=22,
        hover_name="artist",
        hover_data={"genre": True, "interactions": ":,", "x": False, "y": False},
        title=f"Artists in ALS embedding space (t-SNE of whitened {dims}-dim factors, coloured by genre)",
        opacity=0.75,
    )
    figure.update_layout(template="plotly_dark", xaxis_visible=False, yaxis_visible=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(path)


def main():
    model, _, track_ids = load_model(MODEL_DIR / "als.npz")
    catalog = load_music_info(DATA_DIR / "Music Info.csv")
    history = load_user_history(DATA_DIR / "User Listening History.csv")

    vectors, alive = track_vectors(model.item_factors)
    print(f"tracks   {alive.sum():,} usable of {len(alive):,}")

    interactions = artist_interactions(history, catalog)
    artists, matrix = artist_vectors(
        vectors, alive, track_ids, catalog, interactions, MIN_INTERACTIONS
    )
    print(f"artists  {len(artists):,} with >= {MIN_INTERACTIONS:,} interactions")

    reduced, variance = denoise(matrix)
    print(f"\nPCA      {len(variance)} components hold {variance.sum():.1%} of variance")
    print(f"  top 2 of {matrix.shape[1]} hold {variance[:2].sum():.1%} — a few directions dominate")

    shaped = whiten(matrix, WHITEN_COMPONENTS)
    print(f"whitened {matrix.shape[1]} dims -> {shaped.shape[1]} equal-variance directions")

    cosine, corrected = similarities(shaped, k=K)
    off_diagonal = np.isfinite(cosine)
    print(f"\nmean pairwise cosine  {cosine[off_diagonal].mean():+.3f}")
    print(f"hubness               top artist is nearest to {hubness(corrected)} others")

    print("\nnearest neighbours (ranked by cosine)")
    report_neighbors(artists, cosine, corrected)
    report_pairs(artists, cosine)

    plane_pca, pca_variance = project_pca(matrix)
    print(f"\n2-d PCA alone holds {pca_variance:.1%} of variance — too little to plot on")

    plane = project_tsne(shaped)

    genres = artist_genres(catalog, artists)
    baseline = label_baseline(genres)

    tagged = (genres != "").sum()
    print(f"\ngenres   {tagged:,} of {len(artists):,} artists have a trustworthy label")

    print(f"\ngenre agreement @{K}  (ALS never saw a genre tag)")
    print(f"  random baseline     {baseline:.1%}")
    print(f"  {matrix.shape[1]:>3}-dim raw          {neighborhood_agreement(matrix, genres, k=K):.1%}")
    print(f"  {shaped.shape[1]:>3}-dim whitened     {neighborhood_agreement(shaped, genres, k=K):.1%}")
    print(f"  2-d t-SNE           {neighborhood_agreement(plane, genres, k=K):.1%}")
    print(f"  2-d PCA             {neighborhood_agreement(plane_pca, genres, k=K):.1%}")

    print(f"\ntrustworthiness @{K}  (1.0 = every 2-d neighbour is real, 0.5 = chance)")
    print(f"  t-SNE  vs whitened  {structure_preserved(shaped, plane, k=K):.3f}")
    print(f"  PCA    vs raw       {structure_preserved(reduced, plane_pca, k=K):.3f}")

    frame = pd.DataFrame(
        {
            "artist": artists,
            "x": plane[:, 0],
            "y": plane[:, 1],
            "genre": np.where(genres == "", "unknown", genres),
            "interactions": interactions.reindex(artists).to_numpy(),
        }
    )

    path = PLOT_DIR / "artist_map.html"
    plot(frame, path, shaped.shape[1])
    print(f"\nwrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

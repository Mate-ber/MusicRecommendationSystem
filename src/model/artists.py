import numpy as np
import pandas as pd


def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def track_vectors(item_factors):
    vectors = normalize(np.asarray(item_factors, dtype=np.float64))
    alive = np.linalg.norm(vectors, axis=1) > 0

    vectors[alive] -= vectors[alive].mean(axis=0)

    return normalize(vectors), alive


def artist_interactions(history, catalog):
    tagged = history.merge(catalog[["track_id", "artist"]], on="track_id", how="left")
    return tagged.groupby("artist").size()


def artist_vectors(vectors, alive, track_ids, catalog, interactions, min_interactions=1000):
    tracks = pd.DataFrame({"track_id": track_ids, "row": np.arange(len(track_ids))})
    tracks = tracks.merge(catalog[["track_id", "artist"]], on="track_id", how="left")

    keep = interactions[interactions >= min_interactions].index
    tracks = tracks[alive[tracks["row"].to_numpy()] & tracks["artist"].isin(keep)]

    artists = np.sort(tracks["artist"].unique())
    index = {artist: i for i, artist in enumerate(artists)}

    matrix = np.zeros((len(artists), vectors.shape[1]))
    for artist, group in tracks.groupby("artist"):
        matrix[index[artist]] = vectors[group["row"].to_numpy()].mean(axis=0)

    return artists, normalize(matrix)


def whiten(matrix, components=50):
    centered = matrix - matrix.mean(axis=0)
    left, values, _ = np.linalg.svd(centered, full_matrices=False)

    components = min(components, int((values > 1e-12).sum()))

    return normalize(left[:, :components])


def similarities(matrix, k=10):
    cosine = matrix @ matrix.T
    np.fill_diagonal(cosine, -np.inf)

    local = np.sort(cosine, axis=1)[:, -k:].mean(axis=1)
    corrected = 2 * cosine - local[:, None] - local[None, :]
    np.fill_diagonal(corrected, -np.inf)

    return cosine, corrected


def neighbors(artist, artists, cosine, corrected, k=10):
    match = np.flatnonzero(artists == artist)
    if not len(match):
        return None

    row = match[0]
    top = np.argsort(-cosine[row])[:k]

    return pd.DataFrame(
        {"artist": artists[top], "cosine": cosine[row, top], "score": corrected[row, top]}
    )


def hubness(corrected):
    top = np.argmax(corrected, axis=1)
    return np.bincount(top, minlength=len(corrected)).max()

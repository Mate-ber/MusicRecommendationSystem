import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def filter_interactions(history, min_user_plays=5, min_track_plays=5):
    track_counts = history["track_id"].value_counts()
    history = history[history["track_id"].isin(track_counts[track_counts >= min_track_plays].index)]

    user_counts = history["user_id"].value_counts()
    history = history[history["user_id"].isin(user_counts[user_counts >= min_user_plays].index)]

    return history


def confidence(playcount, positive_threshold=2, alpha=40.0, beta=10.0):
    playcount = playcount.astype(np.float32)
    positive = 1.0 + alpha * np.log1p(playcount)
    return np.where(playcount >= positive_threshold, positive, -beta).astype(np.float32)


def build_interactions(history, positive_threshold=2, alpha=40.0, beta=10.0):
    users = pd.Categorical(history["user_id"])
    tracks = pd.Categorical(history["track_id"])

    values = confidence(
        history["playcount"].to_numpy(), positive_threshold, alpha, beta
    )

    matrix = csr_matrix(
        (values, (users.codes, tracks.codes)),
        shape=(len(users.categories), len(tracks.categories)),
        dtype=np.float32,
    )

    return matrix, users.categories.to_numpy(), tracks.categories.to_numpy()

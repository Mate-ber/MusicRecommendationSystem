import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def filter_interactions(history, min_user_plays=5, min_track_plays=5):
    track_counts = history["track_id"].value_counts()
    history = history[history["track_id"].isin(track_counts[track_counts >= min_track_plays].index)]

    user_counts = history["user_id"].value_counts()
    history = history[history["user_id"].isin(user_counts[user_counts >= min_user_plays].index)]

    return history


def encode_interactions(history):
    users = pd.Categorical(history["user_id"])
    tracks = pd.Categorical(history["track_id"])

    return (
        users.codes.astype(np.int32),
        tracks.codes.astype(np.int32),
        history["playcount"].to_numpy().astype(np.float32),
        users.categories.to_numpy(),
        tracks.categories.to_numpy(),
    )


def confidence(playcount, positive_threshold=2, alpha=40.0, beta=10.0):
    playcount = playcount.astype(np.float32)
    positive = 1.0 + alpha * np.log1p(playcount)
    return np.where(playcount >= positive_threshold, positive, -beta).astype(np.float32)


def build_matrix(user_codes, track_codes, values, shape):
    return csr_matrix(
        (values, (user_codes, track_codes)),
        shape=shape,
        dtype=np.float32,
    )


def confidence_grid(min_playcount, thresholds, alphas, betas):
    grid = []
    for threshold in thresholds:
        has_negatives = threshold > min_playcount
        for alpha in alphas:
            for beta in betas if has_negatives else betas[:1]:
                grid.append(
                    {
                        "threshold": threshold,
                        "alpha": alpha,
                        "beta": beta if has_negatives else None,
                    }
                )
    return grid

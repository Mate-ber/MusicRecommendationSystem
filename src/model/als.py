import numpy as np
from implicit.als import AlternatingLeastSquares


def train_als(matrix, factors=64, regularization=0.05, iterations=20, random_state=42):
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        alpha=1.0,
        random_state=random_state,
    )
    model.fit(matrix)
    return model


def save_model(model, user_ids, track_ids, path):
    np.savez(
        path,
        user_factors=model.user_factors,
        item_factors=model.item_factors,
        user_ids=user_ids,
        track_ids=track_ids,
    )


def load_model(path):
    data = np.load(path, allow_pickle=True)

    model = AlternatingLeastSquares(factors=data["user_factors"].shape[1])
    model.user_factors = data["user_factors"]
    model.item_factors = data["item_factors"]

    return model, data["user_ids"], data["track_ids"]

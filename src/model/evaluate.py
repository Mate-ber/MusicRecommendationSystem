import numpy as np


def split_interactions(n_interactions, test_frac=0.2, random_state=42):
    rng = np.random.default_rng(random_state)
    return rng.random(n_interactions) < test_frac


def sample_users(user_codes, n_users, frac, random_state=42):
    if frac >= 1.0:
        return np.ones(len(user_codes), dtype=bool)

    rng = np.random.default_rng(random_state)
    keep = rng.random(n_users) < frac
    return keep[user_codes]


def drop_negatives(matrix):
    matrix = matrix.copy()
    matrix.data[matrix.data < 0] = 0
    matrix.eliminate_zeros()
    return matrix
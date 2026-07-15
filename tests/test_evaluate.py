import numpy as np
from scipy.sparse import csr_matrix

from src.model.evaluate import drop_negatives
from src.model.evaluate import sample_users
from src.model.evaluate import split_interactions


def test_split_interactions_is_deterministic():
    assert np.array_equal(split_interactions(10_000), split_interactions(10_000))


def test_split_interactions_respects_fraction():
    mask = split_interactions(100_000, test_frac=0.2)
    assert abs(mask.mean() - 0.2) < 0.01


def test_sample_users_full_fraction_keeps_all():
    user_codes = np.array([0, 1, 2, 1, 0])
    assert sample_users(user_codes, n_users=3, frac=1.0).all()


def test_sample_users_is_per_user():
    user_codes = np.array([0, 0, 1, 1, 2])
    keep = sample_users(user_codes, n_users=3, frac=0.5)

    # a user is kept or dropped as a whole, never split across interactions
    assert keep[0] == keep[1]
    assert keep[2] == keep[3]


def test_drop_negatives_zeroes_negative_entries():
    matrix = csr_matrix(np.array([[1.0, -5.0], [0.0, 2.0]]))
    dense = drop_negatives(matrix).toarray()

    assert (dense >= 0).all()
    assert dense[0, 0] == 1.0
    assert dense[1, 1] == 2.0
    assert dense[0, 1] == 0.0

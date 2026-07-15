import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import confidence
from src.data.preprocess import confidence_grid
from src.data.preprocess import encode_interactions
from src.data.preprocess import filter_interactions


def test_confidence_positive_and_negative():
    playcount = np.array([1, 2, 5], dtype=np.float32)
    values = confidence(playcount, positive_threshold=2, alpha=40.0, beta=10.0)

    assert values[0] == -10.0  # below threshold -> explicit negative
    assert values[1] == pytest.approx(1 + 40 * np.log1p(2), rel=1e-4)
    assert values[2] == pytest.approx(1 + 40 * np.log1p(5), rel=1e-4)


def test_confidence_threshold_one_has_no_negatives():
    playcount = np.array([1, 1, 3], dtype=np.float32)
    values = confidence(playcount, positive_threshold=1, alpha=20.0, beta=5.0)

    assert (values > 0).all()


def test_confidence_grid_omits_beta_without_negatives():
    grid = confidence_grid(min_playcount=1, thresholds=[1, 2], alphas=[10], betas=[2, 5])

    # threshold=1 with min playcount 1 -> nothing can be negative: one config, no beta
    t1 = [g for g in grid if g["threshold"] == 1]
    assert len(t1) == 1
    assert t1[0]["beta"] is None

    # threshold=2 -> negatives possible: one config per beta
    t2 = [g for g in grid if g["threshold"] == 2]
    assert len(t2) == 2
    assert {g["beta"] for g in t2} == {2, 5}


def test_filter_interactions_drops_sparse_users_and_tracks():
    history = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1", "u1", "u1", "u2"],
            "track_id": ["t1", "t2", "t3", "t4", "t5", "t1"],
            "playcount": [1, 1, 1, 1, 1, 1],
        }
    )

    # only t1 appears twice; with min_track_plays=2 the rest are dropped
    out = filter_interactions(history, min_user_plays=1, min_track_plays=2)

    assert set(out["track_id"]) == {"t1"}
    assert set(out["user_id"]) == {"u1", "u2"}


def test_encode_interactions_codes_and_categories():
    history = pd.DataFrame(
        {
            "user_id": ["b", "a", "b"],
            "track_id": ["y", "x", "x"],
            "playcount": [3, 1, 2],
        }
    )

    user_codes, track_codes, playcounts, user_ids, track_ids = encode_interactions(history)

    assert list(user_ids) == ["a", "b"]  # categories are sorted
    assert list(track_ids) == ["x", "y"]
    assert user_codes.tolist() == [1, 0, 1]
    assert track_codes.tolist() == [1, 0, 0]
    assert playcounts.dtype == np.float32

import numpy as np

from src.model.artists import normalize
from src.model.artists import similarities
from src.model.artists import track_vectors
from src.model.artists import whiten


def test_normalize_unit_rows_and_zero_safe():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))

    assert np.isclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(out[1], 0.0)  # zero row stays zero, no divide-by-zero


def test_track_vectors_normalized_with_alive_mask():
    factors = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    vectors, alive = track_vectors(factors)

    assert alive.tolist() == [True, True, False]
    assert np.allclose(np.linalg.norm(vectors[alive], axis=1), 1.0)


def test_whiten_shape_and_unit_rows():
    rng = np.random.default_rng(0)
    out = whiten(rng.normal(size=(40, 12)), components=5)

    assert out.shape == (40, 5)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_similarities_diagonal_masked_and_symmetric():
    rng = np.random.default_rng(1)
    matrix = normalize(rng.normal(size=(6, 4)))
    cosine, _ = similarities(matrix, k=2)

    assert np.isneginf(np.diag(cosine)).all()
    off = ~np.eye(6, dtype=bool)
    assert np.allclose(cosine[off], cosine.T[off])

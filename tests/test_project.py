import numpy as np

from src.model.project import label_baseline
from src.model.project import neighborhood_agreement


def test_label_baseline_matches_sum_of_squared_shares():
    labels = np.array(["a", "a", "b", ""])  # "" is unknown and excluded

    # known shares: a=2/3, b=1/3  ->  (2/3)^2 + (1/3)^2 = 5/9
    assert np.isclose(label_baseline(labels), 5 / 9)


def test_neighborhood_agreement_perfect_for_separated_clusters():
    cluster_a = np.zeros((6, 3))
    cluster_a[:, 0] = 1.0
    cluster_b = np.zeros((6, 3))
    cluster_b[:, 1] = 10.0

    matrix = np.vstack([cluster_a, cluster_b])
    labels = np.array(["a"] * 6 + ["b"] * 6)

    assert neighborhood_agreement(matrix, labels, k=3) == 1.0

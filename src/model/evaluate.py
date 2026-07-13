import numpy as np
from scipy.sparse import csr_matrix


def split_positives(matrix, test_frac=0.2, random_state=42):
    coo = matrix.tocoo()
    mask = (coo.data > 0) & (
        np.random.default_rng(random_state).random(coo.nnz) < test_frac
    )

    train = csr_matrix(
        (coo.data[~mask], (coo.row[~mask], coo.col[~mask])),
        shape=matrix.shape,
    )
    test = csr_matrix(
        (coo.data[mask], (coo.row[mask], coo.col[mask])),
        shape=matrix.shape,
    )

    return train, test


def drop_negatives(matrix):
    matrix = matrix.copy()
    matrix.data[matrix.data < 0] = 0
    matrix.eliminate_zeros()
    return matrix
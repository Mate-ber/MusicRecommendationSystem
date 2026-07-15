import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness

PCA_COMPONENTS = 50


def denoise(matrix, components=PCA_COMPONENTS, random_state=42):
    components = min(components, *matrix.shape)
    model = PCA(n_components=components, random_state=random_state)
    reduced = model.fit_transform(matrix)

    return reduced, model.explained_variance_ratio_


def project_pca(matrix, random_state=42):
    model = PCA(n_components=2, random_state=random_state)
    return model.fit_transform(matrix), model.explained_variance_ratio_.sum()


def project_tsne(matrix, perplexity=30, random_state=42):
    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        max_iter=1000,
        random_state=random_state,
    )
    return model.fit_transform(matrix)


def neighborhood_agreement(matrix, labels, k=10):
    known = labels != ""
    if known.sum() < k + 1:
        return float("nan")

    vectors = matrix[known]
    tags = labels[known]

    distance = ((vectors**2).sum(1)[:, None] + (vectors**2).sum(1)[None, :]) - 2 * vectors @ vectors.T
    np.fill_diagonal(distance, np.inf)

    top = np.argsort(distance, axis=1)[:, :k]
    return (tags[top] == tags[:, None]).mean()


def label_baseline(labels):
    known = labels[labels != ""]
    shares = np.bincount(np.unique(known, return_inverse=True)[1]) / len(known)
    return (shares**2).sum()


def structure_preserved(source, plane, k=10):
    return trustworthiness(source, plane, n_neighbors=k)

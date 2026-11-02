"""Geometric helpers: cosines, projections, and the difference-in-differences."""

import numpy as np


def cosine(u, v) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


def project_out(X, U):
    """Remove the span of the row(s) of U from every row of X."""
    U = np.atleast_2d(U)
    Q, _ = np.linalg.qr(U.T)  # orthonormal basis of span(U), shape [d, k]
    return X - (X @ Q) @ Q.T


def group_halves(groups, rng):
    """Random split of the unique groups into two halves; returns two masks."""
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    first = set(uniq[: len(uniq) // 2])
    m = np.array([g in first for g in groups])
    return m, ~m


def did_per_group(X, cell_of_row, groups, cells):
    """Per-group difference-in-differences over four cells.

    `cells` = (pp, pn, np_, nn): returns, for each group that has all four,
    X[pp] - X[pn] - X[np_] + X[nn]. With (pp, pn, np_, nn) =
    (lie+false, lie+true, truthful+false, truthful+true) this is the part of
    a lie not explained by adding the instruction effect and the falsehood
    effect. Returns (group_ids, deltas[n_groups, d]).
    """
    weights = dict(zip(cells, (1.0, -1.0, -1.0, 1.0)))
    uniq = np.unique(groups)
    index = {g: i for i, g in enumerate(uniq)}
    delta = np.zeros((len(uniq), X.shape[1]))
    count = np.zeros((len(uniq), len(cells)), dtype=int)
    for r, (g, c) in enumerate(zip(groups, cell_of_row)):
        if c in weights:
            delta[index[g]] += weights[c] * X[r]
            count[index[g], cells.index(c)] += 1
    full = (count == 1).all(1)
    return uniq[full], delta[full]

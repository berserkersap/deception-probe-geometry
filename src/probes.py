"""Linear probes as (unit direction, threshold) pairs in raw activation space.

Keeping every probe in the same raw space, with a unit-norm direction, makes
cosines between probes comparable and lets the threshold be refit on its own.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class Probe:
    w: np.ndarray  # unit direction
    t: float       # predict 1 when X @ w > t

    def score(self, X):
        return X @ self.w

    def predict(self, X):
        return (self.score(X) > self.t).astype(int)

    def with_threshold(self, t):
        return Probe(self.w, float(t))


def dom_probe(X, y) -> Probe:
    """Difference of class means; threshold halfway between the projected means."""
    mu1, mu0 = X[y == 1].mean(0), X[y == 0].mean(0)
    d = mu1 - mu0
    w = d / np.linalg.norm(d)
    return Probe(w, float(0.5 * (mu1 + mu0) @ w))


def logreg_probe(X, y, C=1e-2, seed=0) -> Probe:
    """L2 logistic regression on centred data scaled by one global scalar.

    A single scalar (not per-feature standardisation) keeps the geometry of the
    raw space, so the learned direction can be compared with other probes.
    """
    mu = X.mean(0)
    s = np.sqrt(((X - mu) ** 2).mean())
    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    clf.fit((X - mu) / s, y)
    w = clf.coef_[0] / s
    t = mu @ w - clf.intercept_[0]
    n = np.linalg.norm(w)
    if n < 1e-12:
        # Equal class means (e.g. after erasing their difference) make w = 0
        # the exact optimum: no linear signal. Predict the intercept's class.
        return Probe(np.zeros_like(w), -1.0 if clf.intercept_[0] > 0 else 1.0)
    return Probe(w / n, float(t / n))


def fit_probe(kind, X, y, **kw) -> Probe:
    return {"dom": dom_probe, "lr": logreg_probe}[kind](X, y, **kw)


def accuracy(probe: Probe, X, y) -> float:
    return float((probe.predict(X) == y).mean())


def auroc(scores, y) -> float:
    """Unflipped AUROC: below 0.5 means the direction points the wrong way."""
    return float(roc_auc_score(y, scores))


def best_threshold(scores, y) -> float:
    """Threshold that maximises accuracy of `scores > t` on (scores, y)."""
    order = np.argsort(scores, kind="stable")
    s, ys = scores[order], y[order]
    n = len(s)
    # Split k predicts 1 for items k..n-1. Only splits between distinct scores
    # (or at the ends) are realisable by a threshold.
    ones_at_or_above = np.concatenate([np.cumsum(ys[::-1])[::-1], [0]])
    zeros_below = np.concatenate([[0], np.cumsum(1 - ys)])
    correct = (ones_at_or_above + zeros_below).astype(float)
    valid = np.ones(n + 1, bool)
    valid[1:n] = s[1:] > s[:-1]
    correct[~valid] = -1
    best = np.flatnonzero(correct == correct.max())
    k = int(best[len(best) // 2])  # middle of a plateau of equally good splits
    if k == 0:
        return float(s[0] - 1.0)
    if k == n:
        return float(s[-1] + 1.0)
    return float(0.5 * (s[k - 1] + s[k]))

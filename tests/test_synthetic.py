"""The analysis must tell offset from rotation on data where we know the answer."""

import numpy as np
import pytest

from src.geometry import cosine, did_per_group, group_halves, project_out
from src.probes import (accuracy, auroc, best_threshold, dom_probe,
                        fit_probe, logreg_probe)

D = 512


def unit(rng):
    v = rng.normal(size=D)
    return v / np.linalg.norm(v)


def blobs(rng, direction, n=600, sep=4.0, offset=None):
    y = np.repeat([0, 1], n // 2)
    X = rng.normal(size=(n, D)) + np.outer(y - 0.5, direction) * sep
    if offset is not None:
        X = X + offset
    return X, y


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.mark.parametrize("kind", ["dom", "lr"])
def test_recovers_planted_direction(rng, kind):
    u = unit(rng)
    X, y = blobs(rng, u)
    p = fit_probe(kind, X, y)
    assert cosine(p.w, u) > 0.8
    assert accuracy(p, X, y) > 0.95


@pytest.mark.parametrize("kind", ["dom", "lr"])
def test_offset_is_fixed_by_threshold_refit(rng, kind):
    u = unit(rng)
    Xa, ya = blobs(rng, u)
    Xb, yb = blobs(rng, u, offset=10.0 * u)  # same direction, shifted
    p = fit_probe(kind, Xa, ya)
    assert accuracy(p, Xb, yb) < 0.6               # zero-shot fails
    assert auroc(p.score(Xb), yb) > 0.95           # but the direction is fine
    refit = p.with_threshold(best_threshold(p.score(Xb), yb))
    assert accuracy(refit, Xb, yb) > 0.9


@pytest.mark.parametrize("kind", ["dom", "lr"])
def test_rotation_is_not_fixed_by_threshold_refit(rng, kind):
    u = unit(rng)
    v = project_out(unit(rng)[None], u)[0]
    v /= np.linalg.norm(v)                          # v orthogonal to u
    Xa, ya = blobs(rng, u)
    Xb, yb = blobs(rng, v)
    p = fit_probe(kind, Xa, ya)
    assert abs(auroc(p.score(Xb), yb) - 0.5) < 0.1
    refit = p.with_threshold(best_threshold(p.score(Xb), yb))
    assert accuracy(refit, Xb, yb) < 0.65
    assert abs(cosine(p.w, dom_probe(Xb, yb).w)) < 0.3


def test_flipped_direction_gives_auroc_below_half(rng):
    u = unit(rng)
    Xa, ya = blobs(rng, u)
    Xb, yb = blobs(rng, -u)
    assert auroc(dom_probe(Xa, ya).score(Xb), yb) < 0.05


def test_project_out_removes_the_only_signal(rng):
    u = unit(rng)
    X, y = blobs(rng, u)
    Xp = project_out(X, u)
    assert np.allclose(Xp @ u, 0, atol=1e-8)
    tr, te = slice(0, None, 2), slice(1, None, 2)
    p = logreg_probe(Xp[tr], y[tr])
    assert accuracy(p, Xp[te], y[te]) < 0.65


def test_project_out_leaves_independent_signal(rng):
    u = unit(rng)
    v = project_out(unit(rng)[None], u)[0]
    v /= np.linalg.norm(v)
    y = np.repeat([0, 1], 300)
    X = rng.normal(size=(600, D)) + np.outer(y - 0.5, 4 * u + 4 * v)
    Xp = project_out(X, u)
    tr, te = slice(0, None, 2), slice(1, None, 2)
    assert accuracy(logreg_probe(Xp[tr], y[tr]), Xp[te], y[te]) > 0.9


def test_best_threshold():
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y = np.array([0, 0, 0, 1, 1, 1])
    assert 0.3 < best_threshold(s, y) < 0.7
    assert best_threshold(s, np.ones(6, int)) < 0.1
    assert best_threshold(s, np.zeros(6, int)) >= 0.9


def test_group_halves_are_disjoint_by_group(rng):
    groups = np.repeat(np.arange(50), 6)
    a, b = group_halves(groups, rng)
    assert not set(groups[a]) & set(groups[b])
    assert a.sum() == b.sum() == 150


def test_did_cancels_additive_effects(rng):
    """If a lie = instruction effect + falsehood effect, the DiD is ~0."""
    instr, false, extra = unit(rng), unit(rng), unit(rng)
    rows, cells, groups = [], [], []
    for g in range(200):
        base = rng.normal(size=D)
        for c, vec in (("LF", instr + false), ("LT", instr),
                       ("HF", false), ("HT", 0 * instr)):
            rows.append(base + 3 * vec + 0.1 * rng.normal(size=D))
            cells.append(c)
            groups.append(g)
    X = np.array(rows)
    ids, delta = did_per_group(X, cells, np.array(groups), ("LF", "LT", "HF", "HT"))
    assert len(ids) == 200
    # Main effects have norm 3; what is left is sampling noise (~0.3 here).
    assert np.linalg.norm(delta.mean(0)) < 0.15 * 3
    # Now add a lie-only component and check it is recovered.
    X2 = X + np.array([3 * extra if c == "LF" else 0 * extra for c in cells])
    _, delta2 = did_per_group(X2, cells, np.array(groups), ("LF", "LT", "HF", "HT"))
    assert cosine(delta2.mean(0), extra) > 0.95

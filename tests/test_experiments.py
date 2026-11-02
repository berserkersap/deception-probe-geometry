"""End-to-end check of the experiment functions on a planted world.

Truth is encoded along `v` in Distribution A and in the N/H contexts of B, but
along an orthogonal `r` in the lie (L) context; the lie instruction adds `i`.
E6 should then report: probe A reads N/H but not L, a threshold refit cannot
fix L, and a fresh in-context probe still can.
"""

import numpy as np
import pytest

from src.experiments import (Data, city_folds, context_veracity, grid_scores,
                             interaction, orthogonalization, rotation,
                             summarise_grid, transfer)

D = 64


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    v, r, i = np.linalg.qr(rng.normal(size=(D, 3)))[0].T * 4
    XA, yA, gA = [], [], []
    for c in range(120):
        for f in (0, 1):
            XA.append(rng.normal(size=D) + (f - 0.5) * v)
            yA.append(f)
            gA.append(f"a{c}")
    XB, cell, gB = [], [], []
    for c in range(150):
        for instr in "NHL":
            for f in (0, 1):
                truth = r if instr == "L" else v
                x = rng.normal(size=D) + (f - 0.5) * truth + (instr == "L") * i
                XB.append(x)
                cell.append(instr + ("F" if f else "T"))
                gB.append(f"b{c}")
    gA, gB = np.array(gA), np.array(gB)
    return Data(XA=np.array(XA), yA=np.array(yA), gA=gA, XB=np.array(XB),
                cellB=np.array(cell), gB=gB, foldA=city_folds(gA),
                foldB=city_folds(gB))


def _m(recs, metric):
    return next(r["mean"] for r in recs if r["metric"] == metric)


def test_context_veracity_detects_rotation(world):
    recs = context_veracity(world, "dom", n_reps=5)
    assert _m(recs, "A auroc (N)") > 0.95 and _m(recs, "A auroc (H)") > 0.95
    assert abs(_m(recs, "A auroc (L)") - 0.5) < 0.1
    assert _m(recs, "A threshold-refit acc (L)") < 0.65
    assert _m(recs, "fresh probe acc (L)") > 0.9
    assert _m(recs, "cos ceil L") > 0.8
    assert abs(_m(recs, "cos A~L")) < 0.2 and _m(recs, "cos A~H") > 0.8


def test_all_experiments_run_and_are_sane(world):
    for kind in ("dom", "lr"):
        recs = transfer(world, kind)
        assert _m(recs, "A->A auroc") > 0.95
        assert _m(recs, "B instruction acc") > 0.95
    rot = rotation(world, n_reps=5)
    assert _m(rot, "cos ceil A") > 0.8
    assert abs(_m(rot, "cos A~instruction")) < 0.2
    did = interaction(world, n_perm=200)
    assert _m(did, "did permutation p") < 0.05     # L rotates truth: non-additive
    e5 = orthogonalization(world, "lr")
    assert _m(e5, "instruction leakage acc | erase instr") < 0.6
    grid = summarise_grid(grid_scores(world, "dom"), n_boot=50)
    lie = grid[grid.probe == "lie"].set_index("cell")["mean"]
    assert abs(lie["HT"]) < 1e-9 and abs(lie["LF"] - 1) < 1e-9

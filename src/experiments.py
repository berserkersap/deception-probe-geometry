"""Experiments E0-E5 on cached activations. Each function handles one layer
and returns flat metric records: dicts with experiment, probe, metric, value.

B-grid notation: cell = instruction + answer, e.g. "LF" = lie instruction
with a false answer (the only real lie), "HT" = truthful instruction with a
true answer, "NF" = no instruction with a false answer.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import cosine, did_per_group, group_halves, project_out
from .probes import accuracy, auroc, best_threshold, dom_probe, fit_probe

CELL_ORDER = ["NT", "NF", "HT", "HF", "LT", "LF"]
DID_CELLS = ("LF", "LT", "HF", "HT")


@dataclass
class Data:
    """Activations for one layer plus labels."""
    XA: np.ndarray
    yA: np.ndarray        # 1 = false statement
    gA: np.ndarray        # city
    XB: np.ndarray
    cellB: np.ndarray     # e.g. "LF"
    gB: np.ndarray        # city
    foldA: np.ndarray     # CV fold per A row (grouped by city)
    foldB: np.ndarray     # CV fold per B row (grouped by city)

    @property
    def diag(self):       # HT vs LF: the lie-probe training set
        return np.isin(self.cellB, ["HT", "LF"])

    @property
    def y_lie(self):
        return (self.cellB == "LF").astype(int)

    @property
    def y_falseB(self):
        return np.char.endswith(self.cellB, "F").astype(int)

    @property
    def y_instr(self):     # 1 = lie instruction
        return np.char.startswith(self.cellB, "L").astype(int)


def city_folds(groups, k=5, seed=0):
    """Assign each row the fold of its group; groups are dealt out at random."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])


def _rec(out, experiment, probe, metric, values):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    out.append({"experiment": experiment, "probe": probe, "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0})


def _cv(X, y, folds, kind, mask=None, transform=None, **kw):
    """Grouped CV of a probe; returns per-fold accuracy and AUROC.

    `transform(Xtr, ytr_mask_train_rows) -> f` lets a fold estimate a
    projection on its own training rows only.
    """
    mask = np.ones(len(y), bool) if mask is None else mask
    accs, aucs = [], []
    for f in np.unique(folds):
        tr, te = mask & (folds != f), mask & (folds == f)
        Xtr, Xte = X[tr], X[te]
        if transform is not None:
            proj = transform(~(folds == f))
            Xtr, Xte = proj(Xtr), proj(Xte)
        p = fit_probe(kind, Xtr, y[tr], **kw)
        accs.append(accuracy(p, Xte, y[te]))
        aucs.append(auroc(p.score(Xte), y[te]))
    return np.array(accs), np.array(aucs)


# ---------------------------------------------------------------- E0 / E1

def country_baseline(country, y, folds):
    """How well does the *named country alone* predict the label?

    Country marginals are skewed (e.g. many true statements name China), so a
    probe could partly read country identity. This is the bar to beat.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    X = OneHotEncoder(handle_unknown="ignore").fit(country[:, None])
    accs = []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        clf = LogisticRegression(max_iter=2000).fit(
            X.transform(country[tr, None]), y[tr])
        accs.append(clf.score(X.transform(country[te, None]), y[te]))
    return np.array(accs)


def transfer(d: Data, kind: str):
    """E0 sanity decoders and E1 transfer / offset-vs-rotation."""
    out = []
    acc, auc = _cv(d.XA, d.yA, d.foldA, kind)
    _rec(out, "E1", kind, "A->A acc", acc)
    _rec(out, "E1", kind, "A->A auroc", auc)

    pA = fit_probe(kind, d.XA, d.yA)
    diag, y = d.diag, d.y_lie
    zs_acc, zs_auc, rf_acc, bb_acc, bb_auc = [], [], [], [], []
    for f in np.unique(d.foldB):
        tr, te = diag & (d.foldB != f), diag & (d.foldB == f)
        s_tr, s_te = pA.score(d.XB[tr]), pA.score(d.XB[te])
        zs_acc.append(accuracy(pA, d.XB[te], y[te]))
        zs_auc.append(auroc(s_te, y[te]))
        refit = pA.with_threshold(best_threshold(s_tr, y[tr]))
        rf_acc.append(accuracy(refit, d.XB[te], y[te]))
        pB = fit_probe(kind, d.XB[tr], y[tr])
        bb_acc.append(accuracy(pB, d.XB[te], y[te]))
        bb_auc.append(auroc(pB.score(d.XB[te]), y[te]))
    _rec(out, "E1", kind, "A->B zero-shot acc", zs_acc)
    _rec(out, "E1", kind, "A->B auroc", zs_auc)
    _rec(out, "E1", kind, "A->B threshold-refit acc", rf_acc)
    _rec(out, "E1", kind, "B->B acc", bb_acc)
    _rec(out, "E1", kind, "B->B auroc", bb_auc)

    # Does A's falsehood direction separate true/false answers in every B cell?
    yF = d.y_falseB
    _rec(out, "E1", kind, "A->B veracity auroc (all instr)",
         auroc(pA.score(d.XB), yF))
    for instr in "NHL":
        m = np.char.startswith(d.cellB, instr)
        _rec(out, "E1", kind, f"A->B veracity auroc ({instr})",
             auroc(pA.score(d.XB[m]), yF[m]))

    # E0: confounds are real if both are decodable inside B.
    acc, _ = _cv(d.XB, yF, d.foldB, kind)
    _rec(out, "E0", kind, "B veracity acc", acc)
    instr_rows = np.isin(d.cellB, ["HT", "HF", "LT", "LF"])
    acc, _ = _cv(d.XB, d.y_instr, d.foldB, kind, mask=instr_rows)
    _rec(out, "E0", kind, "B instruction acc", acc)
    return out


# ---------------------------------------------------------------- E2

def _dirs_B(d: Data, rows):
    """Diff-of-means directions inside B, estimated on `rows` only."""
    c = d.cellB
    def dom(pos, neg):
        m1, m0 = rows & np.isin(c, pos), rows & np.isin(c, neg)
        return d.XB[m1].mean(0) - d.XB[m0].mean(0)
    return {
        "lie": dom(["LF"], ["HT"]),
        "veracity": dom(["NF", "HF", "LF"], ["NT", "HT", "LT"]),
        "instruction": dom(["LT", "LF"], ["HT", "HF"]),
    }


def _dom_A(d: Data, rows):
    return d.XA[rows & (d.yA == 1)].mean(0) - d.XA[rows & (d.yA == 0)].mean(0)


def rotation(d: Data, n_reps=20, seed=0):
    """E2: cosines between directions vs. split-half ceilings at matched N.

    Every cosine compares directions estimated on disjoint halves, so shared
    rows (e.g. LF is in both the lie and instruction contrasts) cannot inflate it.
    """
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in ["ceil A", "ceil lie", "ceil veracity",
                            "ceil instruction", "A~lie", "A~veracity",
                            "A~instruction", "lie~instruction", "lie~veracity"]}
    for _ in range(n_reps):
        a1, a2 = group_halves(d.gA, rng)
        b1, b2 = group_halves(d.gB, rng)
        A1, A2 = _dom_A(d, a1), _dom_A(d, a2)
        B1, B2 = _dirs_B(d, b1), _dirs_B(d, b2)
        reps["ceil A"].append(cosine(A1, A2))
        for k in ("lie", "veracity", "instruction"):
            reps[f"ceil {k}"].append(cosine(B1[k], B2[k]))
            reps[f"A~{k}"].append(0.5 * (cosine(A1, B1[k]) + cosine(A2, B2[k])))
        reps["lie~instruction"].append(
            0.5 * (cosine(B1["lie"], B2["instruction"]) +
                   cosine(B2["lie"], B1["instruction"])))
        reps["lie~veracity"].append(
            0.5 * (cosine(B1["lie"], B2["veracity"]) +
                   cosine(B2["lie"], B1["veracity"])))
    out = []
    for k, v in reps.items():
        v = np.array(v)
        out.append({"experiment": "E2", "probe": "dom", "metric": f"cos {k}",
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "p05": float(np.quantile(v, 0.05)),
                    "p95": float(np.quantile(v, 0.95))})
    # Ceiling-normalised cross cosines (like a disattenuated correlation).
    ceil = {k: np.mean(reps[f"ceil {k}"]) for k in
            ("A", "lie", "veracity", "instruction")}
    for a, b in (("A", "lie"), ("A", "veracity"), ("A", "instruction"),
                 ("lie", "instruction"), ("lie", "veracity")):
        key = f"{a}~{b}"
        out.append({"experiment": "E2", "probe": "dom",
                    "metric": f"cos {key} / ceiling",
                    "mean": float(np.mean(reps[key]) /
                                  np.sqrt(max(ceil[a] * ceil[b], 1e-12))),
                    "std": 0.0})
    return out


# ---------------------------------------------------------------- E3

def grid_scores(d: Data, kind: str, project: str = "none"):
    """E3: out-of-fold score of each probe on every B cell, normalised per fold
    so that mean(HT) = 0 and mean(LF) = 1.

    probes: "A" (falsehood probe from Distribution A) and "lie" (trained on
    B's HT vs LF, optionally after erasing directions, see `_erase`).
    Returns a DataFrame with columns probe, city, cell, score.
    """
    pA = fit_probe(kind, d.XA, d.yA)
    rows = []
    for f in np.unique(d.foldB):
        tr, te = d.foldB != f, d.foldB == f
        proj = _erase(d, tr, project)
        XB = proj(d.XB)
        pL = fit_probe(kind, XB[tr & d.diag], d.y_lie[tr & d.diag])
        for name, s in (("A", pA.score(d.XB[te])), ("lie", pL.score(XB[te]))):
            cells = d.cellB[te]
            lo, hi = s[cells == "HT"].mean(), s[cells == "LF"].mean()
            for city, cell, v in zip(d.gB[te], cells, (s - lo) / (hi - lo)):
                rows.append((name, city, cell, v))
    return pd.DataFrame(rows, columns=["probe", "city", "cell", "score"])


def summarise_grid(df: pd.DataFrame, n_boot=1000, seed=0):
    """Cell means with city-bootstrap 95% CIs, plus the additivity residual
    LF - LT - HF + HT (0 if a lie is just instruction + falsehood)."""
    rng = np.random.default_rng(seed)
    out = []
    for probe, g in df.groupby("probe"):
        wide = g.pivot_table(index="city", columns="cell", values="score")
        wide = wide[CELL_ORDER]
        wide["residual"] = wide.LF - wide.LT - wide.HF + wide.HT
        vals = wide.values
        boots = np.array([vals[rng.integers(0, len(vals), len(vals))].mean(0)
                          for _ in range(n_boot)])
        for j, col in enumerate(wide.columns):
            out.append({"probe": probe, "cell": col,
                        "mean": float(vals[:, j].mean()),
                        "lo": float(np.quantile(boots[:, j], 0.025)),
                        "hi": float(np.quantile(boots[:, j], 0.975))})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- E4

def interaction(d: Data, n_perm=10000, seed=0):
    """E4: is a lie more than instruction effect + falsehood effect?

    Per city, delta = x[LF] - x[LT] - x[HF] + x[HT], i.e. the falsehood effect
    under the lie instruction minus the falsehood effect under the truthful one.
    The mean delta over training cities gives a direction; on held-out cities
    we measure the same contrast along it. Null (additivity): each city's
    contrast is symmetric around zero, so we flip its sign at random.
    """
    rng = np.random.default_rng(seed)
    contrasts, cos_A, cos_instr, cos_ver = [], [], [], []
    dA = _dom_A(d, np.ones(len(d.yA), bool))
    for f in np.unique(d.foldB):
        tr, te = d.foldB != f, d.foldB == f
        _, delta = did_per_group(d.XB[tr], d.cellB[tr], d.gB[tr], DID_CELLS)
        u = delta.mean(0)
        u /= np.linalg.norm(u)
        dirs = _dirs_B(d, tr)
        cos_A.append(cosine(u, dA))
        cos_instr.append(cosine(u, dirs["instruction"]))
        cos_ver.append(cosine(u, dirs["veracity"]))
        # held-out: per-city projected scores of the four cells
        Xte, cte, gte = d.XB[te] @ u, d.cellB[te], d.gB[te]
        frame = pd.DataFrame({"g": gte, "c": cte, "s": Xte})
        wide = frame.pivot_table(index="g", columns="c", values="s")[list(DID_CELLS)]
        contrasts.append(wide.values @ np.array([1.0, -1.0, -1.0, 1.0]))
    contrast = np.concatenate(contrasts)
    observed = contrast.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(contrast)))
    null = (signs * contrast).mean(1)
    boots = np.array([contrast[rng.integers(0, len(contrast), len(contrast))].mean()
                      for _ in range(1000)])
    out = [
        {"metric": "did contrast", "mean": float(observed),
         "lo": float(np.quantile(boots, 0.025)),
         "hi": float(np.quantile(boots, 0.975))},
        {"metric": "did effect size d_z", "mean": float(observed / contrast.std(ddof=1))},
        {"metric": "did sign-flip p",
         "mean": float((1 + (null >= observed).sum()) / (1 + n_perm))},
        {"metric": "cos did~A", "mean": float(np.mean(cos_A))},
        {"metric": "cos did~instruction", "mean": float(np.mean(cos_instr))},
        {"metric": "cos did~veracity", "mean": float(np.mean(cos_ver))},
    ]
    for r in out:
        r.update({"experiment": "E4", "probe": "dom"})
        r.setdefault("std", 0.0)
    return out


# ---------------------------------------------------------------- E5

def _erase(d: Data, train_rows, which: str):
    """Projection removing diff-of-means directions, each estimated on
    training rows only (A's falsehood direction uses all of A, which is disjoint
    from B anyway).

    which: "none" | "A" | "A+instr" | "veracityB+instr" | "instr"
    """
    if which == "none":
        return lambda X: X
    dirs = _dirs_B(d, train_rows)
    U = {
        "A": [_dom_A(d, np.ones(len(d.yA), bool))],
        "instr": [dirs["instruction"]],
        "A+instr": [_dom_A(d, np.ones(len(d.yA), bool)), dirs["instruction"]],
        "veracityB+instr": [dirs["veracity"], dirs["instruction"]],
    }[which]
    U = np.array(U)
    return lambda X: project_out(X, U)


ERASURES = ["none", "A", "instr", "A+instr", "veracityB+instr"]


def orthogonalization(d: Data, kind: str = "lr"):
    """E5: retrain the lie probe after erasing directions, plus leakage checks
    (can a fresh probe still read veracity / instruction after erasure?)."""
    out = []
    instr_rows = np.isin(d.cellB, ["HT", "HF", "LT", "LF"])
    for which in ERASURES:
        t = lambda keep, w=which: _erase(d, keep, w)
        acc, auc = _cv(d.XB, d.y_lie, d.foldB, kind, mask=d.diag, transform=t)
        _rec(out, "E5", kind, f"lie acc | erase {which}", acc)
        _rec(out, "E5", kind, f"lie auroc | erase {which}", auc)
        acc, _ = _cv(d.XB, d.y_falseB, d.foldB, kind, transform=t)
        _rec(out, "E5", kind, f"veracity leakage acc | erase {which}", acc)
        acc, _ = _cv(d.XB, d.y_instr, d.foldB, kind, mask=instr_rows, transform=t)
        _rec(out, "E5", kind, f"instruction leakage acc | erase {which}", acc)
    return out


# ---------------------------------------------------------------- E6

CONTEXTS = "NHL"


def _ver_dir(d: Data, rows, ctx):
    """True-vs-false diff-of-means inside one instruction context."""
    m = rows & np.char.startswith(d.cellB, ctx)
    f = np.char.endswith(d.cellB, "F")
    return d.XB[m & f].mean(0) - d.XB[m & ~f].mean(0)


def _unit(v):
    return v / np.linalg.norm(v)


def context_veracity(d: Data, kind: str, n_reps=20, seed=0):
    """E6: does the falsehood probe still read true vs false *inside* each
    instruction context, and if not, is that an offset or a direction change?

    Per context (N / H / L), all on held-out cities:
      - A-probe AUROC and A-probe accuracy after refitting only its threshold;
      - a fresh in-context probe ("is the information still there?");
      - "NH probe": trained on true-vs-false in the N and H contexts only, i.e.
        the honest-context truth direction, scored inside L;
      - "both-context probe": the sum of the unit H and L true-vs-false
        directions (training cities), scored in every context. N is never used
        to build it, and N contains no lies, so success there cannot come from
        reading lie-ness.
    Note that inside L, true-vs-false is LT vs LF and LF is the lie, so the
    L direction mixes falsehood with lie-ness.
    """
    out = []
    yF = d.y_falseB
    ctx_of = {c: np.char.startswith(d.cellB, c) for c in CONTEXTS}
    pA = fit_probe(kind, d.XA, d.yA)
    s = pA.score(d.XB)
    for ctx in CONTEXTS:
        m = ctx_of[ctx]
        refit = []
        for f in np.unique(d.foldB):
            tr, te = m & (d.foldB != f), m & (d.foldB == f)
            t = best_threshold(s[tr], yF[tr])
            refit.append(accuracy(pA.with_threshold(t), d.XB[te], yF[te]))
        fresh, _ = _cv(d.XB, yF, d.foldB, kind, mask=m)
        _rec(out, "E6", kind, f"A auroc ({ctx})", auroc(s[m], yF[m]))
        _rec(out, "E6", kind, f"A threshold-refit acc ({ctx})", refit)
        _rec(out, "E6", kind, f"fresh probe acc ({ctx})", fresh)

    nh, both = [], {c: [] for c in CONTEXTS}
    for f in np.unique(d.foldB):
        tr, te = d.foldB != f, d.foldB == f
        mNH = tr & ~ctx_of["L"]
        p = fit_probe(kind, d.XB[mNH], yF[mNH])
        mL = te & ctx_of["L"]
        nh.append(auroc(p.score(d.XB[mL]), yF[mL]))
        if kind == "dom":
            w = _unit(_unit(_ver_dir(d, tr, "H")) + _unit(_ver_dir(d, tr, "L")))
            for c in CONTEXTS:
                m = te & ctx_of[c]
                both[c].append(auroc(d.XB[m] @ w, yF[m]))
    _rec(out, "E6", kind, "NH probe auroc (L)", nh)
    if kind != "dom":
        return out
    for c in CONTEXTS:
        _rec(out, "E6", kind, f"both-context probe auroc ({c})", both[c])

    # Along A's direction: size of the true/false gap in L relative to H (raw units).
    gap = {c: s[ctx_of[c] & (yF == 1)].mean() - s[ctx_of[c] & (yF == 0)].mean()
           for c in CONTEXTS}
    _rec(out, "E6", kind, "A score gap ratio L/H", gap["L"] / gap["H"])

    # Full-data directions: decomposition and a robustness check against
    # Gemma's few massive-activation dimensions (per-dimension standardised).
    allr = np.ones(len(yF), bool)
    V = {c: _ver_dir(d, allr, c) for c in CONTEXTS}
    V["A"] = _dom_A(d, np.ones(len(d.yA), bool))
    shared = _unit(_unit(V["H"]) + _unit(V["L"]))
    flips = _unit(_unit(V["H"]) - _unit(V["L"]))
    _rec(out, "E6", kind, "cos A~(H+L)", cosine(V["A"], shared))
    _rec(out, "E6", kind, "cos A~(H-L)", cosine(V["A"], flips))
    sd = d.XB.std(0) + 1e-6
    for a, b in (("A", "N"), ("A", "H"), ("A", "L"), ("H", "L")):
        _rec(out, "E6", kind, f"cos {a}~{b} (standardised)",
             cosine(V[a] / sd, V[b] / sd))

    rng = np.random.default_rng(seed)
    pairs = [("A", c) for c in CONTEXTS] + [("N", "H"), ("N", "L"), ("H", "L")]
    reps = {f"ceil {c}": [] for c in ("A",) + tuple(CONTEXTS)}
    reps.update({f"{a}~{b}": [] for a, b in pairs})
    for _ in range(n_reps):
        a1, a2 = group_halves(d.gA, rng)
        b1, b2 = group_halves(d.gB, rng)
        h1 = {c: _ver_dir(d, b1, c) for c in CONTEXTS}
        h2 = {c: _ver_dir(d, b2, c) for c in CONTEXTS}
        h1["A"], h2["A"] = _dom_A(d, a1), _dom_A(d, a2)
        for c in ("A",) + tuple(CONTEXTS):
            reps[f"ceil {c}"].append(cosine(h1[c], h2[c]))
        for a, b in pairs:
            reps[f"{a}~{b}"].append(0.5 * (cosine(h1[a], h2[b]) + cosine(h2[a], h1[b])))
    for k, v in reps.items():
        v = np.array(v)
        out.append({"experiment": "E6", "probe": "dom", "metric": f"cos {k}",
                    "mean": float(v.mean()), "std": float(v.std(ddof=1)),
                    "p05": float(np.quantile(v, 0.05)),
                    "p95": float(np.quantile(v, 0.95))})
    ceil = {k[5:]: np.mean(v) for k, v in reps.items() if k.startswith("ceil ")}
    for a, b in pairs:
        _rec(out, "E6", "dom", f"cos {a}~{b} / ceiling",
             np.mean(reps[f"{a}~{b}"]) / np.sqrt(max(ceil[a] * ceil[b], 1e-12)))
    return out

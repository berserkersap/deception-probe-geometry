"""Static figures for the README, built from the CSVs run_experiments.py writes."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SEQ = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _series(metrics, exp, probe, metric):
    s = metrics[(metrics.experiment == exp) & (metrics.probe == probe) &
                (metrics.metric == metric)].sort_values("layer")
    return s.layer.values, s["mean"].values, s["std"].fillna(0).values


def _line(ax, x, y, color, label, std=None, marker="o"):
    ax.plot(x, y, color=color, label=label, marker=marker, markersize=4)
    if std is not None and np.any(std > 0):
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.15, lw=0)


def _mark_layer(ax, layer):
    ax.axvline(layer, color=INK_2, lw=1, ls=":", zorder=0)


def transfer(metrics, layer, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    panels = [
        ("Threshold-free: AUROC", [
            ("A->A auroc", "A→A (in-distribution)"),
            ("A->B auroc", "A→B (falsehood probe on lies)"),
            ("B->B auroc", "B→B (lie probe, ceiling)")]),
        ("Accuracy: zero-shot vs. threshold refit", [
            ("A->B zero-shot acc", "A→B zero-shot (A's threshold)"),
            ("A->B threshold-refit acc", "A→B threshold refit on B"),
            ("B->B acc", "B→B (lie probe, ceiling)")]),
    ]
    for ax, (title, rows) in zip(axes, panels):
        for i, (metric, label) in enumerate(rows):
            x, y, s = _series(metrics, "E1", "dom", metric)
            _line(ax, x, y, SERIES[i], label, s)
        _mark_layer(ax, layer)
        ax.axhline(0.5, color=INK_2, lw=1, ls="--", zorder=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("layer")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15))
    axes[0].set_ylabel("held-out score (5-fold, grouped by city)")
    axes[0].set_ylim(0.3, 1.02)
    fig.suptitle("Falsehood probe (Distribution A) evaluated on instructed lies "
                 "(B: truthful+true vs lie+false), diff-of-means",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def cosines(metrics, layer, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    e2 = metrics[metrics.experiment == "E2"]

    def band(metric, color, label):
        s = e2[e2.metric == metric].sort_values("layer")
        ax.fill_between(s.layer, s.p05, s.p95, color=color, alpha=0.18, lw=0,
                        label=label)

    band("cos ceil A", "#9a9890", "split-half ceiling, A (5–95%); every\n"
         "direction's ceiling is ≥0.95 from layer 4")
    for i, (metric, label) in enumerate([
            ("cos A~lie", "falsehood(A) vs lie(B)"),
            ("cos A~veracity", "falsehood(A) vs veracity inside B"),
            ("cos A~instruction", "falsehood(A) vs instruction(B)"),
            ("cos lie~instruction", "lie(B) vs instruction(B)")]):
        s = e2[e2.metric == metric].sort_values("layer")
        _line(ax, s.layer.values, s["mean"].values, SERIES[[0, 2, 3, 4][i]],
              label, s["std"].values)
    _mark_layer(ax, layer)
    ax.axhline(0, color=INK_2, lw=1)
    ax.set_ylim(-0.35, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine (diff-of-means directions)")
    ax.set_title("Direction similarity vs. the split-half noise ceiling "
                 "(diff-of-means)", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


GRID_ROWS = [("N", "no instruction"), ("H", "truthful instr."), ("L", "lie instr.")]


def grid(detail, layer, path, panels):
    """3x2 heatmaps of normalised probe scores (HT = 0, LF = 1)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.3))
    for ax, (kind, erase, probe, title) in zip(np.atleast_1d(axes), panels):
        g = detail[(detail.kind == kind) & (detail.erase == erase) &
                   (detail.probe == probe)].set_index("cell")
        M = np.array([[g.loc[r + c, "mean"] for c in "TF"] for r, _ in GRID_ROWS])
        ax.imshow(M, cmap=SEQ, vmin=-0.1, vmax=1.3, aspect="auto")
        for i, (r, _) in enumerate(GRID_ROWS):
            for j, c in enumerate("TF"):
                v, lo, hi = g.loc[r + c, ["mean", "lo", "hi"]]
                ax.text(j, i, f"{v:.2f}\n[{lo:.2f}, {hi:.2f}]", ha="center",
                        va="center", fontsize=9,
                        color="white" if v > 0.75 else INK)
        ax.set_xticks([0, 1], ["true answer", "false answer"])
        ax.set_yticks(range(3), [lab for _, lab in GRID_ROWS])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        res = g.loc["residual"]
        ax.set_title(f"{title}\nadditivity residual {res['mean']:+.2f} "
                     f"[{res.lo:+.2f}, {res.hi:+.2f}]", fontsize=10, loc="left")
    fig.suptitle(f"Mean probe score per cell, layer {layer} "
                 "(0 = truthful+true, 1 = lie+false; 95% city-bootstrap CI)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grid_by_layer(grid_layers, layer, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    cells = [("NF", "no instr. + false"), ("HF", "truthful instr. + false"),
             ("LT", "lie instr. + true"), ("residual", "additivity residual")]
    for ax, probe, title in zip(axes, ("A", "lie"),
                                ("Falsehood probe (A), diff-of-means",
                                 "Lie probe (B: HT vs LF), diff-of-means")):
        g = grid_layers[grid_layers.probe == probe]
        for i, (cell, label) in enumerate(cells):
            s = g[g.cell == cell].sort_values("layer")
            ax.plot(s.layer, s["mean"], color=SERIES[i + 1], marker="o",
                    markersize=4, label=label, ls="--" if cell == "residual" else "-")
            ax.fill_between(s.layer, s.lo, s.hi, color=SERIES[i + 1], alpha=0.15, lw=0)
        ax.axhline(0, color=INK_2, lw=1)
        ax.axhline(1, color=INK_2, lw=1, ls=":")
        _mark_layer(ax, layer)
        ax.set_title(title, loc="left")
        ax.set_xlabel("layer")
    axes[0].set_ylabel("score (0 = truthful+true, 1 = lie+false)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def orthogonalization(metrics, layer, path):
    e5 = metrics[(metrics.experiment == "E5") & (metrics.layer == layer)].set_index("metric")
    erasures = [("none", "nothing"), ("A", "falsehood (A)"),
                ("instr", "instruction (B)"), ("A+instr", "falsehood (A)\n+ instruction"),
                ("veracityB+instr", "veracity (B)\n+ instruction")]
    groups = [("lie acc", "lie probe (HT vs LF)"),
              ("veracity leakage acc", "fresh veracity probe"),
              ("instruction leakage acc", "fresh instruction probe")]
    x = np.arange(len(erasures))
    width = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (key, label) in enumerate(groups):
        vals = [e5.loc[f"{key} | erase {e}", "mean"] for e, _ in erasures]
        errs = [e5.loc[f"{key} | erase {e}", "std"] for e, _ in erasures]
        ax.bar(x + (i - 1) * width, vals, width - 0.03, yerr=errs, color=SERIES[i],
               label=label, error_kw={"ecolor": INK_2, "lw": 1})
    ax.axhline(0.5, color=INK_2, lw=1, ls="--")
    ax.set_xticks(x, [lab for _, lab in erasures])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("held-out accuracy (logistic regression)")
    ax.set_xlabel("directions projected out before retraining")
    ax.set_title(f"Erasing directions, then retraining (layer {layer})", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def context(metrics, layer, path):
    """E6: true vs false inside each instruction context."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    ax = axes[0]
    rows = [
        ("E6", "lr", "A auroc (N)", "falsehood probe (LR): no instruction", "-", 0),
        ("E6", "lr", "A auroc (H)", "falsehood probe (LR): truthful instr.", "-", 1),
        ("E6", "lr", "A auroc (L)", "falsehood probe (LR): lie instr.", "-", 2),
        ("E6", "dom", "A auroc (L)", "falsehood probe (DoM): lie instr.", "--", 2),
        ("E6", "dom", "NH probe auroc (L)",
         "honest-context probe (N+H, DoM): lie instr.", "-", 4),
        ("E6", "dom", "both-context probe auroc (N)",
         "both-context direction (H+L, DoM): no instr. (unseen)", "-", 5),
    ]
    for exp, kind, metric, label, style, ci in rows:
        x, y, sd = _series(metrics, exp, kind, metric)
        ax.plot(x, y, color=SERIES[ci], ls=style, marker="o", markersize=3.5,
                label=label)
        if np.any(sd > 0):
            ax.fill_between(x, y - sd, y + sd, color=SERIES[ci], alpha=0.15, lw=0)
    ax.axhline(0.5, color=INK_2, lw=1, ls="--", zorder=0)
    _mark_layer(ax, layer)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("AUROC, true vs false answer (held-out cities)")
    ax.set_title("Reading true vs false inside each context", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), fontsize=8.5)

    ax = axes[1]
    e6 = metrics[(metrics.experiment == "E6") & (metrics.probe == "dom")]
    c = e6[e6.metric == "cos ceil L"].sort_values("layer")
    ax.fill_between(c.layer, c.p05, c.p95, color="#9a9890", alpha=0.25, lw=0,
                    label="split-half ceiling, lie context (5–95%)")
    for i, (metric, label) in enumerate([
            ("cos A~N", "falsehood(A) vs no-instr. direction"),
            ("cos A~H", "falsehood(A) vs truthful-instr. direction"),
            ("cos A~L", "falsehood(A) vs lie-instr. direction"),
            ("cos H~L", "truthful-instr. vs lie-instr. direction")]):
        s_ = e6[e6.metric == metric].sort_values("layer")
        _line(ax, s_.layer.values, s_["mean"].values, SERIES[[0, 1, 2, 4][i]],
              label, s_["std"].values)
    ax.axhline(0, color=INK_2, lw=1)
    _mark_layer(ax, layer)
    ax.set_ylim(-0.35, 1.02)
    ax.set_xlabel("layer")
    ax.set_ylabel("cosine of true-vs-false directions")
    ax.set_title("Direction of true vs false (diff-of-means, disjoint halves)",
                 loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_all(results: Path, layer: int):
    plots = results / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(results / "metrics.csv")
    detail = pd.read_csv(results / "grid_scores.csv")
    by_layer = pd.read_csv(results / "grid_by_layer.csv")
    transfer(metrics, layer, plots / "transfer_and_recovery.png")
    cosines(metrics, layer, plots / "cosine_vs_ceiling.png")
    grid(detail, layer, plots / "grid_scores.png", [
        ("dom", "none", "A", "Falsehood probe (A)"),
        ("dom", "none", "lie", "Lie probe (B: HT vs LF)"),
        ("lr", "veracityB+instr", "lie",
         "Lie probe after erasing\nveracity + instruction (LR)"),
    ])
    grid_by_layer(by_layer, layer, plots / "grid_by_layer.png")
    orthogonalization(metrics, layer, plots / "orthogonalization.png")
    context(metrics, layer, plots / "context_veracity.png")

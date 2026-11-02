"""Run E0-E5 on cached activations (see extract_activations.py) and write
results/metrics.csv, results/grid_*.csv and results/summary.json.

Analysis is CPU-only and takes a few minutes for all layers.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import (CELL_ORDER, ERASURES, Data, city_folds,
                             context_veracity, country_baseline, grid_scores,
                             interaction,
                             orthogonalization, rotation, summarise_grid,
                             transfer)
from src.plots import make_all


def load(acts_dir: Path):
    A = np.load(acts_dir / "A_acts.npy", mmap_mode="r")
    B = np.load(acts_dir / "B_acts.npy", mmap_mode="r")
    mA = pd.read_csv(acts_dir / "A_meta.csv")
    mB = pd.read_csv(acts_dir / "B_meta.csv")
    mB["cell"] = mB.instr + np.where(mB.answer_true == 1, "T", "F")
    return A, B, mA, mB


def layer_data(A, B, mA, mB, layer, foldA, foldB) -> Data:
    return Data(XA=np.asarray(A[layer], dtype=np.float32),
                yA=mA.is_false.to_numpy(int), gA=np.array(mA.city.tolist()),
                XB=np.asarray(B[layer], dtype=np.float32),
                cellB=np.array(mB.cell.tolist()), gB=np.array(mB.city.tolist()),
                foldA=foldA, foldB=foldB)


def _get(recs, exp, probe, metric):
    return next(r["mean"] for r in recs if r["experiment"] == exp
                and r["probe"] == probe and r["metric"] == metric)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", default="activations")
    ap.add_argument("--out", default="results")
    # Default: every block output; the last hidden state (post final norm) is excluded.
    ap.add_argument("--layers", type=int, nargs="*", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B, mA, mB = load(Path(args.acts))
    layers = args.layers or list(range(1, A.shape[0] - 1))
    foldA = city_folds(np.array(mA.city.tolist()), args.folds, args.seed)
    foldB = city_folds(np.array(mB.city.tolist()), args.folds, args.seed)

    records, grid_rows = [], []
    for layer in layers:
        d = layer_data(A, B, mA, mB, layer, foldA, foldB)
        recs = transfer(d, "dom") + transfer(d, "lr") + rotation(d, seed=args.seed)
        recs += interaction(d, seed=args.seed) + orthogonalization(d, "lr")
        recs += context_veracity(d, "dom", seed=args.seed) + context_veracity(d, "lr")
        for r in recs:
            r["layer"] = layer
        records += recs
        # Cheap per-layer view of the grid for the diff-of-means probes.
        g = summarise_grid(grid_scores(d, "dom"), n_boot=200, seed=args.seed)
        g["layer"] = layer
        grid_rows.append(g)
        e1 = {r["metric"]: r["mean"] for r in recs
              if r["experiment"] == "E1" and r["probe"] == "dom"}
        print(f"layer {layer:2d}  A->A {e1['A->A auroc']:.3f}  "
              f"A->B auroc {e1['A->B auroc']:.3f}  "
              f"zero-shot {e1['A->B zero-shot acc']:.3f}  "
              f"refit {e1['A->B threshold-refit acc']:.3f}  "
              f"B->B {e1['B->B acc']:.3f}  "
              f"A auroc in lie ctx {_get(recs, 'E6', 'dom', 'A auroc (L)'):.3f}")

    metrics = pd.DataFrame(records)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.concat(grid_rows).to_csv(out / "grid_by_layer.csv", index=False)

    # Layer for the detailed grid: chosen on Distribution A only (best A->A
    # diff-of-means CV accuracy, earliest on ties), so B never steers it.
    a_acc = metrics.query("experiment == 'E1' and probe == 'dom' and "
                          "metric == 'A->A acc'").set_index("layer")["mean"]
    best = int(a_acc[a_acc >= a_acc.max() - 1e-9].index.min())
    d = layer_data(A, B, mA, mB, best, foldA, foldB)
    detail = []
    for kind in ("dom", "lr"):
        for which in ERASURES:
            g = summarise_grid(grid_scores(d, kind, project=which), seed=args.seed)
            g["kind"], g["erase"] = kind, which
            detail.append(g)
    detail = pd.concat(detail)
    detail.to_csv(out / "grid_scores.csv", index=False)

    # Country-identity baseline (label predicted from the named country alone).
    diag = mB.cell.isin(["HT", "LF"]).values
    base = {
        "A falsehood": country_baseline(np.array(mA.country.tolist()),
                                        mA.is_false.to_numpy(int), foldA),
        "B lie (HT vs LF)": country_baseline(np.array(mB.country.tolist())[diag],
                                             (mB.cell.to_numpy(str)[diag] == "LF").astype(int),
                                             foldB[diag]),
    }

    def at(layer, exp, probe, metric):
        row = metrics[(metrics.layer == layer) & (metrics.experiment == exp) &
                      (metrics.probe == probe) & (metrics.metric == metric)]
        return float(row["mean"].iloc[0])

    summary = {
        "layers": layers, "selected_layer": best,
        "selection_rule": "argmax A->A diff-of-means CV accuracy (Distribution A only)",
        "n_A_rows": len(mA), "n_B_rows": len(mB),
        "country_baseline_acc": {k: float(v.mean()) for k, v in base.items()},
        "at_selected_layer": {
            f"{exp}/{probe}/{metric}": at(best, exp, probe, metric)
            for exp, probe, metric in metrics[metrics.layer == best][
                ["experiment", "probe", "metric"]].itertuples(index=False)
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    make_all(out, best)
    print(f"selected layer {best}; wrote {out}/metrics.csv, grid_*.csv, "
          f"summary.json, plots/")
    print(detail.query("kind == 'dom' and erase == 'none'")
          .pivot(index="probe", columns="cell", values="mean")
          [CELL_ORDER + ["residual"]].round(2).to_string())


if __name__ == "__main__":
    main()

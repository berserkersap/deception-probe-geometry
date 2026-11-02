"""Redraw results/plots/*.png from existing results (no recomputation)."""

import json
from pathlib import Path

from src.plots import make_all

if __name__ == "__main__":
    results = Path("results")
    layer = json.loads((results / "summary.json").read_text())["selected_layer"]
    make_all(results, layer)
    print(f"wrote {results / 'plots'}")

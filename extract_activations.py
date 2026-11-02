"""Build the datasets, filter to facts the model knows, and cache activations.

Writes to --out (default activations/):
  knowledge.csv          model's free answer per city and whether it was correct
  A_meta.csv, A_acts.npy Distribution A rows and activations [layers, N, d]
  B_meta.csv, B_acts.npy Distribution B grid rows and activations
  manifest.json          model, dtype, versions, pools, layer convention
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import transformers

from src.data import (build_a_rows, build_b_rows, download_cities,
                      knowledge_prompt, load_city_pairs, split_pools,
                      answer_matches)
from src.extract import (final_token_activations, final_token_strings,
                         generate_answers, load_model)


def main():
    ap = argparse.ArgumentParser()
    # Ungated mirror of google/gemma-2-2b-it; pass the Google ID if you have access.
    ap.add_argument("--model", default="unsloth/gemma-2-2b-it")
    ap.add_argument("--out", default="activations")
    ap.add_argument("--frac-a", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = load_city_pairs(download_cities())
    model, tok = load_model(args.model)

    # 1. Knowledge filter: keep cities the model places correctly unprompted.
    answers = generate_answers(model, tok, [knowledge_prompt(c) for c in pairs.city])
    pairs["model_answer"] = answers
    pairs["known"] = [answer_matches(a, c)
                      for a, c in zip(answers, pairs.correct_country)]
    pairs.to_csv(out / "knowledge.csv", index=False)
    known = pairs[pairs.known].reset_index(drop=True)
    print(f"knowledge filter: kept {len(known)}/{len(pairs)} cities")

    # 2. Disjoint pools, then rows.
    pool_a, pool_b = split_pools(known.city, frac_a=args.frac_a, seed=args.seed)
    a_rows = build_a_rows(known[known.city.isin(pool_a)])
    b_rows = build_b_rows(known[known.city.isin(pool_b)])

    # 3. Activations at the final token, which must be the closing period.
    manifest = {
        "model": args.model, "dtype": "bfloat16", "storage_dtype": "float16",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "layer_convention": "index i = output of decoder block i-1; "
                            "0 = embeddings; last index has the final norm applied",
        "n_cities_total": len(pairs), "n_cities_known": len(known),
        "seed": args.seed, "frac_a": args.frac_a,
        "pool_a": pool_a, "pool_b": pool_b,
    }
    for name, rows in (("A", a_rows), ("B", b_rows)):
        finals = set(final_token_strings(tok, rows.text.tolist()))
        if finals != {"."}:
            raise RuntimeError(f"{name}: unexpected final tokens {finals}")
        acts = final_token_activations(model, tok, rows.text.tolist(),
                                       batch_size=args.batch_size)
        max_abs = float(np.abs(acts).max())
        if max_abs > 6e4:
            raise RuntimeError(f"{name}: activations too large for float16")
        np.save(out / f"{name}_acts.npy", acts.astype(np.float16))
        rows.drop(columns="text").to_csv(out / f"{name}_meta.csv", index=False)
        manifest[f"{name}_shape"] = list(acts.shape)
        manifest[f"{name}_max_abs"] = max_abs
        print(f"{name}: {acts.shape}, max |act| {max_abs:.0f}")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

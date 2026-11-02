"""Build the datasets, filter to facts the model knows, and cache activations.

Committed (small, so the exact city pools are reproducible):
  data/knowledge.csv     model's free answer per city and whether it was correct
  data/pools.json        A/B city pools and filter stats; reused on later runs
  results/extraction_manifest.json   model/data revisions, prompts, shapes
Cached (large, gitignored) under --out (default activations/):
  A_meta.csv, A_acts.npy Distribution A rows and activations [layers, N, d]
  B_meta.csv, B_acts.npy Distribution B grid rows and activations
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers

from src.data import (CITIES_URL, INSTRUCTIONS, answer_matches, build_a_rows,
                      build_b_rows, chat_prompt, download_cities,
                      knowledge_prompt, load_city_pairs, question, split_pools)
from src.extract import (final_token_activations, final_token_strings,
                         generate_answers, load_model)

# Ungated mirror of google/gemma-2-2b-it, pinned to a commit.
DEFAULT_MODEL = "unsloth/gemma-2-2b-it"
DEFAULT_REVISION = "457f2e15bf550c227ce6ad86e2ec108d3e42c106"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--out", default="activations")
    ap.add_argument("--frac-a", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--refilter", action="store_true",
                    help="rerun the knowledge filter even if data/pools.json exists")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pools_path, knowledge_path = Path("data/pools.json"), Path("data/knowledge.csv")
    cities_csv = download_cities()
    pairs = load_city_pairs(cities_csv)
    model, tok = load_model(args.model, revision=args.revision)

    # H and L must precede the answer with the same amount of context.
    n_tok = {k: len(tok(v, add_special_tokens=False).input_ids)
             for k, v in INSTRUCTIONS.items()}
    if n_tok["H"] != n_tok["L"]:
        raise RuntimeError(f"instruction token lengths differ: {n_tok}")

    # 1. Knowledge filter (skipped if pools were already fixed by an earlier run).
    if pools_path.exists() and not args.refilter:
        pools = json.loads(pools_path.read_text())
        print(f"reusing {pools_path}")
    else:
        answers = generate_answers(model, tok,
                                   [knowledge_prompt(c) for c in pairs.city])
        pairs["model_answer"] = answers
        pairs["known"] = [answer_matches(a, c)
                          for a, c in zip(answers, pairs.correct_country)]
        pairs[["city", "correct_country", "model_answer", "known"]].to_csv(
            knowledge_path, index=False)
        known = pairs.city[pairs.known]
        pool_a, pool_b = split_pools(known, frac_a=args.frac_a, seed=args.seed)
        pools = {"n_cities_total": len(pairs), "n_cities_known": len(known),
                 "seed": args.seed, "frac_a": args.frac_a,
                 "pool_a": pool_a, "pool_b": pool_b}
        pools_path.write_text(json.dumps(pools, indent=1))
    print(f"knowledge filter: kept {pools['n_cities_known']}/{pools['n_cities_total']}")

    # 2. Rows.
    a_rows = build_a_rows(pairs[pairs.city.isin(pools["pool_a"])])
    b_rows = build_b_rows(pairs[pairs.city.isin(pools["pool_b"])])

    # 3. Activations at the final token, which must be the closing period.
    manifest = {
        "model": args.model, "revision": args.revision,
        "dtype": "bfloat16", "attn_implementation": "eager",
        "storage_dtype": "float16", "batch_size": args.batch_size,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "cities_url": CITIES_URL,
        "cities_sha256": hashlib.sha256(cities_csv.read_bytes()).hexdigest(),
        "layer_convention": "index i = output of decoder block i-1; "
                            "0 = embeddings; last index has the final norm applied",
        "read_position": "final token of the text (the closing '.')",
        "instructions": INSTRUCTIONS, "instruction_token_lengths": n_tok,
        "example_prompt": chat_prompt(INSTRUCTIONS["L"] + question("<city>"),
                                      "The city of <city> is in <country>."),
        "n_cities_known": pools["n_cities_known"],
        "n_pool_a": len(pools["pool_a"]), "n_pool_b": len(pools["pool_b"]),
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

    Path("results").mkdir(exist_ok=True)
    text = json.dumps(manifest, indent=2)
    (out / "manifest.json").write_text(text)
    Path("results/extraction_manifest.json").write_text(text)


if __name__ == "__main__":
    main()

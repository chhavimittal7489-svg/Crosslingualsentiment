import argparse, itertools, os
import pandas as pd
from datasets import load_dataset

# Language folders in ai4bharat/samanantar (each pairs English with that language)
LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--langs", nargs="+", default=["hi", "bn", "ta", "te", "mr"])
    p.add_argument("--n", type=int, default=50_000, help="pairs per language")
    p.add_argument("--out", default="data/raw/samanantar")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for lang in args.langs:
        ds = load_dataset("ai4bharat/samanantar", lang, split="train", streaming=True)
        ds = ds.shuffle(seed=42, buffer_size=10_000)  # otherwise you only get the first N rows
        rows = list(itertools.islice(ds, args.n))
        pd.DataFrame(rows).to_parquet(f"{args.out}/{lang}.parquet", index=False)
        print(f"{lang}: saved {len(rows)} pairs, columns = {list(rows[0].keys())}")

if __name__ == "__main__":
    main()

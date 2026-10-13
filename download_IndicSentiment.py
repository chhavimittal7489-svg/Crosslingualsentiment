"""Download IndicSentiment (ai4bharat/IndicSentiment) and save one clean table per split.

IndicSentiment has NO train split: only `validation` (~156 reviews per language)
and `test` (~1000 per language). The same product reviews appear in every
language (n-way parallel), and each row also carries the original English review.
We use it for evaluation (zero-shot cross-lingual transfer), not for training.

Usage:
    python scripts/download_indicsentiment.py
    python scripts/download_indicsentiment.py --langs hi bn ta te mr
"""
import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "ai4bharat/IndicSentiment"
# as=Assamese bd=Bodo bn=Bengali gu=Gujarati hi=Hindi kn=Kannada ml=Malayalam
# mr=Marathi or=Odia pa=Punjabi ta=Tamil te=Telugu ur=Urdu
LANGS = ["as", "bd", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te", "ur"]
SPLITS = ["validation", "test"]

# Original column name -> our column name
COLUMNS = {"INDIC REVIEW": "text", "ENGLISH REVIEW": "english_text", "LABEL": "label"}


def read_json_file(path):
    """The files may be a single JSON array or JSON Lines; handle both."""
    with open(path, encoding="utf-8") as f:
        is_array = f.read(1024).lstrip().startswith("[")
    return pd.read_json(path, lines=not is_array)


def clean_table(df, lang, split):
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"{split}/{lang}.json is missing {missing}; columns are {list(df.columns)}")
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    df["label"] = df["label"].astype("string").str.strip().str.lower()
    df["text"] = df["text"].astype("string").str.strip()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"] != ""]
    df.insert(0, "lang", lang)
    df.insert(1, "split", split)
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="+", default=LANGS)
    parser.add_argument("--raw-dir", default="data/raw/indicsentiment")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for split in SPLITS:
        tables = []
        for lang in args.langs:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=f"data/{split}/{lang}.json",
                repo_type="dataset",
                local_dir=args.raw_dir,
            )
            tables.append(clean_table(read_json_file(path), lang, split))

        df = pd.concat(tables, ignore_index=True)
        out_path = os.path.join(args.out_dir, f"indicsentiment_{split}.parquet")
        df.to_parquet(out_path, index=False)

        print(f"\n=== {split}: {len(df)} rows -> {out_path}")
        print(pd.crosstab(df["lang"], df["label"], margins=True).to_string())

    print("\nExample row:")
    print(df.iloc[0].to_string())


if __name__ == "__main__":
    main()

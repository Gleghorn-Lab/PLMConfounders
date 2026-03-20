"""Generate summary statistics about BioGRID PPI datasets for supplementary material."""

import argparse
import os
from collections import Counter

import pandas as pd


SIMILARITY_THRESHOLD = 0.4
SPECIES_ID = "biogrid"
STRATEGIES = {"SS": "True", "NS": "False"}
SPLITS = {"train": "train", "valid": "val", "test": "test"}


def build_filename(matching_orgs_str: str, split_suffix: str) -> str:
    return f"processed_datasets/split_with_sim_{SPECIES_ID}_{SIMILARITY_THRESHOLD}_{matching_orgs_str}_{split_suffix}.csv"


def load_cached_splits() -> dict[str, dict[str, pd.DataFrame]] | None:
    result: dict[str, dict[str, pd.DataFrame]] = {}
    for strategy_label, matching_orgs_str in STRATEGIES.items():
        result[strategy_label] = {}
        for split_name, file_suffix in SPLITS.items():
            path = build_filename(matching_orgs_str, file_suffix)
            if not os.path.exists(path):
                return None
            result[strategy_label][split_name] = pd.read_csv(path)
    return result


def compute_split_stats(df: pd.DataFrame) -> dict:
    n_total = len(df)
    n_pos = int((df["labels"] > 0).sum())
    n_neg = int((df["labels"] == 0).sum())

    all_orgs = set(df["OrgA"]).union(set(df["OrgB"]))
    all_proteins = set(df["IdA"]).union(set(df["IdB"]))

    pos_df = df[df["labels"] > 0]
    neg_df = df[df["labels"] == 0]

    pos_intra = int((pos_df["OrgA"] == pos_df["OrgB"]).sum()) if len(pos_df) > 0 else 0
    neg_intra = int((neg_df["OrgA"] == neg_df["OrgB"]).sum()) if len(neg_df) > 0 else 0

    pos_intra_frac = pos_intra / n_pos if n_pos > 0 else 0.0
    neg_intra_frac = neg_intra / n_neg if n_neg > 0 else 0.0

    org_counter: Counter = Counter()
    org_counter.update(df["OrgA"])
    org_counter.update(df["OrgB"])
    top5 = org_counter.most_common(5)

    return {
        "total": n_total,
        "positives": n_pos,
        "negatives": n_neg,
        "unique_organisms": len(all_orgs),
        "unique_proteins": len(all_proteins),
        "pos_intraspecies_frac": pos_intra_frac,
        "pos_interspecies_frac": 1.0 - pos_intra_frac,
        "neg_intraspecies_frac": neg_intra_frac,
        "neg_interspecies_frac": 1.0 - neg_intra_frac,
        "top5_organisms": top5,
    }


def print_stats(strategy: str, split: str, stats: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy}  |  Split: {split}")
    print(f"{'='*60}")
    print(f"  Total examples:      {stats['total']:>10,}")
    print(f"  Positives:           {stats['positives']:>10,}")
    print(f"  Negatives:           {stats['negatives']:>10,}")
    print(f"  Unique organisms:    {stats['unique_organisms']:>10,}")
    print(f"  Unique proteins:     {stats['unique_proteins']:>10,}")
    print(f"  Pos intraspecies:    {stats['pos_intraspecies_frac']:>10.4f}")
    print(f"  Pos interspecies:    {stats['pos_interspecies_frac']:>10.4f}")
    print(f"  Neg intraspecies:    {stats['neg_intraspecies_frac']:>10.4f}")
    print(f"  Neg interspecies:    {stats['neg_interspecies_frac']:>10.4f}")
    print(f"  Top 5 organisms:")
    for org, count in stats["top5_organisms"]:
        print(f"    {org}: {count:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate summary statistics for BioGRID PPI datasets."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dataset_summary.csv",
        help="Path to save the summary CSV.",
    )
    args = parser.parse_args()

    data = load_cached_splits()
    if data is None:
        print(
            "WARNING: Cached split CSVs not found in processed_datasets/. "
            "Run the main experiment first to generate them."
        )
        print("Falling back to raw BioGRID from HuggingFace (no train/valid/test splits).")
        from datasets import load_dataset
        ds = load_dataset("Synthyra/BIOGRID", split="train")
        df = pd.DataFrame(ds)
        df = df.rename(columns={"A": "IdA", "B": "IdB"})
        df["labels"] = 1
        stats = compute_split_stats(df[["IdA", "IdB", "labels", "OrgA", "OrgB"]])
        print_stats("raw", "all", stats)
        return

    rows = []
    for strategy in STRATEGIES:
        for split in SPLITS:
            df = data[strategy][split]
            stats = compute_split_stats(df[["IdA", "IdB", "labels", "OrgA", "OrgB"]])
            print_stats(strategy, split, stats)

            top5_str = "; ".join(f"{org} ({count:,})" for org, count in stats["top5_organisms"])
            rows.append({
                "strategy": strategy,
                "split": split,
                "total": stats["total"],
                "positives": stats["positives"],
                "negatives": stats["negatives"],
                "unique_organisms": stats["unique_organisms"],
                "unique_proteins": stats["unique_proteins"],
                "pos_intraspecies_frac": round(stats["pos_intraspecies_frac"], 4),
                "pos_interspecies_frac": round(stats["pos_interspecies_frac"], 4),
                "neg_intraspecies_frac": round(stats["neg_intraspecies_frac"], 4),
                "neg_interspecies_frac": round(stats["neg_interspecies_frac"], 4),
                "top5_organisms": top5_str,
            })

    summary_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    summary_df.to_csv(args.output, index=False)
    print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    main()

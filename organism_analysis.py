"""Analyze organism distribution in BioGRID to show taxonomic imbalance between positive and negative PPI pairs."""

import argparse
import os
from collections import Counter

import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt


def load_biogrid() -> dict:
    ds = load_dataset("Synthyra/BIOGRID", split="train")
    return ds


def get_pair_key(org_a: str, org_b: str) -> str:
    return " - ".join(sorted([org_a, org_b]))


def compute_same_organism_pct(org_a: list[str], org_b: list[str]) -> float:
    assert len(org_a) == len(org_b)
    same = sum(1 for a, b in zip(org_a, org_b) if a == b)
    return 100.0 * same / len(org_a)


def count_organism_pairs(org_a: list[str], org_b: list[str]) -> Counter:
    counter: Counter = Counter()
    for a, b in zip(org_a, org_b):
        counter[get_pair_key(a, b)] += 1
    return counter


def run_shuffles(
    org_a: list[str],
    org_b: list[str],
    seq_a: list[str],
    seq_b: list[str],
    n_shuffles: int,
    seed: int,
) -> tuple[list[float], Counter]:
    rng = np.random.default_rng(seed)
    n = len(org_a)

    all_seqs = seq_a + seq_b
    all_orgs = org_a + org_b

    # Build sequence -> organism mapping
    seq_to_org: dict[str, str] = {}
    for seq, org in zip(all_seqs, all_orgs):
        seq_to_org[seq] = org

    unique_seqs = list(set(all_seqs))

    same_org_pcts: list[float] = []
    representative_neg_pairs: Counter = Counter()

    for i in range(n_shuffles):
        shuffled = unique_seqs.copy()
        rng.shuffle(shuffled)

        # Create random pairs by pairing consecutive sequences
        neg_a_orgs = []
        neg_b_orgs = []
        for j in range(0, min(2 * n, len(shuffled) - 1), 2):
            neg_a_orgs.append(seq_to_org[shuffled[j]])
            neg_b_orgs.append(seq_to_org[shuffled[j + 1]])

        pct = compute_same_organism_pct(neg_a_orgs, neg_b_orgs)
        same_org_pcts.append(pct)

        if i == 0:
            representative_neg_pairs = count_organism_pairs(neg_a_orgs, neg_b_orgs)

    return same_org_pcts, representative_neg_pairs


def plot_results(
    same_org_pcts: list[float],
    pos_same_org_pct: float,
    pos_pairs: Counter,
    neg_pairs: Counter,
    top_n: int,
    output_path: str,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Histogram of same-organism percentage across shuffles
    ax1.hist(same_org_pcts, bins=20, color="#4878CF", edgecolor="black", alpha=0.8)
    ax1.axvline(pos_same_org_pct, color="red", linestyle="--", linewidth=2, label=f"Positives: {pos_same_org_pct:.1f}%")
    ax1.set_xlabel("Same-Organism Pairs (%)", fontsize=12)
    ax1.set_ylabel("Count (out of 100 shuffles)", fontsize=12)
    ax1.set_title("A) Same-Organism % in Shuffled Negatives vs. Positives", fontsize=13)
    ax1.legend(fontsize=11)

    # Panel B: Grouped bar chart of top organism pairs
    top_keys = [k for k, _ in pos_pairs.most_common(top_n)]
    total_pos = sum(pos_pairs.values())
    total_neg = sum(neg_pairs.values())

    pos_freqs = [100.0 * pos_pairs[k] / total_pos for k in top_keys]
    neg_freqs = [100.0 * neg_pairs.get(k, 0) / total_neg for k in top_keys]

    x = np.arange(len(top_keys))
    width = 0.35

    ax2.bar(x - width / 2, pos_freqs, width, label="Positives", color="#4878CF", edgecolor="black")
    ax2.bar(x + width / 2, neg_freqs, width, label="Shuffled Negatives", color="#E24A33", edgecolor="black")
    ax2.set_xlabel("Organism Pair", fontsize=12)
    ax2.set_ylabel("Frequency (%)", fontsize=12)
    ax2.set_title("B) Top Organism Pair Frequencies: Positives vs. Negatives", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_keys, rotation=45, ha="right", fontsize=9)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze organism distribution in BioGRID PPI data.")
    parser.add_argument(
        "--n_shuffles",
        type=int,
        default=100,
        help="Number of random shuffles for negative pair generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=15,
        help="Number of top organism pairs to show in bar chart.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/organism_distribution.png",
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    print("Loading BioGRID dataset...")
    ds = load_biogrid()

    org_a = ds["OrgA"]
    org_b = ds["OrgB"]
    seq_a = ds["SeqA"]
    seq_b = ds["SeqB"]
    n = len(org_a)

    # Positive pair statistics
    pos_same_org_pct = compute_same_organism_pct(org_a, org_b)
    pos_pairs = count_organism_pairs(org_a, org_b)

    unique_orgs = set(org_a) | set(org_b)
    unique_seqs = set(seq_a) | set(seq_b)

    print(f"\nTotal positive pairs: {n:,}")
    print(f"Unique organisms: {len(unique_orgs):,}")
    print(f"Unique sequences: {len(unique_seqs):,}")
    print(f"Same-organism pairs in positives: {pos_same_org_pct:.2f}%")

    print(f"\nTop 10 organism pairs in positives:")
    total_pos = sum(pos_pairs.values())
    for pair, count in pos_pairs.most_common(10):
        print(f"  {pair}: {count:,} ({100.0 * count / total_pos:.2f}%)")

    # Shuffle analysis
    print(f"\nRunning {args.n_shuffles} shuffles...")
    same_org_pcts, neg_pairs = run_shuffles(
        org_a, org_b, seq_a, seq_b, args.n_shuffles, args.seed
    )

    print(f"Shuffled same-organism %: mean={np.mean(same_org_pcts):.2f}, std={np.std(same_org_pcts):.2f}")
    print(f"Positive same-organism %: {pos_same_org_pct:.2f}")

    print(f"\nTop 10 organism pairs in representative negative set:")
    total_neg = sum(neg_pairs.values())
    for pair, count in neg_pairs.most_common(10):
        print(f"  {pair}: {count:,} ({100.0 * count / total_neg:.2f}%)")

    # Plot
    plot_results(
        same_org_pcts,
        pos_same_org_pct,
        pos_pairs,
        neg_pairs,
        args.top_n,
        args.output,
    )


if __name__ == "__main__":
    main()

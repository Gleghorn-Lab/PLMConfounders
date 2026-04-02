"""Analyze organism distribution in BioGRID to show taxonomic imbalance between positive and negative PPI pairs."""

import argparse
import os
from collections import Counter

import numpy as np
from datasets import load_dataset
from matplotlib import pyplot as plt
from matplotlib.patches import Patch

SPECIES_ABBREV: dict[str, str] = {
    "Homo sapiens": "Hs",
    "Saccharomyces cerevisiae": "Sc",
    "Escherichia coli": "Ec",
    "Schizosaccharomyces pombe": "Sp",
    "Arabidopsis thaliana": "At",
    "Mus musculus": "Mm",
    "Drosophila melanogaster": "Dm",
    "Caenorhabditis elegans": "Ce",
    "Severe acute respiratory syndrome coronavirus 2": "SCV2",
    "Rattus norvegicus": "Rn",
    "Xenopus laevis": "Xl",
    "Danio rerio": "Dr",
    "Gallus gallus": "Gg",
    "Sus scrofa": "Ss",
    "Bos taurus": "Bt",
    "Dictyostelium discoideum": "Dd",
    "Plasmodium falciparum": "Pf",
    "Human immunodeficiency virus 1": "HIV1",
    "Human immunodeficiency virus 2": "HIV2",
}


def abbreviate_species(name: str) -> str:
    if name in SPECIES_ABBREV:
        return SPECIES_ABBREV[name]
    parts = name.split()
    return "".join(p[0].upper() for p in parts if p)


def abbreviate_pair(pair_key: str) -> str:
    parts = pair_key.split(" - ")
    assert len(parts) == 2
    return f"{abbreviate_species(parts[0])}-{abbreviate_species(parts[1])}"


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
    n_shuffles: int,
    seed: int,
) -> tuple[list[float], Counter]:
    rng = np.random.default_rng(seed)

    arr_a = np.array(org_a)
    arr_b = np.array(org_b)

    same_org_pcts: list[float] = []
    representative_neg_pairs: Counter = Counter()

    for i in range(n_shuffles):
        shuf_a = arr_a.copy()
        shuf_b = arr_b.copy()
        rng.shuffle(shuf_a)
        rng.shuffle(shuf_b)

        neg_a_orgs = shuf_a.tolist()
        neg_b_orgs = shuf_b.tolist()

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
    output_path: str,
    min_freq_pct: float = 0.1,
) -> None:
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    # --- Panel A: Same-species vs cross-species stacked bars ---
    neg_mean = np.mean(same_org_pcts)
    neg_std = np.std(same_org_pcts)
    pos_same = pos_same_org_pct
    pos_cross = 100.0 - pos_same
    neg_same = neg_mean
    neg_cross = 100.0 - neg_mean

    c_same = "#2b8c6e"
    c_cross = "#e05e5e"

    bars_same = ax1.bar(
        [0, 1], [pos_same, neg_same],
        color=c_same, width=0.55, label="Intra-species",
    )
    bars_cross = ax1.bar(
        [0, 1], [pos_cross, neg_cross],
        bottom=[pos_same, neg_same],
        color=c_cross, width=0.55, label="Inter-species",
    )
    ax1.errorbar(
        1, neg_same, yerr=neg_std,
        fmt="none", ecolor="black", capsize=5, capthick=1.5, linewidth=1.5,
    )

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Positives", "Negatives"], fontsize=12)
    ax1.set_ylabel("Proportion of pairs (%)", fontsize=12)
    ax1.set_ylim(0, 105)
    ax1.set_title("A", fontsize=14, fontweight="bold", loc="left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for bar_set, values in [(bars_same, [pos_same, neg_same]), (bars_cross, [pos_cross, neg_cross])]:
        for bar, val in zip(bar_set, values):
            if val >= 5:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%",
                    ha="center", va="center", fontsize=10, fontweight="bold", color="white",
                )

    # --- Panel B: Radial bar chart of organism pair frequencies ---
    total_pos = sum(pos_pairs.values())
    total_neg = sum(neg_pairs.values())

    all_keys = set(pos_pairs.keys()) | set(neg_pairs.keys())
    filtered_keys = [
        k for k in all_keys
        if 100.0 * pos_pairs[k] / total_pos >= min_freq_pct
        or 100.0 * neg_pairs[k] / total_neg >= min_freq_pct
    ]
    filtered_keys.sort(key=lambda k: pos_pairs[k], reverse=True)

    n_pairs = len(filtered_keys)
    pos_freqs = np.array([100.0 * pos_pairs[k] / total_pos for k in filtered_keys])
    neg_freqs = np.array([100.0 * neg_pairs[k] / total_neg for k in filtered_keys])
    labels = [abbreviate_pair(k) for k in filtered_keys]

    angles = np.linspace(0, 2 * np.pi, n_pairs, endpoint=False)
    bar_width = 2 * np.pi / n_pairs * 0.8

    c_pos = "#3a7ebf"
    c_neg = "#e8873a"

    # Log-transform to make small bars visible alongside dominant pairs
    r_min = np.log10(0.05)  # center of chart
    log_pos = np.log10(pos_freqs + 0.01) - r_min
    log_neg = np.log10(neg_freqs + 0.01) - r_min
    log_pos = np.maximum(log_pos, 0)
    log_neg = np.maximum(log_neg, 0)

    ax2.bar(angles, log_pos, width=bar_width, color=c_pos, alpha=0.65, label="Positives", zorder=2)
    ax2.bar(angles, log_neg, width=bar_width, color=c_neg, alpha=0.65, label="Negatives", zorder=2)
    ax2.set_rlim(bottom=0)

    ax2.set_xticks(angles)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.tick_params(axis="x", pad=10)
    c_intra = "#2b8c6e"
    c_inter = "#e05e5e"
    for tick_label in ax2.get_xticklabels():
        parts = tick_label.get_text().split("-")
        tick_label.set_color(c_intra if parts[0] == parts[1] else c_inter)
        tick_label.set_fontweight("bold")
    ax2.set_title("B", fontsize=14, fontweight="bold", loc="left", pad=20)
    ax2_legend = ax2.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.35, 1.05))
    ax2.add_artist(ax2_legend)
    intra_inter_handles = [
        Patch(facecolor=c_intra, label="Intra-species"),
        Patch(facecolor=c_inter, label="Inter-species"),
    ]
    ax2.legend(handles=intra_inter_handles, fontsize=9, loc="upper right", bbox_to_anchor=(1.35, 0.9))

    r_ticks = [np.log10(v) - r_min for v in [0.1, 1, 10]]
    ax2.set_rticks(r_ticks)
    ax2.set_yticklabels([""] * len(r_ticks))
    # Draw radial labels manually with background boxes, positioned at bottom of chart
    label_angle = np.pi
    for val, label in zip([0.1, 1, 10], ["0.1%", "1%", "10%"]):
        ax2.text(
            label_angle, np.log10(val) - r_min, label,
            ha="center", va="center", fontsize=8, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", alpha=0.9),
            zorder=10,
        )
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)

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
        "--output",
        type=str,
        default="figures/organism_distribution.png",
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    print("Loading BioGRID dataset...")
    ds = load_biogrid()

    org_a = list(ds["OrgA"])
    org_b = list(ds["OrgB"])
    n = len(org_a)

    # Positive pair statistics
    pos_same_org_pct = compute_same_organism_pct(org_a, org_b)
    pos_pairs = count_organism_pairs(org_a, org_b)

    unique_orgs = set(org_a) | set(org_b)
    unique_proteins = set(ds["A"]) | set(ds["B"])

    print(f"\nTotal positive pairs: {n:,}")
    print(f"Unique organisms: {len(unique_orgs):,}")
    print(f"Unique proteins: {len(unique_proteins):,}")
    print(f"Same-organism pairs in positives: {pos_same_org_pct:.2f}%")

    print(f"\nTop 10 organism pairs in positives:")
    total_pos = sum(pos_pairs.values())
    for pair, count in pos_pairs.most_common(10):
        print(f"  {pair}: {count:,} ({100.0 * count / total_pos:.2f}%)")

    # Shuffle analysis
    print(f"\nRunning {args.n_shuffles} shuffles...")
    same_org_pcts, neg_pairs = run_shuffles(
        org_a, org_b, args.n_shuffles, args.seed
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
        args.output,
    )


if __name__ == "__main__":
    main()

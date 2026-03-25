"""
Detailed analysis of interspecies experiment results.
Loads cached splits + model checkpoints, produces fine-grained metrics.
"""
import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pauc import ROC, ci_auc, compare, plot_roc
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.data import BiogridDataset, BiogridCollator
from model.ppi_model import PPIConfig, PPIModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze interspecies experiment results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/interspecies_experiment",
    )
    parser.add_argument(
        "--embed_path",
        type=str,
        default="esmc600m_biogrid_mv_embeddings.pth",
    )
    parser.add_argument("--plm_path", type=str, default="Synthyra/ESMplusplus_large")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--output_size", type=int, default=64)
    parser.add_argument("--n_tokens", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> PPIModel:
    config = PPIConfig(
        plm_path=args.plm_path,
        input_size=1152,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        expansion_ratio=8 / 3,
        n_tokens=args.n_tokens,
        dropout=args.dropout,
        rotary=True,
        block_type="transformer",
        spectral_norm=False,
        adversarial=False,
        adversarial_num_labels=1,
        add_block_0=False,
    )
    return PPIModel(config)


@torch.no_grad()
def get_predictions(
    model: PPIModel,
    test_df: pd.DataFrame,
    seq_dict: dict[str, str],
    embed_dict: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits, probabilities) arrays aligned with test_df rows."""
    model.eval()
    collator = BiogridCollator(
        embed_dim=1152,
        max_length=args.max_length,
        embedding_dict=embed_dict,
    )
    dataset = BiogridDataset(test_df, seq_dict, eval_mode=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    all_logits = []
    for batch in tqdm(loader, desc="Inference"):
        a = batch["a"].to(device)
        b = batch["b"].to(device)
        a_mask = batch["a_mask"].to(device)
        b_mask = batch["b_mask"].to(device)
        output = model(a, b, a_mask, b_mask)
        all_logits.append(output.logits.squeeze(-1).cpu())

    logits = torch.cat(all_logits).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))
    return logits, probs


def compute_metrics(labels: np.ndarray, logits: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = (logits > 0).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tpr = tp / n_pos if n_pos > 0 else float("nan")
    tnr = tn / n_neg if n_neg > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    has_both = len(np.unique(labels)) > 1
    return {
        "n": len(labels),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "tnr": tnr,
        "ppv": ppv,
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "auroc": float(roc_auc_score(labels, probs)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(labels, probs)) if has_both else float("nan"),
        "mean_prob_pos": float(probs[labels == 1].mean()) if n_pos > 0 else float("nan"),
        "mean_prob_neg": float(probs[labels == 0].mean()) if n_neg > 0 else float("nan"),
        "median_prob_pos": float(np.median(probs[labels == 1])) if n_pos > 0 else float("nan"),
        "median_prob_neg": float(np.median(probs[labels == 0])) if n_neg > 0 else float("nan"),
    }


def print_metrics(name: str, m: dict[str, float]):
    print(f"\n  --- {name} ---")
    print(f"  N={m['n']}  (pos={m['n_pos']}, neg={m['n_neg']})")
    print(f"  TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")
    print(f"  TPR (recall)={m['tpr']:.4f}  TNR (specificity)={m['tnr']:.4f}  PPV (precision)={m['ppv']:.4f}")
    print(f"  Accuracy={m['accuracy']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}")
    print(f"  AUROC={m['auroc']:.4f}  PR-AUC={m['pr_auc']:.4f}")
    print(f"  Mean P(pos)|pos={m['mean_prob_pos']:.4f}  Mean P(pos)|neg={m['mean_prob_neg']:.4f}")
    print(f"  Median P(pos)|pos={m['median_prob_pos']:.4f}  Median P(pos)|neg={m['median_prob_neg']:.4f}")


def analyze_model(
    model_name: str,
    test_df: pd.DataFrame,
    logits: np.ndarray,
    probs: np.ndarray,
):
    labels = test_df["labels"].values
    is_inter = test_df["is_interspecies"].values

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*60}")

    # 1. Overall
    print_metrics("ALL", compute_metrics(labels, logits, probs))

    # 2. By interspecies status (all rows, not subsampled)
    for inter_val, inter_name in [(True, "INTERSPECIES"), (False, "INTRASPECIES")]:
        mask = is_inter == inter_val
        if mask.sum() == 0:
            continue
        sub_labels = labels[mask]
        sub_logits = logits[mask]
        sub_probs = probs[mask]
        print_metrics(f"{inter_name} (all rows)", compute_metrics(sub_labels, sub_logits, sub_probs))

        # Break down further: positives only, negatives only
        pos_mask = mask & (labels == 1)
        neg_mask = mask & (labels == 0)
        if pos_mask.sum() > 0:
            preds_pos = (logits[pos_mask] > 0).astype(int)
            tp = int(preds_pos.sum())
            fn = int((preds_pos == 0).sum())
            print(f"    Positives: n={pos_mask.sum()}  predicted_pos={tp}  predicted_neg={fn}  "
                  f"recall={tp/pos_mask.sum():.4f}  mean_prob={probs[pos_mask].mean():.4f}")
        if neg_mask.sum() > 0:
            preds_neg = (logits[neg_mask] > 0).astype(int)
            fp = int(preds_neg.sum())
            tn = int((preds_neg == 0).sum())
            print(f"    Negatives: n={neg_mask.sum()}  predicted_pos={fp}  predicted_neg={tn}  "
                  f"specificity={tn/neg_mask.sum():.4f}  mean_prob={probs[neg_mask].mean():.4f}")

    # 3. Organism pair breakdown
    org_pairs = test_df.apply(
        lambda r: tuple(sorted([str(r["OrgA"]), str(r["OrgB"])])), axis=1
    )
    test_df = test_df.copy()
    test_df["org_pair"] = org_pairs

    print(f"\n  --- ORGANISM PAIR BREAKDOWN ---")
    pair_counts = test_df["org_pair"].value_counts()
    print(f"  Unique organism pairs: {len(pair_counts)}")
    print(f"  Top 15 pairs:")
    for pair, count in pair_counts.head(15).items():
        pair_mask = (test_df["org_pair"] == pair).values
        pair_labels = labels[pair_mask]
        pair_logits = logits[pair_mask]
        pair_probs = probs[pair_mask]
        pair_preds = (pair_logits > 0).astype(int)
        n_pos = int((pair_labels == 1).sum())
        n_neg = int((pair_labels == 0).sum())
        tp = int(((pair_preds == 1) & (pair_labels == 1)).sum())
        is_cross = pair[0] != pair[1]
        pair_str = f"{pair[0]} / {pair[1]}"
        print(f"    {pair_str:50s}  n={count:5d}  pos={n_pos:5d}  neg={n_neg:5d}  "
              f"TP={tp:5d}  mean_p={pair_probs.mean():.3f}  cross={'Y' if is_cross else 'N'}")

    # 4. Interspecies positives: organism pair breakdown
    inter_pos_mask = (labels == 1) & is_inter
    if inter_pos_mask.sum() > 0:
        print(f"\n  --- INTERSPECIES POSITIVE PAIRS (organism breakdown) ---")
        inter_pos_df = test_df[inter_pos_mask]
        inter_pos_probs = probs[inter_pos_mask]
        inter_pos_logits = logits[inter_pos_mask]
        inter_pos_preds = (inter_pos_logits > 0).astype(int)

        pair_groups = inter_pos_df.groupby("org_pair")
        rows = []
        for pair, group in pair_groups:
            idx = group.index
            mask_idx = np.array([test_df.index.get_loc(i) for i in idx])
            p = probs[mask_idx]
            lg = logits[mask_idx]
            preds = (lg > 0).astype(int)
            rows.append({
                "pair": f"{pair[0]} / {pair[1]}",
                "n": len(group),
                "predicted_pos": int(preds.sum()),
                "recall": preds.sum() / len(group),
                "mean_prob": float(p.mean()),
                "median_prob": float(np.median(p)),
            })
        rows.sort(key=lambda r: r["n"], reverse=True)
        for r in rows[:20]:
            print(f"    {r['pair']:50s}  n={r['n']:5d}  pred_pos={r['predicted_pos']:5d}  "
                  f"recall={r['recall']:.3f}  mean_p={r['mean_prob']:.3f}  med_p={r['median_prob']:.3f}")


def roc_comparison(
    test_df: pd.DataFrame,
    ss_probs: np.ndarray,
    ns_probs: np.ndarray,
    results_dir: str,
):
    """ROC comparison between SS and NS models with DeLong CIs using pauc."""
    labels = test_df["labels"].values
    is_inter = test_df["is_interspecies"].values

    subgroups = [
        ("all", np.ones(len(test_df), dtype=bool)),
        ("interspecies", is_inter),
        ("intraspecies", ~is_inter),
    ]

    print(f"\n{'='*60}")
    print("  ROC COMPARISON (DeLong CIs)")
    print(f"{'='*60}")

    for subgroup_name, mask in subgroups:
        sub_labels = labels[mask]
        sub_ss = ss_probs[mask]
        sub_ns = ns_probs[mask]

        if len(np.unique(sub_labels)) < 2:
            print(f"\n  {subgroup_name.upper()}: skipped (single class)")
            continue

        roc_ss = ROC(sub_labels, sub_ss, name="SS")
        roc_ns = ROC(sub_labels, sub_ns, name="NS")

        ss_lo, ss_hi = ci_auc(roc_ss, conf_level=0.95, method="delong")
        ns_lo, ns_hi = ci_auc(roc_ns, conf_level=0.95, method="delong")

        print(f"\n  {subgroup_name.upper()} (n={mask.sum()}, pos={int((sub_labels == 1).sum())}, neg={int((sub_labels == 0).sum())})")
        print(f"    SS AUROC: {roc_ss.auc:.4f}  [95% CI: {ss_lo:.4f}, {ss_hi:.4f}]")
        print(f"    NS AUROC: {roc_ns.auc:.4f}  [95% CI: {ns_lo:.4f}, {ns_hi:.4f}]")

        result = compare(roc_ss, roc_ns, method="delong", paired=True)
        print(f"    DeLong test: Z={result.stat:.4f}  p={result.p_value:.4e}  delta_AUC={result.estimate:.4f}")

        # Plot ROC curves
        fig, ax = plt.subplots(figsize=(7, 7))
        plot_roc(
            [roc_ss, roc_ns],
            ax=ax,
            title=f"ROC: SS vs NS ({subgroup_name})",
            show_auc=True,
            shade_auc=False,
            plot_ci=True,
            ci_type="sensitivity",
            ci_alpha=0.15,
            annotate_best=True,
            best_method="youden",
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
        ax.legend(loc="lower right")
        fig.tight_layout()
        save_path = os.path.join(results_dir, f"roc_{subgroup_name}.png")
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {save_path}")


def main():
    args = parse_args()
    results_dir = args.results_dir

    # Load cached splits
    cache_path = os.path.join(results_dir, "cached_splits.pkl")
    assert os.path.exists(cache_path), f"Cached splits not found at {cache_path}"
    with open(cache_path, "rb") as f:
        cached = pickle.load(f)

    seq_dict = cached["seq_dict"]
    ss_test = cached["ss_test"]
    ns_test = cached["ns_test"]

    # They share the same test set
    assert ss_test["IdA"].tolist() == ns_test["IdA"].tolist()
    test_df = ss_test

    # Print test set composition
    labels = test_df["labels"].values
    is_inter = test_df["is_interspecies"].values
    print("="*60)
    print("TEST SET COMPOSITION")
    print("="*60)
    print(f"Total: {len(test_df)}")
    print(f"Positives: {(labels == 1).sum()}  Negatives: {(labels == 0).sum()}")
    print(f"Interspecies rows: {is_inter.sum()}  Intraspecies rows: {(~is_inter).sum()}")
    print(f"  Interspecies positives: {((labels == 1) & is_inter).sum()}")
    print(f"  Interspecies negatives: {((labels == 0) & is_inter).sum()}")
    print(f"  Intraspecies positives: {((labels == 1) & (~is_inter)).sum()}")
    print(f"  Intraspecies negatives: {((labels == 0) & (~is_inter)).sum()}")

    # Organism distribution in test
    print(f"\nOrganism pair distribution in test positives:")
    pos_df = test_df[labels == 1].copy()
    pos_df["org_pair"] = pos_df.apply(
        lambda r: tuple(sorted([str(r["OrgA"]), str(r["OrgB"])])), axis=1
    )
    for pair, count in pos_df["org_pair"].value_counts().head(10).items():
        is_cross = pair[0] != pair[1]
        print(f"  {pair[0]} / {pair[1]}: {count} ({'inter' if is_cross else 'intra'})")

    print(f"\nOrganism pair distribution in test negatives:")
    neg_df = test_df[labels == 0].copy()
    neg_df["org_pair"] = neg_df.apply(
        lambda r: tuple(sorted([str(r["OrgA"]), str(r["OrgB"])])), axis=1
    )
    for pair, count in neg_df["org_pair"].value_counts().head(10).items():
        is_cross = pair[0] != pair[1]
        print(f"  {pair[0]} / {pair[1]}: {count} ({'inter' if is_cross else 'intra'})")

    # Load embeddings
    assert os.path.exists(args.embed_path), f"Embeddings not found at {args.embed_path}"
    embed_dict = torch.load(args.embed_path, weights_only=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and evaluate SS model
    ss_model_path = os.path.join(results_dir, "ss_model", "best_model.pt")
    assert os.path.exists(ss_model_path), f"SS model not found at {ss_model_path}"
    ss_model = build_model(args).to(device)
    ss_model.load_state_dict(torch.load(ss_model_path, weights_only=True, map_location=device))
    ss_logits, ss_probs = get_predictions(ss_model, test_df, seq_dict, embed_dict, args, device)
    ss_model.cpu()
    del ss_model

    # Load and evaluate NS model
    ns_model_path = os.path.join(results_dir, "ns_model", "best_model.pt")
    assert os.path.exists(ns_model_path), f"NS model not found at {ns_model_path}"
    ns_model = build_model(args).to(device)
    ns_model.load_state_dict(torch.load(ns_model_path, weights_only=True, map_location=device))
    ns_logits, ns_probs = get_predictions(ns_model, test_df, seq_dict, embed_dict, args, device)
    ns_model.cpu()
    del ns_model

    torch.cuda.empty_cache()

    # Detailed analysis
    analyze_model("SS (Strategic Sampling)", test_df, ss_logits, ss_probs)
    analyze_model("NS (Normal Sampling)", test_df, ns_logits, ns_probs)

    # Direct SS vs NS comparison table
    print(f"\n{'='*60}")
    print("  SS vs NS COMPARISON (threshold=0.5 on probabilities)")
    print(f"{'='*60}")

    for inter_val, name in [(None, "ALL"), (True, "INTERSPECIES"), (False, "INTRASPECIES")]:
        if inter_val is None:
            mask = np.ones(len(test_df), dtype=bool)
        else:
            mask = is_inter == inter_val

        sub_labels = labels[mask]
        pos_sub = sub_labels == 1
        neg_sub = sub_labels == 0

        print(f"\n  {name}: n={mask.sum()} (pos={pos_sub.sum()}, neg={neg_sub.sum()})")
        for model_name, lg, pb in [("SS", ss_logits, ss_probs), ("NS", ns_logits, ns_probs)]:
            sub_lg = lg[mask]
            sub_pb = pb[mask]
            sub_preds = (sub_lg > 0).astype(int)
            has_both = len(np.unique(sub_labels)) > 1
            acc = accuracy_score(sub_labels, sub_preds)
            f1 = f1_score(sub_labels, sub_preds, zero_division=0)
            mcc = matthews_corrcoef(sub_labels, sub_preds)
            auroc = roc_auc_score(sub_labels, sub_pb) if has_both else float("nan")
            tpr = sub_preds[pos_sub].sum() / pos_sub.sum() if pos_sub.sum() > 0 else float("nan")
            tnr = (sub_preds[neg_sub] == 0).sum() / neg_sub.sum() if neg_sub.sum() > 0 else float("nan")
            mean_p_pos = float(sub_pb[pos_sub].mean()) if pos_sub.sum() > 0 else float("nan")
            mean_p_neg = float(sub_pb[neg_sub].mean()) if neg_sub.sum() > 0 else float("nan")
            print(f"    {model_name}: acc={acc:.4f}  f1={f1:.4f}  mcc={mcc:.4f}  "
                  f"auroc={auroc:.4f}  tpr={tpr:.4f}  tnr={tnr:.4f}  "
                  f"mean_p|pos={mean_p_pos:.4f}  mean_p|neg={mean_p_neg:.4f}")

    # ROC comparison with DeLong CIs
    roc_comparison(test_df, ss_probs, ns_probs, results_dir)


if __name__ == "__main__":
    main()

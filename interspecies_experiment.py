"""
Train SS and NS PPI models on BIOGRID-MV, then evaluate on interspecies PPI subgroups.
"""
import argparse
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel

from data.biogrid import generate_negative_ppis
from data.data import BiogridDataset, BiogridCollator
from data.data_clustering import cluster_sequences
from model.ppi_model import PPIConfig, PPIModel
from training.utils import set_seed, AutoGradClipper


global WANDB_AVAILABLE
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interspecies PPI experiment")
    parser.add_argument("--plm_path", type=str, default="Synthyra/ESMplusplus_large")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--output_size", type=int, default=128)
    parser.add_argument("--n_tokens", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min_test_rows", type=int, default=500)
    parser.add_argument("--min_interspecies", type=int, default=100)
    parser.add_argument("--bugfix", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--wandb_token", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="PLMConfounders")
    return parser.parse_args()


def load_biogrid_mv(bugfix: bool) -> tuple[pd.DataFrame, dict[str, str], set[str]]:
    data = load_dataset("Synthyra/BIOGRID-MV-5.0.253", split="train")
    if bugfix:
        data = data.select(range(min(10000, len(data))))

    df = pd.DataFrame(data)
    seq_dict: dict[str, str] = {}
    for _, row in df.iterrows():
        seq_dict[row["A"]] = row["SeqA"]
        seq_dict[row["B"]] = row["SeqB"]

    interaction_set = set(
        "_".join(sorted([a, b])) for a, b in zip(df["A"], df["B"])
    )

    positives = pd.DataFrame({
        "IdA": df["A"],
        "IdB": df["B"],
        "labels": 1,
        "OrgA": df["OrgA"],
        "OrgB": df["OrgB"],
    })
    positives["is_interspecies"] = positives["OrgA"] != positives["OrgB"]
    print(f"Loaded {len(positives)} positives, {positives['is_interspecies'].sum()} interspecies")
    return positives, seq_dict, interaction_set


def interspecies_aware_split(
    positives: pd.DataFrame,
    cluster_dict: dict[str, list[str]],
    min_test_rows: int,
    min_interspecies: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)

    seq_to_cluster: dict[str, str] = {}
    for rep, members in cluster_dict.items():
        for m in members:
            seq_to_cluster[m] = rep

    positives = positives.copy()
    positives["cluster_a"] = positives["IdA"].map(seq_to_cluster)
    positives["cluster_b"] = positives["IdB"].map(seq_to_cluster)

    unmapped_a = positives["cluster_a"].isna().sum()
    unmapped_b = positives["cluster_b"].isna().sum()
    assert unmapped_a == 0, f"{unmapped_a} IdA entries not in cluster_dict"
    assert unmapped_b == 0, f"{unmapped_b} IdB entries not in cluster_dict"

    # Score each cluster by how many interspecies pairs it covers
    cluster_to_inter_count: dict[str, int] = {}
    inter_mask = positives["is_interspecies"].values
    for cluster_id in cluster_dict:
        mask = (positives["cluster_a"].values == cluster_id) | (positives["cluster_b"].values == cluster_id)
        cluster_to_inter_count[cluster_id] = int((mask & inter_mask).sum())

    all_clusters = list(cluster_dict.keys())
    # Sort descending by interspecies count to greedily pick interspecies-rich clusters first
    all_clusters.sort(key=lambda c: cluster_to_inter_count[c], reverse=True)

    def _greedy_select(
        available_clusters: list[str],
        target_rows: int,
        target_inter: int,
    ) -> set[str]:
        selected: set[str] = set()
        for cluster_id in available_clusters:
            selected.add(cluster_id)
            mask = positives["cluster_a"].isin(selected) & positives["cluster_b"].isin(selected)
            subset = positives[mask]
            n_rows = len(subset)
            n_inter = subset["is_interspecies"].sum()
            if n_rows >= target_rows and n_inter >= target_inter:
                break
        return selected

    test_clusters = _greedy_select(all_clusters, min_test_rows, min_interspecies)
    remaining = [c for c in all_clusters if c not in test_clusters]
    valid_clusters = _greedy_select(remaining, min_test_rows, min_interspecies)

    test_mask = positives["cluster_a"].isin(test_clusters) & positives["cluster_b"].isin(test_clusters)
    valid_mask = positives["cluster_a"].isin(valid_clusters) & positives["cluster_b"].isin(valid_clusters)
    excluded = test_clusters | valid_clusters
    train_mask = ~positives["cluster_a"].isin(excluded) & ~positives["cluster_b"].isin(excluded)

    test_df = positives[test_mask].reset_index(drop=True)
    valid_df = positives[valid_mask].reset_index(drop=True)
    train_df = positives[train_mask].reset_index(drop=True)

    print(f"Split sizes  train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
    print(f"Test interspecies positives: {test_df['is_interspecies'].sum()}")
    print(f"Valid interspecies positives: {valid_df['is_interspecies'].sum()}")

    # Verify protein disjointness
    train_prots = set(train_df["IdA"]) | set(train_df["IdB"])
    valid_prots = set(valid_df["IdA"]) | set(valid_df["IdB"])
    test_prots = set(test_df["IdA"]) | set(test_df["IdB"])
    assert len(train_prots & test_prots) == 0, "Train/test protein overlap"
    assert len(train_prots & valid_prots) == 0, "Train/valid protein overlap"
    assert len(valid_prots & test_prots) == 0, "Valid/test protein overlap"

    # Verify cluster disjointness
    train_clusters = set(train_df["cluster_a"]) | set(train_df["cluster_b"])
    valid_clusters_set = set(valid_df["cluster_a"]) | set(valid_df["cluster_b"])
    test_clusters_set = set(test_df["cluster_a"]) | set(test_df["cluster_b"])
    assert len(train_clusters & test_clusters_set) == 0, "Train/test cluster overlap"
    assert len(train_clusters & valid_clusters_set) == 0, "Train/valid cluster overlap"
    assert len(valid_clusters_set & test_clusters_set) == 0, "Valid/test cluster overlap"
    print("Disjointness verified (proteins and clusters)")

    return train_df, valid_df, test_df


def add_negatives(
    train_pos: pd.DataFrame,
    valid_pos: pd.DataFrame,
    test_pos: pd.DataFrame,
    interaction_set: set[str],
    matching_orgs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    policy_name = "SS" if matching_orgs else "NS"
    print(f"\nGenerating negatives for {policy_name} (matching_orgs={matching_orgs})")

    train_neg = generate_negative_ppis(train_pos, interaction_set, matching_orgs=matching_orgs, is_test=False, seed=44)
    valid_neg = generate_negative_ppis(valid_pos, interaction_set, matching_orgs=matching_orgs, is_test=False, seed=43)
    # Test always uses matching_orgs=True
    test_neg = generate_negative_ppis(test_pos, interaction_set, matching_orgs=True, is_test=True, seed=42)

    # Propagate is_interspecies for negatives
    train_neg["is_interspecies"] = train_neg["OrgA"] != train_neg["OrgB"]
    valid_neg["is_interspecies"] = valid_neg["OrgA"] != valid_neg["OrgB"]
    test_neg["is_interspecies"] = test_neg["OrgA"] != test_neg["OrgB"]

    train_df = pd.concat([train_pos, train_neg], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    valid_df = pd.concat([valid_pos, valid_neg], ignore_index=True).sample(frac=1, random_state=43).reset_index(drop=True)
    test_df = pd.concat([test_pos, test_neg], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    for name, df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        print(f"  {name}: {len(df)} rows  pos={int((df['labels'] == 1).sum())}  neg={int((df['labels'] == 0).sum())}")

    return train_df, valid_df, test_df


def embed_sequences(
    seq_dict: dict[str, str],
    plm_path: str,
    max_length: int,
    batch_size: int = 16,
) -> dict[str, torch.Tensor]:
    save_path = "esmc600m_biogrid_mv_embeddings.pth"
    if os.path.exists(save_path):
        print(f"Loading cached embeddings from {save_path}")
        return torch.load(save_path, weights_only=False)

    plm = AutoModel.from_pretrained(plm_path, trust_remote_code=True, dtype=torch.float32).cuda().eval()
    all_seqs = list(set(seq_dict.values()))
    embed_dict = plm.embed_dataset(
        sequences=all_seqs,
        tokenizer=plm.tokenizer,
        batch_size=batch_size,
        max_len=max_length,
        full_embeddings=True,
        embed_dtype=torch.float32,
        num_workers=4,
        sql=False,
        save=True,
        save_path=save_path,
    )
    plm.cpu()
    del plm
    torch.cuda.empty_cache()
    return embed_dict


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


def train_model(
    model: PPIModel,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    seq_dict: dict[str, str],
    embed_dict: dict[str, torch.Tensor],
    args: argparse.Namespace,
    save_dir: str,
) -> PPIModel:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    collator = BiogridCollator(
        embed_dim=1152,
        max_length=args.max_length,
        embedding_dict=embed_dict,
    )
    train_dataset = BiogridDataset(train_df, seq_dict, eval_mode=False)
    valid_dataset = BiogridDataset(valid_df, seq_dict, eval_mode=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    grad_clipper = AutoGradClipper(model)

    best_mcc = -1.0
    global_step = 0
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    for batch in tqdm(train_loader, desc=f"Training ({os.path.basename(save_dir)})"):
        a = batch["a"].to(device)
        b = batch["b"].to(device)
        a_mask = batch["a_mask"].to(device)
        b_mask = batch["b_mask"].to(device)
        labels = batch["labels"].to(device)

        output = model(a, b, a_mask, b_mask)
        loss = criterion(output.logits.squeeze(-1), labels)

        optimizer.zero_grad()
        loss.backward()
        grad_clipper.clip_gradients()
        optimizer.step()

        global_step += 1

        if global_step % 100 == 0:
            if WANDB_AVAILABLE:
                wandb.log({"train/loss": loss.item()}, step=global_step)

        if global_step % 2000 == 0:
            val_mcc = evaluate_mcc(model, valid_loader, device)
            print(f"  Step {global_step}  loss={loss.item():.4f}  val_mcc={val_mcc:.4f}")
            if WANDB_AVAILABLE:
                wandb.log({"valid/mcc": val_mcc}, step=global_step)
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
                print(f"  New best MCC: {best_mcc:.4f}")
            model.train()

    # Final eval
    val_mcc = evaluate_mcc(model, valid_loader, device)
    if val_mcc > best_mcc:
        best_mcc = val_mcc
        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
    print(f"Training complete. Best val MCC: {best_mcc:.4f}")

    # Load best checkpoint
    best_path = os.path.join(save_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))
    return model


@torch.no_grad()
def evaluate_mcc(model: PPIModel, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_logits, all_labels = [], []
    for batch in loader:
        a = batch["a"].to(device)
        b = batch["b"].to(device)
        a_mask = batch["a_mask"].to(device)
        b_mask = batch["b_mask"].to(device)
        labels = batch["labels"]

        output = model(a, b, a_mask, b_mask)
        all_logits.append(output.logits.squeeze(-1).cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (logits > 0).astype(int)
    return float(matthews_corrcoef(labels, preds))


@torch.no_grad()
def full_evaluation(
    model: PPIModel,
    test_df: pd.DataFrame,
    seq_dict: dict[str, str],
    embed_dict: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    collator = BiogridCollator(
        embed_dim=1152,
        max_length=args.max_length,
        embedding_dict=embed_dict,
    )
    test_dataset = BiogridDataset(test_df, seq_dict, eval_mode=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)

    all_logits, all_labels = [], []
    for batch in test_loader:
        a = batch["a"].to(device)
        b = batch["b"].to(device)
        a_mask = batch["a_mask"].to(device)
        b_mask = batch["b_mask"].to(device)
        labels = batch["labels"]

        output = model(a, b, a_mask, b_mask)
        all_logits.append(output.logits.squeeze(-1).cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    is_interspecies = test_df["is_interspecies"].values

    assert len(logits) == len(labels) == len(is_interspecies)

    def _compute_metrics(idx: np.ndarray) -> dict[str, float]:
        lg = logits[idx]
        lb = labels[idx]
        probs = 1.0 / (1.0 + np.exp(-lg))
        preds = (lg > 0).astype(int)
        return {
            "n": len(idx),
            "accuracy": float(accuracy_score(lb, preds)),
            "f1": float(f1_score(lb, preds, zero_division=0)),
            "mcc": float(matthews_corrcoef(lb, preds)),
            "auroc": float(roc_auc_score(lb, probs)) if len(np.unique(lb)) > 1 else float("nan"),
            "pr_auc": float(average_precision_score(lb, probs)) if len(np.unique(lb)) > 1 else float("nan"),
        }

    all_idx = np.arange(len(logits))
    results = {"all": _compute_metrics(all_idx)}

    pos_mask = labels == 1
    neg_mask = labels == 0

    # Interspecies subgroup
    inter_pos = pos_mask & is_interspecies
    n_inter = int(inter_pos.sum())
    if n_inter > 0:
        neg_pool = np.where(neg_mask)[0]
        rng = np.random.RandomState(42)
        neg_sample = rng.choice(neg_pool, size=min(n_inter, len(neg_pool)), replace=False)
        inter_idx = np.concatenate([np.where(inter_pos)[0], neg_sample])
        results["interspecies"] = _compute_metrics(inter_idx)
    else:
        results["interspecies"] = {k: float("nan") for k in ["n", "accuracy", "f1", "mcc", "auroc", "pr_auc"]}

    # Intraspecies subgroup
    intra_pos = pos_mask & ~is_interspecies
    n_intra = int(intra_pos.sum())
    if n_intra > 0:
        neg_pool = np.where(neg_mask)[0]
        rng = np.random.RandomState(43)
        neg_sample = rng.choice(neg_pool, size=min(n_intra, len(neg_pool)), replace=False)
        intra_idx = np.concatenate([np.where(intra_pos)[0], neg_sample])
        results["intraspecies"] = _compute_metrics(intra_idx)
    else:
        results["intraspecies"] = {k: float("nan") for k in ["n", "accuracy", "f1", "mcc", "auroc", "pr_auc"]}

    return results


def main():
    args = parse_args()
    set_seed(42)

    if args.hf_token is not None:
        from huggingface_hub import login as hf_login
        hf_login(args.hf_token)

    if args.wandb_token is not None and WANDB_AVAILABLE:
        wandb.login(key=args.wandb_token)

    results_dir = "results/interspecies_experiment"
    os.makedirs(results_dir, exist_ok=True)

    cache_path = os.path.join(results_dir, "cached_splits.pkl")
    if os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        seq_dict = cached["seq_dict"]
        ss_train = cached["ss_train"]
        ss_valid = cached["ss_valid"]
        ss_test = cached["ss_test"]
        ns_train = cached["ns_train"]
        ns_valid = cached["ns_valid"]
        ns_test = cached["ns_test"]
        print(f"  SS train={len(ss_train)}  NS train={len(ns_train)}  test={len(ss_test)}")
    else:
        # 1. Load data
        positives, seq_dict, interaction_set = load_biogrid_mv(args.bugfix)

        # 2. Cluster
        cluster_dict = cluster_sequences(
            seq_dict,
            method="mmseqs2",
            cluster_percentage=0.4,
            coverage=0.8,
            identifier="biogrid_mv",
            base_path="data",
        )
        print(f"Clustering: {len(seq_dict)} sequences -> {len(cluster_dict)} clusters")

        # 3. Interspecies-aware C3 split
        train_pos, valid_pos, test_pos = interspecies_aware_split(
            positives, cluster_dict, args.min_test_rows, args.min_interspecies,
        )

        # 4. Generate negatives for SS and NS
        ss_train, ss_valid, ss_test = add_negatives(train_pos, valid_pos, test_pos, interaction_set, matching_orgs=True)
        ns_train, ns_valid, ns_test = add_negatives(train_pos, valid_pos, test_pos, interaction_set, matching_orgs=False)

        # Shared test set: test positives are the same, test negatives use matching_orgs=True for both
        assert ss_test["IdA"].tolist() == ns_test["IdA"].tolist(), "Test sets should be identical"

        # Cache processed splits
        with open(cache_path, "wb") as f:
            pickle.dump({
                "seq_dict": seq_dict,
                "ss_train": ss_train, "ss_valid": ss_valid, "ss_test": ss_test,
                "ns_train": ns_train, "ns_valid": ns_valid, "ns_test": ns_test,
            }, f)
        print(f"Cached dataset to {cache_path}")

    # 5. Embed
    embed_dict = embed_sequences(seq_dict, args.plm_path, args.max_length)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 6. Train SS model
    print("\n=== Training SS model ===")
    if WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name="interspecies_SS", config=vars(args))
    set_seed(42)
    ss_model = build_model(args)
    ss_model = train_model(ss_model, ss_train, ss_valid, seq_dict, embed_dict, args, os.path.join(results_dir, "ss_model"))
    if WANDB_AVAILABLE:
        wandb.finish()

    # 7. Train NS model
    print("\n=== Training NS model ===")
    if WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name="interspecies_NS", config=vars(args))
    set_seed(42)
    ns_model = build_model(args)
    ns_model = train_model(ns_model, ns_train, ns_valid, seq_dict, embed_dict, args, os.path.join(results_dir, "ns_model"))

    # 8. Evaluate both on shared test set
    print("\n=== Evaluation ===")
    ss_results = full_evaluation(ss_model, ss_test, seq_dict, embed_dict, args, device)
    ns_results = full_evaluation(ns_model, ns_test, seq_dict, embed_dict, args, device)
    if WANDB_AVAILABLE:
        wandb.finish()

    # Log final results as a summary run
    if WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, name="interspecies_results", config=vars(args))
        for model_name, res in [("SS", ss_results), ("NS", ns_results)]:
            for subgroup in ["all", "interspecies", "intraspecies"]:
                r = res[subgroup]
                for metric_name, metric_val in r.items():
                    if metric_name == "n":
                        continue
                    wandb.log({f"test/{model_name}/{subgroup}/{metric_name}": metric_val})
        wandb.finish()

    # 9. Report
    rows = []
    for subgroup in ["all", "interspecies", "intraspecies"]:
        for model_name, res in [("SS", ss_results), ("NS", ns_results)]:
            r = res[subgroup]
            rows.append({
                "model": model_name,
                "subgroup": subgroup,
                "n": r["n"],
                "accuracy": r["accuracy"],
                "f1": r["f1"],
                "mcc": r["mcc"],
                "auroc": r["auroc"],
                "pr_auc": r["pr_auc"],
            })

    results_df = pd.DataFrame(rows)
    print("\n" + results_df.to_string(index=False))

    csv_path = os.path.join(results_dir, "interspecies_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()

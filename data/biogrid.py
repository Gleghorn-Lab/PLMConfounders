import pandas as pd
import numpy as np
import torch
import random
import subprocess
import os
import psutil
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import defaultdict
from tqdm.auto import tqdm
from scipy.stats import chi2_contingency, entropy


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def generate_negative_ppis(
        df: pd.DataFrame,
        interaction_set: set,
        matching_orgs: bool,
        is_test: bool = False,
        seed: int = 42,
) -> pd.DataFrame:
    set_seed(seed)

    all_orgs = list(set(df['OrgA'].unique()).union(set(df['OrgB'].unique())))
    print(f'All orgs: {all_orgs[:10]}')

    org_counts_a = dict(df['OrgA'].value_counts())
    org_counts_b = dict(df['OrgB'].value_counts())

    print(f'Org counts a: {org_counts_a}')
    print(f'Org counts b: {org_counts_b}')

    total = len(df)
    # Each row contributes two organism observations (OrgA and OrgB)
    # so normalize by the total number of endpoints rather than rows
    total_endpoints = sum(org_counts_a.values()) + sum(org_counts_b.values())
    org_probs = {
        org: (org_counts_a.get(org, 0) + org_counts_b.get(org, 0)) / total_endpoints
        for org in all_orgs
    }
    org_weights = list(org_probs.values())
    print('Org probs:')
    for i, (k, v) in enumerate(org_probs.items()):
        print(f'{k}: {v:.4f}')
        if i > 10:
            break
    
    org_id_dict = {org: set() for org in all_orgs}
    for ida, idb, org_a, org_b in tqdm(zip(df['IdA'], df['IdB'], df['OrgA'], df['OrgB']), desc="Building org seq dict", total=total):
        org_id_dict[org_a].add(ida)
        org_id_dict[org_b].add(idb)

    assert np.isclose(sum(org_probs.values()), 1.0), "Org probs do not sum to 1"

    match_orgs = matching_orgs or is_test

    # Keep a set for uniqueness checks, but maintain ordered lists so that
    # IdA/IdB stay aligned with OrgA/OrgB for each generated row
    negatives_set = set()
    raw_ids_a, raw_ids_b = [], []
    orgs_a, orgs_b = [], []

    attempts = 0
    max_attempts = 100 * total
    pbar = tqdm(desc="Generating negative samples", total=len(df))
    while len(raw_ids_a) < len(df) and attempts < max_attempts:
        attempts += 1
        org_1 = random.choices(all_orgs, weights=org_weights, k=1)[0]
        if match_orgs:
            org_2 = org_1
        else:
            while True:
                org_2 = random.choices(all_orgs, weights=org_weights, k=1)[0]
                if org_2 != org_1:
                    break

        id_1 = random.choice(list(org_id_dict[org_1]))
        id_2 = random.choice(list(org_id_dict[org_2]))

        pair_id = '_'.join(sorted([id_1, id_2]))

        if pair_id not in interaction_set and pair_id not in negatives_set:
            negatives_set.add(pair_id)
            raw_ids_a.append(id_1)
            raw_ids_b.append(id_2)
            orgs_a.append(org_1)
            orgs_b.append(org_2)
        else:
            continue
        pbar.update(1)
        pbar.set_postfix(neg_size=len(negatives_set))
    pbar.close()

    print(f'Generated {len(raw_ids_a)} negative samples')
    labels = [0] * len(raw_ids_a)

    # Randomly swap A/B per row consistently for IDs and organism labels
    neg_id_a, neg_id_b, out_org_a, out_org_b = [], [], [], []
    for i in range(len(raw_ids_a)):
        a_id, b_id = raw_ids_a[i], raw_ids_b[i]
        a_org, b_org = orgs_a[i], orgs_b[i]
        if random.random() < 0.5:
            a_id, b_id = b_id, a_id
            a_org, b_org = b_org, a_org
        neg_id_a.append(a_id)
        neg_id_b.append(b_id)
        out_org_a.append(a_org)
        out_org_b.append(b_org)

    return pd.DataFrame(
        list(zip(neg_id_a, neg_id_b, labels, out_org_a, out_org_b)),
        columns=['IdA', 'IdB', 'labels', 'OrgA', 'OrgB']
    )


def split_with_sim(
        df: pd.DataFrame,
        seq_dict: dict,
        similarity_threshold: float = 0.5,
        min_rows: int = 1000, # number of test clusters
        n: int = 3, # word size, 5 is faster but 3 is more sensitive
        memory_percentage: float = 0.5,
        save: bool = False,
        matching_orgs: bool = False,
        interaction_set: set = None,
        cache_suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(42)
    
    species_id = 'biogrid'
    train_file = f"processed_datasets/split_with_sim_{species_id}_{similarity_threshold}_{matching_orgs}{cache_suffix}_train.csv"
    test_file = f"processed_datasets/split_with_sim_{species_id}_{similarity_threshold}_{matching_orgs}{cache_suffix}_test.csv"
    val_file = f"processed_datasets/split_with_sim_{species_id}_{similarity_threshold}_{matching_orgs}{cache_suffix}_val.csv"

    # Write all_seqs to a FASTA file
    base_path = 'sequence_data'
    fasta_path = f"{base_path}/{species_id}{cache_suffix}.fasta"
    with open(fasta_path, "w") as f:
        for id, seq in seq_dict.items():
            f.write(f">{id}\n{seq}\n")

    # Run cd-hit in Docker
    output_path = f"{base_path}/output_{species_id}_{similarity_threshold}{cache_suffix}"
    cluster_file = f"{base_path}/output_{species_id}_{similarity_threshold}{cache_suffix}.clstr"

    if os.path.exists(output_path):
        print(f'Output file {output_path} already exists')
    else:
        # Build the cd-hit Docker image if not already built
        num_cpu = os.cpu_count() - 4 if os.cpu_count() > 4 else 1
        memory_max = int(memory_percentage * psutil.virtual_memory().total / 1024 / 1024)  # in MB
        print(f'Using {num_cpu} CPUs and {memory_max} MB memory')

        print("Building cd-hit Docker image...")
        docker_image = "cd-hit"
        dockerfile_url = "https://raw.githubusercontent.com/weizhongli/cdhit/master/Docker/Dockerfile"
        # Build the Docker image
        try:
            subprocess.run([
                "docker", "build", "--tag", docker_image, dockerfile_url
            ], check=True)

            print(f'Clustering {len(seq_dict)} sequences')
            subprocess.run([
                "docker", "run",
                "-v", f"{os.getcwd()}:/data",
                "-w", "/data",
                docker_image,
                "cd-hit",
                "-i", fasta_path,
                "-o", output_path,
                "-d", "0",
                "-c", str(similarity_threshold),
                "-n", str(n),
                "-T", str(num_cpu),
                "-M", str(memory_max)
            ], check=True)
        except:
            subprocess.run([
                "sudo", "docker", "build", "--tag", docker_image, dockerfile_url
            ], check=True)

            print(f'Clustering {len(seq_dict)} sequences')
            subprocess.run([
                "sudo", "docker", "run",
                "-v", f"{os.getcwd()}:/data",
                "-w", "/data",
                docker_image,
                "cd-hit",
                "-i", fasta_path,
                "-o", output_path,
                "-d", "0",
                "-c", str(similarity_threshold),
                "-n", str(n),
                "-T", str(num_cpu),
                "-M", str(memory_max)
            ], check=True)    

    # Read the output clusters file
    cluster_dict = defaultdict(list)
    with open(cluster_file, "r") as f:
        for line in tqdm(f, desc="Reading cluster file"):
            if line.startswith(">"):
                cluster_id = line.split('Cluster')[1].split("\n")[0].strip()
            else:
                seq_id = line.split('>')[1].split('...')[0].strip()
                cluster_dict[cluster_id].append(seq_id)

    print(f'Number of unique sequences: {len(seq_dict)}, Number of clusters: {len(cluster_dict)}, Number of rows: {len(df)}')

    # Build a mapping from sequence ID to cluster ID
    seq_id_to_cluster = {}
    for cluster_id, seq_ids in cluster_dict.items():
        for seq_id in seq_ids:
            seq_id_to_cluster[seq_id] = cluster_id

    print(f'Cluster ids: {list(cluster_dict.keys())[:10]}')

    # map each row to their clusters (cluster ID, not list)
    df['cluster_a'] = df['IdA'].map(seq_id_to_cluster)
    df['cluster_b'] = df['IdB'].map(seq_id_to_cluster)

    print(f'Cluster a: {df["cluster_a"].head()}')
    print(f'Cluster b: {df["cluster_b"].head()}')

    # calculate the impact of each cluster on the dataset (vectorized)
    a_counts = df['cluster_a'].value_counts()
    b_counts = df['cluster_b'].value_counts()
    all_clusters = set(a_counts.index).union(b_counts.index)
    cluster_impact = {cluster: a_counts.get(cluster, 0) + b_counts.get(cluster, 0) for cluster in all_clusters}

    # sort the clusters by impact, not actually used anymore
    sorted_clusters = sorted(cluster_impact.items(), key=lambda x: x[1], reverse=True)

    test_clusters = set()
    valid_clusters = set()
    test_mask = np.zeros(len(df), dtype=bool)
    valid_mask = np.zeros(len(df), dtype=bool)
    
    # Randomly shuffle the bottom half clusters
    np.random.shuffle(sorted_clusters)
    
    # Randomly select test clusters from bottom half
    pbar = tqdm(desc="Selecting test clusters", total=len(sorted_clusters))
    for cluster, _ in sorted_clusters:
        test_clusters.add(cluster)
        # Update mask: both cluster_a and cluster_b must be in test_clusters
        new_mask = df["cluster_a"].isin(test_clusters) & df["cluster_b"].isin(test_clusters)
        if new_mask.sum() >= min_rows:
            test_mask = new_mask
            break
        pbar.update(1)
        pbar.set_postfix(test_size=new_mask.sum())
    pbar.close()

    test_df = df[test_mask]
    if len(test_df) == 0:
        raise ValueError("No test rows found")
    else:
        print(f'Test set size: {len(test_df)}')

    print(f'Used {len(test_clusters)} test clusters')

    # Continue selecting validation clusters (same target size as test)
    target_valid_size = len(test_df)
    remaining_clusters = [item for item in sorted_clusters if item[0] not in test_clusters]
    
    pbar = tqdm(desc="Selecting validation clusters", total=len(remaining_clusters))
    for cluster, _ in remaining_clusters:
        valid_clusters.add(cluster)
        # Update mask: both cluster_a and cluster_b must be in val_clusters
        new_mask = df["cluster_a"].isin(valid_clusters) & df["cluster_b"].isin(valid_clusters)
        if new_mask.sum() >= target_valid_size:
            valid_mask = new_mask
            break
        pbar.update(1)
        pbar.set_postfix(valid_size=new_mask.sum(), target=target_valid_size)
    pbar.close()

    valid_df = df[valid_mask]
    if len(valid_df) == 0:
        raise ValueError("No validation rows found")
    else:
        print(f'Validation set size: {len(valid_df)}')

    print(f'Used {len(valid_clusters)} validation clusters')

    before_len = len(df)
    print(f'Train before trimming: {before_len}')

    # remove rows that have any protein in test or validation clusters
    all_excluded_clusters = test_clusters.union(valid_clusters)
    train_df = df[~df["cluster_a"].isin(all_excluded_clusters) & ~df["cluster_b"].isin(all_excluded_clusters)]
    after_len_cluster_trim = len(train_df)

    print(f'Train after trimming: {after_len_cluster_trim}')

    print(f'Test set size: {len(test_df)}')
    print(f'Validation set size: {len(valid_df)}')
    trimmed_len = before_len - after_len_cluster_trim
    print(f'Trimmed {trimmed_len} rows')

    if trimmed_len == 0:
        raise ValueError("No rows were trimmed")
    elif trimmed_len == len(test_df) + len(valid_df):
        print("Warning: ONLY test and validation rows were removed from the training set. We usually expect more training rows than test+val rows to be removed.")

    # Verify disjoint sets
    # For proteins
    test_proteins = set(test_df['IdA']).union(set(test_df['IdB']))
    valid_proteins = set(valid_df['IdA']).union(set(valid_df['IdB']))
    train_proteins = set(train_df['IdA']).union(set(train_df['IdB']))

    print(f'Test proteins: {len(test_proteins)}, Valid proteins: {len(valid_proteins)}, Train proteins: {len(train_proteins)}')

    # Check for overlaps (proteins)
    test_valid_overlap = test_proteins.intersection(valid_proteins)
    test_train_overlap = test_proteins.intersection(train_proteins)
    valid_train_overlap = valid_proteins.intersection(train_proteins)

    if test_valid_overlap:
        print(f'WARNING: {len(test_valid_overlap)} proteins overlap between test and validation')
    if test_train_overlap:
        print(f'WARNING: {len(test_train_overlap)} proteins overlap between test and train')
    if valid_train_overlap:
        print(f'WARNING: {len(valid_train_overlap)} proteins overlap between validation and train')

    if not test_valid_overlap and not test_train_overlap and not valid_train_overlap:
        print('SUCCESS: All three protein sets are completely disjoint!')

    # For clusters
    test_clusters_set = set(test_df['cluster_a']).union(set(test_df['cluster_b']))
    valid_clusters_set = set(valid_df['cluster_a']).union(set(valid_df['cluster_b']))
    train_clusters_set = set(train_df['cluster_a']).union(set(train_df['cluster_b']))

    print(f'Test clusters: {len(test_clusters_set)}, Valid clusters: {len(valid_clusters_set)}, Train clusters: {len(train_clusters_set)}')

    # Check for overlaps (clusters)
    test_valid_cluster_overlap = test_clusters_set.intersection(valid_clusters_set)
    test_train_cluster_overlap = test_clusters_set.intersection(train_clusters_set)
    valid_train_cluster_overlap = valid_clusters_set.intersection(train_clusters_set)

    if test_valid_cluster_overlap:
        print(f'WARNING: {len(test_valid_cluster_overlap)} clusters overlap between test and validation')
    if test_train_cluster_overlap:
        print(f'WARNING: {len(test_train_cluster_overlap)} clusters overlap between test and train')
    if valid_train_cluster_overlap:
        print(f'WARNING: {len(valid_train_cluster_overlap)} clusters overlap between validation and train')

    if not test_valid_cluster_overlap and not test_train_cluster_overlap and not valid_train_cluster_overlap:
        print('SUCCESS: All three cluster sets are completely disjoint!')

    # Generate negative samples for each split
    if interaction_set is not None:
        print("\nGenerating negative samples...")
        
        # Test set: always use matching orgs
        test_negatives = generate_negative_ppis(
            test_df, interaction_set, matching_orgs=True, is_test=True, seed=42
        )
        test_df = pd.concat([test_df, test_negatives], ignore_index=True)
        
        # Validation set: follow matching_orgs parameter
        valid_negatives = generate_negative_ppis(
            valid_df, interaction_set, matching_orgs=matching_orgs, is_test=False, seed=43
        )
        valid_df = pd.concat([valid_df, valid_negatives], ignore_index=True)
        
        # Training set: follow matching_orgs parameter
        train_negatives = generate_negative_ppis(
            train_df, interaction_set, matching_orgs=matching_orgs, is_test=False, seed=44
        )
        train_df = pd.concat([train_df, train_negatives], ignore_index=True)
        
        print(f"\nFinal dataset sizes:")
        print(f"Train: {len(train_df)} (pos: {(train_df['labels'] > 0).sum()}, neg: {(train_df['labels'] == 0).sum()})")
        print(f"Valid: {len(valid_df)} (pos: {(valid_df['labels'] > 0).sum()}, neg: {(valid_df['labels'] == 0).sum()})")
        print(f"Test: {len(test_df)} (pos: {(test_df['labels'] > 0).sum()}, neg: {(test_df['labels'] == 0).sum()})")
    
    # shuffle all three sets
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    test_df = test_df.sample(frac=1).reset_index(drop=True)
    valid_df = valid_df.sample(frac=1).reset_index(drop=True)

    if save:
        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)
        valid_df.to_csv(val_file, index=False)
    return train_df, valid_df, test_df


def get_biogrid_data(
        similarity_threshold: float = 0.5,
        min_rows: int = 1000,
        n: int = 3,
        save: bool = False,
        matching_orgs: bool = False,
        sample_rows: int = None,
        rebuild_negatives: bool = False,
):
    cache_suffix = ""
    if sample_rows is not None:
        cache_suffix = f"_sample{sample_rows}"

    data = load_dataset('Synthyra/BIOGRID', split='train').shuffle(seed=42)

    if sample_rows is not None:
        data = data.select(range(sample_rows))

    id_a = list(data['A'])
    id_b = list(data['B'])
    org_a = list(data['OrgA'])
    org_b = list(data['OrgB'])
    labels = [1] * len(id_a)

    # build set of all interactions
    interaction_set = set('_'.join(sorted([ida, idb])) for ida, idb in zip(id_a, id_b))
    print(f'Interaction set size: {len(interaction_set)}')

    # Use pandas for vectorized assignment for faster seq_dict creation
    df = pd.DataFrame(data)
    seq_dict = pd.Series(df['SeqA'].values, index=df['A']).to_dict()
    seq_dict.update(pd.Series(df['SeqB'].values, index=df['B']).to_dict())

    link_df = pd.DataFrame({'IdA': id_a, 'IdB': id_b, 'labels': labels, 'OrgA': org_a, 'OrgB': org_b})

    os.makedirs("processed_datasets", exist_ok=True)
    species_id = 'biogrid'
    train_file = f"processed_datasets/split_with_sim_{species_id}_{similarity_threshold}_{matching_orgs}{cache_suffix}_train.csv"
    test_file = f"processed_datasets/split_with_sim_{species_id}_{similarity_threshold}_{matching_orgs}{cache_suffix}_test.csv"
    val_file = f"processed_datasets/split_with_sim_{species_id}_{similarity_threshold}_{matching_orgs}{cache_suffix}_val.csv"

    if os.path.exists(train_file) and os.path.exists(test_file) and os.path.exists(val_file):
        print(f'Loading cached train, test, and validation sets from {train_file}, {test_file}, and {val_file}')
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        valid_df = pd.read_csv(val_file)
        rebuild_negatives = False
    else:
        print(f'No cached files found, building dataset')
        train_df, valid_df, test_df = split_with_sim(
            df=link_df,
            seq_dict=seq_dict,
            similarity_threshold=similarity_threshold,
            min_rows=min_rows,
            n=n,
            save=save,
            matching_orgs=matching_orgs,
            interaction_set=interaction_set,
            cache_suffix=cache_suffix,
        )

    # Ensure negatives adhere to the current policy even when loading cached splits
    def _rebuild_with_policy(split_df: pd.DataFrame, must_match: bool, is_test_split: bool, seed_val: int) -> pd.DataFrame:
        positives_only = split_df[split_df['labels'] > 0][['IdA', 'IdB', 'labels', 'OrgA', 'OrgB']].copy()
        regenerated_negs = generate_negative_ppis(
            positives_only,
            interaction_set=interaction_set,
            matching_orgs=must_match,
            is_test=is_test_split,
            seed=seed_val,
        )
        combined = pd.concat([positives_only, regenerated_negs], ignore_index=True)
        combined = combined.sample(frac=1, random_state=seed_val).reset_index(drop=True)
        return combined

    # Optionally rebuild negatives to avoid inconsistencies in cached files
    if rebuild_negatives:
        test_df = _rebuild_with_policy(test_df, must_match=True, is_test_split=True, seed_val=42)
        valid_df = _rebuild_with_policy(valid_df, must_match=matching_orgs, is_test_split=False, seed_val=43)
        train_df = _rebuild_with_policy(train_df, must_match=matching_orgs, is_test_split=False, seed_val=44)

    # =========================
    # Validation checks & plots
    # =========================
    def _assert_neg_match_rule(split_df: pd.DataFrame, expect_match: bool, split_name: str) -> None:
        neg_mask = split_df['labels'] == 0
        if neg_mask.sum() == 0:
            print(f"WARNING: No negatives found in {split_name} for rule check.")
            return
        org_equal = (split_df.loc[neg_mask, 'OrgA'] == split_df.loc[neg_mask, 'OrgB'])
        if expect_match:
            assert bool(org_equal.all()), f"Negatives in {split_name} must have matching organisms."
            print(f"OK: {split_name} negatives have matching organisms as expected.")
        else:
            assert bool((~org_equal).all()), f"Negatives in {split_name} must have non-matching organisms."
            print(f"OK: {split_name} negatives have non-matching organisms as expected.")

    # Test set: always matching orgs per generation logic
    _assert_neg_match_rule(test_df, expect_match=True, split_name='test')
    # Valid/train: follow matching_orgs parameter
    _assert_neg_match_rule(valid_df, expect_match=matching_orgs, split_name='valid')
    _assert_neg_match_rule(train_df, expect_match=matching_orgs, split_name='train')

    # Plot organism distribution comparison (positives vs negatives)
    def _plot_org_distributions(
            split_df: pd.DataFrame,
            split_name: str,
            out_dir: str,
            top_k: int = 15,
    ) -> None:
        os.makedirs(out_dir, exist_ok=True)
        pos_mask = split_df['labels'] > 0
        neg_mask = split_df['labels'] == 0
        # Aggregate counts across OrgA and OrgB
        pos_orgs = pd.concat([split_df.loc[pos_mask, 'OrgA'], split_df.loc[pos_mask, 'OrgB']])
        neg_orgs = pd.concat([split_df.loc[neg_mask, 'OrgA'], split_df.loc[neg_mask, 'OrgB']])
        pos_counts = pos_orgs.value_counts().sort_index()
        neg_counts = neg_orgs.value_counts().sort_index()
        # Align indices
        all_orgs = sorted(set(pos_counts.index).union(set(neg_counts.index)))
        pos_vals = np.array([pos_counts.get(org, 0) for org in all_orgs], dtype=float)
        neg_vals = np.array([neg_counts.get(org, 0) for org in all_orgs], dtype=float)

        # Statistical comparisons (chi-square on counts; JS and KL on distributions)
        table = np.vstack([pos_vals, neg_vals])
        try:
            chi2, p_value, dof, _ = chi2_contingency(table)
        except Exception:
            chi2, p_value, dof = np.nan, np.nan, np.nan
        eps = 1e-12
        pos_total_c = pos_vals.sum()
        neg_total_c = neg_vals.sum()
        pos_dist_full = (pos_vals / pos_total_c) if pos_total_c > 0 else np.zeros_like(pos_vals)
        neg_dist_full = (neg_vals / neg_total_c) if neg_total_c > 0 else np.zeros_like(neg_vals)
        m_dist = 0.5 * (pos_dist_full + neg_dist_full)
        try:
            kl_pos_neg = float(entropy(pos_dist_full + eps, neg_dist_full + eps))
            js_div = 0.5 * float(
                entropy(pos_dist_full + eps, m_dist + eps) +
                entropy(neg_dist_full + eps, m_dist + eps)
            )
        except Exception:
            kl_pos_neg, js_div = np.nan, np.nan
        # Normalize to distributions
        pos_total = pos_vals.sum()
        neg_total = neg_vals.sum()
        pos_dist = (pos_vals / pos_total) if pos_total > 0 else np.zeros_like(pos_vals)
        neg_dist = (neg_vals / neg_total) if neg_total > 0 else np.zeros_like(neg_vals)

        # Rank organisms by combined prevalence and keep top_k; group the rest as 'Other'
        combined = pos_dist + neg_dist
        # Indices of organisms sorted by combined prevalence descending
        sorted_idx = np.argsort(-combined)
        if top_k is None or top_k <= 0 or top_k >= len(all_orgs):
            top_idx = sorted_idx
            include_other = False
        else:
            top_idx = sorted_idx[:top_k]
            include_other = len(all_orgs) > top_k

        top_orgs = [all_orgs[i] for i in top_idx]
        top_pos = pos_dist[top_idx]
        top_neg = neg_dist[top_idx]

        if include_other:
            other_pos = float(max(0.0, 1.0 - top_pos.sum()))
            other_neg = float(max(0.0, 1.0 - top_neg.sum()))
            categories = [str(org) for org in top_orgs] + ["Other"]
            pos_plot = list(top_pos) + [other_pos]
            neg_plot = list(top_neg) + [other_neg]
        else:
            categories = [str(org) for org in top_orgs]
            pos_plot = list(top_pos)
            neg_plot = list(top_neg)

        # Horizontal bar chart with side-by-side bars for readability
        y = np.arange(len(categories))
        height = max(6, 0.35 * len(categories) + 2)
        bar_height = 0.4
        plt.figure(figsize=(10, height))
        plt.barh(y - bar_height/2, pos_plot, height=bar_height, label='Positives')
        plt.barh(y + bar_height/2, neg_plot, height=bar_height, label='Negatives')
        plt.xlabel('Proportion')
        plt.ylabel('Organism')
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()  # Highest at the top
        plt.yticks(y, categories)
        plt.legend(loc='best')
        stats_note = f"chi2={chi2:.2f}, p={p_value:.1e}, JS={js_div:.3f}, KL(P||N)={kl_pos_neg:.3f}"
        plt.title(
            f'Organism distribution (Top {min(top_k, len(all_orgs))}{" + Other" if include_other else ""}): {split_name}\n{stats_note}'
        )
        plt.tight_layout()
        outfile = os.path.join(out_dir, f'{split_name}_organism_distribution.png')
        plt.savefig(outfile, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved organism distribution plot for {split_name} to {outfile}")

        # Save detailed counts/proportions and stats
        out_csv = os.path.join(out_dir, f'{split_name}_organism_distribution.csv')
        df_out = pd.DataFrame({
            'organism': [str(org) for org in all_orgs],
            'pos_count': pos_vals.astype(int),
            'neg_count': neg_vals.astype(int),
            'pos_prop': pos_dist_full,
            'neg_prop': neg_dist_full,
        })
        df_out.to_csv(out_csv, index=False)
        out_stats_csv = os.path.join(out_dir, f'{split_name}_organism_distribution_stats.csv')
        pd.DataFrame([
            {'stat': 'chi2', 'value': chi2},
            {'stat': 'p_value', 'value': p_value},
            {'stat': 'dof', 'value': dof},
            {'stat': 'js_divergence', 'value': js_div},
            {'stat': 'kl_pos_to_neg', 'value': kl_pos_neg},
        ]).to_csv(out_stats_csv, index=False)

    os.makedirs('plots', exist_ok=True)
    plot_dir = os.path.join('plots', 'biogrid_negatives_check')
    _plot_org_distributions(train_df, 'train', plot_dir)
    _plot_org_distributions(valid_df, 'valid', plot_dir)
    _plot_org_distributions(test_df, 'test', plot_dir)

    # Cross-split statistical comparisons (positives-only and negatives-only)
    def _cross_split_stats(split_dfs: dict[str, pd.DataFrame], label_selector: str, out_dir: str) -> None:
        masks = {name: (df['labels'] > 0) if label_selector == 'pos' else (df['labels'] == 0)
                 for name, df in split_dfs.items()}
        # Union of organisms across splits
        org_sets = []
        for name, df in split_dfs.items():
            mask = masks[name]
            orgs = pd.concat([df.loc[mask, 'OrgA'], df.loc[mask, 'OrgB']])
            org_sets.append(set(orgs.values))
        all_orgs = sorted(set().union(*org_sets))
        if len(all_orgs) == 0:
            print(f"No organisms found for cross-split {label_selector} comparison.")
            return

        counts = []
        split_names = []
        for name, df in split_dfs.items():
            mask = masks[name]
            orgs = pd.concat([df.loc[mask, 'OrgA'], df.loc[mask, 'OrgB']])
            vc = orgs.value_counts()
            row = np.array([vc.get(org, 0) for org in all_orgs], dtype=float)
            counts.append(row)
            split_names.append(name)
        table = np.vstack(counts)
        try:
            chi2, p_value, dof, _ = chi2_contingency(table)
        except Exception:
            chi2, p_value, dof = np.nan, np.nan, np.nan

        # Save counts and stats
        os.makedirs(out_dir, exist_ok=True)
        counts_csv = os.path.join(out_dir, f'cross_split_{label_selector}_counts.csv')
        pd.DataFrame(table, index=split_names, columns=[str(org) for org in all_orgs]).to_csv(counts_csv)
        stats_csv = os.path.join(out_dir, f'cross_split_{label_selector}_stats.csv')
        pd.DataFrame([
            {'stat': 'chi2', 'value': chi2},
            {'stat': 'p_value', 'value': p_value},
            {'stat': 'dof', 'value': dof},
            {'stat': 'num_organisms', 'value': len(all_orgs)},
        ]).to_csv(stats_csv, index=False)
        print(f"Saved cross-split {label_selector} stats to {stats_csv}")

    _cross_split_stats({'train': train_df, 'valid': valid_df, 'test': test_df}, 'pos', plot_dir)
    _cross_split_stats({'train': train_df, 'valid': valid_df, 'test': test_df}, 'neg', plot_dir)

    return train_df, valid_df, test_df, seq_dict, interaction_set


if __name__ == "__main__":
    # py -m data.biogrid
    ### If on windows, make sure docker desktop is running
    similarity_threshold = 0.95
    min_rows = 100
    n = 5
    matching_orgs = False
    print('Testing biogrid data...')
    train_df, valid_df, test_df, seq_dict, interaction_set = get_biogrid_data(
        similarity_threshold=similarity_threshold,
        min_rows=min_rows,
        n=n,
        matching_orgs=matching_orgs,
        #sample_rows=10000,
    )
    train_df.to_csv(f'train_df_{matching_orgs}.csv', index=False)
    valid_df.to_csv(f'valid_df_{matching_orgs}.csv', index=False)
    test_df.to_csv(f'test_df_{matching_orgs}.csv', index=False)
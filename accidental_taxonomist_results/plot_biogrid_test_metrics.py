import os
import re
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


SS_LABEL = 'SS'  # Strategic Sampling (matching = True)
NS_LABEL = 'NS'  # Normal Sampling (matching = False)

METRIC_DICT = {
    'balanced_accuracy': 'Accuracy',
    'balanced_recall': 'Recall',
    'balanced_precision': 'Precision',
    'balanced_f1': 'F1',
    'balanced_mcc': 'MCC',
    'balanced_roc_auc': 'ROC AUC',
    'balanced_pr_auc': 'PR AUC',
    'balanced_threshold': 'Threshold',
    'loss': 'Loss',
}


# Match headers and metrics anywhere on the line to allow timestamp prefixes
HEADER_RE = re.compile(r"===\s+(TEST|VALID)\s+METRICS\s+===")
METRIC_RE = re.compile(r"([A-Za-z0-9_\/]+):\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {'true', 't', '1', 'yes', 'y'}
    return False


def parse_metrics_log(path: str) -> Dict[str, List[Dict[str, float]]]:
    """Parse a metrics.log file and extract per-split metrics across entries.

    Returns a dict with lists of metric snapshots for 'TEST' and 'VALID'.
    We'll later use the last TEST snapshot for per-metric plotting.
    """
    results: Dict[str, List[Dict[str, float]]] = {"TEST": [], "VALID": []}
    if not os.path.exists(path):
        return results

    current_section: str = ''
    current_metrics: Dict[str, float] = {}
    with open(path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            header_match = HEADER_RE.search(line)
            if header_match:
                if current_section and current_metrics:
                    results[current_section].append(current_metrics)
                current_section = header_match.group(1)
                current_metrics = {}
                continue

            metric_match = METRIC_RE.search(line)
            if metric_match and current_section:
                key, val = metric_match.group(1), metric_match.group(2)
                try:
                    current_metrics[key] = float(val)
                except ValueError:
                    pass

    if current_section and current_metrics:
        results[current_section].append(current_metrics)

    return results


def discover_run_dirs(root: str, recursive: bool = True) -> List[str]:
    """Find run directories that look like biogrid_*_seed* and contain metrics.log.

    If recursive is True, walk subdirectories; otherwise only scan direct children.
    """
    dirs: List[str] = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            base = os.path.basename(dirpath)
            if base.startswith('biogrid_') and 'seed' in base and 'metrics.log' in filenames:
                dirs.append(dirpath)
    else:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if not os.path.isdir(full):
                continue
            if name.startswith('biogrid_') and 'seed' in name and os.path.exists(os.path.join(full, 'metrics.log')):
                dirs.append(full)
    return sorted(dirs)


def summarize_test_metrics(run_dir: str) -> Tuple[int, bool, Dict[str, float]]:
    """Return (seed, matching_orgs, last_TEST_metrics_dict) for a run directory."""
    log_path = os.path.join(run_dir, 'metrics.log')
    parsed = parse_metrics_log(log_path)

    base = os.path.basename(run_dir.rstrip(os.sep))
    matching_orgs = ('match' in base) and ('nomatch' not in base)
    seed = None
    m = re.search(r"seed(\d+)", base)
    if m:
        try:
            seed = int(m.group(1))
        except Exception:
            seed = None

    last_test: Dict[str, float] = {}
    if parsed['TEST']:
        last_test = parsed['TEST'][-1]

    return seed, bool(matching_orgs), last_test


def build_dataframe_from_runs(run_dirs: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for idx, rd in enumerate(run_dirs):
        seed, matching_orgs, metrics = summarize_test_metrics(rd)
        for metric_name, value in metrics.items():
            rows.append({
                'run_idx': idx,
                'run_dir': rd,
                'seed': seed,
                'matching_orgs': matching_orgs,
                'metric': metric_name,
                'value': float(value),
            })
    return pd.DataFrame(rows)


def _ci95(series: pd.Series) -> Tuple[float, float]:
    values = series.dropna().to_numpy()
    n = values.size
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(values[0]), float(values[0]))
    mean = float(np.mean(values))
    sem = float(np.std(values, ddof=1) / np.sqrt(n))
    ci = 1.96 * sem
    return (mean - ci, mean + ci)


def _significance_stars(p: float) -> str:
    if not np.isfinite(p):
        return 'ns'
    if p < 1e-4:
        return '****'
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'


def sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", name).strip('_')


def plot_ss_ns_violin_for_metric(df: pd.DataFrame, metric_name: str, out_dir: str) -> None:
    """Plot SS vs NS violin for a single TEST metric and save JSON stats."""
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.0)

    palette = {SS_LABEL: '#1f78b4', NS_LABEL: '#e66101'}
    order = [SS_LABEL, NS_LABEL]

    dfp = df.copy()
    dfp['sampling'] = dfp['matching_orgs'].map({True: SS_LABEL, False: NS_LABEL})
    dfp = dfp.dropna(subset=['value'])
    if dfp.empty:
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 4.0), constrained_layout=True)

    # Violin per sampling category
    sns.violinplot(
        data=dfp, x='sampling', y='value', order=order,
        palette=palette, inner='box', linewidth=1.0, cut=0, ax=ax
    )

    # Points overlay (jittered) for raw values
    sns.stripplot(
        data=dfp, x='sampling', y='value', order=order,
        color='black', alpha=0.9, size=3, jitter=0.12, ax=ax, linewidth=0
    )

    # Mean ± 95% CI per group
    means = dfp.groupby('sampling')['value'].mean()
    stds = dfp.groupby('sampling')['value'].std(ddof=1)
    counts = dfp.groupby('sampling')['value'].count().clip(lower=1)
    sems = stds / np.sqrt(counts)
    ci95 = 1.96 * sems
    x_positions = [order.index(cat) for cat in means.index if cat in order]
    ax.errorbar(
        x=x_positions,
        y=[means.get(cat, np.nan) for cat in means.index if cat in order],
        yerr=[ci95.get(cat, 0.0) for cat in means.index if cat in order],
        fmt='o', color='black', ecolor='black', elinewidth=1.2, capsize=4, markersize=4, zorder=5
    )

    # Two-sample t-test between SS and NS
    a = dfp.loc[dfp['sampling'] == SS_LABEL, 'value'].to_numpy()
    b = dfp.loc[dfp['sampling'] == NS_LABEL, 'value'].to_numpy()
    p_value = np.nan
    if a.size > 0 and b.size > 0:
        t_res = ttest_ind(a, b, equal_var=True, nan_policy='omit')
        p_value = float(getattr(t_res, 'pvalue', t_res[1]))
    stars = _significance_stars(p_value)

    # Horizontal significance bracket between SS (x=0) and NS (x=1)
    y_min = float(dfp['value'].min()) if dfp['value'].size else 0.0
    y_max = float(dfp['value'].max()) if dfp['value'].size else 1.0
    y_range = y_max - y_min if y_max > y_min else 1.0
    y_line = y_max + 0.07 * y_range
    y_tick = 0.02 * y_range
    ax.plot([0, 1], [y_line, y_line], color='black', linewidth=1)
    ax.plot([0, 0], [y_line, y_line - y_tick], color='black', linewidth=1)
    ax.plot([1, 1], [y_line, y_line - y_tick], color='black', linewidth=1)
    ax.text(0.5, y_line + 0.02 * y_range, stars, ha='center', va='bottom', fontsize=9, color='black')
    # Numeric p-value
    p_text = f"p={p_value:.1e}" if np.isfinite(p_value) and p_value < 1e-3 else (f"p={p_value:.3f}" if np.isfinite(p_value) else "p=NA")
    ax.text(0.5, y_line + 0.09 * y_range, p_text, ha='center', va='bottom', fontsize=8, color='black')
    # Expand ylim to fit annotations
    ax.set_ylim(bottom=y_min - 0.02 * y_range, top=y_line + 0.14 * y_range)

    # Axis cosmetics
    display_label = METRIC_DICT.get(metric_name, metric_name)
    ax.set_xlabel('Sampling', fontsize=10)
    ax.set_ylabel(display_label, fontsize=10)
    ax.tick_params(axis='both', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Export stats JSONs
    os.makedirs(out_dir, exist_ok=True)
    fname_base = f"biogrid_test_{sanitize_metric_name(display_label)}"
    stats_dir = out_dir
    out = {
        'metric': metric_name,
        'n_'+SS_LABEL: int(a.size),
        'n_'+NS_LABEL: int(b.size),
        'mean_'+SS_LABEL: float(np.nanmean(a)) if a.size else np.nan,
        'mean_'+NS_LABEL: float(np.nanmean(b)) if b.size else np.nan,
        't_test_pvalue': float(p_value) if np.isfinite(p_value) else None,
        'stars': stars,
    }
    with open(os.path.join(stats_dir, f'{fname_base}_tests.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # Summary per group with CI95
    summary: Dict[str, Dict[str, float]] = {}
    for grp, gdf in dfp.groupby('sampling'):
        lo, hi = _ci95(gdf['value'])
        summary[str(grp)] = {
            'n': int(gdf['value'].size),
            'mean': float(gdf['value'].mean()) if gdf['value'].size else np.nan,
            'ci95': [float(lo), float(hi)],
        }
    with open(os.path.join(stats_dir, f'{fname_base}_summary.json'), 'w') as f:
        json.dump({'metric': metric_name, 'stats': summary}, f, indent=2)

    # Save figure
    fig.savefig(os.path.join(out_dir, f'{fname_base}.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Aggregate TEST metrics and plot SS vs NS per metric.')
    parser.add_argument('--runs-root', type=str, default=os.path.join('preprint', 'biogrid_species_experiment'), help='Root directory containing biogrid_*_seed* run folders')
    parser.add_argument('--out-csv', type=str, default=os.path.join('preprint', 'biogrid_species_experiment', 'biogrid_test_metrics_raw.csv'), help='Output CSV path for raw metrics')
    parser.add_argument('--out-dir', type=str, default=os.path.join('preprint'), help='Directory to store per-metric plots and stats JSONs')
    parser.add_argument('--metrics', type=str, nargs='*', default=None, help='Optional list of metric names to include (default: all)')
    parser.add_argument('--exclude', type=str, nargs='*', default=None, help='Optional list of metric names to exclude')
    args = parser.parse_args()

    # Discover run dirs
    run_dirs = discover_run_dirs(args.runs_root, recursive=True)
    if not run_dirs:
        print(f'No run directories with metrics.log found under {args.runs_root}.')
        return

    # Build raw dataframe
    df_runs = build_dataframe_from_runs(run_dirs)
    if df_runs.empty:
        print('Parsed dataframe is empty. Nothing to do.')
        return

    # Save raw CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_runs.to_csv(args.out_csv, index=False)
    print(f'Wrote raw CSV to: {args.out_csv}')

    # Filter metrics if requested
    metrics_present = sorted(df_runs['metric'].dropna().unique().tolist())
    include_set = set(args.metrics) if args.metrics else set(metrics_present)
    exclude_set = set(args.exclude) if args.exclude else set()
    selected_metrics = [m for m in metrics_present if (m in include_set and m not in exclude_set)]
    if not selected_metrics:
        print('No metrics selected after filtering. Skipping plots.')
        return

    # Plot per metric
    for metric in selected_metrics:
        dmf = df_runs[df_runs['metric'] == metric]
        if dmf.empty:
            continue
        plot_ss_ns_violin_for_metric(dmf, metric_name=metric, out_dir=args.out_dir)

    print(f'Saved per-metric plots and JSON stats under: {args.out_dir}')


if __name__ == '__main__':
    main()



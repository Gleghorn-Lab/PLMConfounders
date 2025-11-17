import os
import json
import argparse
from typing import Tuple, Dict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


SS_LABEL = 'SS'  # Strategic Sampling (matching = True)
NS_LABEL = 'NS'  # Normal Sampling (matching = False)


def _to_bool(x) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {'true', 't', '1', 'yes', 'y'}
    return False


def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV and return long-form dataframe with columns:
    ['split', 'sampling', 'value', 'seed'] where
     - split in {'Validation', 'Test'}
     - sampling in {'SS', 'NS'}
    """
    df = pd.read_csv(csv_path)

    # Normalize booleans and labels
    df['matching_orgs'] = df['matching_orgs'].apply(_to_bool)
    df['sampling'] = df['matching_orgs'].map({True: SS_LABEL, False: NS_LABEL})

    # Map metric names to plot splits
    metric_to_split = {
        'valid_best_mcc': 'Validation',
        'test_mcc': 'Test',
    }
    df['split'] = df['metric'].map(metric_to_split)

    # Keep only rows we know how to plot
    df = df[df['split'].notna()].copy()
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])
    return df[['seed', 'split', 'sampling', 'value']]


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


def annotate_stats(ax: plt.Axes, df: pd.DataFrame, order: Tuple[str, str], metric_name: str, stats_dir: str, prefix: str):
    """Overlay mean±95% CI (horizontal), draw vertical significance bracket, export stats JSON."""
    # Mean ± 95% CI points for each category along x (horizontal)
    means = df.groupby('sampling')['value'].mean()
    stds = df.groupby('sampling')['value'].std(ddof=1)
    counts = df.groupby('sampling')['value'].count().clip(lower=1)
    sems = stds / np.sqrt(counts)
    ci95 = 1.96 * sems

    y_positions = [order.index(cat) for cat in means.index if cat in order]
    ax.errorbar(
        x=[means.get(cat, np.nan) for cat in means.index if cat in order],
        y=y_positions,
        xerr=[ci95.get(cat, 0.0) for cat in means.index if cat in order],
        fmt='o', color='black', ecolor='black', elinewidth=1.2, capsize=4, markersize=4, zorder=5
    )

    # Significance bracket rotated vertically between SS (top) and NS (bottom)
    vals_a = df.loc[df['sampling'] == order[0], 'value'].to_numpy()
    vals_b = df.loc[df['sampling'] == order[1], 'value'].to_numpy()
    p_value = np.nan
    if vals_a.size > 0 and vals_b.size > 0:
        t_res = ttest_ind(vals_a, vals_b, equal_var=True, nan_policy='omit')
        p_value = float(getattr(t_res, 'pvalue', t_res[1]))
    stars = _significance_stars(p_value)

    # Place the vertical bracket near the right side within [0,1]
    x_min, x_max = ax.get_xlim()
    # Keep within axis; use 93% of the span
    x_line = x_min + 0.93 * (x_max - x_min)
    x_tick = 0.02 * (x_max - x_min)
    y_top = 0  # SS (first in order)
    y_bottom = 1  # NS (second in order)
    ax.plot([x_line, x_line], [y_top, y_bottom], color='black', linewidth=1)
    ax.plot([x_line, x_line - x_tick], [y_top, y_top], color='black', linewidth=1)
    ax.plot([x_line, x_line - x_tick], [y_bottom, y_bottom], color='black', linewidth=1)
    ax.text(x_line + 0.01 * (x_max - x_min), 0.5 * (y_top + y_bottom), stars, ha='left', va='center', fontsize=9, color='black')

    # Export minimal stats JSONs
    os.makedirs(stats_dir, exist_ok=True)
    out = {
        'metric': metric_name,
        'n_'+order[0]: int(vals_a.size),
        'n_'+order[1]: int(vals_b.size),
        'mean_'+order[0]: float(np.nanmean(vals_a)) if vals_a.size else np.nan,
        'mean_'+order[1]: float(np.nanmean(vals_b)) if vals_b.size else np.nan,
        't_test_pvalue': p_value,
        'stars': stars,
    }
    with open(os.path.join(stats_dir, f'{prefix}_{metric_name}_tests.json'), 'w') as f:
        json.dump(out, f, indent=2)

    # Also save simple summary per group
    summary: Dict[str, Dict[str, float]] = {}
    for grp, gdf in df.groupby('sampling'):
        lo, hi = _ci95(gdf['value'])
        summary[str(grp)] = {
            'n': int(gdf['value'].size),
            'mean': float(gdf['value'].mean()),
            'ci95': [float(lo), float(hi)],
        }
    with open(os.path.join(stats_dir, f'{prefix}_{metric_name}_summary.json'), 'w') as f:
        json.dump({'metric': metric_name, 'stats': summary}, f, indent=2)


def plot_two_column_violin(df: pd.DataFrame, out_path: str, stats_dir: str):
    """Single-axes plot. Y-axis is MCC. X-axis has two categories: Validation and Test.
    For each category, two violins are shown: SS and NS."""
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.0)

    palette = {SS_LABEL: '#1f78b4', NS_LABEL: '#e66101'}
    hue_order = [SS_LABEL, NS_LABEL]
    x_order = ['Validation', 'Test']

    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4.8), constrained_layout=True)

    # Violin per split with SS/NS hue
    sns.violinplot(
        data=df, x='split', y='value', hue='sampling',
        order=x_order, hue_order=hue_order,
        palette=palette, inner='box', linewidth=1.0, cut=0, ax=ax
    )

    # Points overlay (jittered) for raw values
    sns.stripplot(
        data=df, x='split', y='value', hue='sampling', dodge=True,
        order=x_order, hue_order=hue_order,
        color='black', alpha=0.9, size=3, jitter=0.12, ax=ax, linewidth=0
    )

    # Merge legends from both layers: keep a single concise legend
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        # Deduplicate: first two entries correspond to violins; keep SS/NS once
        unique = []
        seen = set()
        for h, l in zip(handles, labels):
            if l in hue_order and l not in seen:
                unique.append((h, l))
                seen.add(l)
        if unique:
            ax.legend([h for h, _ in unique], [l for _, l in unique], frameon=False, ncols=2, loc='upper right', fontsize=9)
        else:
            ax.legend_.remove()
    else:
        try:
            ax.get_legend().remove()
        except Exception:
            pass

    ax.set_xlabel('')
    ax.set_ylabel('MCC', fontsize=10)
    ax.set_ylim(0.0, 1.0)
    # Major ticks every 0.1 with labels; minor ticks every 0.05 without labels
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # Gridlines at both spacings (minor lighter) - keep gray
    ax.grid(axis='y', which='major', linestyle='-', linewidth=0.8, alpha=0.5, color='0.75')
    ax.grid(axis='y', which='minor', linestyle='-', linewidth=0.5, alpha=0.25, color='0.85')
    ax.tick_params(axis='both', labelsize=9)
    ax.tick_params(axis='both', which='both', colors='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Axis lines black
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)

    # Statistical significance per split (Validation/Test): bracket + stars
    # We'll compute two-sample t-test between SS and NS within each split
    def add_bracket_for_split(split_name: str, x_index: int):
        g = df[df['split'] == split_name]
        if g.empty:
            return None
        a = g[g['sampling'] == hue_order[0]]['value'].to_numpy()
        b = g[g['sampling'] == hue_order[1]]['value'].to_numpy()
        if a.size == 0 or b.size == 0:
            return None
        t_res = ttest_ind(a, b, equal_var=True, nan_policy='omit')
        p_value = float(getattr(t_res, 'pvalue', t_res[1]))

        # Determine vertical position for the bracket
        y_max = float(np.nanmax(g['value'].to_numpy())) if g['value'].size else 0.0
        y_range = 1.0  # fixed axis range 0..1
        y_line = min(0.98, y_max + 0.06 * y_range)
        y_tick = 0.02 * y_range

        # Draw a short horizontal bracket centered at category x_index
        half_width = 0.22
        x_left = x_index - half_width
        x_right = x_index + half_width
        ax.plot([x_left, x_right], [y_line, y_line], color='black', linewidth=1)
        ax.plot([x_left, x_left], [y_line, y_line - y_tick], color='black', linewidth=1)
        ax.plot([x_right, x_right], [y_line, y_line - y_tick], color='black', linewidth=1)

        # Numeric p-value label above the stars
        p_text = f"p={p_value:.1e}" if p_value < 1e-3 else f"p={p_value:.3f}"
        y_p = min(0.995, y_line + 0.02)
        ax.text(x_index, y_p, p_text, ha='center', va='bottom', fontsize=8, color='black')

        return {'split': split_name, 'p_value': p_value,
                'mean_'+hue_order[0]: float(np.nanmean(a)), 'mean_'+hue_order[1]: float(np.nanmean(b)),
                'n_'+hue_order[0]: int(a.size), 'n_'+hue_order[1]: int(b.size)}

    stats_outputs = []
    for i, name in enumerate(x_order):
        out = add_bracket_for_split(name, i)
        if out is not None:
            stats_outputs.append(out)

    # Save grouped tests JSON if any
    if stats_outputs:
        os.makedirs(stats_dir, exist_ok=True)
        with open(os.path.join(stats_dir, 'biogrid_ss_ns_group_tests.json'), 'w') as f:
            json.dump(stats_outputs, f, indent=2)

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Create a two-column SS/NS violin plot (Validation, Test).')
    parser.add_argument('--in-csv', type=str, default=os.path.join('preprint', 'biogrid_species_experiment', 'biogrid_matching_orgs_raw.csv'))
    parser.add_argument('--out', type=str, default=os.path.join('preprint', 'biogrid_ss_ns_two_column_violin.png'))
    parser.add_argument('--stats-dir', type=str, default=os.path.join('preprint'))
    args = parser.parse_args()

    df = load_raw_csv(args.in_csv)
    if df.empty:
        print(f'No data loaded from {args.in_csv}. Nothing to plot.')
        return
    plot_two_column_violin(df, out_path=args.out, stats_dir=args.stats_dir)
    print(f'Saved figure to: {args.out}')


if __name__ == '__main__':
    main()



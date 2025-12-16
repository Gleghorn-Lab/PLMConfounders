
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


def process_csv(filepath, metric_name):
    """
    Reads a wandb export csv and calculates mean and std for 'match' (SS) 
    and 'nomatch' (NS) runs across different seeds.
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

    # Filter for the 'Step' column and relevant metric columns
    # We ignore __MIN and __MAX columns
    step_col = 'Step'
    
    # Identify columns
    all_cols = df.columns
    metric_cols = [c for c in all_cols if c != step_col and '__' not in c]
    
    # Classify columns into SS (match) and NS (nomatch)
    # Note: 'nomatch' contains 'match' substring, so check 'nomatch' first or be careful
    ss_cols = [c for c in metric_cols if 'match' in c and 'nomatch' not in c]
    ns_cols = [c for c in metric_cols if 'nomatch' in c]
    
    print(f"File: {os.path.basename(filepath)}")
    print(f"Found {len(ss_cols)} SS (match) columns and {len(ns_cols)} NS (nomatch) columns.")

    if not ss_cols or not ns_cols:
        print("Warning: Could not find both SS and NS columns.")
        return None, None

    # Extract data
    steps = df[step_col]
    
    # Calculate statistics
    ss_data = df[ss_cols]
    ns_data = df[ns_cols]
    

    ss_mean = ss_data.mean(axis=1)
    ss_std = ss_data.std(axis=1)
    ss_sem = ss_data.sem(axis=1)
    
    ns_mean = ns_data.mean(axis=1)
    ns_std = ns_data.std(axis=1)
    ns_sem = ns_data.sem(axis=1)
    

    # Combine into a summary dataframe for easier plotting
    summary_df = pd.DataFrame({
        'Step': steps,
        'SS_mean': ss_mean,
        'SS_std': ss_std,
        'SS_sem': ss_sem,
        'NS_mean': ns_mean,
        'NS_std': ns_std,
        'NS_sem': ns_sem
    })
    
    # Add raw data for plotting individual seeds
    for col in ss_cols:
        summary_df[f'SS_raw_{col}'] = ss_data[col]
    for col in ns_cols:
        summary_df[f'NS_raw_{col}'] = ns_data[col]
    
    return summary_df


def set_style():
    """Sets a publication-ready, sleek style."""
    sns.set_theme(style="ticks", context="talk", font_scale=1.1)
    
    # Custom rcParams for a sleeker look
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], # standard but clean
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.5,
        'grid.color': '#dddddd',
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
        'grid.alpha': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.frameon': False
    })
    

def plot_metrics(loss_df, pos_df, neg_df, output_path='wandb_figure.png'):
    set_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Colors suitable for publication (colorblind friendly if possible)
    color_ss = '#0072B2' # Blue
    color_ns = '#D55E00' # Vermilion/Orange
    color_random = '#555555' # Dark Gray
    
    alpha_fill = 0.2
    alpha_raw = 0.35  # Increased opacity for better visibility of multiple runs
    linewidth = 2.5
    

    # Z-score/t-score for the requested "99.99% CI" or high visibility.
    # For N=5 (df=4), t_0.99995 is approximately 15.54. This will create a very wide "cone".
    ci_factor = 15.54 
    
    # Helper to plot mean and std
    def plot_trend(ax, df, metric_y_label, title, is_loss=False):
        # Grid just on y usually looks cleaner, or both. Let's do both but subtle (handled by rcParams)
        ax.grid(True)
        
        # Apply EMA smoothing (Time-weighted approximation using span on steps)
        # Using span=20 for smoothing
        span = 20
        df['SS_mean_smooth'] = df['SS_mean'].ewm(span=span).mean()
        df['SS_sem_smooth'] = df['SS_sem'].ewm(span=span).mean()
        df['NS_mean_smooth'] = df['NS_mean'].ewm(span=span).mean()
        df['NS_sem_smooth'] = df['NS_sem'].ewm(span=span).mean()
        
        # Plot individual seeds comparison (raw data) - BLACK lines as requested
        # Find columns
        ss_raw_cols = [c for c in df.columns if c.startswith('SS_raw_')]
        ns_raw_cols = [c for c in df.columns if c.startswith('NS_raw_')]
        
        # Plot raw lines first
        for i, col in enumerate(ss_raw_cols):
            ax.plot(df['Step'], df[col], color='black', alpha=0.15, linewidth=1.0, zorder=1)
            
        for i, col in enumerate(ns_raw_cols):
            ax.plot(df['Step'], df[col], color='black', alpha=0.15, linewidth=1.0, zorder=1)
        
        # Plot SS Mean + Extreme CI (Red Cone) -> Now smoothed
        ax.plot(df['Step'], df['SS_mean_smooth'], label='Strategic Sampling (SS)', color=color_ss, linewidth=linewidth, zorder=2)
        ax.fill_between(df['Step'], 
                        df['SS_mean_smooth'] - ci_factor * df['SS_sem_smooth'], 
                        df['SS_mean_smooth'] + ci_factor * df['SS_sem_smooth'], 
                        color='red', alpha=alpha_fill, edgecolor='none', zorder=1, label='99.99% CI')
        
        # Plot NS Mean + Extreme CI (Red Cone) -> Now smoothed
        ax.plot(df['Step'], df['NS_mean_smooth'], label='Normal Sampling (NS)', color=color_ns, linewidth=linewidth, zorder=2)
        ax.fill_between(df['Step'], 
                        df['NS_mean_smooth'] - ci_factor * df['NS_sem_smooth'], 
                        df['NS_mean_smooth'] + ci_factor * df['NS_sem_smooth'], 
                        color='red', alpha=alpha_fill, edgecolor='none', zorder=1)
        

        # Random Baseline for Loss
        if is_loss:
            random_loss = np.log(2)  # ~0.693
            ax.axhline(y=random_loss, color=color_random, linestyle=':', linewidth=2, label='Random (ln 2)', zorder=0)
            # Annotation directly on plot since legend is moved
            ax.text(df['Step'].min(), random_loss + 0.002, 'Random (ln 2)', color=color_random, fontsize=10, va='bottom', fontweight='bold')

        ax.set_title(title, fontweight='bold', pad=15)
        ax.set_xlabel("Training Steps", fontweight='bold')
        ax.set_ylabel(metric_y_label, fontweight='bold')
        
        # Tweak limits to look nice
        sns.despine(ax=ax)

    # Plot 1: Training Loss
    if loss_df is not None:
        plot_trend(axes[0], loss_df, "Cross entropy loss", "Training Loss", is_loss=True)
        # Legend moved to plot 3

    # Plot 2: Average Positive Probability
    if pos_df is not None:
        plot_trend(axes[1], pos_df, "Probability of PPI", "Avg. Positive PPI Prediction")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5) # 0.5 decision boundary

    # Plot 3: Average Negative Probability
    if neg_df is not None:
        plot_trend(axes[2], neg_df, "Probability of PPI", "Avg. Negative PPI Prediction")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5) # 0.5 decision boundary
        

        # Add legend highlighted in a box with specific order: SS, NS, CI
        handles, labels = axes[2].get_legend_handles_labels()
        # Define desired order by label
        order_map = {'Strategic Sampling (SS)': 0, 'Normal Sampling (NS)': 1, '99.99% CI': 2}
        # Sort handles and labels based on the order_map, ignoring those not in map
        sorted_pairs = sorted(
            [(h, l) for h, l in zip(handles, labels) if l in order_map],
            key=lambda pair: order_map[pair[1]]
        )
        if sorted_pairs:
            handles, labels = zip(*sorted_pairs)
            axes[2].legend(handles, labels, loc='upper right', frameon=True, facecolor='white', edgecolor='#333333', framealpha=1.0, fontsize=11, shadow=True, borderpad=1)


    plt.tight_layout()
    
    print(f"Saving figure to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # File paths
    loss_path = os.path.join(base_dir, 'train_loss.csv')
    pos_path = os.path.join(base_dir, 'train_average_positive_prob.csv')
    neg_path = os.path.join(base_dir, 'train_average_negative_prob.csv')
    
    # Process data
    print("Processing Loss Data...")
    loss_df = process_csv(loss_path, 'Loss')
    
    print("Processing Positive Probability Data...")
    pos_df = process_csv(pos_path, 'Pos Prob')
    
    print("Processing Negative Probability Data...")
    neg_df = process_csv(neg_path, 'Neg Prob')
    
    # Plot
    plot_metrics(loss_df, pos_df, neg_df, output_path=os.path.join(base_dir, 'training_trends_comparison.png'))
    print("Done!")


if __name__ == "__main__":
    main()

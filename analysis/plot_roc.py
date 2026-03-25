"""
Publication-quality ROC curve plotting using pauc ROC objects.
Style matches accidental_taxonomist_results/plot_biogrid.py conventions.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from pauc import ROC, ci_auc

SS_COLOR = "#1f78b4"
NS_COLOR = "#e66101"
DEFAULT_COLORS = [SS_COLOR, NS_COLOR]


def _setup_style():
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)


def _style_ax(ax: Axes):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(axis="both", labelsize=9, colors="black")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(axis="both", which="major", linestyle="-", linewidth=0.8, alpha=0.5, color="0.75")
    ax.grid(axis="both", which="minor", linestyle="-", linewidth=0.5, alpha=0.25, color="0.85")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)


def plot_publication_roc(
    rocs: list[ROC],
    ax: Axes | None = None,
    colors: list[str] | None = None,
    show_ci: bool = True,
    ci_method: str = "delong",
    save_path: str | None = None,
    dpi: int = 600,
) -> Axes:
    """Plot publication-quality ROC curves from pauc ROC objects.

    Parameters
    ----------
    rocs : list of pauc.ROC objects, each with a .name attribute for the legend.
    ax : matplotlib Axes, or None to create a new figure.
    colors : per-curve colors. Defaults to SS blue / NS orange.
    show_ci : if True, append 95% CI to legend labels via ci_auc.
    ci_method : "delong" or "bootstrap" for CI computation.
    save_path : if provided, save figure to this path and close it.
    dpi : save resolution (default 600).
    """
    _setup_style()

    if colors is None:
        colors = DEFAULT_COLORS

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    else:
        fig = ax.get_figure()

    # Diagonal reference
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8, color="0.65", zorder=1)

    for i, roc in enumerate(rocs):
        color = colors[i % len(colors)]
        name = roc.name or f"Model {i+1}"

        # Build legend label
        label = f"{name} (AUC = {roc.auc:.3f}"
        if show_ci:
            lo, hi = ci_auc(roc, conf_level=0.95, method=ci_method)
            label += f" [{lo:.3f}, {hi:.3f}]"
        label += ")"

        ax.plot(roc.fpr, roc.tpr, color=color, linewidth=1.5, label=label, zorder=10 + i)

    _style_ax(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if own_fig:
            plt.close(fig)

    return ax

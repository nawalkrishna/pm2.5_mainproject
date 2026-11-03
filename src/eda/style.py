"""Shared figure style for Stage 3+ PNG figures (plain matplotlib, light background).

Colours follow the validated reference palette of the dataviz method: categorical
slots in fixed order (one per state, always the same state -> same colour),
single-hue blue for magnitude, blue/orange with a grey midpoint for signed values.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

STATE_ORDER = ["Kerala", "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Telangana"]
STATE_COLORS = dict(zip(STATE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]))
TEXT, TEXT_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc3f0", "#2a78d6", "#0d3a73"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#e8e7e3", "#eb6834"])


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.labelsize": 9, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 2, "legend.fontsize": 8, "legend.frameon": False, "font.size": 9,
    })


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def label_line_ends(ax, items: list[tuple[str, float]], x: float, min_gap: float = 0.045) -> None:
    """Direct labels at the right end of lines, nudged apart so they never overlap.

    ``items`` is (label, y of the line's last point); ``min_gap`` is in axis data units.
    """
    placed: list[float] = []
    for label, y in sorted(items, key=lambda t: t[1]):
        y_text = max(y, placed[-1] + min_gap) if placed else y
        placed.append(y_text)
        ax.annotate(label, (x, y_text), fontsize=7, color=TEXT_2, va="center")

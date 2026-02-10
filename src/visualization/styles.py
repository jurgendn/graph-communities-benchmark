"""
Matplotlib styling and configuration.
"""

# Default matplotlib configuration
MATPLOTLIB_CONFIG = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def configure_matplotlib():
    """Apply global matplotlib configuration."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(MATPLOTLIB_CONFIG)


class FigureConfig:
    """Figure dimension and styling presets."""

    # Single plot dimensions
    SINGLE_WIDTH = 8
    SINGLE_HEIGHT = 6

    # Grouped plot dimensions
    GROUPED_WIDTH = 10
    GROUPED_HEIGHT = 8

    # Bar chart adaptive width
    @staticmethod
    def bar_width(num_bars: int, min_width: float = 6.0) -> float:
        """Calculate bar chart width based on number of bars."""
        return max(min_width, num_bars * 0.6)

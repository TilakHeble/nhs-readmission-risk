"""Shared matplotlib style so every chart in the project reads as one system."""

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"  # primary series
ORANGE = "#eb6834"  # comparison series
GREY = "#a8a7a2"  # context / de-emphasised
AQUA = "#1baf7a"
YELLOW = "#eda100"
BAND_COLOURS = {  # NHS banding, fixed across every chart
    "A1": "#c2410c",
    "A5": ORANGE,
    "W": GREY,
    "B5": AQUA,
    "B1": "#0f7a55",
}
IMD_COLOURS = {1: "#7a2e0e", 2: "#c2410c", 3: "#eda100", 4: "#5aa0e6", 5: "#1d4f91"}  # most -> least deprived
AGE_COLOURS = {"<16": AQUA, "16-74": BLUE, "75+": ORANGE}


def set_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 24,
            "axes.labelcolor": TEXT_2,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "text.color": TEXT,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "axes.axisbelow": True,
            "text.parse_math": False,
        }
    )


def subtitle(ax, text: str) -> None:
    """Grey one-line subtitle under a left-aligned title."""
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT_2, fontsize=9.5, va="bottom")


def save(fig, name: str) -> None:
    from nhsread.config import ROOT

    out = ROOT / "docs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def fig_title(fig, title: str, sub: str | None = None) -> None:
    """Left-aligned figure title (+ optional grey subtitle) for multi-panel figures.

    Positions are in inches, so title and subtitle never collide at any figure height.
    """
    h = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.08 / h, ha="left", va="top", fontweight="bold", fontsize=13)
    top = 1 - 0.45 / h
    if sub:
        fig.text(0.01, 1 - 0.40 / h, sub, ha="left", va="top", color=TEXT_2, fontsize=9.5)
        top = 1 - 0.55 / h
    fig.tight_layout(rect=(0, 0, 1, top))

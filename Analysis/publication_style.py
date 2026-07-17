"""Shared Matplotlib style for the PRR manuscript figures."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib as mpl
from matplotlib.axes import Axes


def configure_publication_style() -> None:
    """Use a closed-frame, two-column-safe APS figure style."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 13.0,
            "axes.labelsize": 14.0,
            "axes.titlesize": 14.0,
            "axes.titleweight": "normal",
            "axes.linewidth": 1.05,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "0.82",
            "grid.linewidth": 0.65,
            "grid.alpha": 0.55,
            "legend.fontsize": 11.0,
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "lines.linewidth": 2.1,
            "lines.markersize": 5.5,
            "mathtext.fontset": "cm",
            "mathtext.rm": "serif",
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def close_axes(axes: Axes | Iterable[Axes]) -> None:
    """Show a complete box and ticks on all four sides."""
    if isinstance(axes, Axes):
        flat_axes = [axes]
    else:
        try:
            flat_axes = list(axes.flat)  # type: ignore[attr-defined]
        except AttributeError:
            flat_axes = list(axes)

    for axis in flat_axes:
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.05)
        axis.tick_params(top=True, right=True, direction="in", width=1.0)

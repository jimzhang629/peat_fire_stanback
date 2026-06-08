"""Reusable matplotlib style and figure builders for the fire comparison.

Centralising the style here keeps notebooks clean and every figure consistent.
The style follows the user's requests: products encoded by **colour only** (no
redundant marker shapes), **top/right spines removed**, and a **frameless
legend**. The colour cycle is the colour-blind-safe Okabe-Ito palette, since
these figures are for supervisors.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# Okabe-Ito colour-blind-safe qualitative palette.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Categorical colormap for two-product overlay maps (1=A only, 2=B only, 3=both).
OVERLAY_CMAP = ListedColormap(["#1f77b4", "#d62728", "#7e3f9e"])


def set_fire_style() -> None:
    """Apply the project's matplotlib rcParams. Idempotent; call once per session."""
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "figure.autolayout": True,
        }
    )


def plot_annual_series(
    df: pd.DataFrame,
    ylabel: str = "Total burned area (km$^2$)",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Recreate Humber et al. Figure 3: a line per product over years.

    ``df`` is the output of
    :func:`peatfire.fire_comparison.annual_burned_area_series`; only the raw
    km^2 columns are plotted (``*_pct_aoi`` helper columns are ignored).
    Products are distinguished by colour alone, per the style.
    """
    set_fire_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    cols = [c for c in df.columns if not c.endswith("_pct_aoi")]
    for col in cols:
        ax.plot(df.index, df[col], marker="o", markersize=4, label=col)

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(title=None)
    return fig


def plot_agreement_heatmap(
    matrix: pd.DataFrame,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
):
    """Annotated heatmap of a pairwise agreement/correlation matrix.

    ``matrix`` is a symmetric products x products DataFrame from
    :func:`peatfire.fire_comparison.agreement_matrix`.
    """
    set_fire_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    data = matrix.values.astype(float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    # annotate each cell
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < (np.nanmean(data)) else "black",
                    fontsize=9,
                )
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    # heatmaps want all four spines
    for s in ax.spines.values():
        s.set_visible(True)
    return fig


def plot_overlay_map(
    masks: dict,
    aoi_gdf,
    year: int,
    pair: tuple[str, str],
    ax: Optional[plt.Axes] = None,
):
    """Two-product categorical overlay on the common grid.

    ``masks`` is a common-grid stack (name -> binary DataArray) from
    :func:`peatfire.fire_comparison.stack_on_common_grid`. ``pair`` selects the
    two products to compare: 1 = first only, 2 = second only, 3 = both. The AOI
    boundary is drawn on top.
    """
    set_fire_style()
    a_name, b_name = pair
    a = masks[a_name]
    b = masks[b_name]
    av = (a.values > 0)
    bv = (b.values > 0)

    cat = np.zeros(av.shape, dtype="uint8")
    cat[av & ~bv] = 1
    cat[~av & bv] = 2
    cat[av & bv] = 3
    cat_ma = np.ma.masked_where(cat == 0, cat)

    left, bottom, right, top = a.rio.bounds()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    ax.imshow(
        cat_ma, extent=[left, right, bottom, top], origin="upper",
        cmap=OVERLAY_CMAP, vmin=1, vmax=3, interpolation="nearest",
    )
    aoi_gdf.to_crs(a.rio.crs).boundary.plot(ax=ax, color="black", linewidth=0.8)
    ax.set_title(f"Burned area {year}: {a_name} vs {b_name}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(
        handles=[
            Patch(color="#1f77b4", label=f"{a_name} only"),
            Patch(color="#d62728", label=f"{b_name} only"),
            Patch(color="#7e3f9e", label="Both"),
        ],
        loc="lower left",
    )
    return fig

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
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from .fire_comparison import rmse, total_least_squares

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


def plot_product_scatter(
    x,
    y,
    xlabel: str = "Product X",
    ylabel: str = "Product Y",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Scatterplot of two products with a total-least-squares fit and y=x line.

    ``x`` and ``y`` are paired per-unit values (e.g. from
    :func:`peatfire.fire_products_comparison.product_pair_scatter`). Draws the
    points, the **y = x** reference (perfect agreement), and the **TLS** fit, and
    annotates the TLS slope/intercept and the RMSE -- the comparison Humber et al.
    report. Equal aspect so the y=x line reads at 45 degrees.
    """
    set_fire_style()
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
    else:
        fig = ax.figure

    if x.size == 0:
        ax.text(0.5, 0.5, "no overlapping data", ha="center", va="center")
        return fig

    ax.scatter(x, y, s=8, alpha=0.4, color=OKABE_ITO[0], edgecolor="none")
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", label="y = x")

    slope, intercept = total_least_squares(x, y)
    if np.isfinite(slope):
        xs = np.array([lo, hi])
        ax.plot(
            xs, slope * xs + intercept, color=OKABE_ITO[3],
            label=f"TLS: y = {slope:.2f}x + {intercept:.2g}",
        )
    err = rmse(x, y)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"RMSE = {err:.3g}")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="datalim")
    return fig


def plot_temporal_heatmap(
    totals: pd.DataFrame,
    normalize: Optional[str] = "row",
    cmap: str = "magma",
    active_fire: str = "VIIRS",
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
):
    """Humber-style temporal heat map: burned area per product over time.

    ``totals`` is the output of
    :func:`peatfire.fire_products_comparison.period_totals_series` (index = time
    periods, one column per product). Rows become products and columns become
    time, so you can read *when* each product detects fire. The occurrence
    product (``active_fire``, e.g. VIIRS) is placed in the bottom row as an
    independent indicator of when fire actually occurred -- bright cells that
    line up vertically mean a product's burn detections coincide in time with
    active fire.

    Parameters
    ----------
    normalize : {"row", None}
        ``"row"`` (default) scales each product's row to [0, 1] by its own max,
        so the *timing* pattern is comparable across products despite different
        units (km^2 burned area vs VIIRS detection count). ``None`` plots raw
        values on a single shared colour scale (only sensible if every column is
        in the same unit).
    """
    set_fire_style()
    df = totals.dropna(axis=1, how="all").copy()

    # order rows with the active-fire reference last, if present
    cols = list(df.columns)
    if active_fire in cols:
        cols = [c for c in cols if c != active_fire] + [active_fire]
    df = df[cols]

    data = df.to_numpy(dtype="float64").T  # rows = products, cols = time
    if normalize == "row":
        with np.errstate(invalid="ignore"):
            rowmax = np.nanmax(np.where(np.isfinite(data), data, np.nan), axis=1, keepdims=True)
        rowmax[~np.isfinite(rowmax) | (rowmax == 0)] = 1.0
        data = data / rowmax
    data = np.ma.masked_invalid(data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(6, 0.35 * data.shape[1] + 2), 0.6 * data.shape[0] + 1.5))
    else:
        fig = ax.figure

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("0.9")  # NaN periods shown light grey
    im = ax.imshow(data, aspect="auto", cmap=cmap_obj, interpolation="nearest")

    # time labels (thin them out if there are many)
    periods = list(df.index)
    def _lab(p):
        return p.strftime("%Y-%m") if hasattr(p, "strftime") else str(p)
    step = max(1, len(periods) // 24)
    ticks = list(range(0, len(periods), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([_lab(periods[i]) for i in ticks], rotation=90)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)

    # separate the active-fire reference row with a line
    if active_fire in cols:
        ax.axhline(len(cols) - 1.5, color="white", linewidth=2)

    label = "per-row normalised intensity" if normalize == "row" else "value (km$^2$ / count)"
    fig.colorbar(im, ax=ax, shrink=0.8, label=label)
    ax.set_title(title or "Burned area over time, by product (VIIRS active fire = reference)")
    ax.set_xlabel("Time")
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


def plot_consensus_map(
    masks: dict,
    aoi_gdf,
    year: int,
    month: Optional[int] = None,
    cmap: str = "viridis",
    ax: Optional[plt.Axes] = None,
):
    """Multi-product burned-area overlay: colour each cell by how many products burned it.

    The N-product generalisation of :func:`plot_overlay_map`. ``masks`` is a
    common-grid stack (name -> binary DataArray) from
    :func:`peatfire.fire_products_comparison.stack_on_common_grid`, so every
    product is already aligned to the same EPSG:5070 grid. Burned cells are
    coloured 1..N by the number of products that map a burn there (1 = a single
    product, N = unanimous); unburned cells (count 0) are transparent. The AOI
    boundary is drawn on top.

    Reads "where do products agree?": darker/fringe cells are mapped by only one
    product, the brightest cells are the consensus core all products share.
    """
    set_fire_style()
    names = [n for n in masks if masks[n] is not None]
    if not names:
        if ax is None:
            fig, ax = plt.subplots(figsize=(11, 5))
        else:
            fig = ax.figure
        ax.text(0.5, 0.5, "no products for this period", ha="center", va="center")
        return fig

    # masks share the common grid, so they stack cell-for-cell.
    count = np.sum(
        [(masks[n].values > 0).astype("int16") for n in names], axis=0
    )
    n_products = len(names)
    count_ma = np.ma.masked_where(count == 0, count)

    ref = masks[names[0]]
    left, bottom, right, top = ref.rio.bounds()

    # one discrete colour per agreement level 1..N, with integer colourbar ticks.
    colours = plt.get_cmap(cmap)(np.linspace(0.15, 1.0, n_products))
    cmap_obj = ListedColormap(colours)
    norm = BoundaryNorm(np.arange(0.5, n_products + 1.5), cmap_obj.N)

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.figure

    im = ax.imshow(
        count_ma, extent=[left, right, bottom, top], origin="upper",
        cmap=cmap_obj, norm=norm, interpolation="nearest",
    )
    aoi_gdf.to_crs(ref.rio.crs).boundary.plot(ax=ax, color="black", linewidth=0.8)

    label = f"{year}-{month:02d}" if month else str(year)
    ax.set_title(
        f"Burned-area consensus in NC, {label}: {n_products} products "
        f"({', '.join(names)})"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    cbar = fig.colorbar(
        im, ax=ax, ticks=range(1, n_products + 1), shrink=0.8,
        boundaries=np.arange(0.5, n_products + 1.5),
    )
    cbar.set_label("Number of products mapping burn")
    return fig

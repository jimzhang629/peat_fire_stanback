"""Reusable matplotlib style and figure builders for the fire comparison.

Centralising the style here keeps notebooks clean and every figure consistent.
The style follows the user's requests: products encoded by **colour only** (no
redundant marker shapes), **top/right spines removed**, and a **frameless
legend**. The colour cycle is the colour-blind-safe Okabe-Ito palette, since
these figures are for supervisors.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .fire_comparison import (
    ANALYSIS_CRS,
    _as_gdf,
    build_common_grid,
    rasterize_polygons_to_grid,
    rmse,
    stack_on_common_grid,
    total_least_squares,
)
from .reference_sources import load_reference

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
    legend=True
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
    if legend:
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
    # Burned cells are 500 m and sparse (often a few hundred) across a swath
    # hundreds of km wide, so on the full-extent imshow each cell is sub-pixel
    # and antialiasing blends it away. Overlay the burned cells as fixed *pixel*
    # size markers so they stay visible regardless of geographic scale/zoom.
    rows, cols = np.where(cat > 0)
    if rows.size:
        res_x = (right - left) / cat.shape[1]
        res_y = (top - bottom) / cat.shape[0]
        cx = left + (cols + 0.5) * res_x
        cy = top - (rows + 0.5) * res_y
        cell_colours = np.array(OVERLAY_CMAP.colors)[cat[rows, cols] - 1]
        ax.scatter(cx, cy, c=cell_colours, s=10, marker="s", linewidths=0)
    # dissolve to the outer outline and keep it thin/translucent: the raw peat
    # AOI is thousands of intricate polygons whose boundaries otherwise render
    # as a black mesh that buries the coloured cells.
    aoi_gdf.to_crs(a.rio.crs).dissolve().boundary.plot(
        ax=ax, color="black", linewidth=0.4, alpha=0.5,
    )
    # pin to the full raster extent (geopandas autoscales to the tighter AOI
    # bbox, which clips the boundary); equal aspect avoids distortion.
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
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


def plot_raster_map(
    da,
    aoi_gdf,
    title: str = "",
    label: Optional[str] = None,
    aoi_context=None,
    cmap: str = "magma",
    robust: bool = True,
    markers: bool = True,
    marker_size: float = 12,
    ax: Optional[plt.Axes] = None,
):
    """Single-product raster map on the common grid, with the recurring map traps handled.

    Draws a continuous/count ``da`` (severity, VIIRS detection counts, ...) on the
    EPSG:5070 common grid over ``aoi_gdf``, masking unburned (``<= 0``) cells. Bakes
    in the fixes worked out for the fire maps so notebook cells stay one-liners:

    * the AOI is **dissolved** to a single outer outline -- the raw peat extent is
      thousands of polygons whose boundaries otherwise composite into a black mesh
      that buries the data;
    * burned cells are also drawn as fixed **pixel-size markers** so sparse 500 m
      cells survive ``imshow`` downsampling to screen resolution instead of being
      antialiased away;
    * an optional ``aoi_context`` outline (e.g. the NC state boundary) is drawn on
      top, then the view is pinned back to the raster extent so geopandas' autoscale
      cannot zoom the map out to the context bounds.

    ``robust`` matches xarray's option (2nd/98th percentile colour limits), shared by
    the raster and the markers so they read on one colourbar.
    """
    set_fire_style()
    crs = da.rio.crs
    left, bottom, right, top = da.rio.bounds()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    vals = da.values
    burned = np.isfinite(vals) & (vals > 0)
    pos = vals[burned]
    if pos.size and robust:
        vmin, vmax = (float(x) for x in np.nanpercentile(pos, [2, 98]))
    elif pos.size:
        vmin, vmax = float(pos.min()), float(pos.max())
    else:
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    # AOI outline first (dissolved, thin) so it sits under the data.
    aoi_gdf.to_crs(crs).dissolve().boundary.plot(
        ax=ax, color="black", linewidth=0.6, alpha=0.5, zorder=1
    )

    masked = np.where(burned, vals, np.nan)
    mappable = ax.imshow(
        masked, extent=[left, right, bottom, top], origin="upper",
        cmap=cmap, norm=norm, interpolation="nearest", zorder=2,
    )
    if markers and pos.size:
        rows, cols = np.where(burned)
        cx = left + (cols + 0.5) * (right - left) / vals.shape[1]
        cy = top - (rows + 0.5) * (top - bottom) / vals.shape[0]
        mappable = ax.scatter(
            cx, cy, c=vals[rows, cols], cmap=cmap, norm=norm,
            s=marker_size, marker="s", linewidths=0, zorder=3,
        )

    # context outline (e.g. NC) on top; pin limits afterwards so the geopandas
    # autoscale here -- and from the dissolve above -- cannot move the view.
    if aoi_context is not None:
        aoi_context.to_crs(crs).boundary.plot(
            ax=ax, color="0.4", linewidth=0.8, zorder=5
        )
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    default_label = da.attrs.get("long_name")
    if not isinstance(default_label, str):
        default_label = da.name
    fig.colorbar(mappable, ax=ax, label=label or default_label or "")
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
    # geopandas autoscales to the AOI's tight bbox (slightly inside the grid),
    # clipping the boundary at the frame; pin limits to the full raster extent
    # and keep equal aspect so the projection is not distorted.
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)
    ax.set_aspect("equal")

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


def plot_validation_heatmap(
    matrix: pd.DataFrame,
    title: Optional[str] = None,
    cmap: str = "viridis",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cbar_label: str = "recall (1 - omission)",
    ax: Optional[plt.Axes] = None,
):
    """Heatmap of an events x products validation matrix (rows = reference fires).

    ``matrix`` comes from
    :func:`peatfire.fire_products_comparison.recall_matrix` -- one row per
    reference incident (e.g. Pains Bay 2011), one column per product, cells
    holding recall (or precision, etc.). Unlike
    :func:`plot_agreement_heatmap` this is rectangular and the axes are not
    symmetric, so each named fire's per-product strength reads off down a row.
    """
    set_fire_style()
    if ax is None:
        fig, ax = plt.subplots(
            figsize=(1.1 * len(matrix.columns) + 2, 0.45 * len(matrix.index) + 2)
        )
    else:
        fig = ax.figure

    data = matrix.values.astype(float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)
    mid = (vmin + vmax) / 2
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center",
                    color="white" if val < mid else "black", fontsize=9,
                )
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(cbar_label)
    for s in ax.spines.values():
        s.set_visible(True)
    return fig


# ---------------------------------------------------------------------------
# Reusable AOI / reference boundary overlay
# ---------------------------------------------------------------------------
# A "layer" is either a bare GeoDataFrame (drawn with the defaults) or a dict
# carrying the geometry plus per-layer style overrides.
BoundaryLayer = Union[gpd.GeoDataFrame, dict]


def overlay_aoi_boundaries(
    ax: plt.Axes,
    layers,
    crs,
    *,
    default_color: str = "0.4",
    default_linewidth: float = 0.8,
    default_alpha: float = 1.0,
    zorder: int = 5,
    dissolve: bool = False,
    add_legend: bool = False,
    legend_loc: str = "lower left",
):
    """Draw one or more vector boundaries on top of an existing product-map axis.

    This generalises the manual "plot the NC boundary, then restore the
    xlim/ylim" dance the notebooks do by hand: ``geopandas`` autoscales the axis
    to each layer's bounding box, which would otherwise zoom the product map out
    (or in) to that layer. We snapshot the axis limits up front and restore them
    after every layer is drawn, so the product view set by
    :func:`plot_overlay_map` / :func:`plot_raster_map` is preserved.

    Because it just takes "whatever AOIs you want, on whatever product axis", the
    same call works for reference-vs-product (overlay an event perimeter) and
    product-vs-product (overlay the state / peat-extent boundary) maps.

    Parameters
    ----------
    ax : matplotlib Axes
        An axis that already has a product map drawn on it (its current
        ``xlim``/``ylim`` define the view to keep).
    layers : GeoDataFrame | iterable of (GeoDataFrame | dict)
        Each layer is either a bare GeoDataFrame (drawn with the defaults) or a
        dict ``{"gdf": ..., "color": ..., "linewidth": ..., "alpha": ...,
        "label": ..., "dissolve": ...}`` overriding the defaults per layer.
    crs
        Target CRS to reproject every layer into -- pass the product grid CRS
        (e.g. ``stack[name].rio.crs``), which is EPSG:5070 here.
    dissolve : bool
        Default dissolve flag: dissolve a layer to its single outer outline
        before drawing. Good for many-polygon AOIs (the raw peat extent is
        thousands of polygons whose boundaries otherwise render as a black mesh);
        leave ``False`` to keep individual perimeters (e.g. a reference event).

    Returns
    -------
    list of matplotlib handles
        One :class:`~matplotlib.lines.Line2D` proxy per *labelled* layer, so the
        caller can fold them into a combined legend.
    """
    set_fire_style()
    if isinstance(layers, gpd.GeoDataFrame):
        layers = [layers]

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    handles = []
    for layer in layers:
        if isinstance(layer, dict):
            gdf = layer["gdf"]
            color = layer.get("color", default_color)
            lw = layer.get("linewidth", default_linewidth)
            alpha = layer.get("alpha", default_alpha)
            label = layer.get("label")
            do_dissolve = layer.get("dissolve", dissolve)
        else:
            gdf = layer
            color, lw, alpha = default_color, default_linewidth, default_alpha
            label, do_dissolve = None, dissolve

        g = gdf.to_crs(crs)
        if do_dissolve:
            g = g.dissolve()
        g.boundary.plot(ax=ax, color=color, linewidth=lw, alpha=alpha, zorder=zorder)
        if label:
            handles.append(Line2D([0], [0], color=color, lw=lw, label=label))

    # restore the product view (geopandas autoscaled to the overlay bbox above)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if add_legend and handles:
        ax.legend(handles=handles, loc=legend_loc)
    return handles


# ---------------------------------------------------------------------------
# Reference-vs-product map for one chosen event
# ---------------------------------------------------------------------------
def plot_reference_vs_product_for_one_event(
    reference: str,
    product: str,
    aoi,
    event,
    *,
    year: Optional[int] = None,
    month: Optional[int] = None,
    binary: bool = True,
    extent: str = "event",
    buffer_km: float = 10.0,
    aoi_context=None,
    common_grid_res: float = 500.0,
    out_dir: Optional[Union[str, Path]] = None,
    ax: Optional[plt.Axes] = None,
):
    """Map one product against one reference event, on the shared common grid.

    Pulls the four pieces together for a single incident:

    1. load the **reference** source (clipped to ``aoi``) and pick the **event**
       by name (case-insensitive substring; one incident);
    2. time-match and load the **product** for the event's year on a common grid;
    3. rasterise the event's perimeter onto that *same* grid;
    4. draw them, with the event perimeter and an optional context boundary
       overlaid (via :func:`overlay_aoi_boundaries`), and save with the event
       name in the filename.

    For a **burned-area** product (``binary=True``) the figure is the
    reference-vs-product *confusion overlay* -- the same categorical encoding as
    :func:`plot_overlay_map`, reused verbatim, but with one mask being the
    reference: blue = reference only (**omission**, the product missed it), red =
    product only (commission), purple = **both** (hit). For a continuous product
    (``binary=False``: severity / VIIRS counts) the product is drawn with
    :func:`plot_raster_map` and the perimeter is overlaid on top.

    Parameters
    ----------
    reference : str
        A registered reference key (see
        :func:`reference_sources.list_references`), e.g. ``"NIFC_IFPH"``.
    product : str
        A registered product name, e.g. ``"GABAM"``.
    aoi : str | Path | GeoDataFrame
        The clip/backdrop AOI (NC, peat extent, ...), as elsewhere.
    event : str | GeoDataFrame
        Event name (or unique substring) to look up in the reference, or an
        already-selected event GeoDataFrame (its ``_event``/``_year`` columns are
        used if present).
    year : int, optional
        Override the product year; by default the event's own ``_year`` is used.
    extent : {"event", "aoi"}
        ``"event"`` (default) zooms to a ``buffer_km`` window around the event --
        the only view where a single incident's cells are legible. ``"aoi"`` maps
        the whole AOI as the backdrop with the event as a small overlay (the
        full-extent style of the comparison notebook's overlay cell).
    aoi_context : GeoDataFrame, optional
        An extra boundary to draw for geographic reference (e.g. the NC state
        outline), exactly like the manual ``aoi_nc`` overlay in the notebooks.
    out_dir : str | Path, optional
        If given, the figure is saved there as
        ``<reference>_vs_<product>_<event-slug>_<year>.png``.

    Returns
    -------
    matplotlib Figure
    """
    set_fire_style()
    gdf_aoi = _as_gdf(aoi)

    # 1) reference dataset, clipped to the AOI
    ref = load_reference(reference, gdf_aoi)
    if ref is None or ref.empty:
        raise ValueError(
            f"reference {reference!r} has no events within the AOI "
            "(not downloaded, or nothing falls in the extent)."
        )

    # 2) pick one event -- accept a name/substring or an already-selected gdf
    if isinstance(event, gpd.GeoDataFrame):
        event_gdf = event
        event_name = (
            str(event_gdf["_event"].iloc[0]) if "_event" in event_gdf.columns else "event"
        )
    else:
        hit = ref[ref["_event"].str.contains(str(event), case=False, na=False)]
        if hit.empty:
            available = sorted(ref["_event"].dropna().unique().tolist())[:30]
            raise ValueError(
                f"no event matching {event!r} in {reference}. "
                f"available (up to 30): {available}"
            )
        event_name = str(hit["_event"].iloc[0])  # keep a single incident
        event_gdf = hit[hit["_event"] == event_name]

    # 3) the event's year, to time-match the product
    if year is None:
        yrs = pd.to_numeric(event_gdf["_year"], errors="coerce").dropna()
        if yrs.empty:
            raise ValueError(
                f"event {event_name!r} has no parseable year; pass year= explicitly."
            )
        year = int(yrs.iloc[0])

    # 4) grid: zoom to a buffered window around the event, or the whole AOI
    event_5070 = event_gdf.to_crs(ANALYSIS_CRS)
    if extent == "event":
        win_geom = event_5070.geometry.union_all().buffer(buffer_km * 1000.0)
        grid_aoi = gpd.GeoDataFrame(geometry=[win_geom], crs=ANALYSIS_CRS)
    elif extent == "aoi":
        grid_aoi = gdf_aoi
    else:
        raise ValueError(f"extent must be 'event' or 'aoi', got {extent!r}.")
    grid = build_common_grid(grid_aoi, res_m=common_grid_res)
    crs = grid.rio.crs

    # 5) product on that grid + the rasterised reference perimeter (same grid)
    stack = stack_on_common_grid(
        [product], year, grid_aoi, binary=binary, grid=grid, month=month
    )
    ref_mask = rasterize_polygons_to_grid(event_5070, grid)
    prod_da = stack.get(product)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    if binary:
        # Reference-vs-product confusion overlay, reusing plot_overlay_map: the
        # reference plays product "A", so 1 = reference only (omission), 2 =
        # product only (commission), 3 = both (hit). A product absent for the
        # year becomes an all-zero mask so the omission cells still show.
        if prod_da is None:
            prod_da = xr.zeros_like(ref_mask)
        masks = {reference: ref_mask, product: prod_da}
        plot_overlay_map(masks, grid_aoi, year, pair=(reference, product), ax=ax)
        legend_handles = [
            Patch(color=OVERLAY_CMAP.colors[0], label=f"{reference} only (omission)"),
            Patch(color=OVERLAY_CMAP.colors[1], label=f"{product} only (commission)"),
            Patch(color=OVERLAY_CMAP.colors[2], label="both (hit)"),
        ]
    else:
        if prod_da is None:
            ax.text(
                0.5, 0.5, f"{product}: no data for {year}",
                ha="center", va="center", transform=ax.transAxes,
            )
        else:
            plot_raster_map(prod_da, grid_aoi, title="", ax=ax)
        legend_handles = []

    # 6) overlay the actual event perimeter (+ optional context), pinning the view
    layers = [
        {"gdf": event_gdf, "color": "black", "linewidth": 1.4,
         "label": f"{event_name} perimeter"},
    ]
    if aoi_context is not None:
        layers.append(
            {"gdf": aoi_context, "color": "0.4", "linewidth": 0.8, "label": "context"}
        )
    legend_handles += overlay_aoi_boundaries(ax, layers, crs)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower left", fontsize=8)

    ax.set_title(f"{event_name} ({year}): {reference} reference vs {product}")

    # 7) save with the event name in the filename
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^0-9A-Za-z]+", "_", event_name).strip("_") or "event"
        fname = f"{reference}_vs_{product}_{slug}_{year}.png"
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
    return fig

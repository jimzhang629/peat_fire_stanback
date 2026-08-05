"""Diagnostic plots for inspecting the modeling data at each step.

The matching pipeline (``matching.py``) is a sequence of transforms -- register
covariates, pixelate treated + control, sample covariates, match, check balance --
and each step has a picture that tells you whether it did what you think. This
module is those pictures, in one place, sharing the fire-comparison style
(:func:`peatfire.set_fire_style`, the Okabe-Ito palette) so they match the rest
of the deck.

The functions, in pipeline order:

* :func:`plot_covariate_maps` -- each covariate warped onto the grid and drawn
  over NC (the "does this layer have coverage and spatial signal?" check). This is
  where a covariate that is *constant* -- histosol %, pinned near 90 across the
  peat frame -- shows itself as a flat map, i.e. nothing to match on.
* :func:`plot_temporal_covariate_by_year` -- the per-year counterpart of the
  above for a *temporal* covariate (``pdsi`` / ``precip`` / ...): one panel per
  year on a shared, zero-centred scale, so the drought / dry-year signal the DiD
  stage keys off is visible (``plot_covariate_maps`` only maps the static layers).
* :func:`plot_covariate_space` -- treated vs control pixels in a 2-D covariate
  plane, with marginal histograms. A covariate with no spread collapses its axis
  to a line; this is the direct picture of "we are effectively matching on
  elevation alone."
* :func:`plot_covariate_pairs` -- the same idea for >2 covariates: a scatter
  matrix coloured by treatment, so every covariate axis (climate, soil, ...) is
  inspected at once.
* :func:`plot_matched_pairs_covariate` -- treated and its matched control joined
  by a segment in covariate space; short segments = good matches.
* :func:`plot_matched_pairs_geographic` -- the same pairs on the map, so you can
  see *how far* across the landscape each control was drawn from its treated site.
* :func:`plot_candidate_control_pixels` -- the full control candidate pool drawn
  over NC, with the subset actually chosen as controls highlighted on top, so you
  can see which candidates the match drew on and how they sit against the pool.
* :func:`plot_score_map` -- matched pixels on the map coloured by a propensity /
  prognostic score, pairs connected: same colour at both ends of a segment = a
  close score match. The score-based sibling of ``plot_matched_pairs_geographic``.
* :func:`plot_score_overlap` -- treated-vs-control histogram of a score, the
  common-support / positivity check that says whether a match is even feasible.
* :func:`plot_prognostic_trajectory` -- the per-year prognostic risk over time
  (treated vs control), showing the dry/El Nino risk spikes the *per-year*
  prognostic model captures and a single collapsed score cannot.
* :func:`plot_covariate_vs_burn` -- the burn rate against one covariate (binned for
  a continuous layer, per-class for a categorical one), the "is this covariate
  interesting on its own?" picture; :func:`plot_covariates_vs_burn` lays that out
  for every covariate in the frame at once.
* :func:`plot_burned_area_vs_covariate` / :func:`plot_burned_area_covariate_heatmap`
  / :func:`plot_burned_area_and_covariate_by_year` -- the same covariate-vs-fire
  question asked mask-wide and **across years**, in absolute burned hectares off
  :func:`peatfire.modeling.build_mask_frame`: a per-year curve of area against the
  covariate, a year x covariate heat map with burning as colour, and the annual
  burned-area series with the covariate overlaid.
  :func:`plot_burned_area_vs_covariates_over_years` drives all three over a set of
  covariates and saves them.
* :func:`plot_raw_burn_rate_by_year` -- the raw, unadjusted burn rate per calendar
  year for treated vs control: the descriptive signal every model is built to
  explain, before any matching or adjustment.
* :func:`plot_raw_burn_rate_by_event_time` -- the same raw burn rate aligned by
  years since restoration (event time), the model-free companion to the DiD event
  study.
* :func:`plot_event_study` -- the one *temporal* diagnostic: the staggered-DiD ATT
  by time since restoration (event time), pre-period points shaded as the
  parallel-trends check. Consumes ``did.aggregate_att(..., kind="event")``.

Every function returns the matplotlib ``Figure`` (or ``Axes`` where an ``ax`` was
passed) so the caller can save it next to the other modeling figures.
"""

from __future__ import annotations

from typing import Optional, Sequence

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
import pandas as pd

from ..fire_products_comparison.fire_comparison import ANALYSIS_CRS, build_common_grid
from ..fire_products_comparison.plotting import OKABE_ITO, set_fire_style
from .covariates import (
    COVARIATES,
    TEMPORAL_COVARIATES,
    available_covariates,
    available_temporal_covariates,
    covariate_on_grid,
    get_covariate,
    list_covariates,
    temporal_covariate_on_grid,
)
from .frame import DEFAULT_RES_M

# Consistent treated/control colours + labels across every diagnostic here.
TREATED_COLOR = OKABE_ITO[3]   # vermillion
CONTROL_COLOR = OKABE_ITO[0]   # blue


# ---------------------------------------------------------------------------
# Shared treated/control split (mirrors matching.balance_table)
# ---------------------------------------------------------------------------
def _treated_mask(
    pixels: pd.DataFrame,
    treated_col: str,
    restoration_yr_col: str,
) -> pd.Series:
    """Boolean "is treated" per row, robust to which stage produced ``pixels``.

    On a **matched** set (carries ``site_id``) the static ``treated_col`` is
    already the right 0/1 split. On the **unmatched pixel-year panel**,
    ``treated_col`` flips per calendar year, so restoration-*site* membership --
    ``restoration_yr_col`` being non-null -- is the stable group label. This is
    exactly the rule :func:`matching.balance_table` uses, kept identical so the
    plots and the SMDs describe the same two groups.
    """
    if restoration_yr_col in pixels.columns and "site_id" not in pixels.columns:
        return pixels[restoration_yr_col].notna()
    return pixels[treated_col] == 1


def _unique_pixels(pixels: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Collapse a pixel-year panel to one row per physical pixel.

    Static covariates repeat across years, so plotting every pixel-year triple-
    counts a pixel and darkens the scatter meaninglessly. When ``x``/``y`` are
    present we drop to unique pixels first; otherwise we return the frame as-is.
    """
    if {"x", "y"}.issubset(pixels.columns):
        keep = ["x", "y", *[c for c in cols if c in pixels.columns]]
        return pixels.drop_duplicates(subset=["x", "y"])[keep]
    return pixels


# ---------------------------------------------------------------------------
# 1. Covariate coverage maps
# ---------------------------------------------------------------------------
def plot_covariate_maps(
    aoi: gpd.GeoDataFrame,
    names: Optional[Sequence[str]] = None,
    res_m: float = DEFAULT_RES_M,
    ncols: int = 3,
    treated: Optional[gpd.GeoDataFrame] = None,
    aoi_context: Optional[gpd.GeoDataFrame] = None,
    cmap: str = "viridis",
    robust: bool = True,
):
    """Map each covariate over the AOI -- the per-layer coverage / signal check.

    Warps every requested covariate onto the shared grid (continuous = area mean,
    categorical = majority class) and draws it as a panel, with the AOI outline
    and, optionally, the restoration (treated) polygons on top. A covariate that
    comes out visually *flat* (e.g. histosol % pinned near 90 across the peat
    frame) is telling you it carries no spatial signal to match on -- the
    motivation for adding the climate + soil layers.

    Parameters
    ----------
    aoi : GeoDataFrame
        Area to build the grid over and clip to (e.g. the 80% peat frame).
    names : sequence of str, optional
        Covariates to map. Defaults to every covariate currently on disk
        (:func:`available_covariates`).
    treated : GeoDataFrame, optional
        Restoration polygons to outline on each panel for orientation.
    aoi_context : GeoDataFrame, optional
        Extra outline (e.g. the NC boundary) for geographic reference.

    Returns
    -------
    matplotlib Figure
    """
    set_fire_style()
    if names is None:
        names = available_covariates()
    names = list(names)
    if not names:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no covariates on disk", ha="center", va="center")
        return fig

    grid = build_common_grid(aoi, res_m=res_m)
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()

    for ax, name in zip(axes_flat, names):
        da = covariate_on_grid(name, grid, aoi)
        if da is None:
            ax.text(0.5, 0.5, f"{name}\n(not on disk)", ha="center", va="center")
            ax.set_axis_off()
            continue

        vals = da.values.astype("float64")
        finite = np.isfinite(vals)
        if finite.any() and robust:
            vmin, vmax = (float(v) for v in np.nanpercentile(vals[finite], [2, 98]))
        elif finite.any():
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        else:
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:  # a constant layer: widen so imshow still renders
            vmin, vmax = vmin - 0.5, vmax + 0.5

        left, bottom, right, top = da.rio.bounds()
        spread = float(np.nanmax(vals[finite]) - np.nanmin(vals[finite])) if finite.any() else 0.0
        this_cmap = "Greys" if get_covariate(name).role == "categorical" else cmap
        im = ax.imshow(
            np.where(finite, vals, np.nan),
            extent=[left, right, bottom, top], origin="upper",
            cmap=this_cmap, vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        aoi.to_crs(grid.rio.crs).dissolve().boundary.plot(
            ax=ax, color="black", linewidth=0.5, alpha=0.5, zorder=3
        )
        if treated is not None:
            treated.to_crs(grid.rio.crs).boundary.plot(
                ax=ax, color=TREATED_COLOR, linewidth=0.9, zorder=4
            )
        if aoi_context is not None:
            aoi_context.to_crs(grid.rio.crs).boundary.plot(
                ax=ax, color="0.4", linewidth=0.7, zorder=2
            )
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect("equal")
        flat = "  (near-constant!)" if spread < 1e-9 else ""
        ax.set_title(f"{name}{flat}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, shrink=0.8)

    for ax in axes_flat[len(names):]:  # blank any unused panels
        ax.set_axis_off()
    fig.suptitle("Covariate coverage over the peat AOI", y=1.0)
    return fig


def plot_temporal_covariate_by_year(
    aoi: gpd.GeoDataFrame,
    name: str,
    years: Sequence[int],
    res_m: float = DEFAULT_RES_M,
    ncols: int = 3,
    treated: Optional[gpd.GeoDataFrame] = None,
    aoi_context: Optional[gpd.GeoDataFrame] = None,
    cmap: Optional[str] = None,
    center: Optional[float] | str = "auto",
    robust: bool = True,
):
    """Map one *per-year* (temporal) covariate across ``years``, one panel each.

    The temporal counterpart of :func:`plot_covariate_maps`: that function maps the
    *static* covariates in ``COVARIATES`` (the geographic-match layers), so the
    year-specific weather in ``TEMPORAL_COVARIATES`` -- ``precip`` / ``tmax`` /
    ``tmin`` / ``pdsi`` -- never appears there. This draws one of them across the
    panel years so you can *see* the drought / dry-year signal the DiD stage keys
    off (e.g. a ``treated:pdsi`` interaction).

    Every year is warped onto the same grid and drawn on **one shared color scale**
    so panels are comparable across years -- a wet year and a dry year sit on the
    same axis. For a signed, zero-centered index like scPDSI the scale is made
    symmetric about ``center`` (default ``"auto"`` picks 0 when the data straddle
    it) and defaults to the diverging ``RdBu`` colormap (red = drought/negative,
    blue = wet/positive); an unsigned layer like ``precip`` falls back to a robust
    ``viridis`` scale.

    Parameters
    ----------
    aoi : GeoDataFrame
        Area to build the grid over and clip to (e.g. the 80% peat frame).
    name : str
        A temporal covariate name (must be in ``TEMPORAL_COVARIATES``), e.g.
        ``"pdsi"``, ``"precip"``.
    years : sequence of int
        Calendar years to draw; a year with no raster on disk is skipped.
    treated, aoi_context : GeoDataFrame, optional
        Restoration polygons / extra outline to draw on each panel, as in
        :func:`plot_covariate_maps`.
    cmap : str, optional
        Override the colormap. Defaults to ``"RdBu"`` when centered, else
        ``"viridis"``.
    center : float or ``"auto"`` or None
        Diverging midpoint. ``"auto"`` (default) centers on 0 when the data span
        both signs (the scPDSI case) and is off otherwise; a float forces a
        centered symmetric scale; ``None`` disables centering.
    robust : bool
        Clip the color scale to the 2nd/98th percentile (symmetric 98th of the
        absolute value when centered) instead of the raw min/max.

    Returns
    -------
    matplotlib Figure
    """
    if name not in TEMPORAL_COVARIATES:
        raise KeyError(
            f"{name!r} is not a temporal covariate; choose one of "
            f"{sorted(TEMPORAL_COVARIATES)}"
        )
    set_fire_style()
    grid = build_common_grid(aoi, res_m=res_m)

    # Load every year up front so we can fix a single scale shared by all panels.
    layers = {
        year: temporal_covariate_on_grid(name, grid, aoi, year) for year in years
    }
    layers = {year: da for year, da in layers.items() if da is not None}
    if not layers:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5, f"{name}\n(no rasters on disk for {list(years)})",
            ha="center", va="center",
        )
        ax.set_axis_off()
        return fig

    finite_vals = np.concatenate(
        [da.values[np.isfinite(da.values)].astype("float64").ravel()
         for da in layers.values()]
    )
    has_signal = finite_vals.size > 0

    # Resolve the shared scale: symmetric about `center` for a signed index,
    # a plain robust range otherwise.
    if center == "auto":
        center = 0.0 if (has_signal and finite_vals.min() < 0 < finite_vals.max()) else None
    norm = None
    if center is not None and has_signal:
        if robust:
            vabs = float(np.nanpercentile(np.abs(finite_vals - center), 98)) or 1.0
        else:
            vabs = float(np.nanmax(np.abs(finite_vals - center))) or 1.0
        vmin, vmax = center - vabs, center + vabs
        norm = TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        this_cmap = cmap or "RdBu"
    else:
        if has_signal and robust:
            vmin, vmax = (float(v) for v in np.nanpercentile(finite_vals, [2, 98]))
        elif has_signal:
            vmin, vmax = float(finite_vals.min()), float(finite_vals.max())
        else:
            vmin, vmax = 0.0, 1.0
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        this_cmap = cmap or "viridis"

    nrows = int(np.ceil(len(layers) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False,
        layout="constrained",   # places the shared colorbar without overlap
    )
    axes_flat = axes.ravel()

    im = None
    for ax, (year, da) in zip(axes_flat, sorted(layers.items())):
        vals = da.values.astype("float64")
        finite = np.isfinite(vals)
        left, bottom, right, top = da.rio.bounds()
        im = ax.imshow(
            np.where(finite, vals, np.nan),
            extent=[left, right, bottom, top], origin="upper",
            cmap=this_cmap, norm=norm,
            vmin=None if norm is not None else vmin,
            vmax=None if norm is not None else vmax,
            interpolation="nearest",
        )
        aoi.to_crs(grid.rio.crs).dissolve().boundary.plot(
            ax=ax, color="black", linewidth=0.5, alpha=0.5, zorder=3
        )
        if treated is not None:
            treated.to_crs(grid.rio.crs).boundary.plot(
                ax=ax, color=TREATED_COLOR, linewidth=0.9, zorder=4
            )
        if aoi_context is not None:
            aoi_context.to_crs(grid.rio.crs).boundary.plot(
                ax=ax, color="0.4", linewidth=0.7, zorder=2
            )
        ax.set_xlim(left, right)
        ax.set_ylim(bottom, top)
        ax.set_aspect("equal")
        ax.set_title(f"{name} {year}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    for ax in axes_flat[len(layers):]:  # blank any unused panels
        ax.set_axis_off()

    # One shared colorbar down the right -- the whole point is a common scale.
    if im is not None:
        fig.colorbar(
            im, ax=axes.ravel().tolist(), location="right",
            shrink=0.85, aspect=40,
        )
    diverging = " (red = drought, blue = wet)" if norm is not None else ""
    fig.suptitle(f"{name} by year{diverging}")
    return fig


# ---------------------------------------------------------------------------
# 2. Treated vs control in covariate space
# ---------------------------------------------------------------------------
def plot_covariate_space(
    pixels: gpd.GeoDataFrame,
    xcov: str,
    ycov: str,
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    max_control: int = 20000,
    ax: Optional[plt.Axes] = None,
):
    """Scatter treated vs control pixels in the ``(xcov, ycov)`` covariate plane.

    The direct picture of the matching problem: treated (restoration) pixels and
    the control candidate pool, plotted on two covariate axes with marginal
    histograms. If one covariate has no spread its points collapse onto a vertical
    or horizontal line -- so if ``ycov`` is histosol % (pinned near 90) you *see*
    that the match has only the other axis (elevation) to work with, and that
    treated/control overlap poorly there. That is the argument for adding the
    climate / soil covariates.

    Controls are subsampled to ``max_control`` points for legibility (the pool is
    large); treated are always drawn in full and on top.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    is_t = _treated_mask(pixels, treated_col, restoration_yr_col)
    sub = _unique_pixels(pixels.assign(_t=is_t.values), [xcov, ycov, "_t"])
    t = sub[sub["_t"]]
    c = sub[~sub["_t"]]
    if len(c) > max_control:
        c = c.sample(max_control, random_state=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
    else:
        fig = ax.figure

    # main scatter
    ax.scatter(c[xcov], c[ycov], s=8, alpha=0.25, color=CONTROL_COLOR,
               edgecolor="none", label=f"control (n={len(c):,})", zorder=2)
    ax.scatter(t[xcov], t[ycov], s=14, alpha=0.7, color=TREATED_COLOR,
               edgecolor="none", label=f"treated (n={len(t):,})", zorder=3)
    ax.set_xlabel(xcov)
    ax.set_ylabel(ycov)
    ax.legend(loc="best")

    # marginal histograms on shared axes (make the collapsed covariate obvious)
    divider_x = ax.inset_axes([0, 1.02, 1, 0.18], sharex=ax)
    divider_y = ax.inset_axes([1.02, 0, 0.18, 1], sharey=ax)
    for grp, color in ((c, CONTROL_COLOR), (t, TREATED_COLOR)):
        divider_x.hist(grp[xcov].dropna(), bins=30, color=color, alpha=0.5, density=True)
        divider_y.hist(grp[ycov].dropna(), bins=30, color=color, alpha=0.5,
                       density=True, orientation="horizontal")
    divider_x.set_axis_off()
    divider_y.set_axis_off()
    ax.set_title(f"Treated vs control in ({xcov}, {ycov}) space", pad=40)

    # flag a covariate with essentially no spread
    for cov, axis_name in ((xcov, "x"), (ycov, "y")):
        spread = float(np.nanstd(sub[cov].to_numpy(dtype="float64")))
        if spread < 1e-9:
            ax.text(
                0.5, 0.5, f"{cov} is ~constant\n(nothing to match on this axis)",
                transform=ax.transAxes, ha="center", va="center",
                color=TREATED_COLOR, fontsize=10,
            )
    return fig


def plot_covariate_pairs(
    pixels: gpd.GeoDataFrame,
    covariates: Sequence[str],
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    max_control: int = 8000,
):
    """Scatter-matrix of treated vs control across several covariate axes.

    The many-covariate generalisation of :func:`plot_covariate_space`: an
    ``n x n`` grid of pairwise scatters (diagonal = per-covariate histograms),
    coloured by treatment. Use it once climate + soil are attached to see, at a
    glance, which covariates actually separate treated from control (so the match
    has something to work with) and which are redundant or flat.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    covariates = [c for c in covariates if c in pixels.columns]
    n = len(covariates)
    if n == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no covariates present", ha="center", va="center")
        return fig

    is_t = _treated_mask(pixels, treated_col, restoration_yr_col)
    sub = _unique_pixels(pixels.assign(_t=is_t.values), [*covariates, "_t"])
    t = sub[sub["_t"]]
    c = sub[~sub["_t"]]
    if len(c) > max_control:
        c = c.sample(max_control, random_state=0)

    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.6 * n), squeeze=False)
    for i, yi in enumerate(covariates):
        for j, xj in enumerate(covariates):
            ax = axes[i][j]
            if i == j:
                for grp, color in ((c, CONTROL_COLOR), (t, TREATED_COLOR)):
                    ax.hist(grp[xj].dropna(), bins=25, color=color, alpha=0.5, density=True)
            else:
                ax.scatter(c[xj], c[yi], s=5, alpha=0.2, color=CONTROL_COLOR, edgecolor="none")
                ax.scatter(t[xj], t[yi], s=8, alpha=0.6, color=TREATED_COLOR, edgecolor="none")
            if i == n - 1:
                ax.set_xlabel(xj)
            if j == 0:
                ax.set_ylabel(yi)
    handles = [
        plt.Line2D([0], [0], marker="o", ls="", color=TREATED_COLOR, label="treated"),
        plt.Line2D([0], [0], marker="o", ls="", color=CONTROL_COLOR, label="control"),
    ]
    fig.legend(handles=handles, loc="upper right")
    fig.suptitle("Covariate space: treated vs control", y=1.0)
    return fig


# ---------------------------------------------------------------------------
# 3. Matched pairs
# ---------------------------------------------------------------------------
def plot_matched_pairs_covariate(
    matched: gpd.GeoDataFrame,
    xcov: str,
    ycov: str,
    treated_col: str = "treated",
    pair_col: str = "pair_id",
    max_pairs: int = 3000,
    ax: Optional[plt.Axes] = None,
):
    """Draw each matched treated<->control pair as a segment in covariate space.

    After :func:`matching.match_controls`, every treated pixel is joined to its
    control(s) by ``pair_col``. This plots treated (filled) and control (open)
    points on the two covariate axes and connects each pair with a thin segment:
    **short segments = the control is a close twin** on these covariates, long
    segments = a stretched match (raise the caliper concern). It is the
    per-pair companion to the aggregate love plot.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    if pair_col not in matched.columns:
        raise ValueError(
            f"{pair_col!r} not in matched set; run matching.match_controls first."
        )
    t = matched[matched[treated_col] == 1]
    c = matched[matched[treated_col] == 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
    else:
        fig = ax.figure

    # connect pairs: join treated to its control(s) on pair_id
    tp = t.set_index(pair_col)
    pairs_drawn = 0
    for pid, crow in c.set_index(pair_col).iterrows():
        if pid not in tp.index:
            continue
        trow = tp.loc[pid]
        tx = trow[xcov].iloc[0] if hasattr(trow[xcov], "iloc") else trow[xcov]
        ty = trow[ycov].iloc[0] if hasattr(trow[ycov], "iloc") else trow[ycov]
        ax.plot([tx, crow[xcov]], [ty, crow[ycov]], color="0.7", lw=0.5, zorder=1)
        pairs_drawn += 1
        if pairs_drawn >= max_pairs:
            break

    ax.scatter(c[xcov], c[ycov], s=18, facecolor="none", edgecolor=CONTROL_COLOR,
               linewidths=0.8, label="control", zorder=3)
    ax.scatter(t[xcov], t[ycov], s=18, color=TREATED_COLOR, edgecolor="none",
               label="treated", zorder=4)
    ax.set_xlabel(xcov)
    ax.set_ylabel(ycov)
    ax.set_title(f"Matched pairs in ({xcov}, {ycov}) space\n(line = one treated↔control match)")
    ax.legend(loc="best")
    return fig


def plot_matched_pairs_geographic(
    matched: gpd.GeoDataFrame,
    aoi: Optional[gpd.GeoDataFrame] = None,
    treated_col: str = "treated",
    pair_col: str = "pair_id",
    aoi_context: Optional[gpd.GeoDataFrame] = None,
    max_pairs: int = 2000,
    ax: Optional[plt.Axes] = None,
):
    """Map matched treated<->control pairs, connecting each pair across the landscape.

    The geographic complement to :func:`plot_matched_pairs_covariate`: it shows
    *where* each control was drawn from relative to its treated restoration pixel.
    Long connecting lines mean covariate-similar controls were only found far away
    -- a spatial-confounding flag (distant controls may differ in unmeasured ways,
    e.g. climate), which is precisely why adding climate/soil covariates and, if
    needed, an exact-match key matters.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    m = matched.to_crs(ANALYSIS_CRS)
    t = m[m[treated_col] == 1]
    c = m[m[treated_col] == 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    if aoi is not None:
        aoi.to_crs(ANALYSIS_CRS).dissolve().boundary.plot(
            ax=ax, color="0.6", linewidth=0.6, alpha=0.6, zorder=1
        )
    if aoi_context is not None:
        aoi_context.to_crs(ANALYSIS_CRS).boundary.plot(
            ax=ax, color="0.4", linewidth=0.8, zorder=1
        )

    tp = t.set_index(pair_col)
    drawn = 0
    for pid, crow in c.set_index(pair_col).iterrows():
        if pid not in tp.index:
            continue
        trow = tp.loc[pid]
        tgeom = trow.geometry.iloc[0] if hasattr(trow.geometry, "iloc") else trow.geometry
        ax.plot([tgeom.x, crow.geometry.x], [tgeom.y, crow.geometry.y],
                color="0.75", lw=0.4, zorder=2)
        drawn += 1
        if drawn >= max_pairs:
            break

    ax.scatter(c.geometry.x, c.geometry.y, s=10, facecolor="none",
               edgecolor=CONTROL_COLOR, linewidths=0.7, label="control", zorder=3)
    ax.scatter(t.geometry.x, t.geometry.y, s=14, color=TREATED_COLOR,
               edgecolor="none", label="treated", zorder=4)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Matched pairs across NC\n(line = treated↔its matched control)")
    ax.legend(loc="best")
    return fig


def plot_candidate_control_pixels(
    pixels: gpd.GeoDataFrame,
    matched: Optional[gpd.GeoDataFrame] = None,
    aoi: Optional[gpd.GeoDataFrame] = None,
    aoi_context: Optional[gpd.GeoDataFrame] = None,
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    show_treated: bool = True,
    max_candidates: int = 40000,
    ax: Optional[plt.Axes] = None,
):
    """Map the control candidate pool -- and, if matched, the selected controls -- over NC.

    The geographic "where did the controls come from" picture. It draws the full
    control **candidate pool** (every peat pixel outside the treated + spillover
    halo, i.e. ``treated == 0`` in the pre-match set from
    :func:`matching.get_treated_and_control_pixels`) as a faint backdrop, so you
    can see the pool's spatial coverage across the peat frame. When a ``matched``
    set is passed, the subset actually chosen as **controls**
    (:func:`matching.match_controls`, ``treated == 0``) is highlighted on top --
    the direct picture of *which* candidates the match drew on and whether they
    cluster near the treated sites or scatter across the state.

    Complements :func:`plot_matched_pairs_geographic` (which connects each treated
    pixel to its matched control): here the emphasis is the candidate pool as a
    whole against the handful of selected controls, not the pairings.

    Parameters
    ----------
    pixels : GeoDataFrame
        Pre-match pixel (or pixel-year) set from
        :func:`matching.get_treated_and_control_pixels`. The candidate pool is its
        control group; a pixel-year panel is collapsed to unique pixels first.
    matched : GeoDataFrame, optional
        Matched set from :func:`matching.match_controls`. Its ``treated == 0`` rows
        are drawn as the selected controls; when omitted only the pool is shown.
    aoi : GeoDataFrame, optional
        Peat AOI outline (e.g. the 80% peat frame) for orientation.
    aoi_context : GeoDataFrame, optional
        Extra outline (e.g. the NC boundary) for geographic reference.
    show_treated : bool, default True
        Also plot the treated (restoration) pixels for context.
    max_candidates : int, default 40000
        Subsample the candidate-pool scatter to this many points for legibility
        (the pool is large); the selected controls and treated are always full.

    Returns
    -------
    matplotlib Figure
    """
    set_fire_style()
    px = pixels.to_crs(ANALYSIS_CRS)
    is_t = _treated_mask(px, treated_col, restoration_yr_col)
    # Collapse a pixel-year panel to one row per physical pixel, keeping geometry
    # (unlike `_unique_pixels`, which drops it -- we need it for the map).
    uniq = px.assign(_t=is_t.values)
    if {"x", "y"}.issubset(uniq.columns):
        uniq = uniq.drop_duplicates(subset=["x", "y"])
    pool = uniq[~uniq["_t"]]
    treated = uniq[uniq["_t"]]

    pool_plot = pool.sample(max_candidates, random_state=0) if len(pool) > max_candidates else pool

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    if aoi is not None:
        aoi.to_crs(ANALYSIS_CRS).dissolve().boundary.plot(
            ax=ax, color="0.6", linewidth=0.6, alpha=0.6, zorder=1
        )
    if aoi_context is not None:
        aoi_context.to_crs(ANALYSIS_CRS).boundary.plot(
            ax=ax, color="0.4", linewidth=0.8, zorder=1
        )

    ax.scatter(
        pool_plot.geometry.x, pool_plot.geometry.y, s=4, alpha=0.25, color="0.6",
        edgecolor="none", label=f"candidate pool (n={len(pool):,})", zorder=2,
    )
    if matched is not None:
        mc = matched.to_crs(ANALYSIS_CRS)
        sel = mc[mc[treated_col] == 0].drop_duplicates(subset=["x", "y"]) \
            if {"x", "y"}.issubset(mc.columns) else mc[mc[treated_col] == 0]
        ax.scatter(
            sel.geometry.x, sel.geometry.y, s=14, color=CONTROL_COLOR,
            edgecolor="none", alpha=0.8,
            label=f"matched control (n={len(sel):,})", zorder=3,
        )
    if show_treated and len(treated):
        ax.scatter(
            treated.geometry.x, treated.geometry.y, s=14, color=TREATED_COLOR,
            edgecolor="none", label=f"treated (n={len(treated):,})", zorder=4,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Candidate pool and selected controls across NC")
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------------------
# 3b. Score diagnostics (propensity / prognostic)
# ---------------------------------------------------------------------------
def plot_score_map(
    matched: gpd.GeoDataFrame,
    score_col: str,
    scores: Optional[gpd.GeoDataFrame] = None,
    aoi: Optional[gpd.GeoDataFrame] = None,
    treated_col: str = "treated",
    pair_col: str = "pair_id",
    aoi_context: Optional[gpd.GeoDataFrame] = None,
    cmap: str = "viridis",
    max_pairs: int = 2000,
    ax: Optional[plt.Axes] = None,
):
    """Map each matched pixel coloured by a **score**, with matched pairs connected.

    The score-based companion to :func:`plot_matched_pairs_geographic`: instead of
    plain treated/control markers, every pixel is filled by its propensity or
    prognostic score (``score_col`` -- e.g. ``"pscore"``, ``"phat"``, a per-vintage
    ``"psm_2020"`` or a per-year ``"phat_2021"``), and each treated pixel is joined
    to its matched control by a thin segment. A good score match shows the two ends
    of every segment in nearly the **same colour**; a segment whose ends differ in
    colour is a pixel matched to a control of unlike predicted risk. Treated pixels
    are drawn as squares (black edge), controls as circles, so shape reads treatment
    and colour reads the score.

    ``score_col`` is looked up on ``matched`` if present; otherwise pass ``scores``
    (any per-pixel frame with ``x``/``y`` and ``score_col`` -- e.g. the scored panel
    from :func:`add_prognostic_score_series`), and it is joined on ``(x, y)``. This
    matters for the event-time match, whose output carries pixel ids but not the
    (per-cohort, ragged) score columns.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    m = matched.to_crs(ANALYSIS_CRS).copy()
    if score_col not in m.columns:
        if scores is None:
            raise ValueError(
                f"{score_col!r} not on `matched`; pass scores=<per-pixel frame with "
                f"x, y, {score_col!r}> (e.g. the scored panel) to look it up."
            )
        lut = scores.drop_duplicates(subset=["x", "y"])[["x", "y", score_col]]
        m = m.merge(lut, on=["x", "y"], how="left")

    t = m[m[treated_col] == 1]
    c = m[m[treated_col] == 0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 7))
    else:
        fig = ax.figure

    if aoi is not None:
        aoi.to_crs(ANALYSIS_CRS).dissolve().boundary.plot(
            ax=ax, color="0.6", linewidth=0.6, alpha=0.6, zorder=1
        )
    if aoi_context is not None:
        aoi_context.to_crs(ANALYSIS_CRS).boundary.plot(
            ax=ax, color="0.4", linewidth=0.8, zorder=1
        )

    # connect each treated pixel to its matched control(s)
    tp = t.set_index(pair_col)
    drawn = 0
    for pid, crow in c.set_index(pair_col).iterrows():
        if pid not in tp.index:
            continue
        trow = tp.loc[pid]
        tgeom = trow.geometry.iloc[0] if hasattr(trow.geometry, "iloc") else trow.geometry
        ax.plot([tgeom.x, crow.geometry.x], [tgeom.y, crow.geometry.y],
                color="0.75", lw=0.4, zorder=2)
        drawn += 1
        if drawn >= max_pairs:
            break

    # shared colour scale across treated + control so colours are comparable
    vals = m[score_col].to_numpy(dtype="float64")
    finite = vals[np.isfinite(vals)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    ax.scatter(c.geometry.x, c.geometry.y, c=c[score_col], cmap=cmap, vmin=vmin,
               vmax=vmax, s=22, marker="o", edgecolor="0.5", linewidths=0.4,
               label="control", zorder=3)
    sc = ax.scatter(t.geometry.x, t.geometry.y, c=t[score_col], cmap=cmap, vmin=vmin,
                    vmax=vmax, s=34, marker="s", edgecolor="black", linewidths=0.6,
                    label="treated", zorder=4)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=score_col)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"{score_col} per pixel, matched pairs connected\n"
        "(square = treated, circle = control; same colour at both ends = close match)"
    )
    ax.legend(loc="best")
    return fig


def plot_score_overlap(
    pixels: gpd.GeoDataFrame,
    score_col: str,
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    bins: int = 30,
    ax: Optional[plt.Axes] = None,
):
    """Common-support histogram of a score: treated vs control overlap.

    The standard feasibility check for score matching. It overlays the treated and
    control **distributions of ``score_col``** (propensity or prognostic). Healthy
    **overlap** -- the two histograms share a range -- means a control with a
    matching score exists for most treated pixels; a treated mass sitting where no
    controls are (poor common support) is exactly where matching will drop treated
    pixels or stretch calipers. For a *propensity* score this is the canonical
    positivity/overlap plot; for a *prognostic* score it shows whether treated and
    control baseline risk are comparable at all.

    Draw it on the **pre-match** scored panel to judge whether the match is even
    feasible. Returns the matplotlib Figure.
    """
    set_fire_style()
    if score_col not in pixels.columns:
        raise ValueError(f"{score_col!r} not in pixels; run the score builder first.")
    is_t = _treated_mask(pixels, treated_col, restoration_yr_col)
    sub = _unique_pixels(pixels.assign(_t=is_t.values), [score_col, "_t"])
    t = sub.loc[sub["_t"], score_col].dropna()
    c = sub.loc[~sub["_t"], score_col].dropna()

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
    else:
        fig = ax.figure

    lo = float(min(t.min(), c.min())) if len(t) and len(c) else 0.0
    hi = float(max(t.max(), c.max())) if len(t) and len(c) else 1.0
    edges = np.linspace(lo, hi, bins + 1)
    ax.hist(c, bins=edges, color=CONTROL_COLOR, alpha=0.5, density=True,
            label=f"control (n={len(c):,})")
    ax.hist(t, bins=edges, color=TREATED_COLOR, alpha=0.5, density=True,
            label=f"treated (n={len(t):,})")
    # mark the treated range that has no control support
    if len(c):
        c_lo, c_hi = float(c.min()), float(c.max())
        if t.max() > c_hi:
            ax.axvspan(c_hi, float(t.max()), color=TREATED_COLOR, alpha=0.08, zorder=0)
        if t.min() < c_lo:
            ax.axvspan(float(t.min()), c_lo, color=TREATED_COLOR, alpha=0.08, zorder=0)

    ax.set_xlabel(score_col)
    ax.set_ylabel("density")
    ax.set_title(f"Common support: {score_col} (treated vs control)")
    ax.legend(loc="best")
    return fig


def plot_prognostic_trajectory(
    pixels: gpd.GeoDataFrame,
    prefix: str = "phat_",
    response: Optional[str] = "burned",
    year_col: str = "year",
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    ax: Optional[plt.Axes] = None,
):
    """Per-year prognostic fire risk over time -- the point of the *per-year* model.

    Plots the mean **per-year prognostic score** (``prefix<year>`` columns from
    :func:`add_prognostic_score_series`) for treated vs control pixels across the
    study years, with an inter-quartile band. Because a *separate* model is fit each
    year, this curve shows how baseline fire risk **shifts over time** -- the spikes
    are the dry / El Nino years the single collapsed score cannot represent. If the
    per-year ``response`` is present, the observed never-treated burn rate is
    overlaid (dashed) so you can see the prognostic scores tracking the real
    year-to-year fire signal.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    cols = sorted(
        (c for c in pixels.columns
         if c.startswith(prefix) and c[len(prefix):].isdigit()),
        key=lambda c: int(c[len(prefix):]),
    )
    if not cols:
        raise ValueError(
            f"no {prefix!r}<year> columns found; run add_prognostic_score_series first."
        )
    years = [int(c[len(prefix):]) for c in cols]

    is_t = _treated_mask(pixels, treated_col, restoration_yr_col)
    uniq = _unique_pixels(pixels.assign(_t=is_t.values), [*cols, "_t"])
    t = uniq[uniq["_t"]]
    c = uniq[~uniq["_t"]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for grp, color, label in ((c, CONTROL_COLOR, "control"), (t, TREATED_COLOR, "treated")):
        if grp.empty:
            continue
        mean = [grp[col].mean() for col in cols]
        q25 = [grp[col].quantile(0.25) for col in cols]
        q75 = [grp[col].quantile(0.75) for col in cols]
        ax.plot(years, mean, "-o", color=color, label=f"{label} (mean prognostic)", zorder=3)
        ax.fill_between(years, q25, q75, color=color, alpha=0.15, zorder=1)

    # observed never-treated burn rate per year, if the response is available
    if response and response in pixels.columns and year_col in pixels.columns:
        nt = pixels[_treated_mask(pixels, treated_col, restoration_yr_col).values == False]
        obs = nt.groupby(year_col)[response].mean()
        obs = obs.reindex(years)
        ax.plot(years, obs.values, "--", color="0.4", lw=1.2,
                label="observed control burn rate", zorder=2)

    ax.set_xticks(years)
    ax.set_xlabel("year")
    ax.set_ylabel("prognostic fire risk  E[burn | X, untreated]")
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------------------
# 3c. Raw burned-rate signal (unadjusted, descriptive)
# ---------------------------------------------------------------------------
def _restoration_year_per_row(
    frame: pd.DataFrame,
    restoration_yr_col: str,
    site_col: Optional[str],
    is_treated: pd.Series,
) -> pd.Series:
    """Per-row restoration (cohort) year, filling controls from their site's year.

    Treated rows carry their own restoration year in ``restoration_yr_col``
    (e.g. ``End_Yr``). Matched controls usually do not -- they inherit a treated
    site only through a ``site_col`` (``site_id`` in a matched set, ``Proj_Name``
    upstream) -- so we build a ``site -> restoration year`` lookup from the treated
    rows and map it onto every row, which aligns each matched control to its
    treated site's cohort. Rows that resolve to no year (the never-treated control
    pool, whose ``site_col`` is null) stay ``NaN`` and are handled by the caller as
    a flat reference rather than an event-time curve.
    """
    ry = pd.Series(np.nan, index=frame.index, dtype="float64")
    if restoration_yr_col in frame.columns:
        ry = pd.to_numeric(frame[restoration_yr_col], errors="coerce")

    if site_col is None:  # auto-detect the site key (matched vs upstream panel)
        site_col = next((c for c in ("site_id", "Proj_Name") if c in frame.columns), None)
    if site_col is not None and site_col in frame.columns \
            and restoration_yr_col in frame.columns:
        treated_rows = frame.loc[is_treated.values]
        lookup = (
            pd.to_numeric(treated_rows[restoration_yr_col], errors="coerce")
            .groupby(treated_rows[site_col]).first().dropna()
        )
        if not lookup.empty:
            ry = ry.fillna(frame[site_col].map(lookup))
    return ry


def _binomial_se(mean: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Standard error of a 0/1 rate: ``sqrt(p(1-p)/n)`` (NaN where ``n == 0``)."""
    count = count.astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(mean * (1.0 - mean) / count)
    return np.where(count > 0, se, np.nan)


def plot_raw_burn_rate_by_year(
    frame: pd.DataFrame,
    response: str = "burned",
    year_col: str = "year",
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    show_error: bool = True,
    ax: Optional[plt.Axes] = None,
):
    """Raw burn rate per **calendar year**, treated vs control -- the unadjusted signal.

    The plainest descriptive picture of the outcome: the observed mean of
    ``response`` (the 0/1 ``burned`` flag) for each calendar year, split into the
    treated (restoration-site) and control groups with no covariate adjustment,
    matching, or modelling. This is what the DiD and the fitted models are built to
    explain -- eyeball it first to see the level difference between the two groups
    and the shared year-to-year swings (the dry / El Nino spikes) before any
    estimator touches it.

    Group membership is the *stable* treated/control split (:func:`_treated_mask`),
    so a treated pixel counts as treated in every year, including the years before
    its site was restored -- this is a group-level rate over calendar time, not an
    event-time alignment (see :func:`plot_raw_burn_rate_by_event_time` for that).
    With ``show_error`` a binomial standard-error band ``sqrt(p(1-p)/n)`` is shaded
    around each line.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    for col in (response, year_col):
        if col not in frame.columns:
            raise ValueError(f"{col!r} not in frame; need {year_col!r} and {response!r}.")
    is_t = _treated_mask(frame, treated_col, restoration_yr_col)
    df = (
        frame[[year_col, response]]
        .assign(_t=is_t.values)
        .dropna(subset=[response])
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    for grp_val, color, label in (
        (True, TREATED_COLOR, "treated"),
        (False, CONTROL_COLOR, "control"),
    ):
        g = df[df["_t"] == grp_val]
        if g.empty:
            continue
        stat = g.groupby(year_col)[response].agg(["mean", "count"])
        years = stat.index.to_numpy()
        mean = stat["mean"].to_numpy(dtype="float64")
        ax.plot(years, mean, "-o", color=color, zorder=3,
                label=f"{label} (n={len(g):,} pixel-years)")
        if show_error:
            se = _binomial_se(mean, stat["count"].to_numpy())
            ax.fill_between(years, mean - se, mean + se, color=color, alpha=0.15, zorder=1)

    ax.set_xticks(sorted(df[year_col].unique()))
    ax.set_xlabel("calendar year")
    ax.set_ylabel(f"raw burn rate  mean({response})")
    ax.set_title(
        "Raw burn rate by calendar year (unadjusted)\n"
        "treated = restoration sites, control = comparison pixels"
    )
    ax.legend(loc="best")
    return fig


def plot_raw_burn_rate_by_event_time(
    frame: pd.DataFrame,
    response: str = "burned",
    year_col: str = "year",
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    site_col: Optional[str] = None,
    show_error: bool = True,
    ax: Optional[plt.Axes] = None,
):
    """Raw burn rate by **years after treatment** (event time) -- unadjusted.

    The descriptive, model-free sibling of :func:`plot_event_study`. Each row's
    event time is ``year - restoration_year`` (0 = the restoration year), and the
    observed burn rate is averaged within each event-time bin for the treated
    group. Because it is raw means rather than a differenced ATT, the two things to
    read here are the **level shift** in the treated curve around event time 0 (the
    canal blocks going in) and, if controls are aligned, whether treated and
    control moved together *before* it -- the eyeball parallel-trends check.

    Controls are aligned to their treated site's restoration year via ``site_col``
    (``site_id`` in a matched set, else ``Proj_Name``): a matched control inherits
    its site's cohort and so gets its own event-time curve. On the upstream pixel
    panel, where the control pool has no site, controls resolve to no event time
    and are instead drawn as a flat dashed **pooled-mean reference** (their overall
    burn rate), mirroring the never-treated overlay in
    :func:`plot_prognostic_trajectory`. The onset boundary (``-0.5``) and the
    shaded pre-period match :func:`plot_event_study`.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    for col in (response, year_col):
        if col not in frame.columns:
            raise ValueError(f"{col!r} not in frame; need {year_col!r} and {response!r}.")
    is_t = _treated_mask(frame, treated_col, restoration_yr_col)
    ry = _restoration_year_per_row(frame, restoration_yr_col, site_col, is_t)
    event_time = pd.to_numeric(frame[year_col], errors="coerce") - ry
    df = pd.DataFrame({
        "event_time": event_time.round(),
        response: pd.to_numeric(frame[response], errors="coerce").values,
        "_t": is_t.values,
    }).dropna(subset=[response])
    aligned = df.dropna(subset=["event_time"]).copy()
    aligned["event_time"] = aligned["event_time"].astype(int)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    # References: treatment onset and the shaded pre-restoration window.
    onset = -0.5  # boundary between the pre period (-1) and event time 0
    if not aligned.empty:
        et_min = int(aligned["event_time"].min())
        if et_min < 0:
            ax.axvspan(et_min - 0.5, onset, color="0.85", alpha=0.35, zorder=0)
    ax.axvline(onset, color="0.4", lw=1.0, ls=":", zorder=1)

    control_curve_drawn = False
    for grp_val, color, label in (
        (True, TREATED_COLOR, "treated"),
        (False, CONTROL_COLOR, "control"),
    ):
        g = aligned[aligned["_t"] == grp_val]
        if g.empty:
            continue
        stat = g.groupby("event_time")[response].agg(["mean", "count"])
        et = stat.index.to_numpy()
        mean = stat["mean"].to_numpy(dtype="float64")
        ax.plot(et, mean, "-o", color=color, zorder=3,
                label=f"{label} (n={len(g):,} pixel-years)")
        if show_error:
            se = _binomial_se(mean, stat["count"].to_numpy())
            ax.fill_between(et, mean - se, mean + se, color=color, alpha=0.15, zorder=1)
        if grp_val is False:
            control_curve_drawn = True

    # Never-treated controls carry no event time: draw their pooled burn rate as a
    # flat dashed reference so the treated curve has a baseline to read against.
    if not control_curve_drawn:
        ref = df.loc[~df["_t"], response]
        if not ref.empty:
            ax.axhline(float(ref.mean()), color="0.4", lw=1.2, ls="--", zorder=2,
                       label=f"control pooled mean (n={len(ref):,} pixel-years)")

    if not aligned.empty:
        ax.set_xticks(sorted(aligned["event_time"].unique()))
    ax.set_xlabel("event time (years since restoration)")
    ax.set_ylabel(f"raw burn rate  mean({response})")
    ax.set_title(
        "Raw burn rate by years after treatment (unadjusted)\n"
        "event time 0 = restoration year; pre-period is left of the onset line"
    )
    ax.legend(loc="best")
    return fig


# ---------------------------------------------------------------------------
# 3d. Covariate vs burned response (the "is this covariate interesting?" picture)
# ---------------------------------------------------------------------------
def _is_categorical_covariate(
    frame: pd.DataFrame, covariate: str, override: Optional[bool]
) -> bool:
    """Decide whether ``covariate`` should be drawn as categories or binned.

    Honours an explicit ``override`` first; otherwise trusts the covariate
    registry's ``role`` (land cover / drainage class are ``"categorical"``), and
    finally falls back to a dtype/cardinality heuristic for columns that are not
    registered (a non-numeric dtype, or a numeric column with few distinct values,
    reads as categorical). Keeps :func:`plot_covariate_vs_burn` usable on any frame
    column, registered or derived.
    """
    if override is not None:
        return override
    try:
        return get_covariate(covariate).role == "categorical"
    except KeyError:
        pass
    s = frame[covariate]
    return (not pd.api.types.is_numeric_dtype(s)) or s.nunique(dropna=True) <= 12


def _burn_rate_by_bin(
    x: np.ndarray, burned: np.ndarray, bins: int
) -> pd.DataFrame:
    """Mean burn rate within equal-count (quantile) bins of a continuous covariate.

    Returns one row per bin with the bin's mean covariate value ``x`` (the plotting
    abscissa), the mean ``rate`` of ``burned``, the pixel-year ``count``, and the
    binomial ``se``. Quantile bins keep ``count`` roughly balanced so a sparse tail
    does not produce a wild, single-pixel rate; degenerate edges (a covariate with
    fewer distinct values than ``bins``) collapse via ``duplicates="drop"``.
    """
    q = pd.qcut(x, q=min(bins, np.unique(x).size), duplicates="drop")
    g = pd.DataFrame({"x": x, "burned": burned, "bin": q}).groupby("bin", observed=True)
    out = g.agg(x=("x", "mean"), rate=("burned", "mean"), count=("burned", "size"))
    out["se"] = _binomial_se(out["rate"].to_numpy(), out["count"].to_numpy())
    return out.reset_index(drop=True)


def plot_covariate_vs_burn(
    frame: pd.DataFrame,
    covariate: str,
    response: str = "burned",
    bins: int = 12,
    categorical: Optional[bool] = None,
    by_treatment: bool = False,
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    show_error: bool = True,
    max_categories: int = 25,
    ax: Optional[plt.Axes] = None,
):
    """Burn rate vs one covariate -- the covariate-against-fire 2-D picture.

    The descriptive "is this covariate interesting on its own?" plot the project
    notes ask for. Because ``burned`` is a 0/1 pixel-year flag, a raw scatter of it
    against a covariate is uninformative (every point sits on 0 or 1), so this bins
    the covariate and plots the **mean burn rate within each bin** -- i.e. the
    burned *fraction* of pixel-years at that covariate level -- which is the sample
    analogue of the burn *probability* the models estimate:

    * **continuous** covariate (drainage, elevation, climate, ...): equal-count
      (quantile) bins, drawn as a line of burn rate vs the bin's mean covariate
      value, with a binomial standard-error band. A downward slope means fire gets
      *less* likely as the covariate rises (e.g. better-drained/drier pixels burning
      less), the raw shape the fitted coefficient summarises.
    * **categorical** covariate (land cover, drainage class): one point (or bar) of
      burn rate per class, so you can read which classes carry the fire. Classes are
      ordered by burn rate and capped at ``max_categories``.

    Every pixel-*year* is one observation (the frame is **not** collapsed to unique
    pixels), so the rate is a true exposure over the panel. With
    ``by_treatment=True`` the treated and control groups are drawn separately (the
    stable split from :func:`_treated_mask`) -- the picture of whether a covariate's
    fire gradient differs between restored and comparison land, and the visual
    companion to putting that covariate alongside ``treated`` in
    :func:`peatfire.modeling.fit_ols_clustered`.

    Whether the covariate is treated as categorical is auto-detected from the
    covariate registry / dtype; override with ``categorical=True/False``.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    for col in (covariate, response):
        if col not in frame.columns:
            raise ValueError(f"{col!r} not in frame.")
    is_cat = _is_categorical_covariate(frame, covariate, categorical)

    df = frame[[covariate, response]].copy()
    df[response] = pd.to_numeric(df[response], errors="coerce")
    if by_treatment:
        df["_t"] = _treated_mask(frame, treated_col, restoration_yr_col).values
    df = df.dropna(subset=[covariate, response])

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
    else:
        fig = ax.figure

    if by_treatment:
        groups = [
            (df[df["_t"]], TREATED_COLOR, "treated"),
            (df[~df["_t"]], CONTROL_COLOR, "control"),
        ]
    else:
        groups = [(df, TREATED_COLOR, "all pixels")]

    if is_cat:
        # burn rate per class; order categories by the pooled rate for readability
        order = (
            df.groupby(covariate, observed=True)[response].mean()
            .sort_values(ascending=False).index[:max_categories]
        )
        pos = np.arange(len(order))
        width = 0.8 / len(groups)
        for k, (g, color, label) in enumerate(groups):
            if g.empty:
                continue
            stat = g.groupby(covariate, observed=True)[response].agg(["mean", "size"])
            stat = stat.reindex(order)
            rate = stat["mean"].to_numpy(dtype="float64")
            offset = (k - (len(groups) - 1) / 2) * width
            ax.bar(pos + offset, rate, width=width, color=color, alpha=0.75,
                   label=f"{label} (n={int(g.shape[0]):,})", zorder=2)
            if show_error:
                se = _binomial_se(rate, stat["size"].to_numpy())
                ax.errorbar(pos + offset, rate, yerr=se, fmt="none",
                            ecolor="0.3", elinewidth=0.8, capsize=2, zorder=3)
        ax.set_xticks(pos)
        ax.set_xticklabels([str(o) for o in order], rotation=45, ha="right")
    else:
        for g, color, label in groups:
            if g.empty or g[covariate].nunique() < 2:
                continue
            binned = _burn_rate_by_bin(
                g[covariate].to_numpy(dtype="float64"),
                g[response].to_numpy(dtype="float64"),
                bins,
            )
            ax.plot(binned["x"], binned["rate"], "-o", color=color, zorder=3,
                    label=f"{label} (n={int(g.shape[0]):,} pixel-years)")
            if show_error:
                ax.fill_between(
                    binned["x"], binned["rate"] - binned["se"],
                    binned["rate"] + binned["se"], color=color, alpha=0.15, zorder=1,
                )

    ax.set_xlabel(covariate)
    ax.set_ylabel(f"burn rate  mean({response})")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Burn rate vs {covariate}")
    ax.legend(loc="best")
    return fig


def plot_covariates_vs_burn(
    frame: pd.DataFrame,
    covariates: Optional[Sequence[str]] = None,
    response: str = "burned",
    bins: int = 12,
    by_treatment: bool = False,
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    ncols: int = 3,
    max_categories: int = 25,
):
    """Panel of :func:`plot_covariate_vs_burn` over every covariate in the frame.

    The one-call overview: lay out the burn-rate-vs-covariate picture for each
    covariate as its own panel, so drainage and all the other layers can be scanned
    against fire at once (the project note "plot each covariate against burned area
    -- they are interesting in and of themselves"). Defaults to every registered
    covariate present as a column in ``frame`` (static + per-year); pass
    ``covariates`` to pick a subset or a custom order. ``by_treatment`` and ``bins``
    are forwarded to each panel.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    if covariates is None:
        registered = set(COVARIATES) | set(TEMPORAL_COVARIATES)
        covariates = [c for c in frame.columns if c in registered]
    covariates = [c for c in covariates if c in frame.columns]
    if not covariates:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "no covariates in frame", ha="center", va="center")
        return fig

    nrows = int(np.ceil(len(covariates) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.6 * ncols, 4.2 * nrows), squeeze=False
    )
    axes_flat = axes.ravel()
    for ax, name in zip(axes_flat, covariates):
        plot_covariate_vs_burn(
            frame, name, response=response, bins=bins, by_treatment=by_treatment,
            treated_col=treated_col, restoration_yr_col=restoration_yr_col,
            max_categories=max_categories, ax=ax,
        )
    for ax in axes_flat[len(covariates):]:  # blank any unused panels
        ax.set_axis_off()
    fig.suptitle(f"Burn rate vs each covariate  (response = {response})", y=1.0)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3e. Burned AREA vs covariate, across years (mask-wide, descriptive)
#
# The §3d functions above answer "does this covariate shift the burn *rate*?" on
# the modeling frame. These answer the mask-wide, absolute question -- "how many
# hectares of the peat AOI burned, at which covariate levels, in which years?" --
# off `frame.build_mask_frame`. Same binning idea, three views of the same table:
# a covariate-vs-area curve per year, a year x covariate heat map, and the plain
# annual series with the covariate overlaid.
# ---------------------------------------------------------------------------
# What each metric means, for axis/colorbar labels. `area_ha` is the absolute
# burned hectares in a cell of the table; `rate` is the burned *fraction* of that
# cell's pixel-years (exposure-normalised, comparable across bins of unequal
# size); `share` is the cell's percentage of that **year's** total burned area
# (year-normalised: it shows *where* a year's fire sat in covariate space, with
# the year's overall size divided out).
_BURN_METRICS = {
    "area_ha": "burned area (ha)",
    "rate": "burn rate (fraction of pixels)",
    "share": "share of the year's burned area (%)",
}


def _check_metric(metric: str) -> str:
    if metric not in _BURN_METRICS:
        raise ValueError(
            f"unknown metric {metric!r}; choose one of {sorted(_BURN_METRICS)}."
        )
    return _BURN_METRICS[metric]


def _covariate_bin_edges(x: np.ndarray, bins: int) -> np.ndarray:
    """Equal-count (quantile) bin edges over the **pooled** covariate values.

    Pooled, not per-year, on purpose: every year must be binned identically or the
    per-year curves are not comparable and the heat map's rows do not line up.
    Duplicate quantiles (a covariate with few distinct values) collapse, so the
    returned edge count can be smaller than ``bins + 1``.
    """
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("covariate has no finite values to bin.")
    edges = np.unique(np.nanquantile(finite, np.linspace(0.0, 1.0, bins + 1)))
    if edges.size < 2:  # a constant covariate: widen so pd.cut still works
        edges = np.array([edges[0] - 0.5, edges[0] + 0.5])
    return edges


def _burned_area_table(
    frame: pd.DataFrame,
    covariate: str,
    response: str,
    year_col: str,
    area_col: str,
    bins: int,
    is_cat: bool,
) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Aggregate a pixel-year frame to one row per (year, covariate bin).

    The shared aggregation behind all three burned-area views. Returns
    ``(table, edges)`` where ``table`` carries, per cell: ``n_px`` pixels,
    ``burned_px`` burned pixels, ``area_ha`` burned hectares, ``rate`` (=
    ``burned_px / n_px``) with its binomial ``se``, ``share`` (percent of that
    year's burned area), and ``x``, the bin's mean covariate value (the plotting
    abscissa; absent for a categorical covariate). ``edges`` is the quantile bin
    boundary array for a continuous covariate, ``None`` for a categorical one.
    """
    for col in (covariate, response, year_col):
        if col not in frame.columns:
            raise ValueError(f"{col!r} not in frame.")

    df = frame[[year_col, covariate, response]].copy()
    df[response] = pd.to_numeric(df[response], errors="coerce")
    # Per-pixel area: the column build_mask_frame writes, else a caller-supplied
    # constant, else 1 ha (so `area_ha` degrades to a burned-pixel count).
    if area_col in frame.columns:
        df["_area"] = pd.to_numeric(frame[area_col], errors="coerce").values
    else:
        df["_area"] = 1.0
    df = df.dropna(subset=[covariate, response])
    if df.empty:
        raise ValueError(f"no rows with both {covariate!r} and {response!r} present.")

    # A burned-area product's response is already 0/1; `> 0` also makes a severity
    # grid ("any positive severity") behave, so the area sum is always well defined.
    df["_burned"] = (df[response] > 0).astype("float64")
    df["_burned_area"] = df["_burned"] * df["_area"]

    if is_cat:
        df["_bin"] = df[covariate].astype("object")
        edges = None
    else:
        edges = _covariate_bin_edges(df[covariate].to_numpy(dtype="float64"), bins)
        df["_bin"] = pd.cut(df[covariate], bins=edges, include_lowest=True)

    agg = {
        "n_px": ("_burned", "size"),
        "burned_px": ("_burned", "sum"),
        "rate": ("_burned", "mean"),
        "area_ha": ("_burned_area", "sum"),
    }
    if not is_cat:
        agg["x"] = (covariate, "mean")
    out = (
        df.groupby([year_col, "_bin"], observed=True)
        .agg(**agg)
        .reset_index()
        .rename(columns={"_bin": "bin"})
    )
    out["se"] = _binomial_se(out["rate"].to_numpy(), out["n_px"].to_numpy())
    year_total = out.groupby(year_col)["area_ha"].transform("sum")
    out["share"] = np.where(year_total > 0, 100.0 * out["area_ha"] / year_total, np.nan)
    if not is_cat:
        out = out.sort_values([year_col, "x"])
    return out.reset_index(drop=True), edges


def _pool_years(table: pd.DataFrame, is_cat: bool) -> pd.DataFrame:
    """Collapse the per-(year, bin) table over years -- the all-years curve.

    Sums are summed and the rate is re-derived from the pooled counts (**not** a
    mean of per-year rates, which would weight a sparse year like a dense one).
    """
    agg = {"n_px": ("n_px", "sum"), "burned_px": ("burned_px", "sum"),
           "area_ha": ("area_ha", "sum")}
    if not is_cat:
        # Weight each year's bin centre by its pixel count (bins are the same edges
        # every year, so this is essentially the pooled bin mean).
        table = table.assign(_xw=table["x"] * table["n_px"])
        agg["_xw"] = ("_xw", "sum")
    out = table.groupby("bin", observed=True).agg(**agg).reset_index()
    out["rate"] = np.where(out["n_px"] > 0, out["burned_px"] / out["n_px"], np.nan)
    out["se"] = _binomial_se(out["rate"].to_numpy(), out["n_px"].to_numpy())
    total = out["area_ha"].sum()
    out["share"] = 100.0 * out["area_ha"] / total if total > 0 else np.nan
    if not is_cat:
        out["x"] = np.where(out["n_px"] > 0, out["_xw"] / out["n_px"], np.nan)
        out = out.drop(columns="_xw").sort_values("x")
    return out.reset_index(drop=True)


def _year_colors(years: Sequence[int], cmap: str = "viridis") -> dict:
    """One colour per year on a sequential ramp, so time reads as a gradient."""
    cm = plt.get_cmap(cmap)
    n = max(len(years) - 1, 1)
    return {yr: cm(0.12 + 0.76 * i / n) for i, yr in enumerate(sorted(years))}


def _category_order(table: pd.DataFrame, metric: str, max_categories: int) -> list:
    """Classes ordered by their pooled value of ``metric`` (biggest first).

    ``rate`` is re-derived from the pooled counts; ``area_ha`` and ``share`` order
    identically (``share`` is just the area rescaled), so both use the area sum.
    """
    g = table.groupby("bin", observed=True)[["burned_px", "n_px", "area_ha"]].sum()
    pooled = (
        (g["burned_px"] / g["n_px"]) if metric == "rate" else g["area_ha"]
    ).sort_values(ascending=False)
    return list(pooled.index[:max_categories])


def plot_burned_area_vs_covariate(
    frame: pd.DataFrame,
    covariate: str,
    response: str = "burned",
    year_col: str = "year",
    metric: str = "area_ha",
    bins: int = 8,
    categorical: Optional[bool] = None,
    area_col: str = "pixel_area_ha",
    by_year: bool = True,
    show_pooled: bool = True,
    max_categories: int = 25,
    ax: Optional[plt.Axes] = None,
):
    """Burned **area** against one covariate, one curve per year.

    The "covariate on x, burned area on y" picture, over a whole mask rather than a
    treated/control design: bin the covariate once over all years (equal-count
    quantile bins, so every bin holds a comparable number of pixels), then plot the
    burned hectares that fell in each bin. Drawn as **one line per year** on a
    sequential colour ramp (dark = early, bright = late), with the all-years pooled
    curve in black on top -- so a covariate that concentrates fire shows up as a
    sloped/peaked pooled curve, and a year that burned somewhere unusual shows up as
    a line departing from that shape.

    A raw scatter of the response would be uninformative -- ``burned`` is a 0/1
    pixel-year flag, so every point would sit on 0 or 1 -- which is why the covariate
    is binned and the response aggregated within each bin. That is also the
    difference from :func:`plot_covariate_vs_burn`, which asks the *rate* question on
    the treated/control modeling frame; this one is absolute, mask-wide, and split by
    year.

    Parameters
    ----------
    frame : DataFrame
        Pixel-year table with ``covariate``, ``response`` and ``year_col`` -- e.g.
        :func:`peatfire.modeling.build_mask_frame` output (which supplies
        ``pixel_area_ha``), or the treated/control ``panel``.
    metric : {"area_ha", "rate", "share"}
        ``"area_ha"`` (default) plots absolute burned hectares -- the quantity the
        question is usually about, but note it is a *count*, so a bin holding more
        of the landscape can top the chart just by being bigger. ``"rate"`` divides
        that out (burned fraction of the bin's pixels) and is the fairer read of
        "does fire prefer this covariate level". ``"share"`` normalises within each
        year instead (percent of that year's burned area), which puts a big fire
        year and a quiet one on the same axis.
    bins : int
        Number of equal-count bins for a continuous covariate.
    categorical : bool, optional
        Force categorical (per-class) or continuous (binned) treatment. Default
        auto-detects from the covariate registry / dtype.
    by_year : bool
        ``False`` collapses to the single pooled curve.
    show_pooled : bool
        Draw the all-years pooled curve alongside the per-year lines. With
        ``metric="area_ha"`` that curve is the *sum* over years, so it sits well
        above them and compresses their vertical range -- pass ``False`` to zoom
        into the per-year lines (or use ``"rate"`` / ``"share"``, which are on the
        same scale as the years).

    Returns the matplotlib Figure.
    """
    set_fire_style()
    ylabel = _check_metric(metric)
    is_cat = _is_categorical_covariate(frame, covariate, categorical)
    table, _ = _burned_area_table(
        frame, covariate, response, year_col, area_col, bins, is_cat
    )
    pooled = _pool_years(table, is_cat)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.8, 5))
    else:
        fig = ax.figure

    # Categorical: fix a class order (by pooled burned area) and plot on positions.
    if is_cat:
        order = _category_order(table, metric, max_categories)
        pos = {cls: i for i, cls in enumerate(order)}
        table = table[table["bin"].isin(order)].assign(x=lambda d: d["bin"].map(pos))
        pooled = pooled[pooled["bin"].isin(order)].assign(x=lambda d: d["bin"].map(pos))

    # Classes have no natural order, so a connecting line would invent one: draw
    # markers only for a categorical covariate, lines for a continuous one.
    fmt = "o" if is_cat else "-o"

    if by_year:
        years = sorted(table[year_col].unique())
        colors = _year_colors(years)
        for yr in years:
            g = table[table[year_col] == yr].sort_values("x")
            if g.empty:
                continue
            ax.plot(g["x"], g[metric], fmt, color=colors[yr], markersize=4,
                    lw=1.4, alpha=0.9, label=str(int(yr)), zorder=2)
    if show_pooled or not by_year:
        ax.plot(pooled["x"], pooled[metric], fmt, color="black", markersize=6,
                lw=2.0, label="all years", zorder=3)
        if metric == "rate":
            ax.fill_between(pooled["x"], pooled["rate"] - pooled["se"],
                            pooled["rate"] + pooled["se"], color="0.5",
                            alpha=0.2, zorder=1)

    if is_cat:
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(o) for o in order], rotation=45, ha="right")
    ax.set_xlabel(f"{covariate}" + ("" if is_cat else "  (bin mean)"))
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.set_title(f"{ylabel.capitalize()} vs {covariate}")
    ax.legend(loc="best", ncol=2, title="year" if by_year else None)
    return fig


def plot_burned_area_covariate_heatmap(
    frame: pd.DataFrame,
    covariate: str,
    response: str = "burned",
    year_col: str = "year",
    metric: str = "area_ha",
    bins: int = 8,
    categorical: Optional[bool] = None,
    area_col: str = "pixel_area_ha",
    cmap: str = "inferno",
    annotate: bool = False,
    max_categories: int = 25,
    ax: Optional[plt.Axes] = None,
):
    """Year x covariate heat map of burned area -- hotter = more burning.

    The two-dimensional version of :func:`plot_burned_area_vs_covariate`: **time on
    x, covariate level on y, burned area as colour**. Each column is one year, each
    row one covariate bin, and the cell colour is how much burned in that
    (year, covariate-level) combination on a "hot" ramp (dark = little, bright =
    much). It answers the question the per-year curves make you read off overlapping
    lines: *did the covariate level where fire concentrates move between years?* --
    a diagonal or drifting bright band means the covariate-fire relationship is
    year-dependent, a stable bright row means it is not.

    For a continuous covariate the y axis carries the **real covariate values**
    (quantile bin edges), so rows have unequal heights -- that is honest: an
    equal-count bin in a dense part of the covariate range is genuinely narrow.
    Cells with no pixels are left blank (light grey).

    ``metric`` behaves as in :func:`plot_burned_area_vs_covariate`; ``"share"`` is
    often the most readable here, since with ``"area_ha"`` one big fire year can
    dominate the colour scale and flatten every other column to black.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    clabel = _check_metric(metric)
    is_cat = _is_categorical_covariate(frame, covariate, categorical)
    table, edges = _burned_area_table(
        frame, covariate, response, year_col, area_col, bins, is_cat
    )

    years = sorted(table[year_col].unique())
    if is_cat:
        # Rows are classes, ordered by pooled burned area (biggest at the top).
        row_order = _category_order(table, metric, max_categories)[::-1]
        y_edges = np.arange(len(row_order) + 1) - 0.5
    else:
        # Rows are the quantile bins, ascending; use the real edges as the y axis.
        row_order = list(
            table.sort_values("x").drop_duplicates("bin")["bin"]
        )
        y_edges = edges

    grid = np.full((len(row_order), len(years)), np.nan)
    row_ix = {b: i for i, b in enumerate(row_order)}
    col_ix = {y: j for j, y in enumerate(years)}
    for _, r in table.iterrows():
        i, j = row_ix.get(r["bin"]), col_ix.get(r[year_col])
        if i is not None and j is not None:
            grid[i, j] = r[metric]

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.1 * len(years) + 5.0, 5.2))
    else:
        fig = ax.figure

    # One column per year (positions, not calendar arithmetic) so a gap year
    # does not silently stretch a column.
    x_edges = np.arange(len(years) + 1) - 0.5
    this_cmap = plt.get_cmap(cmap).copy()
    this_cmap.set_bad("0.9")
    mesh = ax.pcolormesh(
        x_edges, y_edges, np.ma.masked_invalid(grid),
        cmap=this_cmap, shading="flat",
    )
    fig.colorbar(mesh, ax=ax, shrink=0.9, label=clabel)

    if annotate:
        num_fmt = ",.0f" if metric == "area_ha" else (".3f" if metric == "rate" else ".0f")
        for i in range(grid.shape[0]):
            yc = 0.5 * (y_edges[i] + y_edges[i + 1])
            for j in range(grid.shape[1]):
                if not np.isfinite(grid[i, j]):
                    continue
                # Dark text on a bright cell, light text on a dark one.
                r, g, b, _ = this_cmap(mesh.norm(grid[i, j]))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                ax.text(j, yc, format(grid[i, j], num_fmt), ha="center", va="center",
                        fontsize=8, color="0.1" if lum > 0.55 else "0.95")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(int(y)) for y in years])
    if is_cat:
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels([str(b) for b in row_order])
    ax.set_xlabel("year")
    ax.set_ylabel(covariate)
    ax.set_title(f"{clabel.capitalize()} by year and {covariate}\n(brighter = more burning)")
    return fig


def plot_burned_area_and_covariate_by_year(
    frame: pd.DataFrame,
    covariate: str,
    response: str = "burned",
    year_col: str = "year",
    area_col: str = "pixel_area_ha",
    agg: str = "mean",
    ax: Optional[plt.Axes] = None,
):
    """Annual burned area (bars) with the mask-mean covariate overlaid (line).

    The most direct reading of "burned area over time against a covariate": total
    hectares burned in the mask each year as bars, and that year's **mask-average
    covariate** as a line on a twin axis. For a *temporal* covariate (``pdsi``,
    ``precip``, ``tmax``) this is the picture that actually answers "do the big fire
    years line up with the dry years?", which neither of the binned views shows --
    they bin *within* years, so a year-to-year driver is invisible to them.

    The Pearson and Spearman correlations across years are annotated. Read them as
    description, not inference: with a handful of panel years the estimate is very
    noisy, and it is the DiD's ``treated:pdsi`` interaction that does the real work.

    A **static** covariate is constant across years, so the overlaid line is flat by
    construction and the panel is labelled as such -- for static layers the binned
    views (:func:`plot_burned_area_vs_covariate`,
    :func:`plot_burned_area_covariate_heatmap`) are the informative ones.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    for col in (covariate, response, year_col):
        if col not in frame.columns:
            raise ValueError(f"{col!r} not in frame.")
    if not pd.api.types.is_numeric_dtype(frame[covariate]):
        raise ValueError(
            f"{covariate!r} is not numeric; this view averages the covariate per "
            "year. Use plot_burned_area_covariate_heatmap for a categorical layer."
        )

    df = frame[[year_col, covariate, response]].copy()
    df[response] = pd.to_numeric(df[response], errors="coerce")
    df["_area"] = (
        pd.to_numeric(frame[area_col], errors="coerce").values
        if area_col in frame.columns else 1.0
    )
    df = df.dropna(subset=[response])
    df["_burned_area"] = (df[response] > 0).astype("float64") * df["_area"]

    per_year = df.groupby(year_col).agg(
        area_ha=("_burned_area", "sum"),
        cov=(covariate, agg),
        n_px=(response, "size"),
    ).reset_index()
    years = per_year[year_col].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.bar(years, per_year["area_ha"], width=0.65, color=TREATED_COLOR, alpha=0.75,
           label="burned area", zorder=2)
    ax.set_xlabel("calendar year")
    ax.set_ylabel("burned area (ha)")
    ax.set_xticks(years)
    ax.set_ylim(bottom=0)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(years, per_year["cov"], "-o", color=CONTROL_COLOR, lw=1.8,
             label=f"{covariate} ({agg})", zorder=3)
    ax2.set_ylabel(f"{covariate}  (mask {agg})")

    # Correlations across years -- descriptive only (n = number of panel years).
    a = per_year["area_ha"].to_numpy(dtype="float64")
    c = per_year["cov"].to_numpy(dtype="float64")
    ok = np.isfinite(a) & np.isfinite(c)
    static = ok.sum() > 1 and float(np.nanstd(c[ok])) < 1e-9
    if static:
        note = f"{covariate} is constant across years (a static covariate)"
    elif ok.sum() >= 3:
        pearson = float(np.corrcoef(a[ok], c[ok])[0, 1])
        ranks = lambda v: pd.Series(v).rank().to_numpy()  # noqa: E731
        spearman = float(np.corrcoef(ranks(a[ok]), ranks(c[ok]))[0, 1])
        note = f"r = {pearson:+.2f},  rho = {spearman:+.2f}  (n = {int(ok.sum())} years)"
    else:
        note = "too few years for a correlation"

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper left")
    ax.set_title(f"Burned area per year vs {covariate}\n{note}")
    return fig


# The views the driver below can produce, mapped to the function that draws each.
BURNED_AREA_PLOT_KINDS = ("vs_covariate", "heatmap", "by_year")


def plot_burned_area_vs_covariates_over_years(
    fire_product: str = "FireCCIS311",
    mask: Optional[gpd.GeoDataFrame] = None,
    covariates: Optional[Sequence[str]] = None,
    years: Sequence[int] = range(2019, 2025),
    frame: Optional[pd.DataFrame] = None,
    kinds: Sequence[str] = ("vs_covariate", "heatmap"),
    metric: str = "area_ha",
    bins: int = 8,
    response: str = "burned",
    year_col: str = "year",
    res_m: float = DEFAULT_RES_M,
    out_dir=None,
    dpi: Optional[int] = None,
    close: bool = False,
) -> dict:
    """Draw (and optionally save) the burned-area views for a set of covariates.

    The one-call driver: build the mask's pixel-year table **once** with
    :func:`peatfire.modeling.build_mask_frame`, then lay out the requested views for
    every covariate. Note that year is a *dimension inside* each figure (a colour per
    year, or the heat map's x axis) rather than an outer loop producing one figure
    per year -- comparing years is the whole point, and it only works if they share
    an axis.

    Parameters
    ----------
    fire_product : str
        Registered burned-area product supplying the response (e.g.
        ``"FireCCI51"``, ``"FireCCIS311"``). Also prefixes the saved filenames.
    mask : GeoDataFrame, optional
        Area to tabulate (e.g. the NC 80% histosol peat AOI). Required unless a
        prebuilt ``frame`` is passed.
    covariates : sequence of str, optional
        Covariates to draw. Defaults to every **continuous** registered covariate
        (static + per-year) present in the frame with more than one distinct value.
    years : sequence of int
        Panel years, forwarded to ``build_mask_frame``.
    frame : DataFrame, optional
        A prebuilt pixel-year table to plot instead of reading the rasters again --
        pass the return of a previous call's ``build_mask_frame`` (or the modeling
        panel) when iterating on the figures.
    kinds : sequence of str
        Which views to draw, any of :data:`BURNED_AREA_PLOT_KINDS`:
        ``"vs_covariate"`` (covariate on x, burned area on y, one line per year),
        ``"heatmap"`` (year on x, covariate on y, burned area as colour),
        ``"by_year"`` (annual burned-area bars with the covariate overlaid; skipped
        for categorical covariates, which it cannot average).
    out_dir : path-like, optional
        Directory to save into as ``<fire_product>_<covariate>_<kind>.png``. Not
        saved when omitted.
    close : bool
        Close each figure after saving. Set it when sweeping many covariates, so
        matplotlib does not hold dozens of open figures.

    Returns
    -------
    dict
        ``{(covariate, kind): Figure}`` for every panel drawn (figures are still
        returned when ``close=True``, but they are no longer displayable).
    """
    from pathlib import Path

    from .frame import build_mask_frame

    bad = [k for k in kinds if k not in BURNED_AREA_PLOT_KINDS]
    if bad:
        raise ValueError(f"unknown kind(s) {bad}; choose from {list(BURNED_AREA_PLOT_KINDS)}.")

    if frame is None:
        if mask is None:
            raise ValueError("pass either `mask` (to build the table) or a prebuilt `frame`.")
        frame = build_mask_frame(mask, product=fire_product, years=years, res_m=res_m)

    if covariates is None:
        continuous = set(list_covariates("continuous")) | {
            n for n, spec in TEMPORAL_COVARIATES.items() if spec.role == "continuous"
        }
        covariates = [
            c for c in frame.columns
            if c in continuous and frame[c].nunique(dropna=True) > 1
        ]
    covariates = [c for c in covariates if c in frame.columns]
    if not covariates:
        raise ValueError("no covariates to plot (none of the requested names are frame columns).")

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    figs: dict = {}
    for cov in covariates:
        for kind in kinds:
            if kind == "vs_covariate":
                fig = plot_burned_area_vs_covariate(
                    frame, cov, response=response, year_col=year_col,
                    metric=metric, bins=bins,
                )
            elif kind == "heatmap":
                fig = plot_burned_area_covariate_heatmap(
                    frame, cov, response=response, year_col=year_col,
                    metric=metric, bins=bins,
                )
            else:  # "by_year" -- needs a numeric covariate to average per year
                if not pd.api.types.is_numeric_dtype(frame[cov]):
                    continue
                fig = plot_burned_area_and_covariate_by_year(
                    frame, cov, response=response, year_col=year_col,
                )
            fig.suptitle(fire_product, fontsize=10, color="0.4", x=0.01, ha="left")
            if out_dir is not None:
                fig.savefig(
                    out_dir / f"{fire_product}_{cov}_{kind}.png",
                    bbox_inches="tight", dpi=dpi,
                )
            if close:
                plt.close(fig)
            figs[(cov, kind)] = fig
    return figs


# ---------------------------------------------------------------------------
# 4. Event study (the temporal / parallel-trends picture)
# ---------------------------------------------------------------------------
# Candidate column names for the tidy event-study frame, so this reads the output
# of either DiD backend (`differences` returns a pandas frame; the R `did` path a
# converted frame) without the caller renaming anything. Matched case-insensitively,
# exact name first then substring; pass the *_col arguments to override.
_EVENT_TIME_CANDIDATES = (
    "relative_period", "event_time", "event time", "egt",
    "exposure", "period", "rel_year", "time_to_treatment",
)
_ESTIMATE_CANDIDATES = ("att", "estimate", "coefficient", "coef", "point_estimate", "point")
_LOWER_CANDIDATES = ("lower", "ci_lower", "lower_ci", "conf_low", "conf.low",
                     "cband_lower", "lower_bound")
_UPPER_CANDIDATES = ("upper", "ci_upper", "upper_ci", "conf_high", "conf.high",
                     "cband_upper", "upper_bound")
_SE_CANDIDATES = ("std_error", "std.error", "standard_error", "se", "stderr")


def _match_col(columns, candidates: Sequence[str]) -> Optional[str]:
    """Return the first column matching a candidate (exact, then substring)."""
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:                 # exact match wins
        if cand in lower:
            return lower[cand]
    for cand in candidates:                 # then substring
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    return None


def _tidy_event_study(
    event_study,
    event_time_col,
    estimate_col,
    lower_col,
    upper_col,
    se_col,
    ci_level: float,
) -> pd.DataFrame:
    """Coerce a DiD event-study aggregate into ``event_time/estimate/lower/upper``.

    The output of :func:`peatfire.modeling.did.aggregate_att` (``kind="event"``)
    differs by backend, so this normalises it: it flattens a MultiIndex column
    header, exposes an event-time held in the index, auto-detects the estimate /
    CI / SE columns (override via the ``*_col`` arguments), and -- if only a
    standard error is present -- builds a Wald CI at ``ci_level``.
    """
    df = event_study.copy() if isinstance(event_study, pd.DataFrame) else pd.DataFrame(event_study)

    # Flatten a MultiIndex column header (the `differences` aggregate uses one).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(str(p) for p in tup if str(p) != "").strip("_")
                      for tup in df.columns]

    # Event time often lives in the index (e.g. `relative_period`); expose it.
    et = event_time_col or _match_col(df.columns, _EVENT_TIME_CANDIDATES)
    if et is None:
        df = df.reset_index()
        et = event_time_col or _match_col(df.columns, _EVENT_TIME_CANDIDATES)

    est = estimate_col or _match_col(df.columns, _ESTIMATE_CANDIDATES)
    if et is None or est is None:
        raise ValueError(
            "could not locate the event-time and estimate columns in the "
            f"event-study frame (columns: {list(df.columns)}). Pass "
            "event_time_col=/estimate_col= (and lower_col=/upper_col= or se_col=) "
            "explicitly."
        )

    lo = lower_col or _match_col(df.columns, _LOWER_CANDIDATES)
    hi = upper_col or _match_col(df.columns, _UPPER_CANDIDATES)
    out = pd.DataFrame({
        "event_time": pd.to_numeric(df[et], errors="coerce"),
        "estimate": pd.to_numeric(df[est], errors="coerce"),
    })
    if lo is not None and hi is not None:
        out["lower"] = pd.to_numeric(df[lo], errors="coerce")
        out["upper"] = pd.to_numeric(df[hi], errors="coerce")
    else:
        # No explicit band: build a Wald CI from the standard error.
        se_name = se_col or _match_col(df.columns, _SE_CANDIDATES)
        if se_name is None:
            raise ValueError(
                "event-study frame has neither lower/upper CI columns nor a "
                f"standard-error column (columns: {list(df.columns)}). Pass "
                "lower_col=/upper_col= or se_col= explicitly."
            )
        from statistics import NormalDist

        z = NormalDist().inv_cdf(1 - (1 - ci_level) / 2)
        se = pd.to_numeric(df[se_name], errors="coerce")
        out["lower"] = out["estimate"] - z * se
        out["upper"] = out["estimate"] + z * se

    return out.dropna(subset=["event_time", "estimate"]).sort_values("event_time")


def plot_event_study(
    event_study,
    event_time_col: Optional[str] = None,
    estimate_col: Optional[str] = None,
    lower_col: Optional[str] = None,
    upper_col: Optional[str] = None,
    se_col: Optional[str] = None,
    ci_level: float = 0.95,
    ylabel: str = "ATT on P(burn)",
    treatment_label: str = "restoration",
    ax: Optional[plt.Axes] = None,
):
    """Plot the staggered-DiD event study: burning vs time since restoration.

    The one **temporal** diagnostic in this module, and the picture to eyeball
    before trusting any headline ATT. It consumes the output of
    :func:`peatfire.modeling.did.aggregate_att` with ``kind="event"`` -- the ATT
    by event time (years since a site's restoration) -- and lays it out as
    Callaway-Sant'Anna intends:

    * **event time 0** is each treated site's restoration year (a dashed marker at
      the ``-0.5`` onset boundary, since the period before treatment is the base);
    * **pre-treatment points** (event time ``< 0``, drawn muted over a shaded band)
      are the **parallel-trends test** -- they should straddle zero, i.e. treated
      and control were on the same burning trajectory *before* restoration;
    * **post-treatment points** (event time ``>= 0``, in the treated colour) are the
      dynamic effect -- how the restoration effect on burning evolves after the
      canal blocks go in.

    Because the two DiD backends return differently-shaped aggregates, the frame is
    normalised by :func:`_tidy_event_study`; override the column mapping with the
    ``*_col`` arguments if auto-detection misses. When only a standard error is
    present, a Wald CI at ``ci_level`` is drawn.

    Returns the matplotlib Figure.
    """
    set_fire_style()
    tidy = _tidy_event_study(
        event_study, event_time_col, estimate_col, lower_col, upper_col, se_col, ci_level
    )
    if tidy.empty:
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots()
        ax.text(0.5, 0.5, "no event-study estimates", ha="center", va="center")
        return fig

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    pre = tidy[tidy["event_time"] < 0]
    post = tidy[tidy["event_time"] >= 0]

    # References: no-effect line, treatment onset, and the pre-period test window.
    ax.axhline(0.0, color="0.5", lw=1.0, ls="--", zorder=1)
    onset = -0.5   # boundary between the pre base period (-1) and event time 0
    ax.axvline(onset, color="0.4", lw=1.0, ls=":", zorder=1)
    if not pre.empty:
        ax.axvspan(tidy["event_time"].min() - 0.5, onset,
                   color="0.85", alpha=0.35, zorder=0)

    # Faint path through every point, then coloured points + CIs per period.
    ax.plot(tidy["event_time"], tidy["estimate"], color="0.6", lw=0.8, zorder=2)
    for grp, color, label in (
        (pre, CONTROL_COLOR, "pre (parallel-trends check)"),
        (post, TREATED_COLOR, "post (dynamic ATT)"),
    ):
        if grp.empty:
            continue
        ax.errorbar(
            grp["event_time"], grp["estimate"],
            yerr=[grp["estimate"] - grp["lower"], grp["upper"] - grp["estimate"]],
            fmt="o", color=color, ecolor=color, elinewidth=1.2, capsize=3,
            markersize=5, label=label, zorder=3,
        )

    ax.annotate(
        treatment_label, xy=(onset, 1.0), xycoords=("data", "axes fraction"),
        xytext=(3, -10), textcoords="offset points", va="top", ha="left",
        fontsize=9, color="0.4",
    )
    ax.set_xticks(sorted(tidy["event_time"].unique()))
    ax.set_xlabel("event time (years since restoration)")
    ax.set_ylabel(ylabel)
    ax.set_title(
        "Event study: burning vs time since restoration\n"
        f"({int(round(ci_level * 100))}% CIs; pre-period points test parallel trends)"
    )
    ax.legend(loc="best")
    return fig

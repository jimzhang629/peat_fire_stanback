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
* :func:`plot_covariate_space` -- treated vs control pixels in a 2-D covariate
  plane, with marginal histograms. A covariate with no spread collapses its axis
  to a line; this is the direct picture of "we are effectively matching on
  elevation alone."
* :func:`plot_covariate_pairs` -- the same idea for >2 covariates: a scatter
  matrix coloured by treatment, so every covariate axis (climate, soil, ...) is
  inspected at once.
* :func:`plot_matched_pairs_covariate` -- treated and its matched control joined
  by a segment in covariate space; short segments = good matches.
* :func:`plot_candidate_pixels_geographic` -- the *whole* treated + candidate
  control pixel pool on the map (before matching), coloured by group, so you can
  see where the two populations sit across NC and how the control pool blankets
  the peat frame.
* :func:`plot_matched_pairs_geographic` -- the same pairs on the map, so you can
  see *how far* across the landscape each control was drawn from its treated site.
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
import pandas as pd

from ..fire_products_comparison.fire_comparison import ANALYSIS_CRS, build_common_grid
from ..fire_products_comparison.plotting import OKABE_ITO, set_fire_style
from .covariates import available_covariates, covariate_on_grid, get_covariate
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


def plot_candidate_pixels_geographic(
    pixels: gpd.GeoDataFrame,
    aoi: Optional[gpd.GeoDataFrame] = None,
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    aoi_context: Optional[gpd.GeoDataFrame] = None,
    max_control: int = 40000,
    treated_color: str = TREATED_COLOR,
    control_color: str = CONTROL_COLOR,
    aoi_label: str = ">=80% histosol boundary",
    ax: Optional[plt.Axes] = None,
):
    """Map the treated and candidate-control pixels over NC, coloured by group.

    The geographic companion to :func:`plot_covariate_space`: instead of the two
    covariate axes, it drops every treated (restoration-site) pixel and every
    candidate-control pixel onto the map so you can see *where* the two
    populations sit across the peat frame -- treated in a handful of restoration
    clusters, controls blanketing the rest of the peatland. It consumes the
    unmatched pixel(-year) set from
    :func:`matching.get_treated_and_control_pixels`, so this is the picture
    *before* matching (the pool the match draws controls from), the counterpart
    to :func:`plot_matched_pairs_geographic` after it.

    The treated/candidate split uses the same rule as
    :func:`matching.balance_table` (restoration-site membership on the panel), and
    a pixel-year panel is collapsed to unique pixels first so each physical pixel
    is drawn once. Controls are subsampled to ``max_control`` points for
    legibility (the pool is large); treated are always drawn in full and on top.

    Parameters
    ----------
    pixels : GeoDataFrame
        Treated + candidate-control pixels (EPSG:5070), as returned by
        :func:`matching.get_treated_and_control_pixels`. May be a pixel-year panel.
    aoi : GeoDataFrame, optional
        Peat AOI outline drawn (and labelled) for orientation -- e.g. the
        ``>=80%`` histosol frame the pixels were drawn from.
    aoi_context : GeoDataFrame, optional
        Extra outline (e.g. the NC state boundary) for geographic reference.
    max_control : int, default 40000
        Cap on candidate-control points scatter-plotted (subsampled if exceeded).
    treated_color, control_color : str, optional
        Colours for the treated and candidate-control pixels; default to the
        module's Okabe-Ito :data:`TREATED_COLOR` (vermillion) and
        :data:`CONTROL_COLOR` (blue), matching every other diagnostic here.
    aoi_label : str, default ``">=80% histosol boundary"``
        Legend label for the ``aoi`` outline.

    Returns
    -------
    matplotlib Figure
    """
    set_fire_style()
    is_t = _treated_mask(pixels, treated_col, restoration_yr_col)
    # Collapse a pixel-year panel to one row per physical pixel, keeping x/y for
    # the map, so a pixel repeated across years is plotted once.
    sub = _unique_pixels(pixels.assign(_t=is_t.values), ["_t"])
    t = sub[sub["_t"]]
    c = sub[~sub["_t"]]
    if len(c) > max_control:
        c = c.sample(max_control, random_state=0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure

    handles = []
    if aoi is not None:
        # The peat frame the pixels came from -- e.g. the >=80% histosol boundary.
        aoi.to_crs(ANALYSIS_CRS).dissolve().boundary.plot(
            ax=ax, color="0.35", linewidth=1.0, zorder=4
        )
        handles.append(plt.Line2D([0], [0], color="0.35", lw=1.0, label=aoi_label))
    if aoi_context is not None:
        aoi_context.to_crs(ANALYSIS_CRS).boundary.plot(
            ax=ax, color="0.55", linewidth=0.8, ls="--", zorder=1
        )

    # x/y are the pixel-centroid coordinates on the EPSG:5070 grid, so they line
    # up cell-for-cell with the AOI drawn above.
    handles.append(ax.scatter(
        c["x"], c["y"], s=6, alpha=0.35, color=control_color,
        edgecolor="none", label=f"candidate control (n={len(c):,})", zorder=2))
    handles.append(ax.scatter(
        t["x"], t["y"], s=12, alpha=0.85, color=treated_color,
        edgecolor="none", label=f"treated (n={len(t):,})", zorder=3))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Treated vs candidate-control pixels across NC")
    ax.legend(handles=handles, loc="best", markerscale=2)
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

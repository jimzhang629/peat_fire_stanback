"""Control-pixel matching -- ASSIGNMENT SCAFFOLD (you implement Part A).

Builds the matched treated/control pixel set that :func:`peatfire.build_frame`
consumes. This is the causal-design heart of the study; see
``modeling_notebook_explained.md`` (Part VII) for the staged walkthrough and
the testing recipes, and Part IV for the underlying math.

Layout
------
* **Part A -- functions you implement.** Each is one stage of the assignment.
  The body is ``raise NotImplementedError``; the docstring states the contract
  (inputs, return, and the postconditions its ``check_*`` partner verifies).
* **Part B -- provided test harness.** ``standardized_mean_diff`` and the
  ``check_*`` functions are done for you: call them on your output to verify each
  stage before moving on. You should not need to edit Part B.

Everything is in the analysis CRS (EPSG:5070), so buffers and distances are in
metres. Reuse the toolkit you already have: ``build_common_grid``,
``covariate_on_grid``, ``available_covariates``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ..fire_products_comparison.fire_comparison import ANALYSIS_CRS, build_common_grid
from .covariates import (  # noqa: F401
    available_covariates,
    available_temporal_covariates,
    covariate_on_grid,
    temporal_covariate_on_grid,
)
from .frame import (  # noqa: F401
    DEFAULT_RES_M,
    load_completed_restoration_sites_in_analysis_crs,
)

from scipy.spatial import cKDTree
# A pixel this close to a restoration site may be partially rewetted by it
# (spillover) -- exclude it from the control pool. Tune per Stage 2.
DEFAULT_SPILLOVER_M = 1000.0

def load_treated_units(
    path: Optional[Path] = None,
    restoration_yr_col: str = "End_Yr",
    site_col: str = "Proj_Name",
) -> gpd.GeoDataFrame:
    """Stage 1. Completed restoration polygons, in EPSG:5070, with a restoration year.

    Figure out: which column encodes status; your rule for "completed"; how you
    handle a missing end year; which year is the restoration year (you chose end year).

    Design choices
    --------------
    * "Completed" == has a real restoration *end* year. The raw loader
      :func:`load_completed_restoration_sites_in_analysis_crs` already encodes
      that rule: it reprojects to EPSG:5070 and drops any site whose
      ``restoration_yr_col`` is 0/NaN (a placeholder for "no end year yet"), so
      what comes back is exactly the finished sites.
    * The **restoration year** is the end year, kept under its original column
      name (``restoration_yr_col``, e.g. ``End_Yr``) -- the moment a site flips
      from unrestored to restored, i.e. the event time everything downstream
      (per-year ``treated``, DiD cohort ``g``) keys off. It is not renamed.
    * ``site_id`` is the project name where present, else a positional fallback,
      so every completed site has a stable identifier.

    Contract (verified by :func:`check_treated_units`)
        Returns a GeoDataFrame in EPSG:5070 with at least columns
        ``["site_id", <restoration_yr_col>, "geometry"]``, one row per completed
        site, every ``<restoration_yr_col>`` non-null.
    """
    gdf = load_completed_restoration_sites_in_analysis_crs(
        path, restoration_yr_col=restoration_yr_col
    ).copy()

    # Keep the restoration end year under its own name (guaranteed non-null /
    # non-zero by the loader); cast to a plain integer year.
    gdf[restoration_yr_col] = gdf[restoration_yr_col].astype("int64")

    # Stable per-site id: the project name when it exists, else the row position.
    if site_col in gdf.columns:
        gdf["site_id"] = gdf[site_col].astype(str)
    else:
        gdf["site_id"] = np.arange(len(gdf)).astype(str)

    return gdf.reset_index(drop=True)


def build_candidate_pool(
    peat_aoi: gpd.GeoDataFrame,
    treated: gpd.GeoDataFrame,
    spillover_m: float = DEFAULT_SPILLOVER_M
    ) -> gpd.GeoDataFrame:
    """Stage 2. Peat area that is neither treated nor within the spillover halo.

    Figure out: how to get a peat polygon from the histosol raster (>=80); how to
    *subtract* shapes rather than keep them; what buffer distance represents
    plausible rewetting spillover.

    Contract (verified by :func:`check_candidate_pool`)
        Returns a GeoDataFrame (EPSG:5070) whose geometry does not overlap the
        treated polygons buffered by ``spillover_m``, with positive area.
    """
    # project to the same crs first
    treated = treated.to_crs(ANALYSIS_CRS)
    peat_aoi = peat_aoi.to_crs(ANALYSIS_CRS)

    # exclude pixels in a buffer zone around the treated sites, keep remaining peat pixels as candidates
    exclusion = gpd.GeoDataFrame(geometry=[treated.buffer(spillover_m).union_all()], crs=treated.crs)
    candidates = gpd.overlay(peat_aoi, exclusion, how='difference')

    return candidates

def pixelate(
    polygons: gpd.GeoDataFrame,
    res_m: float = DEFAULT_RES_M,
    grid=None,
    carry: Optional[Sequence[str]] = None,
) -> gpd.GeoDataFrame:
    """
    This projects polygons into res_m and onto a common grid, then gets the centers of each grid pixel within the polygons
    Stage 3. Pixel-centroid points covering ``polygons`` at ``res_m``.

    Figure out: what resolution (match the fire product, ~300 m); how to turn a
    grid into centroid points; how to keep only centroids inside ``polygons``.
    Call this once for the treated area and once for the candidate pool, tag each
    with a ``treated`` (1/0) column, and concatenate.

    ``carry`` optionally propagates polygon attribute columns (e.g. the site's
    restoration year, its ``Proj_Name``) onto every pixel that falls inside that
    polygon. This is how a treated pixel remembers *which* site -- and therefore
    *which restoration year* -- it belongs to, which the calendar-year panel in
    :func:`get_treated_and_control_pixels` needs. A pixel that lands inside two
    overlapping polygons is returned once per polygon (the caller de-duplicates).

    Contract (verified by :func:`check_pixels`)
        Returns a point GeoDataFrame (EPSG:5070) with columns ``["x", "y",
        "geometry", *carry]``; ~ ``area / res_m**2`` rows.
    """
    # reproject polygons to analysis crs first
    polygons = polygons.to_crs(ANALYSIS_CRS)

    # build common grid if not already passed in
    if grid is None:
        grid = build_common_grid(polygons, res_m, ANALYSIS_CRS)

    xs = grid['x'].values # 1d array of column centers
    ys = grid['y'].values # 1d array of row centers

    xx, yy = np.meshgrid(xs, ys) # each grid is nrows, ncols
    xx, yy = xx.ravel(), yy.ravel() # flatten to 1d

    # build gdf of points
    points = gpd.GeoDataFrame(
        {'x': xx, 'y': yy},
        geometry = gpd.points_from_xy(xx, yy),
        crs=grid.rio.crs
    )

    # grab points within polygon, carrying along any requested polygon attributes
    carry = list(carry or [])
    points_in_polygon = gpd.sjoin(points, polygons[['geometry', *carry]], predicate='within')

    return points_in_polygon[['x', 'y', 'geometry', *carry]].reset_index(drop=True)

def _stack_across_years(points: gpd.GeoDataFrame, years: Sequence[int]) -> gpd.GeoDataFrame:
    """Repeat every pixel once per calendar year, adding a ``year`` column.

    The pixel *geometry* does not change from year to year, so we pixelate once
    (an expensive spatial join) and cheaply broadcast the result across ``years``
    rather than re-running the geometry per year.
    """
    frames = []
    for year in years:
        frame = points.copy()
        frame["year"] = int(year)
        frames.append(frame)
    stacked = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(stacked, geometry="geometry", crs=points.crs)


def get_treated_and_control_pixels(
    peat_aoi,
    treated,
    years: Optional[Sequence[int]] = None,
    spillover_m=1000,
    res_m: float = DEFAULT_RES_M,
    treated_col_name="treated",
    restoration_yr_col: str = "End_Yr",
    site_col: str = "Proj_Name",
    drop_pretreatment: bool = False,
):
    """Build the labelled treated/control pixel set on one shared grid.

    Orchestrates Stages 2-3: derive the control candidate pool, lay a single
    common grid over the full peat AOI, pixelate the treated polygons and the
    candidate pool onto that grid, and tag each pixel with its treatment status.
    Because both sets are pixelated against the *same* grid, treated and control
    points are co-registered and directly comparable.

    Calendar-year panel
    -------------------
    Treatment is an *event in time*: a restoration site is only "treated" once its
    restoration year has passed. Pass ``years`` to expand the flat pixel set into a
    tidy pixel-**year** panel (one row per pixel per calendar year) where, for each
    calendar ``year``:

    * a restoration-site pixel is ``treated == 1`` only if its site's restoration
      year is at or before that ``year`` (``years_after_restoration >= 0``); in
      earlier years it is a *not-yet-treated* pixel (``treated == 0``), which the
      staggered-DiD design in :mod:`peatfire.modeling.did` uses as a control;
    * ``years_after_restoration = year - <restoration_yr_col>`` (0 = restoration
      year, positive = years since restoration, negative = years before), matching
      the ``event_year`` convention used in the fire-comparison notebook;
    * every candidate-pool pixel is present in every year with ``treated == 0`` and
      a null ``<restoration_yr_col>`` / ``years_after_restoration``.

    The restoration year keeps its original column name (``restoration_yr_col``,
    e.g. ``End_Yr``) rather than being renamed, so a pixel's restoration-site
    membership is recoverable at any time as ``<restoration_yr_col>.notna()``,
    independent of the per-year ``treated`` status.

    Parameters
    ----------
    peat_aoi : geopandas.GeoDataFrame
        Full peat area of interest (EPSG:5070). Defines the common grid extent
        and is the source area for the control candidate pool.
    treated : geopandas.GeoDataFrame
        Completed restoration polygons (EPSG:5070) -- the treated units. When
        ``years`` is given, must carry ``restoration_yr_col`` (the site's
        restoration year).
    years : sequence of int, optional
        Calendar years to build the panel over (e.g. ``range(2019, 2025)`` for
        FireCCIS311). If ``None`` (default) the legacy time-flat set is returned:
        one row per pixel, ``treated`` = static restoration-site membership.
    spillover_m : float, default 1000
        Buffer (metres) around treated sites excluded from the candidate pool
        to avoid rewetting spillover contamination.
    res_m : float, default 300
        Grid resolution in metres (matched to the ~300 m FireCCIS311 fire product).
    treated_col_name : str, default "treated"
        Name of the 1/0 treatment-status column added to the output.
    restoration_yr_col : str, default "End_Yr"
        Column on ``treated`` holding each site's restoration year; carried onto
        treated pixels under this same name in the panel (not renamed).
    site_col : str, default "Proj_Name"
        Site-identifier column on ``treated`` to carry onto treated pixels (kept if
        present; ignored if absent).
    drop_pretreatment : bool, default False
        If True, drop restoration-site pixel-years *before* their restoration year
        (keep only ``treated == 1`` rows for the treated group). Leave False to
        retain them as not-yet-treated controls for a difference-in-differences.

    Returns
    -------
    geopandas.GeoDataFrame
        Point GeoDataFrame (EPSG:5070). Time-flat (``years is None``): columns
        ``["x", "y", "geometry", <treated_col_name>]``. Calendar-year panel:
        additionally ``["year", <restoration_yr_col>, "years_after_restoration"]``
        and, when present, ``site_col``.
    """
    candidates = build_candidate_pool(peat_aoi, treated, spillover_m)

    grid = build_common_grid(peat_aoi, res_m, ANALYSIS_CRS)

    # --- legacy time-flat behaviour: one row per pixel, static membership ---
    if years is None:
        treated_pts = pixelate(treated, res_m, grid).assign(**{treated_col_name: 1})
        control_pts = pixelate(candidates, res_m, grid).assign(**{treated_col_name: 0})
        return pd.concat([treated_pts, control_pts], ignore_index=True)

    years = list(years)

    # Pixelate the geometry once. Treated pixels carry their site's restoration
    # year (and name) so the panel can decide, per calendar year, whether the site
    # has been restored yet.
    carry = [restoration_yr_col] + ([site_col] if site_col in treated.columns else [])
    treated_pts = pixelate(treated, res_m, grid, carry=carry)
    # A pixel inside two overlapping sites appears once per site -> keep its
    # earliest restoration year (the year it first became treated). We keep the
    # restoration year under its original column name (`restoration_yr_col`, e.g.
    # 'End_Yr') rather than renaming it, so the same name flows through the panel.
    treated_pts = (
        treated_pts.sort_values(restoration_yr_col)
        .drop_duplicates(subset=["x", "y"], keep="first")
        .reset_index(drop=True)
    )
    control_pts = pixelate(candidates, res_m, grid)

    # Broadcast both pixel sets across the calendar years.
    treated_panel = _stack_across_years(treated_pts, years)
    control_panel = _stack_across_years(control_pts, years)

    # Per-year event time and treatment status for restoration-site pixels.
    treated_panel[restoration_yr_col] = treated_panel[restoration_yr_col].astype("float64")
    treated_panel["years_after_restoration"] = (
        treated_panel["year"] - treated_panel[restoration_yr_col]
    )
    treated_panel[treated_col_name] = (
        treated_panel["years_after_restoration"] >= 0
    ).astype(int)
    if drop_pretreatment:
        treated_panel = treated_panel[treated_panel[treated_col_name] == 1].reset_index(
            drop=True
        )

    # Candidate pool: control in every year, no restoration timing.
    control_panel[treated_col_name] = 0
    control_panel[restoration_yr_col] = np.nan
    control_panel["years_after_restoration"] = np.nan

    pixels = pd.concat([treated_panel, control_panel], ignore_index=True)
    return gpd.GeoDataFrame(pixels, geometry="geometry", crs=grid.rio.crs)

def attach_covariates(
    points: gpd.GeoDataFrame,
    names: Optional[Sequence[str]] = None,
    aoi: Optional[gpd.GeoDataFrame] = None,
    res_m: float = DEFAULT_RES_M,
    year_col: str = "year",
    temporal_names: Optional[Sequence[str]] = None,
) -> gpd.GeoDataFrame:
    """Stage 4. Add one column per covariate, sampled at each pixel.

    Figure out: how to read a raster value at a point; which covariates are
    continuous vs categorical (don't average land cover); your rule for pixels
    that are NaN in a covariate.

    Static covariates depend only on where a pixel is ``(x, y)``, not on the
    calendar year. So when ``points`` is a pixel-year panel (the same ``(x, y)``
    repeated across many ``year_col`` values, as produced by
    :func:`get_treated_and_control_pixels` with ``years``), we sample each static
    layer once on the *unique* pixels and broadcast the value back across that
    pixel's years, instead of resampling the same raster once per pixel-year. The
    result is identical, just built without the redundant reads.

    Per-year (temporal) covariates -- year-specific weather, keyed on
    ``(x, y, year)`` -- are sampled per calendar year: for each ``year`` in the
    panel that layer's raster is warped onto the grid once and sampled at that
    year's pixels, then merged back on ``["x", "y", year_col]`` so each pixel-year
    gets *that year's* value. This is the panel-side per-year weather; the outcome
    frame joins the same layers in :func:`peatfire.build_frame`. Note that
    :func:`match_controls` collapses the panel to unique pixels and matches on the
    *static* covariates only, so temporal columns added here are for panel-level
    inspection / direct DiD on the panel, not the geographic match.

    Parameters
    ----------
    names : sequence of str, optional
        Static covariates to attach. Defaults to :func:`available_covariates`.
    temporal_names : sequence of str, optional
        Per-year covariates to attach. Only used when ``points`` carries
        ``year_col``. Defaults to :func:`available_temporal_covariates` for the
        panel's years (every per-year layer present for all those years).

    Contract (verified by :func:`check_covariates`)
        Returns ``points`` with one added column per requested covariate; no
        requested column is entirely NaN (for layers on disk).
    """
    if names is None:
        names = available_covariates()
    if aoi is None:
        aoi = points

    grid = build_common_grid(aoi, res_m, ANALYSIS_CRS)

    points = points.copy()  # don't mutate the caller's gdf in place

    # Static covariates vary only in space -> sample once per distinct pixel.
    unique_px = points[["x", "y"]].drop_duplicates().reset_index(drop=True)
    xi = xr.DataArray(unique_px["x"].values, dims="point")
    yi = xr.DataArray(unique_px["y"].values, dims="point")

    for name in names:
        cov = covariate_on_grid(name, grid, aoi)
        if cov is None:  # not downloaded yet, skip
            continue
        unique_px[name] = cov.sel(x=xi, y=yi, method="nearest").values

    # Broadcast the per-pixel values back onto every (pixel, year) row. For a
    # time-flat input this merge is a no-op join of each pixel onto itself.
    sampled = [c for c in unique_px.columns if c not in ("x", "y")]
    points = points.merge(unique_px[["x", "y", *sampled]], on=["x", "y"], how="left")

    # Per-year (temporal) covariates: sample each layer once per calendar year and
    # join on (x, y, year) so each pixel-year gets that year's weather.
    if year_col in points.columns:
        years = sorted(int(y) for y in points[year_col].dropna().unique())
        if temporal_names is None:
            temporal_names = available_temporal_covariates(years)
        for name in temporal_names:
            per_year = []
            for year in years:
                cov = temporal_covariate_on_grid(name, grid, aoi, year)
                if cov is None:  # that year not built yet -> leave those rows NaN
                    continue
                px = (
                    points.loc[points[year_col] == year, ["x", "y"]]
                    .drop_duplicates()
                    .reset_index(drop=True)
                )
                if px.empty:
                    continue
                txi = xr.DataArray(px["x"].values, dims="point")
                tyi = xr.DataArray(px["y"].values, dims="point")
                px[name] = cov.sel(x=txi, y=tyi, method="nearest").values
                px[year_col] = year
                per_year.append(px)
            if per_year:
                sampled_ty = pd.concat(per_year, ignore_index=True)
                points = points.merge(
                    sampled_ty, on=["x", "y", year_col], how="left"
                )

    return points

def add_matching_scores(
    pixels: gpd.GeoDataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str] = (),
    response: str = "burned",
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    propensity: bool = True,
    prognostic: bool = True,
    pscore_col: str = "pscore",
    phat_col: str = "phat",
    C: float = 1.0,
) -> gpd.GeoDataFrame:
    """Attach a **propensity** and/or **prognostic** score to every pixel.

    The static match in :func:`match_controls` compares treated and control
    pixels on the *raw covariate vector* (Mahalanobis). This function instead
    collapses that vector down to the two scalars that summarise the two channels
    through which a covariate can bias the comparison, exactly the Castro et al.
    (2026) match inputs:

    * **Propensity score** ``e(X) = P(treated | X)`` -- how *treatment-like* a
      pixel looks. Fitted by a (regularised) logistic regression of restoration-site
      membership on the covariates. Matching on it balances the covariates that
      drive *selection into treatment*.
    * **Prognostic score** ``Psi(X) = E[burn | X, untreated]`` -- a pixel's
      *baseline fire risk* absent treatment. Fitted on **control pixels only** (a
      logistic of "did this pixel burn in an untreated year" on the covariates),
      then predicted for everyone. Matching on it balances the covariates that
      drive the *outcome*.

    Matching on **both** (pass the two columns as ``continuous`` to
    :func:`match_controls`, i.e. ``continuous=[pscore_col, phat_col]``) is the
    doubly-robust / bijective match: comparable if *either* score model is right,
    the same logic as ``est_method="dr"`` in :func:`peatfire.modeling.estimate_att`.

    The scores are constant per physical pixel, so they are computed on the
    **unique** ``(x, y)`` pixels (treatment GROUP = restoration-site membership,
    matching :func:`match_controls`) and broadcast back across the pixel-years.

    Parameters
    ----------
    pixels : GeoDataFrame
        Pixel or pixel-year table with ``x``/``y`` and the ``continuous`` (and any
        ``categorical``) covariates already attached (see
        :func:`attach_covariates`). For the **prognostic** score it must also carry
        the per-year ``response`` (0/1 burn) and the per-year ``treated_col`` flag
        so untreated pixel-years -- the ``Y(0)`` observations -- can be identified.
    continuous, categorical : sequence of str
        Covariates fed to the score models: ``continuous`` are standardised;
        ``categorical`` are one-hot encoded. These are the *inputs* to the scores,
        not the match axes -- the match axes become ``pscore_col``/``phat_col``.
    response : str, default ``"burned"``
        Per-year 0/1 fire column used to fit the prognostic score. If absent, the
        prognostic score is skipped with a warning (propensity still runs).
    treated_col : str, default ``"treated"``
        Per-year treatment flag (1 once a pixel's site is restored). Untreated
        rows (``== 0``) are the baseline-fire observations for the prognostic fit.
    restoration_yr_col : str, default ``"End_Yr"``
        Restoration year; its non-null-ness defines the treated *group* for the
        propensity target (a treated pixel is "treated" for scoring in every year,
        even pre-restoration, exactly as in :func:`match_controls`).
    propensity, prognostic : bool
        Toggle each score independently.
    pscore_col, phat_col : str
        Output column names.
    C : float, default 1.0
        Inverse L2-regularisation strength for both logistic fits (sklearn
        convention); the penalty keeps the rare-fire prognostic model from running
        off to infinity under (quasi-)separation.

    Returns
    -------
    GeoDataFrame
        ``pixels`` with ``pscore_col`` and/or ``phat_col`` added (NaN on any pixel
        whose covariates were incomplete -- :func:`match_controls` drops those).

    Notes
    -----
    No leakage into the treatment effect: a treated pixel's prognostic score is
    predicted purely from its covariates, never from its own post-restoration
    burns. The baseline outcome is deliberately the *untreated* years only.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = pixels.copy()
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("pixels must carry 'x' and 'y' pixel coordinates.")

    cont = list(continuous)
    cat = list(categorical)

    # --- one row per physical pixel; treatment GROUP = restoration-site membership
    unique = df.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    if restoration_yr_col in unique.columns:
        grp = unique[restoration_yr_col].notna().to_numpy()
    else:
        grp = (unique[treated_col] == 1).to_numpy()

    # --- keep only pixels with complete covariates for the score models ---------
    ok = unique[cont + cat].notna().all(axis=1).to_numpy() if (cont or cat) else np.ones(len(unique), bool)
    use = unique.loc[ok].reset_index(drop=True)
    grp_use = grp[ok]

    # --- design matrix: standardised continuous + one-hot categorical -----------
    Xc = (
        StandardScaler().fit_transform(use[cont].to_numpy(dtype=float))
        if cont
        else np.empty((len(use), 0))
    )
    if cat:
        Xd = pd.get_dummies(use[cat].astype("category"), drop_first=True)
        X = np.hstack([Xc, Xd.to_numpy(dtype=float)])
    else:
        X = Xc

    # --- propensity: P(treated group | covariates) ------------------------------
    if propensity:
        if len(np.unique(grp_use)) < 2:
            raise ValueError(
                "cannot fit the propensity score: only one treatment class among "
                "pixels with complete covariates (need both treated and control)."
            )
        clf = LogisticRegression(max_iter=1000, C=C)
        clf.fit(X, grp_use.astype(int))
        use[pscore_col] = clf.predict_proba(X)[:, 1]

    # --- prognostic: E[burn | covariates, untreated], fit on controls only ------
    if prognostic:
        if response not in df.columns:
            warnings.warn(
                f"prognostic score skipped: no {response!r} column in pixels -- "
                "sample the fire response onto the panel first (see build_frame / "
                "the DiD notebook cell) to enable it.",
                stacklevel=2,
            )
        else:
            # Baseline (Y(0)) = did this pixel burn in an UNTREATED year? -- all
            # years for controls, pre-restoration years for treated pixels (the
            # per-year `treated_col` is already 0 in those rows).
            untreated = df.loc[df[treated_col] == 0, ["x", "y", response]]
            base = (
                untreated.groupby(["x", "y"])[response]
                .max()
                .rename("_baseline")
                .reset_index()
            )
            use = use.merge(base, on=["x", "y"], how="left")  # left-merge preserves X order
            train = (grp_use == 0) & use["_baseline"].notna().to_numpy()
            yb = use.loc[train, "_baseline"].astype(int).to_numpy()
            if int(train.sum()) < 2 or len(np.unique(yb)) < 2:
                rate = (
                    float(np.nanmean(use["_baseline"]))
                    if use["_baseline"].notna().any()
                    else 0.0
                )
                use[phat_col] = rate
                warnings.warn(
                    "prognostic model fell back to the constant base rate "
                    f"({rate:.4g}): too few control pixels with a defined baseline, "
                    "or no baseline-fire variation to fit on.",
                    stacklevel=2,
                )
            else:
                pmodel = LogisticRegression(max_iter=1000, C=C)
                pmodel.fit(X[train], yb)
                use[phat_col] = pmodel.predict_proba(X)[:, 1]
            use = use.drop(columns="_baseline")

    # --- broadcast the per-pixel scores back across the pixel-years -------------
    score_cols = [c for c in (pscore_col, phat_col) if c in use.columns]
    out = df.merge(use[["x", "y", *score_cols]], on=["x", "y"], how="left")
    if isinstance(df, gpd.GeoDataFrame):
        out = gpd.GeoDataFrame(out, geometry=df.geometry.name, crs=df.crs)
    return out


# ---------------------------------------------------------------------------
# Per-year / per-vintage score series + event-time matching (faithful Castro)
# ---------------------------------------------------------------------------
def _design_matrix(use: pd.DataFrame, continuous: Sequence[str], categorical: Sequence[str]):
    """Standardised continuous + one-hot categorical design matrix for the score fits.

    Built once over *all* rows in ``use`` so the same scaling / dummy columns apply
    when a model is trained on a subset (controls) and then predicted for everyone.
    """
    from sklearn.preprocessing import StandardScaler

    cont, cat = list(continuous), list(categorical)
    Xc = (
        StandardScaler().fit_transform(use[cont].to_numpy(dtype=float))
        if cont
        else np.empty((len(use), 0))
    )
    if cat:
        Xd = pd.get_dummies(use[cat].astype("category"), drop_first=True)
        return np.hstack([Xc, Xd.to_numpy(dtype=float)])
    return Xc


def _burned_wide(df: pd.DataFrame, response: str, year_col: str) -> pd.DataFrame:
    """Pixel x year table of the (0/1) response, indexed by ``(x, y)``, int year cols.

    The source both for per-year prognostic targets and for the pre-construction
    fire-history lags in the event-time match.
    """
    bw = (
        df.dropna(subset=[response])
        .drop_duplicates(["x", "y", year_col])
        .pivot(index=["x", "y"], columns=year_col, values=response)
    )
    bw.columns = [int(c) for c in bw.columns]
    return bw


def add_prognostic_score_series(
    pixels: gpd.GeoDataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str] = (),
    response: str = "burned",
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    year_col: str = "year",
    years: Optional[Sequence[int]] = None,
    prefix: str = "phat_",
    C: float = 1.0,
) -> gpd.GeoDataFrame:
    """Per-**year** prognostic fire-risk scores -- Castro's per-year prognostic model.

    A *separate* logistic fire-risk model is fit for **each outcome year**, calibrated
    **only on never-treated control pixels** (pixels whose ``restoration_yr_col`` is
    NaN -- never restored), then predicted for *all* pixels. Because the fitted
    coefficients differ year to year, the resulting ``phat_<year>`` columns are a
    **year-indexed series**: they encode how baseline fire risk *shifts over time*
    (much higher in dry El Nino / drought years), which a single collapsed prognostic
    score cannot. This is the per-year series the event-time match consumes.

    Unlike the scalar :func:`add_matching_scores`, this does **not** collapse the
    outcome over years -- it is the faithful, time-resolved prognostic score.

    Parameters
    ----------
    pixels : GeoDataFrame
        Pixel-year panel with ``x``/``y``, the covariates, the per-year 0/1
        ``response``, and ``year_col``.
    continuous, categorical : sequence of str
        Prognostic-model inputs: ``continuous`` standardised, ``categorical`` one-hot.
    years : sequence of int, optional
        Outcome years to fit. Defaults to every year with ``response`` coverage.
    prefix : str
        Output columns are ``f"{prefix}{year}"`` (default ``phat_2019`` ...).

    Returns
    -------
    GeoDataFrame
        ``pixels`` with one ``phat_<year>`` column per fitted year (broadcast across
        the pixel-years). A year with no control-pixel fire variation falls back to
        that year's base rate (with a warning).

    Notes
    -----
    Trained on never-treated controls only (Castro: "control pixels that never had a
    block"), so a treated pixel's prognostic score is a pure covariate prediction --
    no leakage of its own post-restoration burns.
    """
    from sklearn.linear_model import LogisticRegression

    if response not in pixels.columns:
        raise ValueError(
            f"need the per-year {response!r} column to fit prognostic scores; sample "
            "the fire response onto the panel first."
        )
    df = pixels.copy()
    cont, cat = list(continuous), list(categorical)

    unique = df.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    if restoration_yr_col in unique.columns:
        never = unique[restoration_yr_col].isna().to_numpy()
    else:  # fall back: a pixel that is control in every year is never-treated
        m = df.groupby(["x", "y"])[treated_col].max()
        never = unique.merge(m.rename("_m").reset_index(), on=["x", "y"], how="left")[
            "_m"
        ].fillna(0).eq(0).to_numpy()

    ok = (
        unique[cont + cat].notna().all(axis=1).to_numpy()
        if (cont or cat)
        else np.ones(len(unique), bool)
    )
    use = unique.loc[ok].reset_index(drop=True)
    never_use = never[ok]
    X = _design_matrix(use, cont, cat)

    bw = _burned_wide(df, response, year_col)
    if years is None:
        years = sorted(bw.columns)

    made: list[str] = []
    for yr in (int(y) for y in years):
        col = f"{prefix}{yr}"
        if yr not in bw.columns:
            warnings.warn(
                f"prognostic: no {response!r} coverage for {yr}; column {col} skipped.",
                stacklevel=2,
            )
            continue
        tgt = use.merge(
            bw[yr].rename("_t").reset_index(), on=["x", "y"], how="left"
        )["_t"].to_numpy()
        train = never_use & ~np.isnan(tgt)
        yb = tgt[train]
        if int(train.sum()) < 2 or len(np.unique(yb)) < 2:
            rate = float(np.nanmean(tgt[never_use])) if never_use.any() else 0.0
            use[col] = 0.0 if not np.isfinite(rate) else rate
            warnings.warn(
                f"prognostic {yr}: fell back to the base rate ({use[col].iloc[0]:.4g}); "
                "too few never-treated controls or no fire variation that year.",
                stacklevel=2,
            )
        else:
            model = LogisticRegression(max_iter=1000, C=C).fit(X[train], yb.astype(int))
            use[col] = model.predict_proba(X)[:, 1]
        made.append(col)

    out = df.merge(use[["x", "y", *made]], on=["x", "y"], how="left")
    if isinstance(df, gpd.GeoDataFrame):
        out = gpd.GeoDataFrame(out, geometry=df.geometry.name, crs=df.crs)
    return out


def add_propensity_score_series(
    pixels: gpd.GeoDataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str] = (),
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
    prefix: str = "psm_",
    C: float = 1.0,
) -> gpd.GeoDataFrame:
    """Per-**vintage** propensity scores -- Castro's per-construction-vintage psm.

    One logistic is fit **per restoration cohort** ``g`` (the pixels restored in year
    ``g`` as positives, the never-treated controls as negatives), then **projected
    onto every pixel**. So the coefficients are cohort-specific -- they capture the
    siting decisions of each construction vintage -- rather than varying by outcome
    year. The result is a ``psm_<g>`` column per vintage; the event-time match for
    cohort ``g`` uses ``psm_<g>``.

    (Castro fit per *province x vintage*; NC has no province layer, so this is
    per-vintage. Add a province/region stratum by fitting within its groups if one
    ever exists.)

    Parameters
    ----------
    pixels : GeoDataFrame
        Pixel(-year) panel with ``x``/``y``, the covariates, and ``restoration_yr_col``
        (the vintage; NaN for never-treated controls).
    continuous, categorical : sequence of str
        Propensity-model inputs: ``continuous`` standardised, ``categorical`` one-hot.
    prefix : str
        Output columns are ``f"{prefix}{g}"`` (default ``psm_2020`` ...).

    Returns
    -------
    GeoDataFrame
        ``pixels`` with one ``psm_<g>`` column per restoration vintage (projected onto
        all pixels, broadcast across the pixel-years).
    """
    from sklearn.linear_model import LogisticRegression

    if restoration_yr_col not in pixels.columns:
        raise ValueError(
            f"need {restoration_yr_col!r} to define construction vintages."
        )
    df = pixels.copy()
    cont, cat = list(continuous), list(categorical)

    unique = df.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    ry = unique[restoration_yr_col].to_numpy(dtype=float)  # NaN == never-treated
    never = np.isnan(ry)
    ok = (
        unique[cont + cat].notna().all(axis=1).to_numpy()
        if (cont or cat)
        else np.ones(len(unique), bool)
    )
    use = unique.loc[ok].reset_index(drop=True)
    ry_use, never_use = ry[ok], never[ok]
    X = _design_matrix(use, cont, cat)

    vintages = sorted(int(g) for g in np.unique(ry_use[~never_use]))
    if not vintages:
        raise ValueError("no treated vintages found (all pixels never-treated).")

    made: list[str] = []
    for g in vintages:
        col = f"{prefix}{g}"
        # positives = vintage g; negatives = never-treated controls; other vintages
        # are excluded from training but still get a projected score.
        target = np.where(ry_use == g, 1.0, np.where(never_use, 0.0, np.nan))
        train = ~np.isnan(target)
        if len(np.unique(target[train])) < 2:
            rate = float(np.mean(ry_use == g))
            use[col] = rate
            warnings.warn(
                f"propensity vintage {g}: fell back to a constant ({rate:.4g}); "
                "not enough contrast to fit.",
                stacklevel=2,
            )
        else:
            model = LogisticRegression(max_iter=1000, C=C).fit(
                X[train], target[train].astype(int)
            )
            use[col] = model.predict_proba(X)[:, 1]
        made.append(col)

    out = df.merge(use[["x", "y", *made]], on=["x", "y"], how="left")
    if isinstance(df, gpd.GeoDataFrame):
        out = gpd.GeoDataFrame(out, geometry=df.geometry.name, crs=df.crs)
    return out


def match_controls_event_time(
    pixels: gpd.GeoDataFrame,
    response: str = "burned",
    categorical: Sequence[str] = (),
    restoration_yr_col: str = "End_Yr",
    site_col: str = "Proj_Name",
    year_col: str = "year",
    treated_col: str = "treated",
    phat_prefix: str = "phat_",
    psm_prefix: str = "psm_",
    pre_lags: Sequence[int] = (1, 2),
    drought_year: Optional[int] = 2015,
    post_horizon: Optional[int] = None,
    caliper: float = 1.0,
    k: int = 1,
    max_dist_m: Optional[float] = None,
    carry: Sequence[str] = (),
    lag_prefix: str = "fire_y",
) -> gpd.GeoDataFrame:
    """Event-time 1:1 Mahalanobis match on Castro's **trajectory** vector (per vintage).

    Faithful to Castro's matching: rather than a single collapsed score, a treated
    pixel is matched to the control whose whole **time series of predicted risk**
    looks like its own. Matching is done **per construction vintage** ``g`` (so the
    event window aligns), one-to-one nearest-neighbour on Mahalanobis distance,
    **without replacement** within a vintage, and (if ``categorical`` is given)
    **exact-matched within each categorical class** (e.g. land cover) -- controls of
    a different class are never eligible.

    For each vintage ``g`` the matching vector is assembled, **event-time anchored**:

    * **pre-construction fire history** -- actual burn at years ``g - pre_lags``
      (e.g. ``g-1``, ``g-2``) and at ``drought_year`` (a fixed benchmark), each a
      separate dimension;
    * **per-year prognostic scores** ``phat_<t>`` for ``t = g .. last year`` (from
      :func:`add_prognostic_score_series`) -- the forward risk trajectory;
    * the **vintage propensity** ``psm_<g>`` (from :func:`add_propensity_score_series`).

    Any component whose calendar year is not in the panel (e.g. a pre-2019 lag or the
    2015 drought under 2019-2024 fire coverage) is simply omitted from that vintage's
    vector. Requires the ``phat_<year>`` and ``psm_<g>`` columns to already be on
    ``pixels`` -- run the two ``add_*_series`` builders first.

    Notes
    -----
    * **No** exact-match on ``site_col`` (few NC sites; cluster SEs on it downstream
      instead). Exact matching is applied only to ``categorical``.
    * Controls are matched without replacement *within* a vintage, but the never-
      treated pool is offered afresh to each vintage (per-vintage matching), so a
      strong control may serve different vintages -- mirroring Castro.
    * ``site_id`` on the output is the treated pixel's restoration site (controls
      inherit it); ``pair_id`` is globally unique across vintages; ``cohort`` records
      the vintage each pair was matched under.

    Returns
    -------
    GeoDataFrame
        Treated + matched control pixels across all vintages, with
        ``[unit_id, site_id, pair_id, cohort, treated, match_distance, x, y,
        geometry]`` plus ``restoration_yr_col`` and any ``carry`` columns -- ready for
        :func:`assemble_units` / :func:`peatfire.modeling.restrict_panel_to_matched`.
    """
    if restoration_yr_col not in pixels.columns:
        raise ValueError(f"need {restoration_yr_col!r} to define construction vintages.")

    df = pixels.copy()
    bw = _burned_wide(df, response, year_col)
    avail = set(bw.columns)

    unique = df.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    # attach burned-by-year columns (int-named) as the fire-history lag source
    unique = unique.merge(bw.reset_index(), on=["x", "y"], how="left")

    vintages = sorted(
        int(g)
        for g in unique.loc[unique[restoration_yr_col].notna(), restoration_yr_col].unique()
    )
    if not vintages:
        raise ValueError("no treated vintages to match.")

    id_cols = ["unit_id", "site_id", "pair_id", treated_col, "match_distance",
               "x", "y", "geometry"]
    keep_out = id_cols + [restoration_yr_col, "cohort"] + [c for c in carry]

    parts: list[gpd.GeoDataFrame] = []
    pair_offset = 0
    for g in vintages:
        cols: list[str] = []
        # pre-construction fire-history lags + drought benchmark (actual burn)
        lag_years = [g - int(l) for l in pre_lags]
        if drought_year is not None:
            lag_years.append(int(drought_year))
        for ly in lag_years:
            if ly in avail:
                cn = f"{lag_prefix}{ly}"
                if cn not in unique.columns:
                    unique[cn] = unique[ly]
                cols.append(cn)
        # forward per-year prognostic trajectory (construction -> end)
        post = [y for y in sorted(avail) if y >= g]
        if post_horizon is not None:
            post = post[:post_horizon]
        cols += [f"{phat_prefix}{py}" for py in post if f"{phat_prefix}{py}" in unique.columns]
        # vintage propensity
        if f"{psm_prefix}{g}" in unique.columns:
            cols.append(f"{psm_prefix}{g}")

        cols = list(dict.fromkeys(cols))
        if not cols:
            warnings.warn(
                f"cohort {g}: no event-time match variables available "
                "(missing phat_/psm_ columns and no in-panel fire lags); skipped.",
                stacklevel=2,
            )
            continue

        # this vintage's treated pixels + the never-treated control pool
        sub = unique[
            (unique[restoration_yr_col] == g) | (unique[restoration_yr_col].isna())
        ].copy()

        m = match_controls(
            sub,
            continuous=cols,
            categorical=categorical,        # exact-match on land cover etc.
            caliper=caliper,
            k=k,
            replace=False,
            max_dist_m=max_dist_m,
            restoration_yr_col=restoration_yr_col,
            site_col=site_col,
            carry=carry,
        )
        if len(m) == 0:
            continue
        m["cohort"] = g
        m["pair_id"] = m["pair_id"] + pair_offset
        pair_offset = int(m["pair_id"].max()) + 1
        parts.append(m[[c for c in keep_out if c in m.columns]])

    if not parts:
        raise ValueError(
            "no vintage produced matches -- check that add_prognostic_score_series / "
            "add_propensity_score_series were run and that fire coverage overlaps the "
            "event windows."
        )
    matched = pd.concat(parts, ignore_index=True)
    matched = gpd.GeoDataFrame(matched, geometry="geometry", crs=parts[0].crs)
    matched["unit_id"] = np.arange(len(matched))
    return matched


def match_controls(
    pixels: gpd.GeoDataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str] = (),
    caliper: float = 1.0,
    k: int = 1,
    treated_col: str = "treated",
    replace: bool = False,
    restoration_yr_col: str = "End_Yr",
    site_col: str = "Proj_Name",
    max_dist_m: Optional[float] = None,
    carry: Sequence[str] = (),
) -> gpd.GeoDataFrame:
    """Stage 5. Pair each treated pixel with its nearest control(s), optionally
    only among controls physically within ``max_dist_m`` metres.

    Why each step is here:

    * **Standardize before distances.** Elevation (metres, ~0-50) and histosol %
      (0-100) live on different scales; raw Euclidean distance would be dominated
      by whichever covariate has the bigger numeric spread. We whiten first so
      every covariate contributes on equal footing.
    * **Mahalanobis, not plain Euclidean.** Elevation and drainage/coast distance
      are *correlated*, so Euclidean distance double-counts the shared signal.
      Whitening by ``cov^{-1/2}`` decorrelates the covariates, turning plain
      Euclidean distance in the whitened space into Mahalanobis distance in the
      original space.
    * **Exact-match categoricals.** A categorical like land cover doesn't belong
      in a numeric distance (the codes are labels, not magnitudes), so we
      *stratify*: a treated pixel is only ever matched to controls of the **same
      class**, and the continuous distance is computed within that class.
    * **Caliper.** A distance ceiling (in whitened / SD units): a treated pixel
      with no control inside the caliper is dropped rather than matched to a poor
      twin. How many get dropped is a design diagnostic -- it's printed.
    * **Geographic caliper (``max_dist_m``).** A *separate* ceiling, in **metres**
      (not SD units), on how physically far a control may be from its treated
      pixel to be an eligible candidate -- the spatial restriction from Castro et
      al. (2026). We apply it **geography-first**: for each treated pixel we take
      only the controls within ``max_dist_m`` (a KD-tree radius query on the
      EPSG:5070 ``x``/``y`` coordinates), *then* rank those by covariate distance.
      Doing eligibility before the covariate ranking -- rather than post-filtering
      a global nearest-neighbour search -- means a geographically close, decent
      covariate match is never missed just because it fell outside a fixed
      neighbour budget. ``None`` (default) imposes no spatial restriction.
    * **Replacement.** ``replace=False`` (default) gives disjoint controls (each
      control used at most once); ``replace=True`` lets a good control serve
      several treated pixels (better balance, but reused controls -- the checker
      warns, not errors).
    * **Carry-through columns (``carry``).** Extra columns to copy onto every
      matched row *besides* the match axes. This matters when you match on
      **derived** axes -- e.g. the propensity/prognostic scores from
      :func:`add_matching_scores` (``continuous=["pscore", "phat"]``): the raw
      covariates (elevation, histosol %, ...) are then *not* match axes, so pass
      them here to keep them on the output for the raw-covariate love plot
      (:func:`balance_table`) and for ``build_frame`` downstream. Ignored if a
      named column is absent.

    Clustering key
    --------------
    ``site_id`` is the **restoration site** (``site_col``, e.g. ``Proj_Name``): a
    treated pixel carries its own site, and its matched control(s) *inherit that
    same site*. So all treated pixels of one restoration project and the controls
    matched to them form a single ``site_id`` group. That is the level standard
    errors should cluster on downstream (:func:`fit_logit_clustered`), because the
    effective sample size is the handful of restoration sites, not the thousands
    of correlated pixels within them. Each matched pair also gets a ``pair_id`` if
    you want the finer stratum.

    Contract (verified by :func:`check_matches`)
        Returns treated + matched control pixels as one GeoDataFrame with columns
        ``["unit_id", "site_id", "pair_id", treated_col, "match_distance", ...]``;
        control rows carry their distance to the treated partner (<= ``caliper``).
    """
    df = pixels.copy()

    # --- 1. Define the treated GROUP (not the per-year flag) --------------
    # If this is a year panel, `treated_col` flips per year and each pixel
    # repeats. Matching is on static covariates, so collapse to unique pixels
    # and label the group by restoration-site membership.
    if restoration_yr_col in df.columns:
        # one row per physical pixel for the matching step
        cross = df.drop_duplicates(subset=["x", "y"]).copy()
        cross["_grp_treated"] = cross[restoration_yr_col].notna().astype(int)
    else:
        cross = df.copy()
        cross["_grp_treated"] = cross[treated_col].astype(int)

    # --- 2. Build the covariate matrix, drop rows NaN in any covariate ----
    cont = list(continuous)
    cross = cross.dropna(subset=cont).reset_index(drop=True)
    X = cross[cont].to_numpy(dtype=float)

    # --- 3. Whiten so plain Euclidean == Mahalanobis ---------------------
    # W = cov^{-1/2}; Xw = (X - mean) @ W. In Xw, ordinary Euclidean distance
    # equals Mahalanobis distance in the original covariate space.
    mean = X.mean(axis=0)
    Xc = X - mean
    # np.cov wants variables in rows; rowvar=False keeps them in columns.
    cov = np.atleast_2d(np.cov(Xc, rowvar=False))
    # Ridge the diagonal so a (near-)singular covariance still inverts -- e.g. if
    # two covariates are collinear, or a class has too few pixels to estimate it.
    cov += 1e-6 * np.eye(cov.shape[0])
    # Symmetric eigendecomposition -> inverse square root: cov = V diag(w) V^T,
    # so cov^{-1/2} = V diag(w^{-1/2}) V^T.
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)  # guard tiny/negative eigenvalues
    W = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    Xw = Xc @ W  # whitened, shape (n_pixels, n_cont)

    cross_w = cross.assign(**{f"_z{i}": Xw[:, i] for i in range(Xw.shape[1])})
    zcols = [f"_z{i}" for i in range(Xw.shape[1])]

    # Columns each output row keeps: coordinates, geometry, the covariates, the
    # exact-match keys, the restoration year for the panel/DiD (when present), and
    # any caller-requested carry-through columns (e.g. the raw covariates behind a
    # score-based match). Dedup while preserving order; keep only columns present.
    extra = [c for c in (restoration_yr_col, *carry) if c in cross_w.columns]
    keep_cols = list(dict.fromkeys(["x", "y", "geometry", *cont, *list(categorical), *extra]))
    have_site = site_col in cross_w.columns

    def _record(row, is_treated: int, site_id, pair_id: int, distance: float) -> dict:
        rec = {col: row[col] for col in keep_cols}
        rec[treated_col] = is_treated
        rec["site_id"] = site_id      # restoration site (the cluster key)
        rec["pair_id"] = pair_id      # this treated pixel + its control(s)
        rec["match_distance"] = distance
        return rec

    # --- 4. Exact-match on categoricals: match WITHIN each class ----------
    # Group so a treated pixel only sees controls of the same land cover.
    group_keys = list(categorical) if categorical else None
    groups = cross_w.groupby(group_keys) if group_keys else [((), cross_w)]

    pairs = []  # collect matched treated + control rows
    n_dropped_dist = 0     # treated dropped: no control within max_dist_m
    n_dropped_caliper = 0  # treated dropped: none within the covariate caliper
    pair_counter = 0  # unique id per matched treated pixel (the pair/stratum)
    for _, g in groups:
        t = g[g["_grp_treated"] == 1]
        c = g[g["_grp_treated"] == 0].reset_index(drop=True)
        if len(t) == 0 or len(c) == 0:
            n_dropped_caliper += len(t)  # treated with no same-class controls
            continue

        # Whitened covariates (for the Mahalanobis distance) and physical
        # coordinates (metres, for the geographic caliper), as positional arrays
        # aligned to `t` / `c`.
        c_z = c[zcols].to_numpy()
        c_xy = c[["x", "y"]].to_numpy()
        t_z = t[zcols].to_numpy()
        t_xy = t[["x", "y"]].to_numpy()

        # --- 5a. Geography FIRST: eligible controls per treated pixel --------
        # For each treated pixel, restrict to controls within `max_dist_m` metres
        # BEFORE ranking on covariates, so a nearby, decent covariate match is
        # never missed for falling outside a fixed neighbour budget. `None` ->
        # every control is eligible (no spatial restriction).
        if max_dist_m is not None:
            tree = cKDTree(c_xy)
            eligible_per_t = tree.query_ball_point(t_xy, r=max_dist_m)
        else:
            all_pos = np.arange(len(c))
            eligible_per_t = [all_pos] * len(t)

        # --- 5b. Covariate caliper + assemble pairs -------------------------
        used: set[int] = set()  # control positions already claimed (no-replace)
        for i in range(len(t)):
            eligible = np.asarray(eligible_per_t[i], dtype=int)
            if eligible.size == 0:
                n_dropped_dist += 1  # no control within max_dist_m of this pixel
                continue

            # Covariates SECOND: Mahalanobis (whitened Euclidean) distance to the
            # eligible controls only, ordered nearest-first so the caliper `break`
            # below stays valid.
            d_elig = np.linalg.norm(c_z[eligible] - t_z[i], axis=1)
            order = np.argsort(d_elig)

            chosen = []  # (control_position, distance) picked for this treated
            for j in order:
                cpos, d = int(eligible[j]), float(d_elig[j])
                if d > caliper:
                    break  # covariate-sorted -> the rest are too far, stop
                if not replace and cpos in used:
                    continue
                chosen.append((cpos, d))
                if not replace:
                    used.add(cpos)
                if len(chosen) >= k:
                    break
            if not chosen:
                n_dropped_caliper += 1  # eligible by distance, none within caliper
                continue

            pid = pair_counter
            pair_counter += 1
            t_row = t.iloc[i]
            # The cluster key: this treated pixel's restoration site. Controls
            # inherit it, so every pixel compared against a site joins that site's
            # cluster. Fall back to the pair id if no site column is present.
            site_val = t_row[site_col] if have_site and pd.notna(t_row[site_col]) else pid
            pairs.append(_record(t_row, 1, site_val, pid, 0.0))  # treated: distance 0
            for cpos, d in chosen:
                pairs.append(_record(c.iloc[cpos], 0, site_val, pid, d))

    # --- 6. Assemble output the checker wants ----------------------------
    # Columns required by check_matches: unit_id, site_id, treated_col,
    # match_distance. site_id is the restoration site (cluster key); pair_id is the
    # finer matched stratum; control rows carry their distance to the treated partner.
    matched = gpd.GeoDataFrame(pairs, geometry="geometry", crs=cross.crs)
    matched["unit_id"] = np.arange(len(matched))

    n_treated = int((matched[treated_col] == 1).sum())
    n_control = int((matched[treated_col] == 0).sum())
    n_dropped = n_dropped_dist + n_dropped_caliper
    print(
        f"[match] {n_treated} treated matched to {n_control} controls "
        f"(k={k}, caliper={caliper}, replace={replace}, max_dist_m={max_dist_m}); "
        f"dropped {n_dropped} treated pixels "
        f"({n_dropped_dist} with no control within max_dist_m, "
        f"{n_dropped_caliper} with none inside the covariate caliper)"
    )
    return matched


def balance_table(
    pixels: gpd.GeoDataFrame,
    continuous: Sequence[str],
    treated_col: str = "treated",
    restoration_yr_col: str = "End_Yr",
) -> pd.DataFrame:
    """Stage 6. Standardized mean difference per covariate for a labelled set.

    Call it on the *unmatched* candidate pool and again on the *matched* set to
    get the before/after comparison. Use the provided
    :func:`standardized_mean_diff`.

    Contract
        Returns a DataFrame indexed by covariate with an ``smd`` column.
    """
    df = pixels

    # Define the two groups consistently with match_controls. On the *unmatched*
    # pixel-year panel `treated_col` flips per year, so collapse to unique pixels
    # and split by restoration-site membership; on a matched set (which carries a
    # `site_id`) the static `treated_col` is already the right split.
    if restoration_yr_col in df.columns and "site_id" not in df.columns:
        df = df.drop_duplicates(subset=["x", "y"])
        is_treated = df[restoration_yr_col].notna()
    else:
        is_treated = df[treated_col] == 1

    smds = {}
    for name in continuous:
        smds[name] = standardized_mean_diff(
            df.loc[is_treated, name], df.loc[~is_treated, name]
        )
    return pd.DataFrame({"smd": pd.Series(smds)})


def plot_balance(before: pd.DataFrame, after: pd.DataFrame, ax=None):
    """Stage 6. Love plot: |SMD| per covariate, before vs after matching.

    The figure that justifies the design -- after-matching points should collapse
    toward zero. Style with :func:`peatfire.set_fire_style`.

    Returns the matplotlib ``Axes`` so the caller can save the figure.
    """
    import matplotlib.pyplot as plt

    from ..fire_products_comparison.plotting import set_fire_style

    set_fire_style()

    covs = list(before.index)
    ypos = np.arange(len(covs))
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 0.6 * len(covs) + 1.5))

    # One dot per covariate for each of before / after; |SMD| shrinking toward 0
    # after matching is the whole point of the plot.
    ax.scatter(before["smd"].abs(), ypos, s=70, label="before matching", zorder=3)
    ax.scatter(
        after.reindex(covs)["smd"].abs(), ypos, s=70, label="after matching", zorder=3
    )
    # Connect each pair so the reader's eye follows the improvement per covariate.
    for y, b, a in zip(ypos, before["smd"].abs(), after.reindex(covs)["smd"].abs()):
        ax.plot([b, a], [y, y], color="0.7", lw=1, zorder=1)

    ax.axvline(0.1, ls="--", color="0.4", lw=1, label="|SMD| = 0.1 (balanced)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(covs)
    ax.set_xlabel("|standardized mean difference|")
    ax.set_title("Covariate balance: before vs after matching")
    ax.set_xlim(left=0)
    ax.legend()
    return ax


def assemble_units(matched: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Stage 7. Shape the matched pixels into the `units` build_frame expects.

    ``site_id`` is the restoration site (the cluster key from
    :func:`match_controls`); ``pair_id`` (the finer matched stratum) is carried
    through too when present, in case you want to cluster on the pair instead.

    The covariate columns matching sampled and balanced on (elevation, histosol
    %, ...) are **carried through unchanged**. This matters: ``matched`` pixels
    are pixel-*centroid* POINTs, and re-deriving covariates from rasters by
    clipping to those zero-area points nulls every pixel (see
    :func:`peatfire.build_frame`). Passing the already-sampled values through
    both avoids that trap and guarantees the model adjusts for exactly the values
    the matching balanced on.

    Contract
        Returns a GeoDataFrame with at least ``["unit_id", "site_id", "treated",
        "geometry"]`` (plus ``pair_id`` and any sampled covariate columns) --
        exactly what :func:`peatfire.build_frame` consumes.
    """
    required = ["unit_id", "site_id", "treated", "geometry"]
    missing = [c for c in required if c not in matched.columns]
    if missing:
        raise ValueError(
            f"assemble_units: matched set is missing {missing}; run match_controls() first."
        )

    # Keep the required columns first, then every other column matching produced
    # (covariates, x/y, restoration year, pair_id, ...) so build_frame can reuse
    # the covariate values instead of re-sampling rasters at zero-area points.
    keep = required + [c for c in matched.columns if c not in required]
    units = matched[keep].copy()
    # `site_id` is the restoration site (a project name); leave it as-is so
    # build_frame clusters on the site. `treated` must be a clean 0/1 integer.
    units["treated"] = units["treated"].astype("int64")
    return gpd.GeoDataFrame(units, geometry="geometry", crs=matched.crs).reset_index(
        drop=True
    )


# ===========================================================================
# PART B -- provided test harness (you should not need to edit below)
# ===========================================================================
def standardized_mean_diff(treated_vals, control_vals) -> float:
    """SMD = (mean_t - mean_c) / pooled SD. |SMD| < 0.1 is the usual "balanced".

    NaNs are dropped. Returns ``nan`` if either group is empty or has no spread.
    """
    t = np.asarray(treated_vals, dtype="float64")
    c = np.asarray(control_vals, dtype="float64")
    t = t[~np.isnan(t)]
    c = c[~np.isnan(c)]
    if t.size == 0 or c.size == 0:
        return float("nan")
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1)) / 2.0)
    if pooled_sd == 0:
        return float("nan")
    return float((t.mean() - c.mean()) / pooled_sd)


def _require_cols(df, cols, where):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise AssertionError(f"{where}: missing required columns {missing}.")


def check_treated_units(
    treated: gpd.GeoDataFrame, restoration_yr_col: str = "End_Yr"
) -> None:
    """Verify Stage 1's contract; raises AssertionError on the first failure."""
    _require_cols(treated, ["site_id", restoration_yr_col, "geometry"], "treated units")
    assert len(treated) > 0, "no treated (completed) sites returned."
    assert treated.crs is not None and treated.crs.to_epsg() == 5070, (
        f"treated CRS must be EPSG:5070, got {treated.crs}."
    )
    n_null = treated[restoration_yr_col].isna().sum()
    assert n_null == 0, f"{n_null} treated sites have a null {restoration_yr_col}."
    print(
        f"[Stage 1 OK] {len(treated)} completed sites, all with a {restoration_yr_col}."
    )


def check_candidate_pool(
    candidates: gpd.GeoDataFrame,
    treated: gpd.GeoDataFrame,
    spillover_m: float = DEFAULT_SPILLOVER_M,
) -> None:
    """Verify Stage 2: candidates avoid treated + halo and have positive area."""
    total = candidates.area.sum()
    assert total > 0, "candidate pool has zero area."
    halo = treated.to_crs(candidates.crs).buffer(spillover_m).union_all()
    overlap = candidates.geometry.intersection(halo).area.sum()
    frac = overlap / total
    assert frac < 1e-6, (
        f"candidate pool overlaps the treated+spillover halo "
        f"({frac:.2%} of its area) -- the difference/buffer step leaked."
    )
    print(f"[Stage 2 OK] candidate area {total/1e6:,.1f} km^2, ~0 overlap with halo.")


def check_pixels(
    points: gpd.GeoDataFrame,
    treated_polys: gpd.GeoDataFrame,
    treated_col: str = "treated",
) -> None:
    """Verify Stage 3: treated pixels fall inside restoration polygons."""
    _require_cols(points, ["x", "y", treated_col, "geometry"], "pixels")
    vals = set(pd.unique(points[treated_col].dropna()))
    assert vals <= {0, 1}, f"{treated_col} must be 0/1, saw {vals}."
    treated_pts = points[points[treated_col] == 1]
    assert len(treated_pts) > 0, "no treated pixels."
    joined = gpd.sjoin(
        treated_pts, treated_polys.to_crs(points.crs)[["geometry"]], predicate="within"
    )
    frac_in = joined.index.nunique() / len(treated_pts)
    assert frac_in > 0.99, (
        f"only {frac_in:.1%} of treated pixels fall inside restoration polygons."
    )
    print(
        f"[Stage 3 OK] {len(points):,} pixels "
        f"({len(treated_pts):,} treated, all inside restoration polygons)."
    )


def check_covariates(points: gpd.GeoDataFrame, names: Sequence[str]) -> None:
    """Verify Stage 4: covariate columns exist, aren't all-NaN, ranges are sane."""
    _require_cols(points, list(names), "covariate pixels")
    for name in names:
        assert not points[name].isna().all(), f"covariate {name!r} is entirely NaN."
        frac_nan = points[name].isna().mean()
        if frac_nan > 0.5:
            warnings.warn(f"{name}: {frac_nan:.0%} of pixels are NaN.", stacklevel=2)
    if "histosol_pct" in names:
        v = points["histosol_pct"].dropna()
        assert v.between(0, 100).all(), "histosol_pct outside [0, 100]."
    print(f"[Stage 4 OK] attached {list(names)} with sane ranges.")


def check_matches(
    matched: gpd.GeoDataFrame,
    caliper: float,
    treated_col: str = "treated",
    dist_col: str = "match_distance",
    id_col: str = "unit_id",
) -> None:
    """Verify Stage 5: every control is within the caliper; report drop count."""
    _require_cols(matched, [id_col, "site_id", treated_col, dist_col], "matched")
    controls = matched[matched[treated_col] == 0]
    treated = matched[matched[treated_col] == 1]
    assert len(controls) > 0 and len(treated) > 0, "matched set missing a class."
    worst = controls[dist_col].max()
    assert worst <= caliper + 1e-9, (
        f"a control lies {worst:.3g} away, beyond the caliper {caliper:g}."
    )
    if controls[id_col].duplicated().any():
        warnings.warn(
            "some control pixels are reused -- fine if you matched WITH "
            "replacement, a bug if you intended without.",
            stacklevel=2,
        )
    print(
        f"[Stage 5 OK] {len(treated):,} treated matched to {len(controls):,} "
        f"controls; max distance {worst:.3g} <= caliper {caliper:g}."
    )


def check_balance(before: pd.DataFrame, after: pd.DataFrame, threshold: float = 0.1) -> None:
    """Verify Stage 6: |SMD| shrank and clears the balance threshold post-match."""
    _require_cols(before, ["smd"], "before-balance")
    _require_cols(after, ["smd"], "after-balance")
    b = before["smd"].abs()
    a = after["smd"].abs()
    improved = (a <= b + 1e-9).all()
    assert improved, "matching did not reduce |SMD| for every covariate."
    worst = a.max()
    if worst > threshold:
        warnings.warn(
            f"post-match |SMD| max is {worst:.3f} > {threshold} -- residual "
            f"imbalance; consider a tighter caliper or more covariates.",
            stacklevel=2,
        )
    print(f"[Stage 6 OK] balance improved; post-match max |SMD| = {worst:.3f}.")

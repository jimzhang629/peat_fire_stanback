"""Control-pixel matching -- ASSIGNMENT SCAFFOLD (you implement Part A).

Builds the matched treated/control pixel set that :func:`peatfire.build_frame`
consumes. This is the causal-design heart of the study and is intentionally left
for you to implement; see ``matching_assignment.md`` for the staged walkthrough,
guiding questions, and hints.

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
from .covariates import available_covariates, covariate_on_grid  # noqa: F401
from .frame import DEFAULT_RES_M  # noqa: F401

from sklearn.neighbors import NearestNeighbors
# A pixel this close to a restoration site may be partially rewetted by it
# (spillover) -- exclude it from the control pool. Tune per Stage 2.
DEFAULT_SPILLOVER_M = 1000.0

# hm this is unnecessary i think - load_completed_restoration_sites_in_analysis_crs does this already
def load_treated_units(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Stage 1. Completed restoration polygons, in EPSG:5070, with a pivot year.

    Figure out: which column encodes status; your rule for "completed"; how you
    handle a missing end year; which year becomes the pivot (you chose end year).

    Contract (verified by :func:`check_treated_units`)
        Returns a GeoDataFrame in EPSG:5070 with at least columns
        ``["site_id", "pivot_year", "geometry"]``, one row per completed site,
        every ``pivot_year`` non-null.
    """
    raise NotImplementedError(
        "Stage 1: load restoration sites, keep completed, attach a pivot_year. "
        "See load_completed_restoration_sites_in_analysis_crs() in frame.py for the raw loader."
    )


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
      year is at or before that ``year`` (``years_after_treatment >= 0``); in
      earlier years it is a *not-yet-treated* pixel (``treated == 0``), which the
      staggered-DiD design in :mod:`peatfire.modeling.did` uses as a control;
    * ``years_after_treatment = year - restoration_year`` (0 = restoration year,
      positive = years since restoration, negative = years before), matching the
      ``event_year`` convention used in the fire-comparison notebook;
    * every candidate-pool pixel is present in every year with ``treated == 0`` and
      a null ``restoration_year`` / ``years_after_treatment``.

    A pixel's restoration-site membership is therefore recoverable at any time as
    ``restoration_year.notna()``, independent of the per-year ``treated`` status.

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
        Column on ``treated`` holding each site's restoration (pivot) year; carried
        onto treated pixels and renamed ``restoration_year`` in the panel.
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
        additionally ``["year", "restoration_year", "years_after_treatment"]`` and,
        when present, ``site_col``.
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
    # earliest restoration year (the year it first became treated).
    treated_pts = (
        treated_pts.sort_values(restoration_yr_col)
        .drop_duplicates(subset=["x", "y"], keep="first")
        .reset_index(drop=True)
        .rename(columns={restoration_yr_col: "restoration_year"})
    )
    control_pts = pixelate(candidates, res_m, grid)

    # Broadcast both pixel sets across the calendar years.
    treated_panel = _stack_across_years(treated_pts, years)
    control_panel = _stack_across_years(control_pts, years)

    # Per-year event time and treatment status for restoration-site pixels.
    treated_panel["restoration_year"] = treated_panel["restoration_year"].astype("float64")
    treated_panel["years_after_treatment"] = (
        treated_panel["year"] - treated_panel["restoration_year"]
    )
    treated_panel[treated_col_name] = (
        treated_panel["years_after_treatment"] >= 0
    ).astype(int)
    if drop_pretreatment:
        treated_panel = treated_panel[treated_panel[treated_col_name] == 1].reset_index(
            drop=True
        )

    # Candidate pool: control in every year, no restoration timing.
    control_panel[treated_col_name] = 0
    control_panel["restoration_year"] = np.nan
    control_panel["years_after_treatment"] = np.nan

    pixels = pd.concat([treated_panel, control_panel], ignore_index=True)
    return gpd.GeoDataFrame(pixels, geometry="geometry", crs=grid.rio.crs)

def attach_covariates(
    points: gpd.GeoDataFrame,
    names: Optional[Sequence[str]] = None,
    aoi: Optional[gpd.GeoDataFrame] = None,
    res_m: float = DEFAULT_RES_M,
    year_col: str = "year",
) -> gpd.GeoDataFrame:
    """Stage 4. Add one column per covariate, sampled at each pixel.

    Figure out: how to read a raster value at a point; which covariates are
    continuous vs categorical (don't average land cover); your rule for pixels
    that are NaN in a covariate.

    The covariates registered today are *static* -- they depend only on where a
    pixel is ``(x, y)``, not on the calendar year. So when ``points`` is a
    pixel-year panel (the same ``(x, y)`` repeated across many ``year_col`` values,
    as produced by :func:`get_treated_and_control_pixels` with ``years``), we
    sample each layer once on the *unique* pixels and broadcast the value back
    across that pixel's years, instead of resampling the same raster once per
    pixel-year. The result is identical, just built without the redundant reads.

    Per-year (temporal) covariates -- climate such as PRISM/Daymet -- are keyed on
    ``(x, y, year)`` and are a documented TODO below: when those layers land they
    are sampled here using ``points[year_col]`` and joined on the year too.

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

    # TODO(temporal covariates): per-year climate layers (PRISM/Daymet) are keyed
    # on (x, y, year); when downloaded, sample them per `year_col` and merge on
    # ["x", "y", year_col] here so each pixel-year gets that year's weather.

    return points

def match_controls(
    pixels: gpd.GeoDataFrame,
    continuous: Sequence[str],
    categorical: Sequence[str] = (),
    caliper: float = 1.0,
    k: int = 1,
    treated_col: str = "treated",
) -> gpd.GeoDataFrame:
    """Stage 5. Pair each treated pixel with its nearest control(s).

    Figure out: why standardize before computing distances; Euclidean vs
    Mahalanobis (what does Mahalanobis fix given elevation<->coast are
    correlated?); how to *exact-match* on ``categorical`` instead of putting it in
    the distance; what a caliper means and in what units; with/without
    replacement and the ``k`` ratio.

    Contract (verified by :func:`check_matches`)
        Returns treated + matched control pixels as one GeoDataFrame with columns
        ``["unit_id", "site_id", treated_col, "match_distance", ...]`` where
        ``site_id`` is the matched stratum (pair) and control rows carry their
        distance to the treated partner (<= ``caliper``).
    """
    df = pixels.copy()

    # --- 1. Define the treated GROUP (not the per-year flag) --------------
    # If this is a year panel, `treated_col` flips per year and each pixel
    # repeats. Matching is on static covariates, so collapse to unique pixels
    # and label the group by restoration-site membership.
    if "restoration_year" in df.columns:
        is_treated = df["restoration_year"].notna()
        # one row per physical pixel for the matching step
        cross = df.drop_duplicates(subset=["x", "y"]).copy()
        cross["_grp_treated"] = cross["restoration_year"].notna().astype(int)
    else:
        cross = df.copy()
        cross["_grp_treated"] = cross[treated_col].astype(int)

    # --- 2. Build the covariate matrix, drop rows NaN in any covariate ----
    cont = list(continuous)
    cross = cross.dropna(subset=cont).reset_index(drop=True)
    X = cross[cont].to_numpy(dtype=float)

    # --- 3. Whiten so plain Euclidean == Mahalanobis ---------------------
    # cov of the pooled covariates; W = cov^{-1/2}; Xw = (X - mean) @ W
    # TODO(you): compute mean, covariance, and the inverse-sqrt (hint:
    #   np.linalg.eigh on the covariance, or scipy.linalg.sqrtm of the inverse).
    #   Guard against a singular covariance (add a tiny ridge to the diagonal).
    Xw = ...  # whitened, shape (n_pixels, n_cont)
    cross_w = cross.assign(**{f"_z{i}": Xw[:, i] for i in range(Xw.shape[1])})
    zcols = [f"_z{i}" for i in range(Xw.shape[1])]

    # --- 4. Exact-match on categoricals: match WITHIN each class ----------
    # Group so a treated pixel only sees controls of the same land cover.
    group_keys = list(categorical) if categorical else None
    groups = cross_w.groupby(group_keys) if group_keys else [((), cross_w)]

    pairs = []  # collect matched (treated_row, control_row, distance)
    n_dropped = 0
    for _, g in groups:
        t = g[g["_grp_treated"] == 1]
        c = g[g["_grp_treated"] == 0]
        if len(t) == 0 or len(c) == 0:
            n_dropped += len(t)   # treated with no same-class controls
            continue

        nn = NearestNeighbors(n_neighbors=min(k, len(c)))
        nn.fit(c[zcols].to_numpy())
        dist, idx = nn.kneighbors(t[zcols].to_numpy())  # (n_t, k) each

        # --- 5. caliper + assemble pairs ---------------------------------
        # TODO(you): for each treated row i and its k neighbours:
        #   - keep only neighbours with dist <= caliper (drop the treated
        #     pixel entirely if none qualify -> increment n_dropped)
        #   - decide replacement: with -> reuse controls freely; without ->
        #     don't let a control index be claimed twice (track used idx)
        #   - record: treated row, matched control row(s), the distance,
        #     and a shared site_id for this stratum/pair
        ...

    # --- 6. Assemble output the checker wants ----------------------------
    # Columns required by check_matches: unit_id, site_id, treated_col,
    # match_distance. Treated rows: distance 0 (or NaN); control rows: their
    # distance to the treated partner. site_id = the pair/stratum id.
    # TODO(you): turn `pairs` into rows, keep geometry, set the 4 columns.
    matched = ...

    print(f"[match] dropped {n_dropped} treated pixels with no control in caliper")

    # --- 7. (optional) re-expand to the year panel -----------------------
    # site_id / match_distance are time-invariant, so if you want the full
    # panel back, merge these assignments onto `df` by ["x", "y"].
    return matched
    

def balance_table(
    pixels: gpd.GeoDataFrame,
    continuous: Sequence[str],
    treated_col: str = "treated",
) -> pd.DataFrame:
    """Stage 6. Standardized mean difference per covariate for a labelled set.

    Call it on the *unmatched* candidate pool and again on the *matched* set to
    get the before/after comparison. Use the provided
    :func:`standardized_mean_diff`.

    Contract
        Returns a DataFrame indexed by covariate with an ``smd`` column.
    """
    raise NotImplementedError(
        "Stage 6: for each covariate, split by treated_col and call "
        "standardized_mean_diff(treated_vals, control_vals)."
    )


def plot_balance(before: pd.DataFrame, after: pd.DataFrame):
    """Stage 6. Love plot: |SMD| per covariate, before vs after matching.

    The figure that justifies the design -- after-matching points should collapse
    toward zero. Style with :func:`peatfire.set_fire_style`.
    """
    raise NotImplementedError("Stage 6: scatter |smd| for `before` and `after`.")


def assemble_units(matched: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Stage 7. Shape the matched pixels into the `units` build_frame expects.

    Contract
        Returns a GeoDataFrame with ``["unit_id", "site_id", "treated",
        "geometry"]`` -- exactly what :func:`peatfire.build_frame` consumes.
    """
    raise NotImplementedError(
        "Stage 7: select/rename the matched columns; then build_frame(units)."
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


def check_treated_units(treated: gpd.GeoDataFrame) -> None:
    """Verify Stage 1's contract; raises AssertionError on the first failure."""
    _require_cols(treated, ["site_id", "pivot_year", "geometry"], "treated units")
    assert len(treated) > 0, "no treated (completed) sites returned."
    assert treated.crs is not None and treated.crs.to_epsg() == 5070, (
        f"treated CRS must be EPSG:5070, got {treated.crs}."
    )
    n_null = treated["pivot_year"].isna().sum()
    assert n_null == 0, f"{n_null} treated sites have a null pivot_year."
    print(f"[Stage 1 OK] {len(treated)} completed sites, all with a pivot_year.")


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

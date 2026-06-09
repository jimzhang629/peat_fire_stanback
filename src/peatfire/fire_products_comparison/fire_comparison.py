"""Compare fire products against each other within an arbitrary AOI.

This is the *analysis layer*. It consumes the standardized representations from
:mod:`peatfire.fire_products` and produces the numbers and grids behind the
figures:

* annual total burned area per product -- both at each product's **native**
  resolution and on a shared **common grid** (so the resolution confound is
  visible and controllable);
* pairwise **agreement matrices** -- binary (Jaccard / Cohen's kappa /
  percent agreement) for burned-area maps, and correlation (Pearson / Spearman)
  for severity grids or for the year-to-year co-variation of totals;
* the per-product common-grid stacks used to draw overlay maps.

Everything hangs off one entry point, :func:`compare_fire_products`, whose first
argument is an AOI -- a shapefile/GeoPackage path or a GeoDataFrame. The same
call works for the whole state, for peatlands, or for non-peatlands.

Design choices (see ``decisions.md``):

* **Analysis CRS is fixed at EPSG:5070** (NAD83 / CONUS Albers Equal Area), not
  the AOI's CRS. Area is only meaningful in an equal-area projection, and a
  fixed CRS keeps results comparable across different AOIs. The AOI's own CRS is
  used only as a clip mask.
* **Common grid uses "max" aggregation** ("any sub-cell burn lights the cell"),
  following Humber et al. (2019), so coarse and fine products are compared on
  identical cells.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling

from ..preproc.data_loading import data_path  # noqa: F401  (convenience re-export)
from .fire_products import (
    _files_for_period,
    get_spec,
    list_products,
    load_points,
    load_standardized,
)

ANALYSIS_CRS = "EPSG:5070"  # NAD83 / CONUS Albers Equal Area
AOILike = Union[str, Path, gpd.GeoDataFrame]


# ---------------------------------------------------------------------------
# AOI handling
# ---------------------------------------------------------------------------
def _as_gdf(aoi: AOILike) -> gpd.GeoDataFrame:
    """Accept a path or a GeoDataFrame and always return a GeoDataFrame."""
    if isinstance(aoi, gpd.GeoDataFrame):
        return aoi
    return gpd.read_file(aoi)


# ---------------------------------------------------------------------------
# Time unit handling (annual vs monthly)
# ---------------------------------------------------------------------------
def _expand_periods(years: Iterable[int], temporal_unit: str):
    """Expand a list of years into ``(year, month)`` periods.

    ``temporal_unit="year"`` -> one period per year with ``month=None`` (the
    annual behaviour). ``temporal_unit="month"`` -> twelve periods per year.
    """
    years = list(years)
    if temporal_unit == "year":
        return [(y, None) for y in years]
    if temporal_unit == "month":
        return [(y, m) for y in years for m in range(1, 13)]
    raise ValueError(f"temporal_unit must be 'year' or 'month', got {temporal_unit!r}.")


def _period_label(year: int, month: Optional[int]):
    """Index label for a period: the int year (annual) or a month-start Timestamp."""
    return year if month is None else pd.Timestamp(year=year, month=month, day=1)


def _monthly_capable(product: str) -> bool:
    """True if a product can be compared at monthly resolution.

    Monthly rasters carry a ``month_parser``; occurrence points (VIIRS) are
    filtered by date. Annual-only products are excluded from monthly mode.
    """
    spec = get_spec(product)
    return spec.month_parser is not None or spec.family == "occurrence"


# ---------------------------------------------------------------------------
# Area metric (moved verbatim in behaviour from the sandbox notebook)
# ---------------------------------------------------------------------------
def burned_area_km2(
    binary: xr.DataArray, equal_area: str = ANALYSIS_CRS, resolution: float = 30
) -> float:
    """Total burned area (km^2) of a boolean mask.

    ``binary`` is a boolean DataArray (``True`` = burned) with a CRS set. If the
    array is in a geographic CRS (degrees) it is first reprojected to
    ``equal_area`` at ``resolution`` metres so that every pixel has a constant,
    meaningful area; then burned pixels are counted and multiplied by the pixel
    area.
    """
    da = binary.astype("uint8")
    if da.rio.crs.is_geographic:
        da = da.rio.reproject(
            equal_area, resolution=resolution, resampling=Resampling.nearest
        )
    t = da.rio.transform()
    return float((da.values == 1).sum()) * abs(t.a * t.e) / 1e6


# ---------------------------------------------------------------------------
# Common grid (the resolution-confound fix)
# ---------------------------------------------------------------------------
def build_common_grid(
    aoi: gpd.GeoDataFrame, res_m: float = 500.0, crs: str = ANALYSIS_CRS
) -> xr.DataArray:
    """Build an empty reference grid covering ``aoi`` at ``res_m`` in ``crs``.

    Every product is later resampled onto *this* product-independent grid, so no
    single product's native grid privileges the comparison.
    """
    bounds = aoi.to_crs(crs).total_bounds  # (minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bounds
    ncols = int(np.ceil((maxx - minx) / res_m))
    nrows = int(np.ceil((maxy - miny) / res_m))
    # pixel-centre coordinates; y descending so origin is upper-left
    xs = minx + (np.arange(ncols) + 0.5) * res_m
    ys = maxy - (np.arange(nrows) + 0.5) * res_m
    grid = xr.DataArray(
        np.zeros((nrows, ncols), dtype="float32"),
        coords={"y": ys, "x": xs},
        dims=("y", "x"),
    )
    transform = Affine(res_m, 0.0, minx, 0.0, -res_m, maxy)
    grid = grid.rio.write_crs(crs)
    grid = grid.rio.write_transform(transform)
    return grid


def _resampling(how: str) -> Resampling:
    return {
        "max": Resampling.max,
        "min": Resampling.min,
        "mean": Resampling.average,
        "nearest": Resampling.nearest,
    }[how]


def to_common_grid(
    da: xr.DataArray, grid: xr.DataArray, how: str = "max"
) -> xr.DataArray:
    """Resample ``da`` onto ``grid`` via ``reproject_match``.

    ``how="max"`` implements "any sub-cell burn lights the cell" for binary
    masks; use ``"mean"`` (area-weighted average) for continuous severity.
    """
    return da.rio.reproject_match(grid, resampling=_resampling(how))


def rasterize_points_to_grid(
    pts: gpd.GeoDataFrame, grid: xr.DataArray, agg: str = "count"
) -> xr.DataArray:
    """Bin point detections onto ``grid``: per-cell ``count`` or binary ``any``.

    Points are reprojected to the grid CRS and assigned to cells by their
    coordinates. ``agg="count"`` returns detections per cell (for correlation);
    ``agg="any"`` returns a 0/1 presence mask (so VIIRS can join binary
    agreement matrices).
    """
    crs = grid.rio.crs
    pts = pts.to_crs(crs)
    xs = pts.geometry.x.values
    ys = pts.geometry.y.values

    gx = grid["x"].values
    gy = grid["y"].values
    res_x = abs(gx[1] - gx[0]) if gx.size > 1 else 1.0
    res_y = abs(gy[1] - gy[0]) if gy.size > 1 else 1.0
    # x edges ascending; y descending (matches build_common_grid)
    x_edges = np.append(gx - res_x / 2, gx[-1] + res_x / 2)
    y_edges = np.append(gy + res_y / 2, gy[-1] - res_y / 2)

    counts = np.zeros((gy.size, gx.size), dtype="float32")
    # digitize: column from x_edges (ascending), row from descending y
    col = np.searchsorted(x_edges, xs, side="right") - 1
    row = np.searchsorted(-y_edges, -ys, side="right") - 1
    inside = (col >= 0) & (col < gx.size) & (row >= 0) & (row < gy.size)
    np.add.at(counts, (row[inside], col[inside]), 1.0)

    if agg == "any":
        counts = (counts > 0).astype("float32")
    out = xr.DataArray(counts, coords={"y": gy, "x": gx}, dims=("y", "x"))
    return out.rio.write_crs(crs)


# ---------------------------------------------------------------------------
# Annual time series (native AND common-grid)
# ---------------------------------------------------------------------------
def annual_burned_area_series(
    products: Iterable[str],
    years: Iterable[int],
    aoi: gpd.GeoDataFrame,
    mode: str = "native",
    common_grid_res: float = 500.0,
    temporal_unit: str = "year",
) -> pd.DataFrame:
    """Total burned area (km^2) per product, indexed by period.

    ``mode="native"`` uses each product's own pixel area (reproducing Humber
    Figure 3 directly). ``mode="common_grid"`` first resamples every product to a
    shared ``common_grid_res`` grid (``how="max"``), so cell area is identical
    across products -- the fair, resolution-controlled comparison.

    ``temporal_unit="year"`` indexes by integer year (the default). With
    ``temporal_unit="month"`` the series is monthly (indexed by a month-start
    Timestamp) and **annual-only products are dropped** -- only products with a
    ``month_parser`` contribute.

    A ``<product>_pct_aoi`` column is also returned for each product, giving
    burned area as a percentage of the AOI area (comparable across AOIs of
    different size).
    """
    periods = _expand_periods(years, temporal_unit)
    grid = build_common_grid(aoi, res_m=common_grid_res) if mode == "common_grid" else None
    aoi_km2 = float(aoi.to_crs(ANALYSIS_CRS).area.sum()) / 1e6

    rows: dict = {_period_label(y, m): {} for (y, m) in periods}
    for product in products:
        spec = get_spec(product)
        if spec.family != "burned_area":
            continue
        if temporal_unit == "month" and spec.month_parser is None:
            continue  # annual-only product: not available at monthly resolution
        for year, month in periods:
            mask = load_standardized(product, year, aoi, month)
            if mask is None:
                continue
            if mode == "common_grid":
                mask = to_common_grid(mask.astype("float32"), grid, how="max") > 0
                mask = mask.rio.write_crs(grid.rio.crs)
            km2 = burned_area_km2(mask)
            label = _period_label(year, month)
            rows[label][product] = km2
            rows[label][f"{product}_pct_aoi"] = (
                100.0 * km2 / aoi_km2 if aoi_km2 else np.nan
            )

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "period" if temporal_unit == "month" else "year"
    return df


# ---------------------------------------------------------------------------
# Common-grid stacks + agreement
# ---------------------------------------------------------------------------
def stack_on_common_grid(
    products: Iterable[str],
    year: int,
    aoi: gpd.GeoDataFrame,
    common_grid_res: float = 500.0,
    binary: bool = True,
    grid: Optional[xr.DataArray] = None,
    month: Optional[int] = None,
) -> dict[str, xr.DataArray]:
    """Load each product for a year (or month) and align it to one shared grid.

    ``binary=True`` returns burned/unburned masks (occurrence products via
    presence); ``binary=False`` returns continuous severity grids. ``month`` (if
    given) restricts to a single calendar month -- annual-only products then have
    no data and are omitted, as are any products absent for the period.
    """
    if grid is None:
        grid = build_common_grid(aoi, res_m=common_grid_res)
    out: dict[str, xr.DataArray] = {}
    for product in products:
        spec = get_spec(product)
        if spec.family == "occurrence":
            pts = load_points(product, year, aoi, month)
            if pts is None:
                continue
            out[product] = rasterize_points_to_grid(
                pts, grid, agg="any" if binary else "count"
            )
            continue
        da = load_standardized(product, year, aoi, month)
        if da is None:
            continue
        how = "max" if binary else "mean"
        g = to_common_grid(da.astype("float32"), grid, how=how)
        out[product] = (g > 0).astype("float32") if binary else g
    return out


def _valid_pair(a: np.ndarray, b: np.ndarray):
    """Flatten and keep only cells where both arrays are finite."""
    a = a.ravel()
    b = b.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def _binary_scores(a: np.ndarray, b: np.ndarray, method: str) -> float:
    a, b = _valid_pair(a, b)
    a = a > 0
    b = b > 0
    if a.size == 0:
        return np.nan
    if method in ("jaccard", "iou"):
        union = (a | b).sum()
        return float((a & b).sum() / union) if union else np.nan
    if method == "percent_agreement":
        return float((a == b).mean())
    if method == "kappa":
        po = (a == b).mean()
        pa, pb = a.mean(), b.mean()
        pe = pa * pb + (1 - pa) * (1 - pb)
        return float((po - pe) / (1 - pe)) if (1 - pe) else np.nan
    raise ValueError(f"Unknown binary method {method!r}.")


def _corr_score(a: np.ndarray, b: np.ndarray, method: str) -> float:
    a, b = _valid_pair(a, b)
    if a.size < 2:
        return np.nan
    if method == "pearson":
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    if method == "spearman":
        ra = pd.Series(a).rank().values
        rb = pd.Series(b).rank().values
        if ra.std() == 0 or rb.std() == 0:
            return np.nan
        return float(np.corrcoef(ra, rb)[0, 1])
    raise ValueError(f"Unknown correlation method {method!r}.")


def total_least_squares(x, y) -> tuple[float, float]:
    """Total least squares (orthogonal) regression of ``y`` on ``x``.

    Unlike ordinary least squares (which minimises *vertical* distance and
    assumes ``x`` is error-free), TLS minimises the *perpendicular* distance to
    the line -- the right choice when both products carry measurement error, as
    in Humber et al. (2019). Returns ``(slope, intercept)``.

    Note the fit is symmetric only up to reflection: swapping x and y gives the
    reciprocal slope (the same line reflected across y=x).
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return (np.nan, np.nan)
    xb, yb = x.mean(), y.mean()
    dx, dy = x - xb, y - yb
    sxx = np.mean(dx * dx)
    syy = np.mean(dy * dy)
    sxy = np.mean(dx * dy)
    if sxy == 0:
        return (np.nan, np.nan)
    slope = (syy - sxx + np.sqrt((syy - sxx) ** 2 + 4 * sxy ** 2)) / (2 * sxy)
    intercept = yb - slope * xb
    return (float(slope), float(intercept))


def rmse(x, y) -> float:
    """Root-mean-square error between two products (symmetric; 0 = identical).

    Humber reports this alongside the TLS line, compared against the y=x line.
    """
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((x[m] - y[m]) ** 2)))


# methods whose scoring is on continuous values (not binary masks)
_CONTINUOUS_METHODS = ("pearson", "spearman", "tls_slope", "rmse")


def _ordered_score(x: np.ndarray, y: np.ndarray, method: str, same: bool) -> float:
    """Score one ordered pair (x as predictor, y as response).

    Symmetric methods ignore the ordering; ``tls_slope`` does not. ``same`` marks
    the diagonal (a product vs itself) so it gets the natural identity value.
    """
    if method in ("jaccard", "iou", "kappa", "percent_agreement"):
        return 1.0 if same else _binary_scores(x, y, method)
    if method in ("pearson", "spearman"):
        return 1.0 if same else _corr_score(x, y, method)
    if method == "tls_slope":
        return 1.0 if same else total_least_squares(x, y)[0]
    if method == "rmse":
        return 0.0 if same else rmse(x, y)
    raise ValueError(f"Unknown method {method!r}.")


def period_totals_series(
    products: Iterable[str],
    years: Iterable[int],
    aoi: gpd.GeoDataFrame,
    temporal_unit: str = "year",
    common_grid_res: float = 500.0,
) -> pd.DataFrame:
    """Per-period total for each product, indexed by period.

    Burned-area products contribute their **common-grid burned area (km^2)**;
    occurrence products (VIIRS) contribute their **active-fire detection count**.
    Including counts lets active fire join the temporal scatter / correlation /
    TLS against the burned-area products, the quantitative analogue of Humber's
    temporal heat maps. (Severity products are skipped -- they have no scalar
    total.) Units differ by column, which is fine for correlation/TLS but means
    RMSE across families is not meaningful.
    """
    periods = _expand_periods(years, temporal_unit)
    grid = build_common_grid(aoi, res_m=common_grid_res)
    rows: dict = {_period_label(y, m): {} for (y, m) in periods}
    for product in products:
        spec = get_spec(product)
        if spec.family == "severity":
            continue
        if temporal_unit == "month" and not _monthly_capable(product):
            continue
        has_files = bool(_files_for_period(get_spec(product), periods[0][0])) or \
            spec.directory.exists()
        if not has_files:
            continue
        for year, month in periods:
            label = _period_label(year, month)
            if spec.family == "occurrence":
                pts = load_points(product, year, aoi, month)
                rows[label][product] = 0 if pts is None else int(len(pts))
            else:  # burned_area
                mask = load_standardized(product, year, aoi, month)
                if mask is None:
                    continue
                mask = to_common_grid(mask.astype("float32"), grid, how="max") > 0
                mask = mask.rio.write_crs(grid.rio.crs)
                rows[label][product] = burned_area_km2(mask)

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "period" if temporal_unit == "month" else "year"
    return df


def agreement_matrix(
    products: Iterable[str],
    years: Iterable[int],
    aoi: gpd.GeoDataFrame,
    method: str = "jaccard",
    common_grid_res: float = 500.0,
    pooling: str = "cells",
    temporal_unit: str = "year",
) -> pd.DataFrame:
    """Pairwise agreement among products as a products x products DataFrame.

    Parameters
    ----------
    method : str
        Binary-mask methods: ``"jaccard"``/``"iou"``, ``"kappa"``,
        ``"percent_agreement"``. Continuous methods: ``"pearson"``,
        ``"spearman"``, ``"tls_slope"`` (total least squares slope), ``"rmse"``.
        TLS and RMSE follow Humber et al.'s scatterplot statistics; ``tls_slope``
        near 1 means the two products agree on magnitude, and the matrix is
        *asymmetric* (entry [a, b] regresses with a as x, b as y).
    pooling : str
        ``"cells"`` pools all grid cells across every period (spatial agreement
        -- "do they burn the same places?"). ``"years"`` uses each product's
        per-period total (temporal co-variation -- "do they rise and fall
        together?"), and includes occurrence products (VIIRS) via detection count.
    temporal_unit : str
        ``"year"`` (default) compares per year; ``"month"`` compares per calendar
        month and drops annual-only products.
    """
    products = list(products)
    binary = method in ("jaccard", "iou", "kappa", "percent_agreement")

    if pooling == "years":
        # one value per product per period (BA km^2, or VIIRS detection count)
        df = period_totals_series(
            products, years, aoi, temporal_unit=temporal_unit,
            common_grid_res=common_grid_res,
        )
        names = [p for p in products if p in df.columns]
        vecs = {p: df[p].values.astype("float64") for p in names}
    else:
        # pooling == "cells": concatenate flattened cells across all periods
        periods = _expand_periods(years, temporal_unit)
        pooled: dict[str, list[np.ndarray]] = {p: [] for p in products}
        grid = build_common_grid(aoi, res_m=common_grid_res)
        for year, month in periods:
            stack = stack_on_common_grid(
                products, year, aoi, binary=binary, grid=grid, month=month
            )
            for p in products:
                if p in stack:
                    pooled[p].append(stack[p].values.astype("float64"))
        vecs = {p: np.concatenate(arrs) for p, arrs in pooled.items() if arrs}
        names = [p for p in products if p in vecs]

    mat = pd.DataFrame(index=names, columns=names, dtype="float64")
    for a in names:
        for b in names:
            mat.loc[a, b] = _ordered_score(vecs[a], vecs[b], method, same=(a == b))
    return mat


def product_pair_scatter(
    product_x: str,
    product_y: str,
    year: int,
    aoi: gpd.GeoDataFrame,
    month: Optional[int] = None,
    common_grid_res: float = 500.0,
):
    """Paired per-cell values for a two-product scatterplot in one period.

    Each point is one common-grid cell -- the per-spatial-unit analogue of
    Humber's per-TSA scatter points -- holding the *burned fraction* of that cell
    for each product (continuous in [0, 1]; for VIIRS, detections per cell).
    Returns ``(x, y)`` 1-D arrays over cells where both products are finite.
    Feed these to :func:`total_least_squares`, :func:`rmse`, or
    :func:`peatfire.fire_products_comparison.plot_product_scatter`.
    """
    gdf = _as_gdf(aoi)
    grid = build_common_grid(gdf, res_m=common_grid_res)
    stack = stack_on_common_grid(
        [product_x, product_y], year, gdf, binary=False, grid=grid, month=month
    )
    if product_x not in stack or product_y not in stack:
        return (np.array([]), np.array([]))
    a = stack[product_x].values.ravel()
    b = stack[product_y].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def compare_fire_products(
    aoi: AOILike,
    products: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    metric: str = "burned_area",
    common_grid_res: float = 500.0,
    agreement_methods: Iterable[str] = ("jaccard", "kappa"),
    overlay_years: Optional[Iterable] = None,
    out_dir: Optional[Union[str, Path]] = None,
    temporal_unit: str = "year",
) -> dict:
    """Run the full comparison for one AOI and return a dict of results.

    Parameters
    ----------
    aoi : str | Path | GeoDataFrame
        Shapefile/GeoPackage path or GeoDataFrame. Used both to clip every
        product and to define the common-grid extent, so the same call works for
        NC, peatlands, or non-peatlands.
    products : iterable of str, optional
        Defaults to every registered product in the chosen ``metric`` family,
        plus ``VIIRS`` (occurrence) so it can appear in the agreement matrices.
    metric : str
        ``"burned_area"`` or ``"severity"``.
    agreement_methods : iterable of str
        Which spatial-agreement metrics to compute. A temporal correlation
        matrix (pooling="years") is always added for burned area.
    overlay_years : iterable, optional
        Periods to build overlay stacks for. In annual mode these are integer
        years; in monthly mode pass ``(year, month)`` tuples.
    temporal_unit : str
        ``"year"`` (default) compares per year. ``"month"`` compares per calendar
        month and **drops annual-only products** (GABAM, USGS_BA, SE_FireMap,
        MTBS), keeping the monthly products plus VIIRS -- matching the monthly
        timing analysis in Humber et al.

    Returns
    -------
    dict
        Keys include ``"area_native"``, ``"area_common_grid"`` (DataFrames),
        ``"agreement_<method>"`` and ``"correlation_years"`` (DataFrames), and
        ``"stacks"`` (per-overlay-period common-grid dicts). If ``out_dir`` is
        given, CSVs are written there too.
    """
    gdf = _as_gdf(aoi)
    if metric not in ("burned_area", "severity"):
        raise ValueError(f"Unknown metric {metric!r}.")
    if products is None:
        # the chosen family, plus occurrence products (VIIRS) so they appear in
        # every agreement matrix as an independent check.
        products = list_products(metric) + list_products("occurrence")
    products = list(products)
    if temporal_unit == "month":
        # monthly resolution: keep only monthly-capable products (drops annual-only).
        products = [p for p in products if _monthly_capable(p)]
    if years is None:
        years = range(2001, 2022)
    years = list(years)

    results: dict = {}
    family_products = [p for p in products if get_spec(p).family == metric]

    if metric == "burned_area":
        results["area_native"] = annual_burned_area_series(
            family_products, years, gdf, mode="native", temporal_unit=temporal_unit
        )
        results["area_common_grid"] = annual_burned_area_series(
            family_products, years, gdf, mode="common_grid",
            common_grid_res=common_grid_res, temporal_unit=temporal_unit,
        )
        # temporal co-variation of per-period totals. Pearson includes VIIRS
        # (scale-invariant); TLS slope / RMSE compare magnitudes so they stay on
        # the burned-area products, which share km^2 units.
        results["correlation_years"] = agreement_matrix(
            products, years, gdf, method="pearson", pooling="years",
            common_grid_res=common_grid_res, temporal_unit=temporal_unit,
        )
        results["tls_slope_years"] = agreement_matrix(
            family_products, years, gdf, method="tls_slope", pooling="years",
            common_grid_res=common_grid_res, temporal_unit=temporal_unit,
        )
        results["rmse_years"] = agreement_matrix(
            family_products, years, gdf, method="rmse", pooling="years",
            common_grid_res=common_grid_res, temporal_unit=temporal_unit,
        )
        for m in agreement_methods:
            results[f"agreement_{m}"] = agreement_matrix(
                products, years, gdf, method=m, pooling="cells",
                common_grid_res=common_grid_res, temporal_unit=temporal_unit,
            )
    else:  # severity
        results["agreement_spearman"] = agreement_matrix(
            products, years, gdf, method="spearman", pooling="cells",
            common_grid_res=common_grid_res, temporal_unit=temporal_unit,
        )

    if overlay_years:
        # entries are int years (annual) or (year, month) tuples (monthly)
        stacks: dict = {}
        for entry in overlay_years:
            yr, mo = entry if isinstance(entry, tuple) else (entry, None)
            stacks[entry] = stack_on_common_grid(
                family_products, yr, gdf, common_grid_res=common_grid_res,
                binary=(metric == "burned_area"), month=mo,
            )
        results["stacks"] = stacks

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, val in results.items():
            if isinstance(val, pd.DataFrame):
                val.to_csv(out_dir / f"{key}.csv")

    return results

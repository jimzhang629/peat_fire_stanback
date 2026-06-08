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

from .data_loading import data_path  # noqa: F401  (convenience re-export for callers)
from .fire_products import get_spec, list_products, load_points, load_standardized

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
) -> pd.DataFrame:
    """Total annual burned area (km^2) per product, indexed by year.

    ``mode="native"`` uses each product's own pixel area (reproducing Humber
    Figure 3 directly). ``mode="common_grid"`` first resamples every product to a
    shared ``common_grid_res`` grid (``how="max"``), so cell area is identical
    across products -- the fair, resolution-controlled comparison.

    A ``<product>_pct_aoi`` column is also returned for each product, giving
    burned area as a percentage of the AOI area (comparable across AOIs of
    different size).
    """
    years = list(years)
    grid = build_common_grid(aoi, res_m=common_grid_res) if mode == "common_grid" else None
    aoi_km2 = float(aoi.to_crs(ANALYSIS_CRS).area.sum()) / 1e6

    rows: dict[int, dict[str, float]] = {y: {} for y in years}
    for product in products:
        if get_spec(product).family != "burned_area":
            continue
        for year in years:
            mask = load_standardized(product, year, aoi)
            if mask is None:
                continue
            if mode == "common_grid":
                mask = to_common_grid(mask.astype("float32"), grid, how="max") > 0
                mask = mask.rio.write_crs(grid.rio.crs)
            km2 = burned_area_km2(mask)
            rows[year][product] = km2
            rows[year][f"{product}_pct_aoi"] = 100.0 * km2 / aoi_km2 if aoi_km2 else np.nan

    df = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    df.index.name = "year"
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
) -> dict[str, xr.DataArray]:
    """Load each product for ``year`` and align it to one shared grid.

    ``binary=True`` returns burned/unburned masks (occurrence products via
    presence); ``binary=False`` returns continuous severity grids. Products
    absent for the year are omitted from the returned dict.
    """
    if grid is None:
        grid = build_common_grid(aoi, res_m=common_grid_res)
    out: dict[str, xr.DataArray] = {}
    for product in products:
        spec = get_spec(product)
        if spec.family == "occurrence":
            pts = load_points(product, year, aoi)
            if pts is None:
                continue
            out[product] = rasterize_points_to_grid(
                pts, grid, agg="any" if binary else "count"
            )
            continue
        da = load_standardized(product, year, aoi)
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


def agreement_matrix(
    products: Iterable[str],
    years: Iterable[int],
    aoi: gpd.GeoDataFrame,
    method: str = "jaccard",
    common_grid_res: float = 500.0,
    pooling: str = "cells",
) -> pd.DataFrame:
    """Pairwise agreement among products as a symmetric DataFrame.

    Parameters
    ----------
    method : str
        ``"jaccard"``/``"iou"``, ``"kappa"``, ``"percent_agreement"`` operate on
        binary burned masks. ``"pearson"``/``"spearman"`` operate on continuous
        values (severity grids when ``pooling="cells"``, or annual totals when
        ``pooling="years"``).
    pooling : str
        ``"cells"`` pools all grid cells across ``years`` into paired vectors
        (spatial agreement -- "do they burn the same places?").
        ``"years"`` correlates each product's annual total across years
        (temporal co-variation -- "do they rise and fall together?").
    """
    products = list(products)
    binary = method in ("jaccard", "iou", "kappa", "percent_agreement")

    if pooling == "years":
        # correlate annual totals; works for any family that yields a yearly
        # scalar -- here, burned-area products via common-grid totals.
        df = annual_burned_area_series(
            products, years, aoi, mode="common_grid", common_grid_res=common_grid_res
        )
        cols = [p for p in products if p in df.columns]
        return df[cols].corr(method="spearman" if method == "spearman" else "pearson")

    # pooling == "cells": concatenate flattened cells across all years
    pooled: dict[str, list[np.ndarray]] = {p: [] for p in products}
    grid = build_common_grid(aoi, res_m=common_grid_res)
    for year in years:
        stack = stack_on_common_grid(
            products, year, aoi, binary=binary, grid=grid
        )
        for p in products:
            if p in stack:
                pooled[p].append(stack[p].values.astype("float64"))

    vecs = {
        p: np.concatenate(arrs) for p, arrs in pooled.items() if arrs
    }
    names = [p for p in products if p in vecs]
    mat = pd.DataFrame(index=names, columns=names, dtype="float64")
    score = _binary_scores if binary else _corr_score
    for i, a in enumerate(names):
        for b in names[i:]:
            val = 1.0 if a == b and binary else score(vecs[a], vecs[b], method)
            mat.loc[a, b] = val
            mat.loc[b, a] = val
    return mat


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
    overlay_years: Optional[Iterable[int]] = None,
    out_dir: Optional[Union[str, Path]] = None,
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

    Returns
    -------
    dict
        Keys include ``"area_native"``, ``"area_common_grid"`` (DataFrames),
        ``"agreement_<method>"`` and ``"correlation_years"`` (DataFrames), and
        ``"stacks"`` (per-overlay-year common-grid dicts). If ``out_dir`` is
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
    if years is None:
        years = range(2001, 2022)
    years = list(years)

    results: dict = {}
    family_products = [p for p in products if get_spec(p).family == metric]

    if metric == "burned_area":
        results["area_native"] = annual_burned_area_series(
            family_products, years, gdf, mode="native"
        )
        results["area_common_grid"] = annual_burned_area_series(
            family_products, years, gdf, mode="common_grid",
            common_grid_res=common_grid_res,
        )
        results["correlation_years"] = agreement_matrix(
            family_products, years, gdf, method="pearson", pooling="years",
            common_grid_res=common_grid_res,
        )
        for m in agreement_methods:
            results[f"agreement_{m}"] = agreement_matrix(
                products, years, gdf, method=m, pooling="cells",
                common_grid_res=common_grid_res,
            )
    else:  # severity
        results["agreement_spearman"] = agreement_matrix(
            products, years, gdf, method="spearman", pooling="cells",
            common_grid_res=common_grid_res,
        )

    if overlay_years:
        results["stacks"] = {
            y: stack_on_common_grid(
                family_products, y, gdf, common_grid_res=common_grid_res,
                binary=(metric == "burned_area"),
            )
            for y in overlay_years
        }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, val in results.items():
            if isinstance(val, pd.DataFrame):
                val.to_csv(out_dir / f"{key}.csv")

    return results

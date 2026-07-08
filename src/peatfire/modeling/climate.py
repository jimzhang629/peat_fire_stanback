"""Turn GHCN weather-station points into gridded climate covariates.

The project's climate source (``src/get_climate&soil_data.R``) pulls **GHCN
daily** station records -- per-station ``TMAX``/``TMIN``/``PRCP`` time series with
a point geometry -- not a raster. The matching / frame pipeline, on the other
hand, samples every covariate as a value *at a pixel* on the shared EPSG:5070
grid (like elevation or histosol %). This module bridges the two: it collapses
the station records into a small number of **climate normals** per station and
**interpolates** them onto the common grid, writing one GeoTIFF per element so
they register in :mod:`peatfire.modeling.covariates` exactly like any other
raster covariate.

Why *normals* (a long-run average), not the weather of a single year?

* **Matching** compares *sites*, and a site's climate is a stable, near
  time-invariant characteristic -- the coast-to-inland precipitation gradient,
  the warmth of the lower coastal plain. That spatial signal is exactly what is
  missing when the only matchable covariates are elevation + a near-constant
  histosol %, and it is what breaks the "match on elevation alone" degeneracy.
  A normal (e.g. the 1991-2020 mean) is the right *static* per-pixel covariate
  for that.
* **Year-specific** weather (was 2023 a drought year here?) is a *temporal*
  covariate: it belongs in the outcome / DiD stage as ``treated:precip`` or a
  Castro-style per-year climate term, not in the geographic match. That path is
  the documented TODO in ``matching.attach_covariates`` / ``frame.build_frame``;
  this module deliberately produces the static normals the match needs today.

Interpolation is **inverse-distance weighting** (IDW): station coverage in the
NC coastal plain is sparse and irregular, and IDW is the transparent, dependency
-light choice (a KD-tree nearest-neighbour query + distance weights) that never
extrapolates outside the station range -- appropriate when we only need a smooth
climate surface to *match on*, not a calibrated climatology. Swap in kriging
later if a variogram is warranted; the covariate contract (a GeoTIFF on the grid)
does not change.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ..fire_products_comparison.fire_comparison import ANALYSIS_CRS, build_common_grid
from ..preproc.data_loading import data_path

# GHCN daily uses -9999 as its missing sentinel; guard against it leaking into
# means even after the R export (which usually, but not always, drops it).
GHCN_MISSING = -9999.0

# Where build_and_write_climate_normals writes its rasters. Matches the
# ``("processed", "climate")`` root the covariate specs read from.
CLIMATE_DIR_PARTS = ("processed", "climate")


# ---------------------------------------------------------------------------
# 1. Load station points
# ---------------------------------------------------------------------------
def load_ghcn_stations(
    path,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    station_col: str = "STATION",
    source_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Read exported GHCN records into a GeoDataFrame in the analysis CRS.

    Accepts whatever the R script's ``nc.prcp`` / ``nc.max`` / ``nc.min`` frames
    are exported as -- a tidy CSV/parquet with one row per (station, day) and a
    station longitude/latitude, or an already-spatial file (GeoPackage/GeoJSON/
    shapefile). Everything downstream keys off ``station_col`` and the geometry,
    so column names for the *values* (PRCP/TMAX/...) are chosen later.

    Parameters
    ----------
    path : str | Path
        File to read. ``.csv``/``.txt`` -> pandas + point geometry from
        ``lon_col``/``lat_col``; anything else -> :func:`geopandas.read_file`.
    lon_col, lat_col : str
        Longitude / latitude columns (only used for the tabular path).
    station_col : str
        Per-station identifier, carried through so records can be collapsed to
        one normal per station.
    source_crs : str
        CRS of the incoming lon/lat (GHCN is WGS84 / EPSG:4326).

    Returns
    -------
    GeoDataFrame
        Reprojected to :data:`ANALYSIS_CRS` (EPSG:5070; metres), one row per
        input record, with the value columns preserved.
    """
    path = Path(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
        missing = [c for c in (lon_col, lat_col) if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path.name}: expected lon/lat columns {missing} not found; "
                f"pass lon_col=/lat_col= (have {list(df.columns)})."
            )
        gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
            crs=source_crs,
        )
    else:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(source_crs)

    if station_col not in gdf.columns:
        # Fall back to a per-geometry id so the collapse step still works.
        gdf[station_col] = gdf.geometry.astype(str)
    return gdf.to_crs(ANALYSIS_CRS)


# ---------------------------------------------------------------------------
# 2. Collapse station records -> one normal value per station
# ---------------------------------------------------------------------------
def station_normals(
    records: gpd.GeoDataFrame,
    value_col: str,
    station_col: str = "STATION",
    year_col: Optional[str] = "YEAR",
    baseline: Optional[tuple[int, int]] = None,
    annual_reduce: str = "mean",
) -> gpd.GeoDataFrame:
    """Reduce per-day station records to a single normal per station.

    Two-step reduction, so the normal means what you expect:

    1. **within each year**, reduce the daily values to an annual figure with
       ``annual_reduce`` -- ``"sum"`` for precipitation (annual total rainfall),
       ``"mean"`` for a temperature level, ``"max"`` for an extreme;
    2. **across years** (optionally restricted to ``baseline``), average those
       annual figures into the station's normal.

    If the records are already one row per station (no ``year_col`` or it is
    absent), step 1 is skipped and the value is reduced straight to the mean.

    Parameters
    ----------
    records : GeoDataFrame
        Output of :func:`load_ghcn_stations` (must carry ``value_col``,
        ``station_col``, a geometry, and -- for the annual step -- ``year_col``).
    value_col : str
        The measured column to summarise (e.g. ``"PRCP"``, ``"TMAX"``).
    baseline : (int, int), optional
        Inclusive ``(start_year, end_year)`` climatology window (e.g.
        ``(1991, 2020)``). ``None`` uses every year present.
    annual_reduce : {"sum", "mean", "max", "min"}
        How daily values combine into an annual figure (step 1).

    Returns
    -------
    GeoDataFrame
        One row per station: ``[station_col, value_col, geometry]`` with the
        station's normal in ``value_col`` and one representative point geometry.
    """
    df = records.copy()
    df = df[df[value_col] != GHCN_MISSING]
    df = df.dropna(subset=[value_col])

    has_year = year_col is not None and year_col in df.columns
    if has_year and baseline is not None:
        lo, hi = baseline
        df = df[(df[year_col] >= lo) & (df[year_col] <= hi)]
    if df.empty:
        raise ValueError(
            f"no {value_col!r} records left after filtering "
            f"(missing sentinel / baseline {baseline})."
        )

    reduce_fn = {"sum": "sum", "mean": "mean", "max": "max", "min": "min"}[annual_reduce]
    if has_year:
        annual = (
            df.groupby([station_col, year_col])[value_col].agg(reduce_fn).reset_index()
        )
        normal = annual.groupby(station_col)[value_col].mean()
    else:
        normal = df.groupby(station_col)[value_col].mean()

    # one representative point per station (stations don't move)
    geom = df.groupby(station_col).geometry.first()
    out = gpd.GeoDataFrame(
        {station_col: normal.index, value_col: normal.values},
        geometry=geom.reindex(normal.index).values,
        crs=records.crs,
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Inverse-distance-weighting interpolation to the grid
# ---------------------------------------------------------------------------
def idw_to_grid(
    stations: gpd.GeoDataFrame,
    value_col: str,
    grid: xr.DataArray,
    power: float = 2.0,
    k: int = 8,
    smoothing: float = 0.0,
) -> xr.DataArray:
    """Interpolate point ``value_col`` onto ``grid`` by inverse-distance weighting.

    Each grid-cell centre is estimated as a distance-weighted average of its ``k``
    nearest stations, weight ``1 / (d**power + smoothing)``. A cell that sits
    exactly on a station takes that station's value. IDW never produces a value
    outside the observed station range, which is what we want for a covariate we
    only use to *match* on.

    Parameters
    ----------
    stations : GeoDataFrame
        One row per station with a point geometry and ``value_col`` (e.g. the
        output of :func:`station_normals`). Reprojected to the grid CRS.
    grid : xr.DataArray
        Target grid from :func:`peatfire.build_common_grid`.
    power : float
        IDW exponent; higher = more local (nearby stations dominate).
    k : int
        Number of nearest stations averaged per cell (capped at the station
        count).
    smoothing : float
        Added to squared distance to avoid divide-by-zero and soften the surface.

    Returns
    -------
    xr.DataArray
        ``value_col`` interpolated onto ``grid`` (same ``y``/``x``/CRS), so it can
        be handed straight to :func:`peatfire.to_common_grid` consumers or saved.
    """
    from scipy.spatial import cKDTree

    pts = stations.to_crs(grid.rio.crs)
    vals = pts[value_col].to_numpy(dtype="float64")
    keep = np.isfinite(vals)
    xy = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])[keep]
    vals = vals[keep]
    if len(vals) == 0:
        raise ValueError(f"no finite {value_col!r} station values to interpolate.")

    tree = cKDTree(xy)
    k_eff = int(min(k, len(vals)))

    gx = grid["x"].values
    gy = grid["y"].values
    mesh_x, mesh_y = np.meshgrid(gx, gy)
    query = np.column_stack([mesh_x.ravel(), mesh_y.ravel()])

    dist, idx = tree.query(query, k=k_eff)
    if k_eff == 1:  # cKDTree squeezes the last axis when k == 1
        dist = dist[:, None]
        idx = idx[:, None]

    # Exact hits: distance 0 -> take that station's value directly.
    weights = 1.0 / (dist ** power + smoothing)
    exact = dist == 0.0
    if exact.any():
        weights[exact.any(axis=1)] = 0.0
        weights[exact] = 1.0

    neigh_vals = vals[idx]
    est = np.sum(weights * neigh_vals, axis=1) / np.sum(weights, axis=1)
    est = est.reshape(mesh_x.shape).astype("float32")

    out = xr.DataArray(est, coords={"y": gy, "x": gx}, dims=("y", "x"), name=value_col)
    return out.rio.write_crs(grid.rio.crs)


# ---------------------------------------------------------------------------
# 4. Build + persist climate-normal rasters (so they become covariates)
# ---------------------------------------------------------------------------
# Each entry: output covariate name -> how to build its normal from the station
# records. `records` is the exported GHCN table for that element; `value_col` its
# measured column; `annual_reduce` how daily -> annual (sum for rain, mean for
# temperature level). These names are what covariates.py registers.
DEFAULT_CLIMATE_ELEMENTS: dict[str, dict] = {
    "precip_normal": {"value_col": "PRCP", "annual_reduce": "sum"},
    "tmax_normal": {"value_col": "TMAX", "annual_reduce": "mean"},
    "tmin_normal": {"value_col": "TMIN", "annual_reduce": "mean"},
}


def build_climate_normals(
    records_by_element: Mapping[str, gpd.GeoDataFrame],
    aoi: gpd.GeoDataFrame,
    elements: Optional[Mapping[str, dict]] = None,
    res_m: float = 300.0,
    station_col: str = "STATION",
    year_col: Optional[str] = "YEAR",
    baseline: Optional[tuple[int, int]] = None,
    power: float = 2.0,
    k: int = 8,
) -> dict[str, xr.DataArray]:
    """Build one interpolated climate-normal grid per element (no disk writes).

    Ties steps 2-3 together for several elements at once, all on **one** common
    grid over ``aoi`` so every climate layer aligns cell-for-cell with the others
    (and, at the same ``res_m``, with the fire response).

    Parameters
    ----------
    records_by_element : mapping ``covariate_name -> station records``
        For each output covariate, the loaded GHCN records (from
        :func:`load_ghcn_stations`) carrying that element's value column. The
        same records frame can be reused for several elements if it holds several
        value columns.
    elements : mapping, optional
        ``covariate_name -> {"value_col", "annual_reduce"}`` recipe. Defaults to
        :data:`DEFAULT_CLIMATE_ELEMENTS` (precip total, mean tmax, mean tmin).

    Returns
    -------
    dict ``covariate_name -> DataArray`` on the shared grid.
    """
    elements = elements or DEFAULT_CLIMATE_ELEMENTS
    grid = build_common_grid(aoi, res_m=res_m)

    out: dict[str, xr.DataArray] = {}
    for name, recipe in elements.items():
        records = records_by_element.get(name)
        if records is None:
            continue
        stn = station_normals(
            records,
            value_col=recipe["value_col"],
            station_col=station_col,
            year_col=year_col,
            baseline=baseline,
            annual_reduce=recipe.get("annual_reduce", "mean"),
        )
        da = idw_to_grid(stn, recipe["value_col"], grid, power=power, k=k)
        da.name = name
        out[name] = da
    return out


def write_climate_normals(
    normals: Mapping[str, xr.DataArray],
    out_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Save built normals as ``<name>_nc.tif`` under ``processed/climate/``.

    The filenames match the globs registered in :mod:`peatfire.modeling.covariates`
    (``precip_normal_nc.tif`` etc.), so once written they attach to the frame
    automatically like elevation or histosol %.

    Returns a mapping ``name -> written path``.
    """
    out_dir = Path(out_dir) if out_dir is not None else data_path(*CLIMATE_DIR_PARTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, da in normals.items():
        path = out_dir / f"{name}_nc.tif"
        da.rio.to_raster(path)
        written[name] = path
    return written

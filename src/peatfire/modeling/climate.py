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
from ..preproc.data_loading import data_path, read_vector_resilient

# Common longitude / latitude column spellings to auto-detect in an export.
_LON_CANDIDATES = ("longitude", "lon", "long", "x", "LONGITUDE", "LON")
_LAT_CANDIDATES = ("latitude", "lat", "y", "LATITUDE", "LAT")
# Common geometry column names in an `sf` export (used as a fallback when there
# are no plain lon/lat columns -- see _extract_point_coords).
_GEOM_CANDIDATES = ("geometry", "geom", "the_geom", "SHAPE", "geometry.x")

# GHCN daily uses -9999 as its missing sentinel; guard against it leaking into
# means even after the R export (which usually, but not always, drops it).
GHCN_MISSING = -9999.0

# Where build_and_write_climate_normals writes its rasters. Matches the
# ``("processed", "climate")`` root the covariate specs read from.
CLIMATE_DIR_PARTS = ("processed", "climate")

# Where build_annual_climate writes its per-year rasters. A subdirectory of the
# climate root so the year-specific weather (precip_2020_nc.tif, ...) never
# collides with the normals (precip_normal_nc.tif) and matches the
# ``("processed", "climate", "annual")`` root the temporal covariate specs read.
ANNUAL_CLIMATE_DIR_PARTS = ("processed", "climate", "annual")


# ---------------------------------------------------------------------------
# 1. Load station points
# ---------------------------------------------------------------------------
def _read_r_serialisation(path: Path) -> pd.DataFrame:
    """Read an ``.Rds``/``.RData`` GHCN export into a DataFrame.

    Two readers, tried in order, because the project's exports break the fast one:

    * ``pyreadr`` (a C ``librdata`` wrapper) is quick and returns clean dtypes,
      but ``librdata`` only understands *plain* data frames of atomic columns. The
      ``nc_*_long.Rds`` files are ``dplyr`` joins against ``nc.climate$spatial``
      (an ``sf`` object), so they carry a **geometry list-column** -- and modern R
      also stores integer/real sequences as compact ALTREP vectors. ``librdata``
      supports neither, and raises ``LibrdataError: Invalid file, or file has
      unsupported features`` (or a ``PyreadrError``) on exactly these files.
    * ``rdata`` is a pure-Python reader that *does* handle ALTREP vectors, nested
      lists, and list-columns. Unknown R classes (an ``sf`` ``sfc`` geometry
      column, a ``Date``, ...) come back as their underlying values with a warning
      rather than aborting the read, so the atomic lon/lat/value columns the rest
      of this module needs survive intact.

    So we let ``pyreadr`` handle the simple case and fall back to ``rdata`` for the
    ``sf``/ALTREP files, only requiring ``rdata`` be installed when it is actually
    needed.
    """
    try:
        import pyreadr  # optional dep; the fast path for plain data frames
    except ImportError:
        pyreadr = None

    if pyreadr is not None:
        try:
            result = pyreadr.read_r(str(path))
            # An .Rds holds one unnamed object (key None); .RData may hold several.
            key = None if None in result else next(iter(result))
            return pd.DataFrame(result[key])
        except Exception as exc:  # noqa: BLE001 -- librdata surfaces several types
            # librdata can't parse this one (geometry list-column / ALTREP). Fall
            # through to the pure-Python reader rather than failing outright.
            pyreadr_error = exc
    else:
        pyreadr_error = None

    try:
        import rdata  # pure-Python R reader; handles what librdata cannot
    except ImportError as exc:
        hint = (
            f"{path.name}: could not be read by pyreadr/librdata "
            f"({pyreadr_error}). " if pyreadr_error is not None
            else f"{path.name}: reading R serialisations needs a reader. "
        )
        raise ImportError(
            hint + "This export uses features librdata does not support (an sf "
            "geometry list-column and/or ALTREP vectors); install the pure-Python "
            "fallback with `pip install rdata` to read it."
        ) from exc

    # Unknown R classes (sf `sfc`, Date, ...) warn and return raw values; that is
    # fine here -- we only need the atomic columns -- so keep the warnings quiet.
    import warnings

    suffix = path.suffix.lower()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if suffix == ".rds":
            obj = rdata.read_rds(str(path))
        else:  # .rdata / .rda hold a name -> object mapping
            objects = rdata.read_rda(str(path))
            # Mirror pyreadr's "first object" behaviour, preferring a real frame.
            obj = next(
                (v for v in objects.values() if isinstance(v, pd.DataFrame)),
                next(iter(objects.values())),
            )
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)


def _read_records(path: Path) -> pd.DataFrame:
    """Read a GHCN export into a plain DataFrame, format inferred from suffix.

    Supports the R serialisations the project actually produces (``.Rds`` via
    ``pyreadr`` with an ``rdata`` fallback) alongside the portable tabular formats,
    so the ``nc_prcp_long`` / ``nc_tmax_long`` / ``nc_tmin_long`` ``.Rds`` files
    can be read without a round-trip through R.
    """
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".rds", ".rdata", ".rda"}:
        return _read_r_serialisation(path)
    raise ValueError(
        f"{path.name}: unsupported climate export {suffix!r}. Use .Rds "
        "(pyreadr/rdata), .csv, .parquet, or a spatial file (.gpkg/.geojson/.shp)."
    )


def _find_col(df: pd.DataFrame, explicit: str, candidates: tuple) -> Optional[str]:
    """Return ``explicit`` if present, else the first case-insensitive candidate."""
    if explicit in df.columns:
        return explicit
    lower = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def _extract_point_coords(col: pd.Series) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Recover ``(lon, lat)`` arrays from an ``sf``/``sfc`` geometry column.

    The project's ``nc_*_long.Rds`` files are ``sf`` objects, so an export that
    kept no separate lon/lat columns still carries the station location in its
    geometry list-column. The pure-Python ``rdata`` reader does not know the
    ``sfg`` class, so each ``POINT`` comes back as its underlying numeric pair
    (an ``[lon, lat]`` array) rather than a shapely object -- exactly the values
    we need. This also handles the friendlier shapes a different export might
    produce: already-parsed shapely geometries or ``"POINT (lon lat)"`` WKT
    strings.

    Returns ``None`` when the column is empty or holds nothing coordinate-like,
    so the caller can fall back to its "no location found" error.
    """
    from shapely.geometry.base import BaseGeometry

    values = col.to_numpy()
    sample = next(
        (v for v in values if v is not None and not (np.isscalar(v) and pd.isna(v))),
        None,
    )
    if sample is None:
        return None

    # 1) already-parsed shapely geometries (e.g. a real GeometryDtype column).
    if isinstance(sample, BaseGeometry):
        gs = gpd.GeoSeries(values)
        return gs.x.to_numpy(dtype="float64"), gs.y.to_numpy(dtype="float64")

    # 2) WKT text, e.g. "POINT (-76.5 35.1)".
    if isinstance(sample, str):
        from shapely import wkt

        try:
            gs = gpd.GeoSeries([wkt.loads(v) for v in values])
        except Exception:  # noqa: BLE001 -- not WKT; give the caller its error
            return None
        return gs.x.to_numpy(dtype="float64"), gs.y.to_numpy(dtype="float64")

    # 3) raw coordinate pairs from an `sfc` list-column: each element is an
    #    [lon, lat] array/list/tuple (sf stores POINTs as c(X, Y) = c(lon, lat)).
    try:
        arr = np.array([np.asarray(v, dtype="float64").ravel() for v in values])
    except (TypeError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, 0], arr[:, 1]


def load_ghcn_stations(
    path,
    lon_col: str = "longitude",
    lat_col: str = "latitude",
    station_col: str = "STATION",
    source_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Read exported GHCN records into a GeoDataFrame in the analysis CRS.

    Accepts whatever the R script's ``nc.prcp`` / ``nc.max`` / ``nc.min`` frames
    are saved as: the project's ``.Rds`` (read via ``pyreadr``), a tidy CSV /
    parquet with one row per (station, day) and a station longitude/latitude, or
    an already-spatial file (GeoPackage/GeoJSON/shapefile). Everything downstream
    keys off ``station_col`` and the geometry, so the *value* columns
    (PRCP/TMAX/...) are chosen later.

    Parameters
    ----------
    path : str | Path
        File to read. ``.Rds``/``.csv``/``.parquet`` -> table + point geometry
        from ``lon_col``/``lat_col`` (spellings auto-detected); a spatial file ->
        read directly (resiliently, to survive read-only GeoPackages).
    lon_col, lat_col : str
        Longitude / latitude columns for the tabular path. If the exact names are
        absent, common spellings (``LONGITUDE``/``lon``/... , ``LATITUDE``/...)
        are tried; failing that, a geometry list-column (an ``sf``/``sfc`` export,
        as the project's ``nc_*_long.Rds`` files are) is used to recover lon/lat
        before erroring.
    station_col : str
        Per-station identifier, carried through so records collapse to one normal
        per station.
    source_crs : str
        CRS of the incoming lon/lat (GHCN is WGS84 / EPSG:4326).

    Returns
    -------
    GeoDataFrame
        Reprojected to :data:`ANALYSIS_CRS` (EPSG:5070; metres), one row per
        input record, with the value columns preserved.
    """
    path = Path(path)
    spatial_suffixes = {".gpkg", ".geojson", ".json", ".shp"}

    if path.suffix.lower() in spatial_suffixes:
        gdf = read_vector_resilient(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(source_crs)
    else:
        df = _read_records(path)
        # R exports label columns with numpy str scalars (np.str_); normalise to
        # plain str so column names print cleanly and match reliably downstream.
        df.columns = [str(c) for c in df.columns]
        lon = _find_col(df, lon_col, _LON_CANDIDATES)
        lat = _find_col(df, lat_col, _LAT_CANDIDATES)
        if lon is not None and lat is not None:
            geometry = gpd.points_from_xy(df[lon].to_numpy(), df[lat].to_numpy())
        else:
            # No plain lon/lat: fall back to the geometry list-column an `sf`
            # export carries (the project's nc_*_long.Rds are sf objects).
            geom_col = _find_col(df, "geometry", _GEOM_CANDIDATES)
            coords = (
                _extract_point_coords(df[geom_col]) if geom_col is not None else None
            )
            if coords is None:
                raise ValueError(
                    f"{path.name}: could not find longitude/latitude columns "
                    f"(looked for {lon_col!r}/{lat_col!r} and common spellings) "
                    f"or a usable geometry column. Columns present: "
                    f"{list(df.columns)}. If the export is an `sf` object without "
                    "plain lon/lat, add LONGITUDE/LATITUDE columns in R before "
                    "saving (e.g. cbind(sf::st_coordinates(...)))."
                )
            geometry = gpd.points_from_xy(*coords)
        # Drop any raw geometry list-column so it can't collide with the active
        # geometry set below (the tabular readers never yield a real GeometryDtype).
        geom_names = {c.lower() for c in _GEOM_CANDIDATES}
        stale = [c for c in df.columns if str(c).lower() in geom_names]
        if stale:
            df = df.drop(columns=stale)
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=source_crs)

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


# Growing degree days (base 5 C) normal, built with the same station -> annual
# value -> IDW path as the temperature normals, but from Cat's monthly climate
# export (clim_monthly.gpkg, get_climate&soil_data_updated.R) rather than the daily
# GHCN frames. That file already carries a per-station-year `totalGDD` (repeated
# across the year's 12 month rows), so the within-year reduce is a mean -- a no-op
# over the duplicates that yields the year's GDD -- and the across-years mean gives
# the growing-season-warmth normal. clim_monthly.gpkg has no STATION column, so
# load_ghcn_stations falls back to a per-point id (its lowercase year column means
# passing year_col="year"). A static site characteristic, so it registers in
# covariates.COVARIATES (gdd_normal), not the per-year TEMPORAL_COVARIATES.
DEFAULT_GDD_ELEMENTS: dict[str, dict] = {
    "gdd_normal": {"value_col": "totalGDD", "annual_reduce": "mean"},
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


# Per-year (temporal) counterparts of DEFAULT_CLIMATE_ELEMENTS. Same value
# columns and daily->annual reductions (precip total, mean tmax/tmin), but named
# for the temporal covariates they build (precip / tmax / tmin, registered in
# covariates.TEMPORAL_COVARIATES) rather than the *_normal static covariates.
DEFAULT_ANNUAL_ELEMENTS: dict[str, dict] = {
    "precip": {"value_col": "PRCP", "annual_reduce": "sum"},
    "tmax": {"value_col": "TMAX", "annual_reduce": "mean"},
    "tmin": {"value_col": "TMIN", "annual_reduce": "mean"},
}


# scPDSI drought index, built with the same station -> annual-value -> IDW path as
# the other annual elements, but from its own long table (nc_pdsi_long.Rds, one row
# per station-MONTH) rather than the daily GHCN records. scPDSI is already a monthly
# index, so a year's value is the MEAN of that year's monthly scPDSI (the annual
# drought level; ~0 = normal, negative = drought). It is intentionally *only* a
# per-year (temporal) covariate: scPDSI is self-calibrating, so a long-run normal is
# ~0 everywhere and carries no spatial signal to match on -- hence no `*_normal`
# counterpart in DEFAULT_CLIMATE_ELEMENTS. Pass this to build_annual_climate as
# `elements=` alongside a {"pdsi": pdsi_records} mapping.
DEFAULT_PDSI_ELEMENTS: dict[str, dict] = {
    "pdsi": {"value_col": "scPDSI", "annual_reduce": "mean"},
}


def build_annual_climate(
    records_by_element: Mapping[str, gpd.GeoDataFrame],
    aoi: gpd.GeoDataFrame,
    years: Iterable[int],
    elements: Optional[Mapping[str, dict]] = None,
    res_m: float = 300.0,
    station_col: str = "STATION",
    year_col: Optional[str] = "YEAR",
    power: float = 2.0,
    k: int = 8,
) -> dict[str, dict[int, xr.DataArray]]:
    """Build one interpolated climate grid per element **per calendar year**.

    The temporal counterpart of :func:`build_climate_normals`. It reuses exactly
    the same station -> annual-value -> IDW machinery, but instead of averaging the
    annual values across years into one *normal*, it interpolates each single
    year's annual value onto the grid -- the year-specific weather the outcome /
    DiD stage needs (e.g. a ``treated:precip`` dry-year interaction), as opposed to
    the stable normals the geographic match keys off.

    Each year is produced by reducing that station's daily records within the year
    (``annual_reduce`` -- sum for precip, mean for a temperature level), i.e.
    :func:`station_normals` restricted to a single-year ``baseline=(year, year)``,
    then IDW-interpolated onto one common grid over ``aoi`` (shared across years so
    every layer aligns cell-for-cell with the normals and the fire response).

    Parameters
    ----------
    records_by_element : mapping ``element_name -> station records``
        For each element (``"precip"``/``"tmax"``/``"tmin"`` by default), the
        loaded GHCN records (from :func:`load_ghcn_stations`) carrying that
        element's value column and a ``year_col``. The same records used for the
        normals can be reused here.
    years : iterable of int
        Calendar years to build a grid for (e.g. ``range(2019, 2025)``). A year
        with no station records for an element is skipped (with a warning).
    elements : mapping, optional
        ``element_name -> {"value_col", "annual_reduce"}`` recipe. Defaults to
        :data:`DEFAULT_ANNUAL_ELEMENTS`.

    Returns
    -------
    dict ``element_name -> {year -> DataArray}`` on the shared grid.
    """
    import warnings

    elements = elements or DEFAULT_ANNUAL_ELEMENTS
    years = list(years)
    grid = build_common_grid(aoi, res_m=res_m)

    out: dict[str, dict[int, xr.DataArray]] = {}
    for name, recipe in elements.items():
        records = records_by_element.get(name)
        if records is None:
            continue
        per_year: dict[int, xr.DataArray] = {}
        for year in years:
            try:
                stn = station_normals(
                    records,
                    value_col=recipe["value_col"],
                    station_col=station_col,
                    year_col=year_col,
                    baseline=(year, year),  # single-year window -> that year's value
                    annual_reduce=recipe.get("annual_reduce", "mean"),
                )
            except ValueError:
                # No records for this element/year after filtering -> skip the year.
                warnings.warn(
                    f"{name}: no records for {year} -- skipping that year.",
                    stacklevel=2,
                )
                continue
            da = idw_to_grid(stn, recipe["value_col"], grid, power=power, k=k)
            da.name = f"{name}_{year}"
            per_year[year] = da
        out[name] = per_year
    return out


def write_annual_climate(
    annual: Mapping[str, Mapping[int, xr.DataArray]],
    out_dir: Optional[Path] = None,
) -> dict[tuple[str, int], Path]:
    """Save per-year grids as ``<element>_<year>_nc.tif`` under ``processed/climate/annual/``.

    The filenames match the ``{year}`` globs registered in
    :data:`peatfire.modeling.covariates.TEMPORAL_COVARIATES`
    (``precip_{year}_nc.tif`` etc.), so once written they attach as temporal
    covariates automatically -- the temporal analogue of
    :func:`write_climate_normals`.

    Returns a mapping ``(element_name, year) -> written path``.
    """
    out_dir = (
        Path(out_dir) if out_dir is not None else data_path(*ANNUAL_CLIMATE_DIR_PARTS)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[tuple[str, int], Path] = {}
    for name, per_year in annual.items():
        for year, da in per_year.items():
            path = out_dir / f"{name}_{year}_nc.tif"
            da.rio.to_raster(path)
            written[(name, int(year))] = path
    return written


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

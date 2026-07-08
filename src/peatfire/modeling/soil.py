"""Turn the SSURGO soil **polygons** into gridded soil covariates.

The project's soil layer (``src/get_climate&soil_data.R`` -> gSSURGO) is a
**vector GeoPackage** of soil map-unit polygons carrying attribute columns
(organic matter, available water capacity, drainage class, ...), not a raster.
The matching / frame pipeline samples every covariate as a value *at a grid cell*
(like elevation), so this module **rasterises** the chosen polygon attributes onto
the shared analysis grid and writes one GeoTIFF per attribute under
``processed/soil/ssurgo/``. Those filenames match the specs registered in
:mod:`peatfire.modeling.covariates`, so once written they attach to the frame
automatically -- the same build-then-register pattern
:mod:`peatfire.modeling.climate` uses for the GHCN normals.

Continuous attributes (organic matter, AWC) are burned as their numeric value;
categorical attributes (drainage class) are factorised to integer **codes** and
burned as codes (the covariate loader then majority-aggregates them, and the
printed legend maps codes back to class names).

Because the SSURGO export's exact column names vary with how it was assembled,
:func:`inspect_soil_columns` prints what is available and :func:`build_soil_rasters`
takes an explicit ``attributes`` mapping (with sensible gSSURGO defaults) and
skips -- with a clear message -- any column it cannot find.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from ..fire_products_comparison.fire_comparison import ANALYSIS_CRS, build_common_grid
from ..preproc.data_loading import data_path, read_vector_resilient

# Where build_soil_rasters writes, matching the covariates.py soil specs' globs.
SOIL_DIR_PARTS = ("processed", "soil", "ssurgo")

# output covariate name -> how to build it. `column` is the SSURGO attribute to
# read (override after inspecting your export); `filename` matches the registered
# glob in covariates.py; `role` drives continuous (burn value) vs categorical
# (factorise to codes). Defaults use standard gSSURGO representative-value names.
DEFAULT_SOIL_ATTRIBUTES: dict[str, dict] = {
    "soil_organic_matter": {
        "column": "om_r", "role": "continuous",
        "filename": "ssurgo_organic_matter_nc.tif",
    },
    "soil_awc": {
        "column": "awc_r", "role": "continuous",
        "filename": "ssurgo_awc_nc.tif",
    },
    "soil_drainage_class": {
        "column": "drainagecl", "role": "categorical",
        "filename": "ssurgo_drainage_class_nc.tif",
    },
}


def inspect_soil_columns(gpkg_path=None) -> pd.DataFrame:
    """Print + return the soil GeoPackage's columns (dtype, #unique, sample).

    Run this first: SSURGO exports differ, so use the output to fill in the
    ``attributes`` mapping for :func:`build_soil_rasters` (which numeric column is
    organic matter / AWC, which column is the drainage class).
    """
    if gpkg_path is None:
        gpkg_path = data_path("interim", "soil", "ssurgo", "nc_soil_ssurgo.gpkg")
    gdf = read_vector_resilient(gpkg_path)
    rows = []
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        s = gdf[col]
        sample = s.dropna().unique()[:4]
        rows.append(
            {"column": col, "dtype": str(s.dtype), "n_unique": int(s.nunique()),
             "n_null": int(s.isna().sum()), "sample": list(sample)}
        )
    table = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 60, "display.width", 160):
        print(f"{Path(gpkg_path).name}: {len(gdf):,} polygons, {len(table)} attribute columns")
        print(table.to_string(index=False))
    return table


def _rasterize_attribute(
    gdf: gpd.GeoDataFrame, column: str, role: str, grid: xr.DataArray
):
    """Burn one polygon attribute onto ``grid``; returns (DataArray, legend).

    Continuous -> float values; categorical -> integer codes (legend maps
    ``code -> class``). Cells no polygon covers are NaN.
    """
    from rasterio.features import rasterize

    legend: Optional[dict] = None
    if role == "categorical":
        codes, uniques = pd.factorize(gdf[column], sort=True)
        values = codes.astype("float64")
        values[codes < 0] = np.nan  # factorize marks NaN as -1
        legend = {i: str(u) for i, u in enumerate(uniques)}
    else:
        values = pd.to_numeric(gdf[column], errors="coerce").to_numpy(dtype="float64")

    shapes = [
        (geom, val)
        for geom, val in zip(gdf.geometry, values)
        if geom is not None and not geom.is_empty and np.isfinite(val)
    ]
    out_shape = (grid.sizes["y"], grid.sizes["x"])
    if not shapes:
        arr = np.full(out_shape, np.nan, dtype="float32")
    else:
        arr = rasterize(
            shapes,
            out_shape=out_shape,
            transform=grid.rio.transform(),
            fill=np.nan,
            all_touched=False,  # cell centre in polygon -> avoids edge inflation
            dtype="float32",
        )
    da = xr.DataArray(
        arr, coords={"y": grid["y"].values, "x": grid["x"].values}, dims=("y", "x")
    )
    return da.rio.write_crs(grid.rio.crs), legend


def build_soil_rasters(
    gpkg_path=None,
    aoi: gpd.GeoDataFrame = None,
    attributes: Optional[Mapping[str, dict]] = None,
    res_m: float = 300.0,
    out_dir: Optional[Path] = None,
) -> dict[str, Path]:
    """Rasterise SSURGO polygon attributes to covariate GeoTIFFs.

    Reads the soil GeoPackage (resiliently -- survives read-only-database
    errors), reprojects to the analysis CRS, and for each requested attribute
    burns it onto one shared grid over ``aoi`` at ``res_m``, writing
    ``processed/soil/ssurgo/<filename>`` so it registers as the matching soil
    covariate. A requested column that is not in the export is skipped with a
    message (run :func:`inspect_soil_columns` to get the real names).

    Parameters
    ----------
    gpkg_path : str | Path, optional
        The SSURGO GeoPackage (default ``interim/soil/ssurgo/nc_soil_ssurgo.gpkg``).
    aoi : GeoDataFrame
        Area defining the output grid extent (e.g. the 80% peat frame).
    attributes : mapping, optional
        ``covariate_name -> {"column", "role", "filename"}``. Defaults to
        :data:`DEFAULT_SOIL_ATTRIBUTES` (organic matter, AWC, drainage class).
    res_m : float
        Output raster resolution in metres (match the modeling grid, ~300 m).

    Returns
    -------
    dict ``covariate_name -> written path`` for the attributes that were built.
    """
    if gpkg_path is None:
        gpkg_path = data_path("interim", "soil", "ssurgo", "nc_soil_ssurgo.gpkg")
    if aoi is None:
        raise ValueError("build_soil_rasters needs an `aoi` to define the grid.")
    attributes = attributes or DEFAULT_SOIL_ATTRIBUTES
    out_dir = Path(out_dir) if out_dir is not None else data_path(*SOIL_DIR_PARTS)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = read_vector_resilient(gpkg_path).to_crs(ANALYSIS_CRS)
    grid = build_common_grid(aoi, res_m=res_m)

    written: dict[str, Path] = {}
    for name, spec in attributes.items():
        column = spec["column"]
        if column not in gdf.columns:
            print(
                f"[soil] {name}: column {column!r} not in the GeoPackage -- skipping. "
                f"Run inspect_soil_columns() and pass the right column."
            )
            continue
        da, legend = _rasterize_attribute(gdf, column, spec.get("role", "continuous"), grid)
        path = out_dir / spec["filename"]
        da.rio.to_raster(path)
        written[name] = path
        msg = f"[soil] wrote {name} <- {column!r} -> {path.name}"
        if legend is not None:
            msg += f"; classes {legend}"
        print(msg)
    return written

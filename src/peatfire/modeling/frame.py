"""Build the tidy pixel-year modeling frame.

This is the *assembly layer*. Given a set of analysis **units** (treated
restoration polygons + matched control polygons -- built upstream by the matching
step, which is intentionally *not* in this module) and the layers registered in
:mod:`peatfire.modeling.covariates`, it produces one tidy table:

    one row per (unit-pixel x year), columns
    [unit_id, site_id, treated, year, x, y, <covariates...>, burned, ...]

ready to hand to :mod:`peatfire.modeling.models`.

Design choices (see ``decisions.md`` / ``modeling_roadmap.md``):

* The frame is built on the **same EPSG:5070 common grid** the fire-product
  comparison uses (:func:`peatfire.build_common_grid`), so no new CRS/area
  decisions and every layer aligns cell-for-cell.
* The **response is swappable**: it comes from ``load_standardized(product,
  year, aoi)``, so switching ``product`` from ``"FireCCIS311"`` to a severity
  product changes only the ``y`` column, not this builder.
* **Matching lives upstream.** ``build_frame`` takes an already-assembled
  ``units`` GeoDataFrame (with ``treated`` and ``site_id`` columns); it does not
  decide which pixels are controls. That keeps the causal design (who is a
  control, matched on what) separate and explicit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import xarray as xr

from ..preproc.data_loading import data_path
from ..fire_products_comparison.fire_comparison import (
    ANALYSIS_CRS,
    build_common_grid,
    to_common_grid,
)
from ..fire_products_comparison.fire_products import load_standardized
from .covariates import available_covariates, covariate_on_grid

# FireCCIS311 is ~300 m; default the modeling grid to its native resolution so
# the response is not needlessly upsampled. Override per call as needed.
DEFAULT_RES_M = 300.0


# ---------------------------------------------------------------------------
# Treatment layer (restoration sites)
# ---------------------------------------------------------------------------
def load_restoration_sites(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load peat-restoration polygons as a gdf
    
    Parameters
    ----------
    path : Path
        Path to the restoration sites shapefile. Has a default location.
        
    Returns
    -------
    gdf : gpd.GeoDataFrame
        The loaded restoration sites as a gdf
    """
    if path is None:
        path = data_path(
            "processed",
            "peat_restoration",
            "NC_Pocosin_Restoration_Sites_2026",
            "NC_Pocosin_Restoration_Sites_2026.shp",
        )
        
    gdf = gpd.read_file(path)
    
    return gdf

def load_restoration_sites_in_analysis_crs(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load peat-restoration polygons as a gdf in ANALYSIS_CRS
    
    Parameters
    ----------
    path : Path
        Path to the restoration sites shapefile. Has a default location.
        
    Returns
    -------
    gdf : gpd.GeoDataFrame
        The loaded restoration sites as a gdf in ANALYSIS_CRS
    """
    gdf = load_restoration_sites(path)
    gdf = gdf.to_crs(ANALYSIS_CRS)
    
    return gdf

def load_completed_restoration_sites_in_analysis_crs(path: Optional[Path] = None, restoration_yr_col: str = 'End_Yr') -> gpd.GeoDataFrame:
    """Load the completed peatland-restoration polygons, reprojected to the analysis CRS and dropping rows with 0's in the restoration_yr_col.
    
    Parameters
    ----------
    path : Path
        Path to the restoration sites shapefile. Has a default location.
    restoration_yr_col : str
        The column to treat as the restoration year (i.e., 'Start_Yr' or 'End_Yr'). If a row is 0 in this column, it will be dropped.
    
    Returns
    -------
    gdf : gpd.GeoDataFrame
        The gdf with rows marked as 0 in the restoration_yr_col dropped and reprojected to the ANALYSIS_CRS
    """
    gdf = load_restoration_sites_in_analysis_crs(path)

    # set the restoration year as the 'End_Yr' rather than the 'Start_Yr' for now, and replace placeholder 0's with nan
    # some sites have 'Status_202' = 'Completed' but no 'End_Yr' yet. Ask Eric about these, but leave them out for now.
    gdf[restoration_yr_col] = gdf[restoration_yr_col].replace(0, np.nan)
    gdf = gdf.dropna(subset=restoration_yr_col)
    return gdf

# ---------------------------------------------------------------------------
# Grid <-> unit labelling
# ---------------------------------------------------------------------------
def _rasterize_values(
    gdf: gpd.GeoDataFrame, values: Sequence, grid: xr.DataArray, fill=np.nan
) -> np.ndarray:
    """Burn ``values`` (one per feature) onto ``grid``'s cells (all_touched).

    ``all_touched=True`` matches the toolkit's "any sub-cell lights the cell"
    convention so small units survive the coarse grid. Later features win on
    overlap -- callers should pass non-overlapping units (treated + matched
    controls are disjoint by construction).
    """
    shapes = zip(gdf.to_crs(grid.rio.crs).geometry, values)
    return rasterio.features.rasterize(
        shapes,
        out_shape=(grid.rio.height, grid.rio.width),
        transform=grid.rio.transform(),
        fill=fill,
        all_touched=True,
        dtype="float64",
    )


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------
def build_frame(
    units: gpd.GeoDataFrame,
    product: str = "FireCCIS311",
    years: Iterable[int] = range(2019, 2025),
    covariate_names: Optional[Sequence[str]] = None,
    res_m: float = DEFAULT_RES_M,
    unit_id_col: str = "unit_id",
    site_id_col: str = "Proj_Name",
    treated_col: str = "treated",
) -> pd.DataFrame:
    """Assemble the tidy pixel-year modeling frame.

    Parameters
    ----------
    units : GeoDataFrame
        Treated restoration polygons + matched control polygons, already chosen
        upstream. Must carry ``treated`` (1/0), a ``site_id`` (matching stratum),
        and a per-row ``unit_id``. Reprojected internally to the analysis CRS.
    product : str
        Registered fire product supplying the response (default FireCCIS311).
        Swap for a severity product to model severity instead -- nothing else
        changes.
    years : iterable of int
        Years to stack (FireCCIS311 covers 2019-2024).
    covariate_names : sequence of str, optional
        Which registered covariates to attach. Defaults to every covariate whose
        file is currently on disk (:func:`available_covariates`), so this runs on
        elevation + histosol % today and picks up the rest as they download.
    res_m : float
        Grid resolution in metres (default = FireCCIS311 native ~300 m).

    Returns
    -------
    DataFrame
        One row per unit-cell-year, static covariates repeated across years, with
        a boolean/float ``burned`` response. Cells outside every unit are dropped.
        (Temporal covariates -- climate -- are a documented TODO below.)
    """
    units = units.to_crs(ANALYSIS_CRS).reset_index(drop=True)
    if unit_id_col not in units:
        units = units.assign(**{unit_id_col: np.arange(len(units))})
    for col in (site_id_col, treated_col):
        if col not in units:
            raise ValueError(
                f"units is missing required column {col!r}; the matching step must "
                f"supply {treated_col!r}, {site_id_col!r} (and ideally {unit_id_col!r})."
            )

    grid = build_common_grid(units, res_m=res_m)

    # --- which grid cell belongs to which unit (and its treated/site labels) ---
    unit_ids = _rasterize_values(units, units[unit_id_col], grid)
    treated = _rasterize_values(units, units[treated_col], grid)
    site_ids = _rasterize_values(units, units[site_id_col].astype("category").cat.codes, grid)
    in_unit = ~np.isnan(unit_ids)  # cells covered by some analysis unit

    ys, xs = xr.broadcast(grid["y"], grid["x"])
    base = {
        "unit_id": unit_ids[in_unit],
        "site_id": site_ids[in_unit],
        "treated": treated[in_unit],
        "y": ys.values[in_unit],
        "x": xs.values[in_unit],
    }

    # --- static covariates (repeated across years) ---
    if covariate_names is None:
        covariate_names = available_covariates()
    for name in covariate_names:
        cov = covariate_on_grid(name, grid, units)
        if cov is None:
            continue  # not downloaded yet -> covariates.py already warned
        base[name] = cov.values[in_unit]

    static = pd.DataFrame(base)

    # --- response, per year (the swappable DV) ---
    rows = []
    for year in years:
        resp = load_standardized(product, year, units)
        if resp is None:
            continue  # product absent for this year -> skip
        burned = to_common_grid(resp.astype("float32"), grid, how="max")
        year_df = static.copy()
        year_df["year"] = year
        year_df["burned"] = burned.values[in_unit]
        # TODO(temporal covariates): join per-year climate (PRISM/Daymet) here,
        # keyed on (x, y, year), once those layers are downloaded.
        rows.append(year_df)

    if not rows:
        raise ValueError(f"No {product} data found for years {list(years)}.")
    frame = pd.concat(rows, ignore_index=True)
    # burned is NaN where the product had no coverage; drop those cell-years.
    return frame.dropna(subset=["burned"]).reset_index(drop=True)

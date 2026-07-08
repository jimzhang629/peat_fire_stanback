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

**SSURGO is relational, not flat.** The polygon layer carries only a map-unit key
(``MUKEY``) plus a few identifiers -- the soil *properties* live in separate,
attribute-only tables in the same GeoPackage: drainage class in ``component``
(one map unit -> many components), organic matter / AWC in ``chorizon`` (one
component -> many depth horizons). So a property has to be **aggregated down to
one representative value per map unit** before it can be joined to the polygons
and rasterised. :func:`build_soil_rasters` does this automatically: it looks a
requested column up across the relational tables and reduces the one-to-many rows
(component-percent-weighted over components, horizon-thickness-weighted over
horizons; "dominant condition" for categoricals) to a per-``MUKEY`` value.

Because the SSURGO export's exact column names vary with how it was assembled,
:func:`inspect_soil_columns` prints what is available (across the relational
tables, not just the polygons) and :func:`build_soil_rasters` takes an explicit
``attributes`` mapping (with sensible gSSURGO defaults) and skips -- with a clear
message -- any column it cannot find.
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

# SSURGO relational schema. The polygon layer spells the key MUKEY (uppercase);
# the attribute tables spell it mukey. These are the join keys / weights used to
# collapse the one-map-unit -> many-components -> many-horizons rows to one value.
_MUKEY = "mukey"                             # map-unit key (polygons: MUKEY)
_COKEY = "cokey"                             # component key (links component<->chorizon)
_COMPPCT = "comppct_r"                       # component % of the map unit (weight)
_HZ_TOP, _HZ_BOT = "hzdept_r", "hzdepb_r"    # horizon top / bottom depth, cm (weight)
# Tables searched (in order) for a requested attribute column, and how each joins
# back to a map unit. "component"/"chorizon" need aggregating; the rest are 1:mukey.
_ATTR_TABLES = ("component", "chorizon", "muaggatt", "mapunit")

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


def _col_ci(df, name: str) -> Optional[str]:
    """Return ``df``'s column matching ``name`` case-insensitively, or ``None``."""
    if name in df.columns:
        return name
    lower = {str(c).lower(): c for c in df.columns}
    return lower.get(name.lower())


def _list_layers(gpkg_path) -> list[str]:
    """Names of every layer/table in the GeoPackage (spatial and attribute-only)."""
    try:
        import pyogrio

        return [str(n) for n in pyogrio.list_layers(gpkg_path)[:, 0]]
    except Exception:  # noqa: BLE001 -- fall back to fiona if pyogrio is unavailable
        import fiona

        return list(fiona.listlayers(str(gpkg_path)))


def _spatial_layer(gpkg_path) -> Optional[str]:
    """Name of the polygon layer, so we can read it without the multi-layer warning."""
    try:
        import pyogrio

        info = pyogrio.list_layers(gpkg_path)  # [[name, geometry_type], ...]
        for name, geom_type in info:
            if geom_type:  # non-empty geometry type == the spatial layer
                return str(name)
    except Exception:  # noqa: BLE001
        pass
    return None


def _read_gpkg_table(gpkg_path, layer: str) -> pd.DataFrame:
    """Read one GeoPackage attribute table (no geometry), resilient to read-only DBs.

    Mirrors :func:`read_vector_resilient`'s writable-copy workaround, but reads a
    *named* layer with the geometry dropped (these SSURGO tables have none), so the
    relational ``component`` / ``chorizon`` tables come back as plain DataFrames.
    """
    import pyogrio

    def _read(p) -> pd.DataFrame:
        return pd.DataFrame(pyogrio.read_dataframe(p, layer=layer, read_geometry=False))

    try:
        return _read(gpkg_path)
    except Exception:  # noqa: BLE001 -- readonly-db: retry from a writable copy
        import os
        import shutil
        import tempfile

        src = Path(gpkg_path)
        tmp = Path(tempfile.mkdtemp(prefix="peatfire_gpkg_"))
        dst = tmp / src.name
        shutil.copy2(src, dst)
        for side in ("-wal", "-shm"):
            s = src.with_name(src.name + side)
            if s.exists():
                shutil.copy2(s, tmp / s.name)
        os.chmod(dst, 0o644)
        return _read(dst)


def _weighted_mean_by(df: pd.DataFrame, key, value, weight) -> pd.Series:
    """Per-``key`` weighted mean of ``value`` by ``weight`` (mean if all weights 0)."""
    d = df[[key]].copy()
    d["_v"] = pd.to_numeric(df[value], errors="coerce")
    d["_w"] = pd.to_numeric(df[weight], errors="coerce").fillna(0.0).clip(lower=0)
    d = d.dropna(subset=["_v"])
    d["_wv"] = d["_v"] * d["_w"]
    g = d.groupby(key).agg(wv=("_wv", "sum"), w=("_w", "sum"), mv=("_v", "mean"))
    return g["wv"].where(g["w"] > 0, g["mv"]) / g["w"].where(g["w"] > 0, 1.0)


def mukey_attribute(
    gpkg_path,
    column: str,
    role: str = "continuous",
    depth_cm: Optional[float] = None,
) -> tuple[Optional[pd.Series], Optional[str]]:
    """Aggregate an SSURGO attribute to one value per map unit (``MUKEY``).

    Locates ``column`` in the relational tables and reduces the one-map-unit ->
    many-components(-> many-horizons) rows to a single representative value:

    * **horizon-level continuous** (``om_r``, ``awc_r`` in ``chorizon``):
      horizon-thickness-weighted mean over each component's horizons (optionally
      only the top ``depth_cm``), then component-percent-weighted mean over the
      map unit's components;
    * **component-level continuous** (in ``component``): component-percent-weighted
      mean;
    * **categorical** (``drainagecl``): *dominant condition* -- the class holding
      the largest total component percent in the map unit;
    * a column already living in a per-map-unit table (``muaggatt``, ``mapunit``):
      taken directly.

    Returns ``(series indexed by str(mukey), source_layer)``, or ``(None, None)``
    if the column is in none of the searched tables.
    """
    layers = {name.lower() for name in _list_layers(gpkg_path)}

    def _load(name: str) -> Optional[pd.DataFrame]:
        return _read_gpkg_table(gpkg_path, name) if name in layers else None

    comp = _load("component")
    comp_mukey = _col_ci(comp, _MUKEY) if comp is not None else None
    comp_cokey = _col_ci(comp, _COKEY) if comp is not None else None
    comp_pct = _col_ci(comp, _COMPPCT) if comp is not None else None
    if comp is not None and comp_pct is None:
        # No component-percent column: weight every component equally.
        comp = comp.copy()
        comp["_comppct_fallback"] = 1.0
        comp_pct = "_comppct_fallback"

    # 1) component table -----------------------------------------------------
    if comp is not None and comp_mukey is not None:
        comp_col = _col_ci(comp, column)
        if comp_col is not None:
            if role == "categorical":
                d = comp[[comp_mukey, comp_col]].copy()
                d["_pct"] = pd.to_numeric(comp[comp_pct], errors="coerce").fillna(0.0)
                d = d.dropna(subset=[comp_col])
                totals = d.groupby([comp_mukey, comp_col])["_pct"].sum().reset_index()
                dom = totals.loc[totals.groupby(comp_mukey)["_pct"].idxmax()]
                series = dom.set_index(comp_mukey)[comp_col].astype(str)
            else:
                series = _weighted_mean_by(comp, comp_mukey, comp_col, comp_pct)
            series.index = series.index.astype(str)
            return series, "component"

    # 2) chorizon table (needs component to reach a map unit) ----------------
    hz = _load("chorizon")
    if (
        hz is not None
        and comp is not None
        and comp_mukey is not None
        and comp_cokey is not None
    ):
        hz_col = _col_ci(hz, column)
        hz_cokey = _col_ci(hz, _COKEY)
        hz_top, hz_bot = _col_ci(hz, _HZ_TOP), _col_ci(hz, _HZ_BOT)
        if hz_col is not None and hz_cokey is not None:
            d = hz[[hz_cokey]].copy()
            d["_v"] = pd.to_numeric(hz[hz_col], errors="coerce")
            top = pd.to_numeric(hz[hz_top], errors="coerce") if hz_top else 0.0
            bot = pd.to_numeric(hz[hz_bot], errors="coerce") if hz_bot else np.nan
            if depth_cm is not None:  # restrict the weight to the top depth_cm
                top = np.minimum(top, depth_cm)
                bot = np.minimum(bot, depth_cm)
            thickness = (bot - top)
            # No usable depths -> fall back to an unweighted mean (weight 1).
            d["_w"] = pd.Series(thickness, index=hz.index).fillna(1.0).clip(lower=0)
            d = d.dropna(subset=["_v"])
            per_comp = _weighted_mean_by(
                d.rename(columns={hz_cokey: "_k"}), "_k", "_v", "_w"
            )
            # component-percent-weighted mean of the per-component values. Match
            # cokey as strings on both sides so an int/str dtype gap can't break
            # the join between the horizon and component tables.
            per_comp_by_str = {str(k): v for k, v in per_comp.items()}
            c = comp[[comp_mukey]].copy()
            c["_v"] = comp[comp_cokey].astype(str).map(per_comp_by_str)
            c["_pct"] = comp[comp_pct]
            series = _weighted_mean_by(c.dropna(subset=["_v"]), comp_mukey, "_v", "_pct")
            series.index = series.index.astype(str)
            return series, "chorizon"

    # 3) already per-map-unit tables (muaggatt, mapunit) ---------------------
    for name in ("muaggatt", "mapunit"):
        t = _load(name)
        if t is None:
            continue
        col = _col_ci(t, column)
        mk = _col_ci(t, _MUKEY)
        if col is not None and mk is not None:
            s = t.dropna(subset=[col]).drop_duplicates(mk).set_index(mk)[col]
            if role != "categorical":
                s = pd.to_numeric(s, errors="coerce").dropna()
            else:
                s = s.astype(str)
            s.index = s.index.astype(str)
            return s, name

    return None, None


def inspect_soil_columns(gpkg_path=None) -> pd.DataFrame:
    """Print + return the soil GeoPackage's columns (dtype, #unique, sample).

    Run this first: SSURGO exports differ, so use the output to fill in the
    ``attributes`` mapping for :func:`build_soil_rasters` (which numeric column is
    organic matter / AWC, which column is the drainage class).

    The polygon layer only holds identifiers (``MUKEY`` + a few keys); the soil
    *properties* live in the relational tables (``component``, ``chorizon``, ...).
    So besides the polygon columns this also **locates each default attribute**
    across those tables and prints which table/column to pass to
    :func:`build_soil_rasters`.
    """
    if gpkg_path is None:
        gpkg_path = data_path("interim", "soil", "ssurgo", "nc_soil_ssurgo.gpkg")

    # Read the polygon layer by name so the multi-layer read doesn't warn.
    spatial = _spatial_layer(gpkg_path)
    gdf = (
        read_vector_resilient(gpkg_path, layer=spatial)
        if spatial is not None
        else read_vector_resilient(gpkg_path)
    )
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
        print(f"{Path(gpkg_path).name}: {len(gdf):,} polygons, {len(table)} polygon columns")
        print(table.to_string(index=False))

        # Where do the requested soil properties actually live? Search the
        # relational tables so the caller knows a column exists even though it is
        # not on the polygons, and confirm build_soil_rasters will find it.
        layers = _list_layers(gpkg_path)
        print(f"\n{len(layers)} layers total; soil properties are in the attribute tables:")
        for name, spec in DEFAULT_SOIL_ATTRIBUTES.items():
            col = spec["column"]
            _, src = mukey_attribute(gpkg_path, col, spec.get("role", "continuous"))
            where = f"found in {src!r} (aggregated to MUKEY)" if src else "NOT FOUND"
            print(f"  {name:<22} column {col!r:<14} -> {where}")
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
    covariate.

    A requested column need **not** be on the polygons: SSURGO keeps the soil
    properties in relational tables (drainage class in ``component``, organic
    matter / AWC in ``chorizon``), so a column missing from the polygon layer is
    looked up and **aggregated to one value per map unit** via
    :func:`mukey_attribute`, then joined back onto the polygons by ``MUKEY`` before
    rasterising. Only a column found in *no* table is skipped, with a message (run
    :func:`inspect_soil_columns` to see the real names).

    Parameters
    ----------
    gpkg_path : str | Path, optional
        The SSURGO GeoPackage (default ``interim/soil/ssurgo/nc_soil_ssurgo.gpkg``).
    aoi : GeoDataFrame
        Area defining the output grid extent (e.g. the 80% peat frame).
    attributes : mapping, optional
        ``covariate_name -> {"column", "role", "filename"[, "depth_cm"]}``.
        ``depth_cm`` (optional) caps the horizon depth window when aggregating a
        ``chorizon`` property. Defaults to :data:`DEFAULT_SOIL_ATTRIBUTES`
        (organic matter, AWC, drainage class).
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

    # Read the polygon layer by name so the multi-layer read doesn't warn.
    spatial = _spatial_layer(gpkg_path)
    gdf = (
        read_vector_resilient(gpkg_path, layer=spatial)
        if spatial is not None
        else read_vector_resilient(gpkg_path)
    ).to_crs(ANALYSIS_CRS)
    mukey_col = _col_ci(gdf, _MUKEY)  # polygon key for joining aggregated values
    grid = build_common_grid(aoi, res_m=res_m)

    written: dict[str, Path] = {}
    for name, spec in attributes.items():
        column = spec["column"]
        role = spec.get("role", "continuous")
        source = "polygon layer"

        if column in gdf.columns:
            burn_col = column
        else:
            # Not on the polygons -> aggregate it out of the relational tables and
            # join back by MUKEY.
            series, src = mukey_attribute(gpkg_path, column, role, spec.get("depth_cm"))
            if series is None:
                print(
                    f"[soil] {name}: column {column!r} not found in the polygon "
                    f"layer or the relational tables -- skipping. Run "
                    f"inspect_soil_columns() and pass the right column."
                )
                continue
            if mukey_col is None:
                print(
                    f"[soil] {name}: {column!r} lives in table {src!r} but the "
                    f"polygon layer has no MUKEY to join it on -- skipping."
                )
                continue
            burn_col = f"_soil_{name}"
            gdf[burn_col] = gdf[mukey_col].astype(str).map(series)
            matched = int(gdf[burn_col].notna().sum())
            if matched == 0:
                print(
                    f"[soil] {name}: aggregated {column!r} from {src!r} but no "
                    f"MUKEY matched the polygons -- skipping."
                )
                continue
            source = f"{src!r} aggregated to MUKEY ({matched:,}/{len(gdf):,} polygons)"

        da, legend = _rasterize_attribute(gdf, burn_col, role, grid)
        path = out_dir / spec["filename"]
        da.rio.to_raster(path)
        written[name] = path
        msg = f"[soil] wrote {name} <- {column!r} [{source}] -> {path.name}"
        if legend is not None:
            msg += f"; classes {legend}"
        print(msg)
    return written

"""Environmental-covariate registry and standardized loaders.

This is the *data layer* of the modeling pipeline, and it deliberately mirrors
:mod:`peatfire.fire_products_comparison.fire_products`: the messy per-layer
details (where a covariate lives on disk, its band/nodata, whether it is a
*continuous* field that should be averaged onto a coarser grid or a *categorical*
field that should take the majority class) are isolated behind a small registry
of :class:`CovariateSpec` objects.

Downstream (:mod:`peatfire.modeling.frame`) never needs those details: it asks
for ``covariate_on_grid(name, grid, aoi)`` and gets back a DataArray aligned to
the shared analysis grid, aggregated the right way for that layer.

Adding a covariate is one :class:`CovariateSpec` in :data:`COVARIATES`; no loader
code changes. Layers that are not downloaded yet are registered anyway -- their
loaders simply warn and return ``None`` (exactly like the fire registry), so the
frame builder runs on whatever is on disk today (elevation + histosol %) and
picks up land cover / drainage / management as those land.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from ..preproc.data_loading import clip_raster_to_mask, data_path
from ..fire_products_comparison.fire_comparison import to_common_grid


# ---------------------------------------------------------------------------
# Covariate specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CovariateSpec:
    """Everything the generic loaders need to read one covariate raster.

    Parameters
    ----------
    name : str
        Short identifier and output column name, e.g. ``"elevation"``.
    role : str
        ``"continuous"`` (elevation, histosol %, HAND) or ``"categorical"``
        (land cover, management class). Drives the default grid aggregation:
        continuous -> area ``"mean"`` (the roadmap's *area-weighted* covariate),
        categorical -> ``"mode"`` (majority class) so a coarse cell keeps a valid
        class instead of a meaningless fractional average.
    root_parts : tuple of str
        Arguments forwarded to :func:`peatfire.data_path` to locate the layer's
        directory, e.g. ``("processed", "topography", "GLO30_DEM")``.
    glob : str
        Filename glob within that directory (the first match is used).
    band : int, optional
        1-based band to select from a multi-band raster. ``None`` = single band.
    nodata : tuple of float, optional
        Undeclared fill sentinel(s) to null after reading, for files whose nodata
        is not stored in the GeoTIFF header (so ``masked=True`` misses it).
    transform : callable, optional
        ``DataArray -> DataArray`` applied after read (e.g. unit rescaling).
    resample : str, optional
        Override the role-derived grid aggregation (``"mean"``/``"mode"``/
        ``"min"``/``"max"``). Leave ``None`` to use the role default.
    native_res_m : float, optional
        Documentation only; the true pixel size is read from each file.
    source : str, optional
        Human-readable provenance note.
    """

    name: str
    role: str
    root_parts: tuple
    glob: str
    band: Optional[int] = None
    nodata: Optional[tuple] = None
    transform: Optional[Callable[[xr.DataArray], xr.DataArray]] = None
    resample: Optional[str] = None
    native_res_m: Optional[float] = None
    source: Optional[str] = None

    @property
    def directory(self) -> Path:
        return data_path(*self.root_parts)

    @property
    def how(self) -> str:
        """Grid aggregation for this layer (explicit override or role default)."""
        if self.resample is not None:
            return self.resample
        return "mean" if self.role == "continuous" else "mode"

# ---------------------------------------------------------------------------
# Registry
#
# Paths/globs match the processed layout in download_and_clip_data.ipynb. Only
# the three layers currently on disk (elevation, histosol %, and -- as the
# treatment, handled in frame.py -- restoration) are live; the rest are
# registered so they attach automatically once downloaded (their loaders skip
# with a warning until then, mirroring the fire-product registry).
# ---------------------------------------------------------------------------
COVARIATES: dict[str, CovariateSpec] = {
    "elevation": CovariateSpec(
        name="elevation",
        role="continuous",
        # /data/processed/topography/GLO30_DEM/GLO30_DEM.tif
        root_parts=("processed", "topography", "GLO30_DEM"),
        glob="GLO30_DEM_nc.tif",
        native_res_m=30.0,
        source="Copernicus GLO-30 DEM (metres above sea level).",
    ),
    "histosol_pct": CovariateSpec(
        name="histosol_pct",
        role="continuous",
        # /data/processed/peat_extent/conus_80_histosol_major_percentage_nc.tif
        # gSSURGO % of pixel mapped as a major-component histosol (H%, 0-100).
        root_parts=("processed", "peat_extent"),
        glob="conus_80_histosol_major_percentage_nc.tif",
        native_res_m=30.0,
        source="gSSURGO major-histosol percentage (Lilleskov et al. 2025).",
    ),
    # --- climate normals (interpolated from GHCN stations by modeling.climate) --
    # Static per-pixel climatology (a stable *site* characteristic), the spatial
    # signal the match needs beyond elevation + near-constant histosol %. Built and
    # written to processed/climate/ by climate.build_climate_normals +
    # write_climate_normals; year-specific weather stays a temporal (DiD) covariate.
    "precip_normal": CovariateSpec(
        name="precip_normal",
        role="continuous",
        # /data/processed/climate/precip_normal_nc.tif  (mean annual PRCP, mm)
        root_parts=("processed", "climate"),
        glob="precip_normal_nc.tif",
        native_res_m=300.0,
        source="GHCN daily PRCP normal, IDW-interpolated (modeling.climate).",
    ),
    "tmax_normal": CovariateSpec(
        name="tmax_normal",
        role="continuous",
        # /data/processed/climate/tmax_normal_nc.tif  (mean annual TMAX, deg C)
        root_parts=("processed", "climate"),
        glob="tmax_normal_nc.tif",
        native_res_m=300.0,
        source="GHCN daily TMAX normal, IDW-interpolated (modeling.climate).",
    ),
    "tmin_normal": CovariateSpec(
        name="tmin_normal",
        role="continuous",
        # /data/processed/climate/tmin_normal_nc.tif  (mean annual TMIN, deg C)
        root_parts=("processed", "climate"),
        glob="tmin_normal_nc.tif",
        native_res_m=300.0,
        source="GHCN daily TMIN normal, IDW-interpolated (modeling.climate).",
    ),
    "gdd_normal": CovariateSpec(
        name="gdd_normal",
        role="continuous",
        # /data/processed/climate/gdd_normal_nc.tif  (mean annual growing degree
        # days, base 5 C). Growing-season warmth as a static site characteristic.
        root_parts=("processed", "climate"),
        glob="gdd_normal_nc.tif",
        native_res_m=300.0,
        source="GHCN growing degree days (base 5C) normal, IDW-interpolated from "
        "clim_monthly.gpkg (get_climate&soil_data_updated.R).",
    ),
    # --- SSURGO soil properties (gridded rasters on disk) -----------------------
    # gSSURGO map-unit properties rasterised to 30 m. Continuous properties are
    # area-averaged onto the grid; drainage class is categorical (majority class).
    # Globs match a `processed/soil/` layout -- rename the glob if your SSURGO
    # export uses different filenames; an absent file simply skips (with a warning).
    "soil_organic_matter": CovariateSpec(
        name="soil_organic_matter",
        role="continuous",
        # /data/processed/soil/ssurgo/ssurgo_organic_matter_nc.tif  (% by weight)
        root_parts=("processed", "soil", "ssurgo"),
        glob="ssurgo_organic_matter*_nc.tif",
        native_res_m=30.0,
        source="gSSURGO organic-matter content (om_r), fuel-load proxy.",
    ),
    "soil_awc": CovariateSpec(
        name="soil_awc",
        role="continuous",
        # /data/processed/soil/ssurgo/ssurgo_awc_nc.tif  (available water capacity)
        root_parts=("processed", "soil", "ssurgo"),
        glob="ssurgo_awc*_nc.tif",
        native_res_m=30.0,
        source="gSSURGO available water capacity (awc_r), moisture-retention proxy.",
    ),
    "soil_drainage_class": CovariateSpec(
        name="soil_drainage_class",
        role="categorical",
        # /data/processed/soil/ssurgo/ssurgo_drainage_class_nc.tif  (coded classes)
        root_parts=("processed", "soil", "ssurgo"),
        glob="ssurgo_drainage_class*_nc.tif",
        native_res_m=30.0,
        source="gSSURGO drainage class (drainagecl), coded ordinal -> majority class.",
    ),
    # --- extra soil properties from Cat's aggregated soil_database.gpkg ----------
    # Built by soil.build_soil_database_rasters from soil_database.gpkg
    # (get_climate&soil_data_updated.R). These augment the raw-SSURGO layers above.
    "soil_site_index": CovariateSpec(
        name="soil_site_index",
        role="continuous",
        # /data/processed/soil/ssurgo/ssurgo_site_index_nc.tif
        root_parts=("processed", "soil", "ssurgo"),
        glob="ssurgo_site_index*_nc.tif",
        native_res_m=30.0,
        source="SSURGO site index (coforprod.siteindex.r), forest-productivity proxy.",
    ),
    "soil_water_table_depth": CovariateSpec(
        name="soil_water_table_depth",
        role="continuous",
        # /data/processed/soil/ssurgo/ssurgo_water_table_depth_nc.tif  (cm)
        root_parts=("processed", "soil", "ssurgo"),
        glob="ssurgo_water_table_depth*_nc.tif",
        native_res_m=30.0,
        source="SSURGO water-table depth, Apr-Jun min (muaggatt.wtdepaprjunmin), cm.",
    ),
    # --- not yet downloaded: registered so they attach automatically later ---
    "land_cover": CovariateSpec(
        name="land_cover",
        role="categorical",
        # LandFire Existing Vegetation Type (2024). Categorical -> majority class.
        root_parts=("processed", "land_cover", "LF2024_EVT_NC"),
        glob="landfire_evt_*_nc.tif",
        native_res_m=30.0,
        source="LandFire EVT (https://landfire.gov/vegetation/evt).",
    ),
    "drainage": CovariateSpec(
        name="drainage",
        role="continuous",
        # NASA HAND (Height Above Nearest Drainage), 30 m. Higher = better drained
        # / drier. Continuous -> area-weighted mean per unit
        # (modeling_notebook_explained.md, Part V).
        root_parts=("processed", "drainage"),
        glob="hand_nc.tif",
        native_res_m=30.0,
        source="NASA HAND 30 m (drainage/wetness proxy).",
    ),
    "management": CovariateSpec(
        name="management",
        role="categorical",
        # SEUS industrial forest management raster (binary/categorical).
        root_parts=("processed", "management", "seus_forest_mgmt"),
        glob="seus_forest_mgmt_*_nc.tif",
        native_res_m=30.0,
        source="SEUS Forest Mgmt Map (Fibreboard/Nature sdata2018165).",
    ),
    # NOTE: distance-to-coast is *derived* (from the NC boundary / coastline).
    # Per-year climate (year-specific weather) is *temporal* and lives in the
    # separate TEMPORAL_COVARIATES registry below, keyed on (x, y, year).
}


# ---------------------------------------------------------------------------
# Temporal (per-year) covariates
#
# Same idea as COVARIATES, but the file depends on the calendar year: the spec's
# ``glob`` is a *template* containing ``{year}`` (e.g. ``precip_{year}_nc.tif``),
# filled in per year before globbing. These are the year-specific weather layers
# (built by modeling.climate.build_annual_climate) that belong in the outcome /
# DiD stage -- a `treated:precip` dry-year interaction -- rather than in the
# geographic match, which keys off the *static* climate normals in COVARIATES.
# Reusing CovariateSpec keeps "one spec per covariate": a static glob has no
# ``{year}`` so ``.format(year=...)`` is a harmless no-op, and a temporal glob is
# resolved to a concrete filename by the temporal loaders below.
# ---------------------------------------------------------------------------
TEMPORAL_COVARIATES: dict[str, CovariateSpec] = {
    "precip": CovariateSpec(
        name="precip",
        role="continuous",
        # /data/processed/climate/annual/precip_<year>_nc.tif  (annual PRCP total, mm)
        root_parts=("processed", "climate", "annual"),
        glob="precip_{year}_nc.tif",
        native_res_m=300.0,
        source="GHCN daily PRCP annual total, IDW-interpolated (modeling.climate).",
    ),
    "tmax": CovariateSpec(
        name="tmax",
        role="continuous",
        # /data/processed/climate/annual/tmax_<year>_nc.tif  (mean daily TMAX, deg C)
        root_parts=("processed", "climate", "annual"),
        glob="tmax_{year}_nc.tif",
        native_res_m=300.0,
        source="GHCN daily TMAX annual mean, IDW-interpolated (modeling.climate).",
    ),
    "tmin": CovariateSpec(
        name="tmin",
        role="continuous",
        # /data/processed/climate/annual/tmin_<year>_nc.tif  (mean daily TMIN, deg C)
        root_parts=("processed", "climate", "annual"),
        glob="tmin_{year}_nc.tif",
        native_res_m=300.0,
        source="GHCN daily TMIN annual mean, IDW-interpolated (modeling.climate).",
    ),
    "pdsi": CovariateSpec(
        name="pdsi",
        role="continuous",
        # /data/processed/climate/annual/pdsi_<year>_nc.tif  (annual mean scPDSI)
        # Self-calibrating Palmer Drought Severity Index: ~0 = normal, negative =
        # drought. Built per year (its normal is ~0 everywhere, so there is no
        # static-normal counterpart in COVARIATES) for a treated:pdsi drought-year
        # interaction, the drought analogue of the treated:precip term.
        root_parts=("processed", "climate", "annual"),
        glob="pdsi_{year}_nc.tif",
        native_res_m=300.0,
        source="scPDSI (Palmer drought index) annual mean, IDW-interpolated "
        "from GHCN precip + Hargreaves PET (modeling.climate / get_climate&soil_data.R).",
    ),
}


def get_covariate(name: str) -> CovariateSpec:
    """Return the :class:`CovariateSpec` for ``name`` (clear error if unknown)."""
    try:
        return COVARIATES[name]
    except KeyError:
        raise KeyError(
            f"Unknown covariate {name!r}. Registered: {sorted(COVARIATES)}"
        )


def list_covariates(role: Optional[str] = None) -> list[str]:
    """List registered covariate names, optionally filtered by ``role``."""
    return [
        name
        for name, spec in COVARIATES.items()
        if role is None or spec.role == role
    ]


def get_temporal_covariate(name: str) -> CovariateSpec:
    """Return the per-year :class:`CovariateSpec` for ``name`` (clear error if unknown)."""
    try:
        return TEMPORAL_COVARIATES[name]
    except KeyError:
        raise KeyError(
            f"Unknown temporal covariate {name!r}. "
            f"Registered: {sorted(TEMPORAL_COVARIATES)}"
        )


def list_temporal_covariates() -> list[str]:
    """List registered per-year (temporal) covariate names."""
    return list(TEMPORAL_COVARIATES)


def available_covariates() -> list[str]:
    """Names whose file is actually present on disk right now.

    Lets the frame builder default to "every covariate we have" without the
    caller having to know which downloads have landed.
    """
    return [name for name, spec in COVARIATES.items() if _covariate_file(spec)]


def available_temporal_covariates(years: Sequence[int]) -> list[str]:
    """Per-year covariate names that have a raster for **every** year in ``years``.

    The counterpart to :func:`available_covariates` for the temporal registry, so
    the frame builder / matcher can default to "every per-year layer we have for
    this panel" without the caller tracking which years have been built. Requiring
    all requested years present avoids silently attaching a half-built layer (which
    would null the missing pixel-years); request a name explicitly to attach a
    partially-built layer, where missing years fall back to NaN.
    """
    years = list(years)
    out = []
    for name, spec in TEMPORAL_COVARIATES.items():
        if not spec.directory.exists():
            continue
        if all(
            sorted(spec.directory.glob(spec.glob.format(year=year))) for year in years
        ):
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _covariate_file(spec: CovariateSpec) -> Optional[Path]:
    """First file matching the spec's glob, or ``None`` (with a warning) if the
    directory or file is absent -- so not-yet-downloaded layers skip gracefully."""
    if not spec.directory.exists():
        warnings.warn(
            f"{spec.name}: directory {spec.directory} does not exist -- skipping.",
            stacklevel=2,
        )
        return None
    files = sorted(spec.directory.glob(spec.glob))
    if not files:
        warnings.warn(
            f"{spec.name}: no file matches {spec.glob} in {spec.directory} -- skipping.",
            stacklevel=2,
        )
        return None
    return files[0]


def _temporal_covariate_file(spec: CovariateSpec, year: int) -> Optional[Path]:
    """First file matching the spec's per-year glob for ``year`` (or ``None``).

    Fills the ``{year}`` template in ``spec.glob`` before globbing, so one spec
    covers every year on disk. Warns and returns ``None`` when the directory or a
    given year's file is absent -- so a not-yet-built year skips gracefully, the
    same way :func:`_covariate_file` handles a not-yet-downloaded static layer.
    """
    if not spec.directory.exists():
        warnings.warn(
            f"{spec.name}: directory {spec.directory} does not exist -- skipping.",
            stacklevel=2,
        )
        return None
    glob = spec.glob.format(year=year)
    files = sorted(spec.directory.glob(glob))
    if not files:
        warnings.warn(
            f"{spec.name}: no file matches {glob} in {spec.directory} -- skipping.",
            stacklevel=2,
        )
        return None
    return files[0]


def _standardize(da: xr.DataArray, spec: CovariateSpec, name: str) -> xr.DataArray:
    """Squeeze to ``(y, x)``, select ``band``, null ``nodata``, apply ``transform``.

    The shared read-standardization step for both static and per-year covariates.
    """
    if spec.band is not None:
        da = da.sel(band=spec.band, drop=True)
    elif "band" in da.dims:
        da = da.squeeze("band", drop=True)
    if spec.nodata is not None:
        da = da.where(~da.isin(list(spec.nodata)))
    if spec.transform is not None:
        da = spec.transform(da)
    da.name = name
    return da


def load_covariate(name: str, aoi) -> Optional[xr.DataArray]:
    """Clipped, standardized 2-D DataArray for one covariate (or ``None``).

    Reads the layer, clips to ``aoi`` (native CRS preserved), squeezes to
    ``(y, x)``, selects ``band`` if multi-band, nulls any undeclared ``nodata``
    sentinels, and applies ``transform``. Mirrors ``fire_products._clip_raster``.
    """
    spec = get_covariate(name)
    path = _covariate_file(spec)
    if path is None:
        return None
    return _standardize(clip_raster_to_mask(path, aoi), spec, name)


def load_temporal_covariate(name: str, aoi, year: int) -> Optional[xr.DataArray]:
    """Clipped, standardized 2-D DataArray for one per-year covariate (or ``None``).

    The temporal counterpart of :func:`load_covariate`: resolves the spec's
    ``{year}`` glob to that calendar year's raster, then applies the identical
    clip + standardization. Returns ``None`` (with a warning) if that year is not
    on disk.
    """
    spec = get_temporal_covariate(name)
    path = _temporal_covariate_file(spec, year)
    if path is None:
        return None
    return _standardize(clip_raster_to_mask(path, aoi), spec, name)


def covariate_on_grid(name: str, grid: xr.DataArray, aoi) -> Optional[xr.DataArray]:
    """Load a covariate and warp it onto the shared analysis ``grid``.

    Continuous layers are area-averaged (``how="mean"``), categorical layers take
    the majority class (``how="mode"``); override per layer via ``spec.resample``.
    Returns ``None`` if the layer is not on disk. ``grid`` comes from
    :func:`peatfire.build_common_grid` so every covariate lands on exactly the
    same EPSG:5070 cells as the fire response.
    """
    da = load_covariate(name, aoi)
    if da is None:
        return None
    spec = get_covariate(name)
    return to_common_grid(da.astype("float32"), grid, how=spec.how)


def temporal_covariate_on_grid(
    name: str, grid: xr.DataArray, aoi, year: int
) -> Optional[xr.DataArray]:
    """Load one per-year covariate for ``year`` and warp it onto ``grid``.

    The temporal counterpart of :func:`covariate_on_grid`: same grid-aggregation
    contract, but for the calendar year's weather raster. Returns ``None`` if that
    year's layer is not on disk, so callers sampling a panel can skip missing years.
    """
    da = load_temporal_covariate(name, aoi, year)
    if da is None:
        return None
    spec = get_temporal_covariate(name)
    return to_common_grid(da.astype("float32"), grid, how=spec.how)

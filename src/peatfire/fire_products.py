"""Per-product knowledge and standardized loaders for fire data products.

This is the *data layer* of the fire-product comparison toolkit. It isolates the
messy, product-specific details -- where each product lives on disk, its native
CRS and resolution, how a "burned" pixel is encoded, and whether it is delivered
monthly or annually -- behind a small registry of :class:`ProductSpec` objects
and a handful of generic loaders.

Downstream code (:mod:`peatfire.fire_comparison`) never needs to know these
details: it asks for ``load_standardized(product, year, aoi)`` and gets back a
standardized representation (an annual boolean burned mask, a continuous
severity grid, or a points GeoDataFrame).

Adding a new product is a matter of adding one :class:`ProductSpec` to
:data:`FIRE_PRODUCTS`; no loader code changes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from .data_loading import data_path


# ---------------------------------------------------------------------------
# Product specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProductSpec:
    """Everything the generic loaders need to read one fire product.

    Parameters
    ----------
    name : str
        Short identifier, e.g. ``"MCD64A1"``. Used as the registry key and as
        the column/label name in outputs.
    family : str
        One of ``"burned_area"``, ``"severity"``, ``"occurrence"``. Drives which
        loader :func:`load_standardized` dispatches to and which comparison a
        product participates in.
    kind : str
        ``"raster"`` or ``"vector"``.
    native_res_m : float
        Nominal pixel size in metres (used only for documentation / sensitivity
        analysis; areas are computed from the actual affine transform).
    temporal : str
        ``"monthly"``, ``"annual"`` or ``"events"`` -- controls how files are
        grouped into a single year.
    root_parts : tuple of str
        Arguments forwarded to :func:`peatfire.data_path` to locate the
        product's directory, e.g. ``("processed", "fire", "MCD64A1_061")``.
    glob : str
        Filename glob within that directory, e.g. ``"MCD64A1_*_nc.tif"``.
    year_parser : callable, optional
        ``Path -> int`` mapping a file to its year. Required for raster products
        whose year is encoded in the filename. ``None`` for vector products that
        are filtered by a date column instead.
    burn_predicate : callable, optional
        ``DataArray -> boolean DataArray`` marking burned pixels. Required for
        ``burned_area`` rasters.
    value_predicate : callable, optional
        ``DataArray -> float DataArray`` extracting the continuous severity
        metric (e.g. dNBR, CBI). Required for ``severity`` products.
    date_field : str, optional
        For vector ``occurrence`` products, the attribute holding the
        acquisition date (used to filter to a year).
    frp_field : str, optional
        For vector ``occurrence`` products, the Fire Radiative Power column.
    native_crs : str, optional
        Documentation only; the actual CRS is read from each file.
    """

    name: str
    family: str
    kind: str
    native_res_m: float
    temporal: str
    root_parts: tuple
    glob: str
    year_parser: Optional[Callable[[Path], int]] = None
    burn_predicate: Optional[Callable[[xr.DataArray], xr.DataArray]] = None
    value_predicate: Optional[Callable[[xr.DataArray], xr.DataArray]] = None
    date_field: Optional[str] = None
    frp_field: Optional[str] = None
    native_crs: Optional[str] = None

    @property
    def directory(self) -> Path:
        return data_path(*self.root_parts)


# ---------------------------------------------------------------------------
# Filename year parsers (kept tiny and named so specs read clearly)
# ---------------------------------------------------------------------------
def _mcd64_year(p: Path) -> int:
    # 'MCD64A1_A2017001_nc' -> 'A2017001' -> 2017
    return int(p.stem.split("_")[1][1:5])


def _gabam_year(p: Path) -> int:
    # 'gabam_2011_nc' -> 2011
    return int(p.stem.split("_")[1])


def _second_token_year(p: Path) -> int:
    # generic '<prod>_<year>_...' helper for products we haven't downloaded yet
    return int(p.stem.split("_")[1])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Products with data already in the repo (MCD64A1, GABAM, VIIRS) have concrete
# specs. The others are placeholders: their globs / encodings are marked TODO
# and must be confirmed once the data is downloaded. Until then the loaders skip
# them gracefully (see ``_files_for_year``).
FIRE_PRODUCTS: dict[str, ProductSpec] = {
    "MCD64A1": ProductSpec(
        name="MCD64A1",
        family="burned_area",
        kind="raster",
        native_res_m=500.0,
        temporal="monthly",
        root_parts=("processed", "fire", "MCD64A1_061"),
        glob="MCD64A1_*_nc.tif",
        year_parser=_mcd64_year,
        # BurnDate is day-of-year 1-366 for burned pixels; 0/-1/-2/NaN unburned.
        burn_predicate=lambda da: da > 0,
        native_crs="Sinusoidal",
    ),
    "GABAM": ProductSpec(
        name="GABAM",
        family="burned_area",
        kind="raster",
        native_res_m=30.0,
        temporal="annual",
        root_parts=("raw", "fire", "gabam"),
        glob="gabam_*_nc.tif",
        year_parser=_gabam_year,
        burn_predicate=lambda da: da == 1,
        native_crs="EPSG:4326",
    ),
    "VIIRS": ProductSpec(
        name="VIIRS",
        family="occurrence",
        kind="vector",
        native_res_m=375.0,
        temporal="events",
        root_parts=("processed", "fire", "viirs"),
        glob="viirs_snpp_archive_nc.gpkg",
        date_field="ACQ_DATE",
        frp_field="FRP",
        native_crs="EPSG:4326",
    ),
    # ---- placeholders: confirm globs/encodings when data lands ----
    "FireCCI51": ProductSpec(
        name="FireCCI51",
        family="burned_area",
        kind="raster",
        native_res_m=250.0,
        temporal="monthly",
        root_parts=("processed", "fire", "FireCCI51"),
        glob="firecci51_*_nc.tif",  # TODO confirm
        year_parser=_second_token_year,
        burn_predicate=lambda da: da > 0,  # TODO confirm (date-of-first-detection)
        native_crs="EPSG:4326",
    ),
    "USGS_BA": ProductSpec(
        name="USGS_BA",
        family="burned_area",
        kind="raster",
        native_res_m=30.0,
        temporal="annual",
        root_parts=("processed", "fire", "usgs_lba"),
        glob="usgs_lba_*_nc.tif",  # TODO confirm
        year_parser=_second_token_year,
        burn_predicate=lambda da: da > 0,  # TODO confirm (burn classification)
        native_crs="EPSG:5070",
    ),
    "SE_FireMap": ProductSpec(
        name="SE_FireMap",
        family="severity",
        kind="raster",
        native_res_m=30.0,
        temporal="annual",
        root_parts=("processed", "fire", "se_firemap"),
        glob="se_firemap_*_nc.tif",  # TODO confirm
        year_parser=_second_token_year,
        value_predicate=lambda da: da,  # continuous CBI 0-3; TODO confirm band
        native_crs="EPSG:5070",
    ),
    "MOSEV": ProductSpec(
        name="MOSEV",
        family="severity",
        kind="raster",
        native_res_m=500.0,
        temporal="monthly",
        root_parts=("processed", "fire", "mosev"),
        glob="mosev_*_nc.tif",  # TODO confirm
        year_parser=_second_token_year,
        value_predicate=lambda da: da,  # dNBR; TODO confirm band
        native_crs="Sinusoidal",
    ),
    "MTBS": ProductSpec(
        name="MTBS",
        family="severity",
        kind="raster",
        native_res_m=30.0,
        temporal="annual",
        root_parts=("processed", "fire", "mtbs"),
        glob="mtbs_*_nc.tif",  # TODO confirm
        year_parser=_second_token_year,
        value_predicate=lambda da: da,  # dNBR / thematic class; TODO confirm
        native_crs="EPSG:5070",
    ),
}


def get_spec(product: str) -> ProductSpec:
    """Return the :class:`ProductSpec` for ``product`` (clear error if unknown)."""
    try:
        return FIRE_PRODUCTS[product]
    except KeyError:
        raise KeyError(
            f"Unknown product {product!r}. Registered products: "
            f"{sorted(FIRE_PRODUCTS)}"
        )


def list_products(family: Optional[str] = None) -> list[str]:
    """List registered product names, optionally filtered by ``family``."""
    return [
        name
        for name, spec in FIRE_PRODUCTS.items()
        if family is None or spec.family == family
    ]


# ---------------------------------------------------------------------------
# In-memory clipping
#
# data_loading.clip_raster_to_mask / clip_vector_to_mask write to disk, which we
# do not want to do once per (product, year). These private helpers do the same
# clip in memory, leaving the tested disk-writing functions untouched.
# ---------------------------------------------------------------------------
def _clip_raster_mem(path: Path, aoi: gpd.GeoDataFrame) -> xr.DataArray:
    da = rioxarray.open_rasterio(path, masked=True).squeeze("band", drop=True)
    aoi_in_crs = aoi.to_crs(da.rio.crs)
    return da.rio.clip(aoi_in_crs.geometry, aoi_in_crs.crs)


def _clip_vector_mem(path: Path, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    aoi_in_crs = aoi.to_crs(gdf.crs)
    return gpd.sjoin(
        gdf, aoi_in_crs[["geometry"]], predicate="within"
    ).drop(columns="index_right")


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------
def _files_for_year(spec: ProductSpec, year: int) -> list[Path]:
    """Return the file(s) for ``spec`` in ``year`` (``[]`` if none/absent).

    A missing directory or empty result is *not* an error: the product is simply
    skipped with a warning, so a comparison can run on whatever is downloaded.
    """
    if not spec.directory.exists():
        warnings.warn(
            f"{spec.name}: directory {spec.directory} does not exist -- skipping.",
            stacklevel=2,
        )
        return []
    files = sorted(spec.directory.glob(spec.glob))
    if spec.year_parser is None:
        return files  # vector products are filtered by date later
    matched = [f for f in files if spec.year_parser(f) == year]
    return matched


# ---------------------------------------------------------------------------
# Standardized loaders
# ---------------------------------------------------------------------------
def load_binary_annual(
    product: str, year: int, aoi: gpd.GeoDataFrame
) -> Optional[xr.DataArray]:
    """Annual boolean burned mask (``True`` = burned) for one product/year.

    Monthly products are OR'd into a single annual mask (a pixel that burned in
    *any* month counts as burned for the year), mirroring the original sandbox
    logic. The result is clipped to ``aoi`` with its native CRS preserved.

    Returns ``None`` if the product has no data for ``year`` (caller skips it).
    """
    spec = get_spec(product)
    if spec.burn_predicate is None:
        raise ValueError(f"{product} has no burn_predicate (not a burned-area product).")

    files = _files_for_year(spec, year)
    if not files:
        return None

    masks = []
    for f in files:
        clipped = _clip_raster_mem(f, aoi)
        masks.append(spec.burn_predicate(clipped))

    if len(masks) == 1:
        annual = masks[0]
    else:
        annual = xr.concat(masks, dim="month").max("month")
    annual = annual.rio.write_crs(masks[0].rio.crs)
    annual.name = product
    return annual


def load_continuous_annual(
    product: str, year: int, aoi: gpd.GeoDataFrame
) -> Optional[xr.DataArray]:
    """Annual continuous severity grid (e.g. CBI / dNBR) for one product/year.

    Monthly severity products are aggregated by per-pixel maximum. Returns
    ``None`` if there is no data for ``year``.
    """
    spec = get_spec(product)
    if spec.value_predicate is None:
        raise ValueError(f"{product} has no value_predicate (not a severity product).")

    files = _files_for_year(spec, year)
    if not files:
        return None

    vals = [spec.value_predicate(_clip_raster_mem(f, aoi)) for f in files]
    if len(vals) == 1:
        annual = vals[0]
    else:
        annual = xr.concat(vals, dim="month").max("month")
    annual = annual.rio.write_crs(vals[0].rio.crs)
    annual.name = product
    return annual


def load_points(
    product: str, year: int, aoi: gpd.GeoDataFrame
) -> Optional[gpd.GeoDataFrame]:
    """Active-fire point detections (with FRP) for one product/year.

    The whole clipped layer is read, then filtered to ``year`` using
    ``spec.date_field``. Returns ``None`` if the file is absent or no detections
    fall in the year.
    """
    spec = get_spec(product)
    files = _files_for_year(spec, year)
    if not files:
        return None

    gdf = _clip_vector_mem(files[0], aoi)
    if spec.date_field and spec.date_field in gdf.columns:
        dates = gpd.pd.to_datetime(gdf[spec.date_field], errors="coerce")
        gdf = gdf[dates.dt.year == year]
    if gdf.empty:
        return None
    return gdf


def load_standardized(product: str, year: int, aoi: gpd.GeoDataFrame):
    """Dispatch to the right loader based on the product's ``family``.

    Returns an annual boolean mask (burned_area), a continuous grid (severity),
    a points GeoDataFrame (occurrence), or ``None`` when data is absent.
    """
    spec = get_spec(product)
    if spec.family == "burned_area":
        return load_binary_annual(product, year, aoi)
    if spec.family == "severity":
        return load_continuous_annual(product, year, aoi)
    if spec.family == "occurrence":
        return load_points(product, year, aoi)
    raise ValueError(f"Unknown family {spec.family!r} for product {product!r}.")

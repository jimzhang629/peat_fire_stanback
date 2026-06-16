"""Ground-truth / reference fire sources for *validating* the candidate products.

Where :mod:`fire_products` registers the remotely-sensed products we are trying
to evaluate, this module registers the independent **reference** datasets we
score them against -- fire-incident perimeters and prescribed-burn footprints
that record, on the ground, where a fire actually happened.

Currently registered (all vector):

* ``NIFC_IFPH`` -- NIFC Interagency Fire Perimeter History ("All Years"), the
  authoritative long-record perimeter compilation. The default reference for the
  large pocosin incidents Margaret listed (Evans Road 2008, Pains Bay 2011,
  Juniper Road 2011, Whipping Creek 2016, Great Lakes 2023, ...).
* ``GEOMAC`` -- NIFC Historic GeoMAC Perimeters Combined 2000-2018. Overlaps
  IFPH over 2000-2018, so it is a *cross-check* on those years, not a source to
  pool with IFPH (the same fire appears in both -- pooling double-counts it).
* ``TNC_SANDHILLS`` -- TNC NC Sandhills program preserve burn history (longleaf
  uplands; smaller controlled burns -- a stress test of small-fire omission).
* ``TNC_COASTAL_PLAIN`` -- TNC NC Coastal Plain program preserve burn history
  (the peat-relevant program: pocosin / wet-pine burns).
* ``NCWRC_RX`` -- NC Wildlife Resources Commission prescribed-fire footprints
  (recent only; another small-fire reference).

The two NIFC sources carry **different attribute schemas** (IFPH uses uppercase
``FIRE_YEAR``/``INCIDENT``/``GIS_ACRES``; GeoMAC uses lowercase
``fireyear``/``incidentname``/``gisacres``), which is why each spec lists
candidate column names per role.

Why a *separate* registry from :data:`fire_products.FIRE_PRODUCTS`
------------------------------------------------------------------
A reference is **truth within its own footprint**: we are confident a fire
occurred inside a mapped perimeter, so these sources give a clean
**omission / recall** measurement (did the product catch a known fire?). They are
emphatically *not* a spatially-exhaustive census of every fire in the
landscape -- NIFC only maps large incidents, TNC only maps its own preserves --
so a product pixel *outside* every perimeter is **not** necessarily a false
alarm. That asymmetry (good for recall, only conditionally usable for precision)
is the reason references live apart from the products under test. See
:mod:`validation` for how the metrics encode it.

Field-name tolerance
--------------------
Agency vector exports rename their columns constantly (``FIRE_YEAR`` vs
``attr_FireDiscoveryDateTime``, ``INCIDENT`` vs ``poly_IncidentName``, ...). Each
:class:`ReferenceSpec` therefore lists *candidate* column names per role and the
loader picks the first one present, so a spec keeps working whichever NIFC export
lands on disk.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd

from ..preproc.data_loading import clip_vector_to_mask, data_path


# ---------------------------------------------------------------------------
# Reference specification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReferenceSpec:
    """Everything the loader needs to read one reference (ground-truth) source.

    Parameters
    ----------
    name : str
        Short identifier / registry key, e.g. ``"NIFC"``.
    kind : str
        ``"perimeter"`` (polygons -- rasterised to a burned mask) or ``"points"``
        (point ignitions -- binned to the grid). Most references are perimeters.
    root_parts : tuple of str
        Arguments to :func:`peatfire.data_path` locating the source's directory,
        e.g. ``("processed", "fire", "reference", "nifc")``.
    glob : str
        Filename glob within that directory. May match several files (e.g. the
        two NIFC perimeter exports Margaret linked); they are concatenated.
    date_fields : tuple of str
        Candidate column names holding an event *date* (parsed with
        :func:`pandas.to_datetime`). The first present is used to derive the
        event year/month when ``year_fields`` is absent.
    year_fields : tuple of str
        Candidate column names already holding an integer fire *year* (e.g. NIFC
        ``FIRE_YEAR``). Preferred over ``date_fields`` when present -- cheaper and
        avoids date-parse ambiguity.
    event_fields : tuple of str
        Candidate column names holding the incident *name* (for per-event
        analysis and for matching Margaret's named fires).
    area_fields : tuple of str
        Candidate column names holding the reported burned *acreage* (for the
        event table; never used for the geometric metrics, which come from the
        rasterised polygon).
    native_crs : str, optional
        Documentation only; the actual CRS is read from each file.
    """

    name: str
    kind: str
    root_parts: tuple
    glob: str
    date_fields: tuple = ()
    year_fields: tuple = ()
    event_fields: tuple = ()
    area_fields: tuple = ()
    native_crs: Optional[str] = None

    @property
    def directory(self) -> Path:
        return data_path(*self.root_parts)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
# Folder layout mirrors the products: data/processed/fire/reference/<source>/.
# A source whose directory is missing simply skips with a warning (like the
# product loaders), so a validation run works on whatever has been downloaded.
REFERENCE_SOURCES: dict[str, ReferenceSpec] = {
    "NIFC_IFPH": ReferenceSpec(
        name="NIFC_IFPH",
        kind="perimeter",
        # raw: data/raw/fire/InterAgencyFirePerimeterHistory_All_Years_View_*.gpkg
        # clip to NC + standardise filename into the reference folder.
        root_parts=("processed", "fire", "reference", "nifc_ifph"),
        glob="nifc_ifph_*.gpkg",
        # IFPH carries an integer FIRE_YEAR_INT plus a (string) FIRE_YEAR;
        # DATE_CUR is the "current as of" date. Prefer the explicit integer year.
        date_fields=("DATE_CUR", "FIRE_DATE", "DISCOVERY_DATE"),
        year_fields=("FIRE_YEAR_INT", "FIRE_YEAR"),
        event_fields=("INCIDENT", "FIRE_NAME", "INCIDENT_NAME"),
        area_fields=("GIS_ACRES", "ACRES"),
        native_crs="EPSG:3857",  # ArcGIS Hub export is Web Mercator; reprojected to 5070 for analysis
    ),
    "GEOMAC": ReferenceSpec(
        name="GEOMAC",
        kind="perimeter",
        # raw: data/raw/fire/Historic_Geomac_Perimeters_Combined_2000_2018_*.gpkg
        root_parts=("processed", "fire", "reference", "geomac"),
        glob="geomac_*.gpkg",
        # GeoMAC uses lowercase fields and a perimeter datetime.
        date_fields=(
            "latestperimeterdatetime",
            "perimeterdatetime",
            "datecurrent",
        ),
        year_fields=("fireyear",),
        event_fields=("incidentname", "fire_name"),
        area_fields=("gisacres", "acres"),
        native_crs="EPSG:3857",  # ArcGIS Hub export is Web Mercator; reprojected to 5070 for analysis
    ),
    # Margaret's two TNC programs arrive as shapefiles
    # (TNC_NC_Sandhills_Fire_History_2025.shp,
    #  TNC_NC_Coastal_Plain_Fire_History_2025.shp); clip each to a processed gpkg.
    # NOTE: the *_fields candidate lists below are best-guess until the shapefile
    # columns are confirmed -- shapefiles also truncate field names to 10 chars,
    # so verify and extend these tuples (one-line change, no logic touched).
    "TNC_SANDHILLS": ReferenceSpec(
        name="TNC_SANDHILLS",
        kind="perimeter",
        root_parts=("processed", "fire", "reference", "tnc_burns"),
        glob="tnc_burns_*.gpkg",
        date_fields=("BurnDate", "Burn_Date", "DATE", "FireDate"),
        year_fields=("Year", "BurnYear", "FY"),
        event_fields=("Name", "BurnUnit", "Unit", "Preserve", "BURN_NAME"),
        area_fields=("Acres", "GIS_ACRES", "ACRES"),
        native_crs="EPSG:4326",
        root_parts=("processed", "fire", "reference", "tnc_sandhills"),
        glob="tnc_sandhills_*.gpkg",
        date_fields=("BurnDate", "Burn_Date", "DATE", "FireDate", "Date"),
        year_fields=("Year", "BurnYear", "FY", "FireYear"),
        event_fields=("Name", "BurnUnit", "Unit", "Preserve", "BURN_NAME", "UnitName"),
        area_fields=("Acres", "GIS_ACRES", "ACRES", "Acreage"),
        native_crs=None,  # read from the shapefile (often NC State Plane / NAD83)
    ),
    "TNC_COASTAL_PLAIN": ReferenceSpec(
        name="TNC_COASTAL_PLAIN",
        kind="perimeter",
        root_parts=("processed", "fire", "reference", "tnc_coastal_plain"),
        glob="tnc_coastal_plain_*.gpkg",
        date_fields=("BurnDate", "Burn_Date", "DATE", "FireDate", "Date"),
        year_fields=("Year", "BurnYear", "FY", "FireYear"),
        event_fields=("Name", "BurnUnit", "Unit", "Preserve", "BURN_NAME", "UnitName"),
        area_fields=("Acres", "GIS_ACRES", "ACRES", "Acreage"),
        native_crs=None,  # read from the shapefile (often NC State Plane / NAD83)
    ),
    "NCWRC_RX": ReferenceSpec(
        name="NCWRC_RX",
        kind="perimeter",
        root_parts=("processed", "fire", "reference", "ncwrc_rx"),
        glob="ncwrc_rx_*.gpkg",
        date_fields=("BurnDate", "DATE", "IgnitionDate", "Date_"),
        year_fields=("Year", "BurnYear"),
        event_fields=("Name", "Unit", "GameLand", "BURN_NAME"),
        area_fields=("Acres", "GIS_ACRES", "ACRES"),
        native_crs="EPSG:4326",
    ),
}


def get_reference(name: str) -> ReferenceSpec:
    """Return the :class:`ReferenceSpec` for ``name`` (clear error if unknown)."""
    try:
        return REFERENCE_SOURCES[name]
    except KeyError:
        raise KeyError(
            f"Unknown reference {name!r}. Registered references: "
            f"{sorted(REFERENCE_SOURCES)}"
        )


def list_references() -> list[str]:
    """List registered reference-source names."""
    return list(REFERENCE_SOURCES)


# ---------------------------------------------------------------------------
# Column-role resolution (tolerant to the agencies' shifting field names)
# ---------------------------------------------------------------------------
def _first_present(gdf: gpd.GeoDataFrame, candidates: tuple) -> Optional[str]:
    """Return the first of ``candidates`` that is a column of ``gdf`` (or None)."""
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def _derive_year(gdf: gpd.GeoDataFrame, spec: ReferenceSpec) -> pd.Series:
    """Best-effort integer fire year per row: a year column if present, else the
    year parsed from a date column. NaN where neither is available."""
    yfield = _first_present(gdf, spec.year_fields)
    if yfield is not None:
        return pd.to_numeric(gdf[yfield], errors="coerce")
    dfield = _first_present(gdf, spec.date_fields)
    if dfield is not None:
        return pd.to_datetime(gdf[dfield], errors="coerce").dt.year
    return pd.Series([pd.NA] * len(gdf), index=gdf.index, dtype="Float64")


def _derive_month(gdf: gpd.GeoDataFrame, spec: ReferenceSpec) -> pd.Series:
    """Best-effort integer month per row from a date column; NaN if none/parse-fail."""
    dfield = _first_present(gdf, spec.date_fields)
    if dfield is not None:
        return pd.to_datetime(gdf[dfield], errors="coerce").dt.month
    return pd.Series([pd.NA] * len(gdf), index=gdf.index, dtype="Float64")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _read_all(spec: ReferenceSpec) -> Optional[gpd.GeoDataFrame]:
    """Read and concatenate every file matching ``spec.glob`` in its directory.

    Returns ``None`` (with a warning) if the directory or files are absent, so a
    validation run degrades gracefully to whatever has been downloaded -- the
    same contract as the product loaders.
    """
    if not spec.directory.exists():
        warnings.warn(
            f"{spec.name}: directory {spec.directory} does not exist -- skipping.",
            stacklevel=2,
        )
        return None
    files = sorted(spec.directory.glob(spec.glob))
    if not files:
        warnings.warn(
            f"{spec.name}: no files match {spec.glob} in {spec.directory} -- skipping.",
            stacklevel=2,
        )
        return None
    gdfs = [gpd.read_file(f) for f in files]
    if len(gdfs) == 1:
        return gdfs[0]
    # concat in the first file's CRS so geometries stay aligned.
    base_crs = gdfs[0].crs
    gdfs = [g.to_crs(base_crs) for g in gdfs]
    return gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), crs=base_crs
    )


def load_reference(
    name: str,
    aoi: gpd.GeoDataFrame,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Load a reference source, clipped to ``aoi`` and (optionally) time-filtered.

    The returned GeoDataFrame is augmented with three standardized columns the
    validation layer relies on, regardless of the source's native schema:

    * ``_event`` -- incident name (or ``"<NAME>_<i>"`` fallback),
    * ``_year``  -- integer fire year,
    * ``_month`` -- integer fire month (may be NaN if the source has no date),
    * ``_area``  -- reported acreage (may be NaN).

    ``year`` / ``month`` filter to events in that period (rows with an unparseable
    year are dropped only when ``year`` is requested). Returns ``None`` if the
    source is absent or nothing falls in ``aoi``/the period.
    """
    spec = get_reference(name)
    gdf = _read_all(spec)
    if gdf is None or gdf.empty:
        return None

    gdf = clip_vector_to_mask(gdf, aoi)
    if gdf.empty:
        return None

    gdf = gdf.copy()
    gdf["_year"] = _derive_year(gdf, spec)
    gdf["_month"] = _derive_month(gdf, spec)
    efield = _first_present(gdf, spec.event_fields)
    if efield is not None:
        gdf["_event"] = gdf[efield].astype("string")
    else:
        gdf["_event"] = pd.Series(
            [f"{spec.name}_{i}" for i in range(len(gdf))], index=gdf.index
        )
    afield = _first_present(gdf, spec.area_fields)
    gdf["_area"] = pd.to_numeric(gdf[afield], errors="coerce") if afield else pd.NA

    if year is not None:
        gdf = gdf[gdf["_year"] == year]
        if month is not None:
            gdf = gdf[gdf["_month"] == month]
    if gdf.empty:
        return None
    return gdf


def reference_event_table(
    name: str, aoi: gpd.GeoDataFrame
) -> Optional[pd.DataFrame]:
    """Tidy inventory of the reference's events within ``aoi``.

    One row per source feature with ``event``/``year``/``month``/``acres`` -- the
    menu of incidents you can validate against (and the place to spot Margaret's
    named fires). Returns ``None`` if the source is absent.
    """
    gdf = load_reference(name, aoi)
    if gdf is None:
        return None
    out = pd.DataFrame(
        {
            "source": name,
            "event": gdf["_event"].values,
            "year": gdf["_year"].values,
            "month": gdf["_month"].values,
            "acres": gdf["_area"].values,
        }
    )
    return out.sort_values(["year", "event"]).reset_index(drop=True)

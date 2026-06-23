"""Validate candidate fire products against independent ground-truth references.

Where :mod:`fire_comparison` answers *"do the products agree with each other?"*
(product-vs-product, no truth), this module answers *"is a product **right**?"*
-- it scores each product against the reference fire perimeters registered in
:mod:`reference_sources` (NIFC large-fire perimeters, TNC preserve burns, NCWRC
prescribed fires).

The metrics and their caveats (the methodological core)
-------------------------------------------------------
For a reference event we rasterise its perimeter onto the common grid (the
"reference burned" cells) and load each product on the *same* grid, then count:

* **recall = TP / (TP + FN)** -- of the reference's burned cells, the fraction the
  product also flags. This is **1 - omission** and is the **headline metric**: it
  is clean because we are confident a fire occurred inside the perimeter.
* **precision = TP / (TP + FP)** -- of the product's burned cells *in the analysis
  window*, the fraction inside the perimeter. Reported **only per-event, inside a
  buffered window** around the incident, and flagged as *conditional*, because:

  1. the references are **not a complete census** -- a product burn outside the
     perimeter may be a real small fire the reference never recorded, not a false
     alarm (so a global precision would be meaningless); and
  2. a perimeter is an **outer boundary**, not a burned-pixel map -- it includes
     unburned islands, so it *overestimates* truth, which depresses precision and
     inflates omission. Treat the perimeter as an upper bound on the true burn.

Two further deliberate choices:

* **Time-matched, per-event.** Each incident is scored against the product layer
  for *its own year* (annual matching by default, so a product is not penalised
  for sub-monthly timing). Events are *not* dissolved into a single timeless
  union -- that would let a detection in one year "catch" a fire in another.
* **Occurrence products (VIIRS) are judged at the event level.** Active fire is
  points detecting a burning front, not burned area, so per-cell recall is the
  wrong question; ``detected`` (did any cell of the product overlap the
  perimeter?) is the meaningful one. VIIRS also only exists 2012+, so it is
  simply absent (skipped) for the older incidents.

Everything reuses the common-grid machinery in :mod:`fire_comparison`
(:func:`build_common_grid`, :func:`stack_on_common_grid`,
:func:`rasterize_polygons_to_grid`) so a product is scored on exactly the grid it
is compared on elsewhere.
"""

from __future__ import annotations

from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from .fire_comparison import (
    ANALYSIS_CRS,
    build_common_grid,
    rasterize_points_to_grid,
    rasterize_polygons_to_grid,
    to_common_grid,
)
from .fire_products import get_spec, list_products, load_points, load_standardized
from .reference_sources import get_reference, load_reference


# ---------------------------------------------------------------------------
# Confusion of one product mask against one reference mask
# ---------------------------------------------------------------------------
def confusion_scores(
    product_mask: xr.DataArray, reference_mask: xr.DataArray
) -> dict:
    """Cell-wise confusion of a product's burned mask vs a reference burned mask.

    Both arrays must already be on the **same grid** (same shape/transform); this
    function only counts cells. Returns a dict with ``tp``/``fn``/``fp`` cell
    counts, ``n_ref``/``n_prod`` totals, ``recall`` and ``precision`` (NaN when
    their denominator is empty), and ``detected`` (any overlap at all).

    ``recall`` is the trustworthy number; ``precision`` is only meaningful inside
    a bounded analysis window and carries the caveats in the module docstring.
    """
    p = np.asarray(product_mask.values).ravel() > 0
    r = np.asarray(reference_mask.values).ravel() > 0
    tp = int((p & r).sum())
    fn = int((~p & r).sum())
    fp = int((p & ~r).sum())
    n_ref = tp + fn
    n_prod = tp + fp
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "n_ref": n_ref,
        "n_prod": n_prod,
        "recall": tp / n_ref if n_ref else np.nan,
        "precision": tp / n_prod if n_prod else np.nan,
        "detected": bool(tp > 0),
    }


# ---------------------------------------------------------------------------
# Per-event validation
# ---------------------------------------------------------------------------
def _window_gdf(event_geom_5070, buffer_m: float) -> gpd.GeoDataFrame:
    """A buffered analysis window (in EPSG:5070) around one event geometry.

    Products are clipped to this window and precision is measured within it, so
    precision captures burns *near* the incident (spatial displacement / false
    alarms in the neighbourhood) rather than being trivially 1 from clipping to
    the perimeter itself.
    """
    win = event_geom_5070.buffer(buffer_m)
    return gpd.GeoDataFrame(geometry=[win], crs=ANALYSIS_CRS)


def _preload_products(
    products: Iterable[str],
    year: int,
    month: Optional[int],
    clip_aoi: gpd.GeoDataFrame,
) -> dict:
    """Load each product ONCE for a period, clipped to ``clip_aoi`` (native grid).

    Returns ``{product: (kind, data)}`` where ``kind`` is ``"raster"`` (a burned
    mask in the product's native grid) or ``"points"`` (an active-fire
    GeoDataFrame), and ``data`` is ``None`` when the product has no data for the
    period. Only burned-area and occurrence products are loaded; severity is not
    validated against perimeters.

    Loading here -- once per period rather than once per event -- is what lets
    many same-year events reuse a single disk read (see :func:`validate_products`).
    """
    out: dict = {}
    for product in products:
        spec = get_spec(product)
        if spec.family == "burned_area":
            out[product] = ("raster", load_standardized(product, year, clip_aoi, month))
        elif spec.family == "occurrence":
            out[product] = ("points", load_points(product, year, clip_aoi, month))
        # severity: not scored against perimeters -> skipped
    return out


def _score_event(
    event_gdf: gpd.GeoDataFrame,
    preloaded: dict,
    year: int,
    month: Optional[int],
    buffer_km: float,
    common_grid_res: float,
) -> pd.DataFrame:
    """Score already-loaded products against ONE event on its window grid.

    The result is identical to loading per-event: the buffered window grid, the
    rasterised reference mask, and the metrics are the same -- the products are
    simply reprojected from the in-memory period arrays in ``preloaded`` (from
    :func:`_preload_products`) instead of being re-read from disk.
    """
    geom_5070 = event_gdf.to_crs(ANALYSIS_CRS).geometry.union_all()
    window = _window_gdf(geom_5070, buffer_km * 1000.0)
    grid = build_common_grid(window, res_m=common_grid_res)

    ref_mask = rasterize_polygons_to_grid(event_gdf.to_crs(ANALYSIS_CRS), grid)
    if float(ref_mask.sum()) == 0:
        return pd.DataFrame()  # perimeter too small for the grid; nothing to score

    event_name = (
        str(event_gdf["_event"].iloc[0]) if "_event" in event_gdf.columns else "event"
    )
    rows = []
    for product, (kind, data) in preloaded.items():
        if data is None:
            continue  # product absent for this period (e.g. VIIRS pre-2012)
        if kind == "raster":
            prod_mask = to_common_grid(data.astype("float32"), grid, how="max") > 0
            prod_mask = prod_mask.rio.write_crs(grid.rio.crs)
        else:  # points (occurrence): per-cell presence
            prod_mask = rasterize_points_to_grid(data, grid, agg="any")
        scores = confusion_scores(prod_mask, ref_mask)
        rows.append(
            {
                "event": event_name,
                "year": year,
                "month": month,
                "product": product,
                "family": get_spec(product).family,
                **scores,
            }
        )
    return pd.DataFrame(rows)


def validate_event(
    event_gdf: gpd.GeoDataFrame,
    products: Iterable[str],
    year: int,
    month: Optional[int] = None,
    buffer_km: float = 5.0,
    common_grid_res: float = 500.0,
) -> pd.DataFrame:
    """Score every product against ONE reference event (already time-resolved).

    ``event_gdf`` holds the event's perimeter row(s) (its ``_event``/``_year``
    columns are carried into the output if present). A buffered window grid is
    built around the event, the perimeter is rasterised to the reference mask,
    each product is loaded on that grid for ``year`` (``month`` optional), and
    :func:`confusion_scores` is computed per product.

    Returns one row per product present for the period; empty if no product has
    data (e.g. a pre-2012 event with only VIIRS requested).
    """
    products = list(products)
    geom_5070 = event_gdf.to_crs(ANALYSIS_CRS).geometry.union_all()
    window = _window_gdf(geom_5070, buffer_km * 1000.0)
    preloaded = _preload_products(products, year, month, window)
    return _score_event(event_gdf, preloaded, year, month, buffer_km, common_grid_res)


def validate_products(
    reference: str,
    aoi: gpd.GeoDataFrame,
    products: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    match: str = "year",
    buffer_km: float = 5.0,
    common_grid_res: float = 500.0,
    min_acres: float = 0.0,
) -> pd.DataFrame:
    """Validate products against every event of a reference source within ``aoi``.

    Loads the reference (:func:`reference_sources.load_reference`) and scores every
    event, time-matched to its own year (and month, if ``match="month"`` and the
    source carries a date). Events are grouped by period so each product is read
    from disk **once per period** rather than once per event -- so a source with
    hundreds of small fires in a few years (TNC/NCWRC) stays fast. The per-event
    result is identical to :func:`validate_event` (same window grid and metrics).

    Parameters
    ----------
    reference : str
        A key of :data:`reference_sources.REFERENCE_SOURCES` (e.g. ``"NIFC_IFPH"``).
    products : iterable of str, optional
        Defaults to every registered burned-area product plus ``VIIRS``. Severity
        products are skipped (perimeters carry no severity -- use the cross-comparison
        in :mod:`fire_comparison`, restricted to reference cells, for those).
    years : iterable of int, optional
        Restrict to events in these years; default uses every event's own year.
    match : str
        ``"year"`` (default) matches each event against the product's annual mask
        (forgiving of within-year timing); ``"month"`` matches by the event's
        month when the source supplies a date.
    min_acres : float
        Drop reference events smaller than this reported acreage (NaN acreage is
        kept). Useful to focus on mappable incidents.

    Returns
    -------
    DataFrame
        Tidy, one row per (event, product): ``recall``, ``precision``,
        ``detected``, cell counts, year/month. Empty if the reference is absent
        or no events match. Feed to :func:`summarize_validation` for per-product
        roll-ups.
    """
    if products is None:
        products = list_products("burned_area") + list_products("occurrence")
    products = list(products)

    ref = load_reference(reference, aoi)
    if ref is None:
        return pd.DataFrame()

    if min_acres:
        keep = ref["_area"].isna() | (ref["_area"] >= min_acres)
        ref = ref[keep]
    ref = ref[ref["_year"].notna()]
    if years is not None:
        years = set(int(y) for y in years)
        ref = ref[ref["_year"].astype(int).isin(years)]
    if ref.empty:
        return pd.DataFrame()

    # Group events by their matched period and load each product ONCE per period
    # (clipped to the buffered bounding box of that period's events), then score
    # every event in the period from the in-memory arrays. This turns disk reads
    # from O(events) into O(periods) -- the big win when a source has many small
    # fires clustered in a handful of years (TNC/NCWRC). Results are identical to
    # scoring each event independently (same window grid and metrics).
    pkey_year = ref["_year"].astype(int)
    if match == "month":
        pkey_month = ref["_month"].apply(lambda m: int(m) if pd.notna(m) else None)
    else:
        pkey_month = pd.Series([None] * len(ref), index=ref.index, dtype="object")
    ref = ref.assign(_pkey_year=pkey_year.values, _pkey_month=pkey_month.values)

    frames = []
    for (yr, mo), group in ref.groupby(
        ["_pkey_year", "_pkey_month"], dropna=False, sort=True
    ):
        yr = int(yr)
        mo = int(mo) if (mo is not None and pd.notna(mo)) else None
        # one clip per period: the buffered bbox covering all of its events
        region_geom = (
            group.to_crs(ANALYSIS_CRS).buffer(buffer_km * 1000.0).union_all().envelope
        )
        region = gpd.GeoDataFrame(geometry=[region_geom], crs=ANALYSIS_CRS)
        preloaded = _preload_products(products, yr, mo, region)
        for idx in group.index:
            frame = _score_event(
                group.loc[[idx]], preloaded, yr, mo, buffer_km, common_grid_res
            )
            if not frame.empty:
                frame.insert(0, "source", reference)
                frames.append(frame)
        del preloaded  # release this period's rasters before loading the next

    return (
        pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    )


# ---------------------------------------------------------------------------
# Roll-ups
# ---------------------------------------------------------------------------
def summarize_validation(detail: pd.DataFrame) -> pd.DataFrame:
    """Per-product roll-up of an event-level validation table.

    Two complementary recalls, because pooling matters:

    * ``recall_pooled`` -- ``sum(tp)/sum(n_ref)`` over all events (cell-weighted;
      dominated by the big fires, which is right for "what fraction of *burned
      area* did the product see?").
    * ``recall_mean_event`` -- the simple mean of per-event recall (each fire
      counts once, regardless of size -- "how does the product do on a *typical*
      incident?", which surfaces small-fire omission).

    Also ``detection_rate`` (fraction of events the product caught at all) and
    the mean within-window precision (conditional -- see module docstring).
    """
    if detail.empty:
        return pd.DataFrame()
    g = detail.groupby("product", sort=False)
    out = pd.DataFrame(
        {
            "n_events": g.size(),
            "recall_pooled": g["tp"].sum() / g["n_ref"].sum(),
            "recall_mean_event": g["recall"].mean(),
            "detection_rate": g["detected"].mean(),
            "precision_mean_event": g["precision"].mean(),
        }
    )
    return out.sort_values("recall_pooled", ascending=False)


def recall_matrix(detail: pd.DataFrame, value: str = "recall") -> pd.DataFrame:
    """Pivot the event-level table to an events x products matrix of ``value``.

    Handy for a heatmap (events down, products across) so per-incident strengths
    and weaknesses -- e.g. which product misses the escaped-Rx Dad Fire -- read
    off at a glance. ``value`` is any numeric column (``"recall"``,
    ``"precision"``, ``"detected"``, ...).
    """
    if detail.empty:
        return pd.DataFrame()
    return detail.pivot_table(
        index="event", columns="product", values=value, aggfunc="mean"
    )

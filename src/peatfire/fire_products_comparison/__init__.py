"""Compare fire data products (burned area, severity, occurrence) within an AOI.

Public API::

    from peatfire.fire_products_comparison import compare_fire_products
    from peatfire.fire_products_comparison import set_fire_style, plot_annual_series

Submodules are imported in dependency order (products -> comparison -> plotting).
"""

from .fire_products import (
    FIRE_PRODUCTS,
    ProductSpec,
    get_spec,
    list_products,
    load_standardized,
)
from .fire_comparison import (
    ANALYSIS_CRS,
    agreement_matrix,
    annual_burned_area_series,
    build_common_grid,
    burned_area_km2,
    compare_fire_products,
    period_totals_series,
    product_pair_scatter,
    rasterize_points_to_grid,
    rasterize_polygons_to_grid,
    rmse,
    stack_on_common_grid,
    to_common_grid,
    total_least_squares,
)
from .reference_sources import (
    REFERENCE_SOURCES,
    ReferenceSpec,
    get_reference,
    list_references,
    load_reference,
    reference_event_table,
)
from .validation import (
    confusion_scores,
    recall_matrix,
    summarize_validation,
    validate_event,
    validate_products,
)
from .plotting import (
    overlay_aoi_boundaries,
    plot_agreement_heatmap,
    plot_annual_series,
    plot_consensus_map,
    plot_event_perimeter,
    plot_overlay_map,
    plot_product_scatter,
    plot_raster_map,
    plot_reference_vs_product_for_one_event,
    plot_temporal_heatmap,
    plot_validation_heatmap,
    set_fire_style,
)

__all__ = [
    "FIRE_PRODUCTS",
    "ProductSpec",
    "get_spec",
    "list_products",
    "load_standardized",
    "ANALYSIS_CRS",
    "agreement_matrix",
    "annual_burned_area_series",
    "build_common_grid",
    "burned_area_km2",
    "compare_fire_products",
    "period_totals_series",
    "product_pair_scatter",
    "rasterize_points_to_grid",
    "rasterize_polygons_to_grid",
    "rmse",
    "stack_on_common_grid",
    "to_common_grid",
    "total_least_squares",
    "REFERENCE_SOURCES",
    "ReferenceSpec",
    "get_reference",
    "list_references",
    "load_reference",
    "reference_event_table",
    "confusion_scores",
    "recall_matrix",
    "summarize_validation",
    "validate_event",
    "validate_products",
    "overlay_aoi_boundaries",
    "plot_agreement_heatmap",
    "plot_annual_series",
    "plot_consensus_map",
    "plot_event_perimeter",
    "plot_overlay_map",
    "plot_product_scatter",
    "plot_raster_map",
    "plot_reference_vs_product_for_one_event",
    "plot_temporal_heatmap",
    "plot_validation_heatmap",
    "set_fire_style",
]

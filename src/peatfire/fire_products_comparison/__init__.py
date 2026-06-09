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
    rmse,
    stack_on_common_grid,
    total_least_squares,
)
from .plotting import (
    plot_agreement_heatmap,
    plot_annual_series,
    plot_overlay_map,
    plot_product_scatter,
    plot_temporal_heatmap,
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
    "rmse",
    "stack_on_common_grid",
    "total_least_squares",
    "plot_agreement_heatmap",
    "plot_annual_series",
    "plot_overlay_map",
    "plot_product_scatter",
    "plot_temporal_heatmap",
    "set_fire_style",
]

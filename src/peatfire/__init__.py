"""Importable helpers for the NC peatland fire project.

Re-exports the data-loading helpers and the fire-product comparison API so
notebooks and scripts can do::

    from peatfire import data_path, load_csv
    from peatfire import compare_fire_products, set_fire_style

The geospatial comparison API (``compare_fire_products`` and friends) requires
the optional geospatial stack in ``environment.yml`` (geopandas, rioxarray,
rasterio, matplotlib, ...). It is imported lazily below so that importing
``peatfire`` for the lightweight ``data_loading`` helpers still works in an
environment without those packages installed.
"""

from .preproc.data_loading import (
    DATA_DIR,
    PROJECT_ROOT,
    data_path,
    get_key,
    load_csv,
    load_env,
)

__all__ = [
    "DATA_DIR",
    "PROJECT_ROOT",
    "data_path",
    "get_key",
    "load_csv",
    "load_env",
]

try:  # optional geospatial API
    from .fire_products_comparison.fire_products import (
        FIRE_PRODUCTS,
        ProductSpec,
        get_spec,
        list_products,
        load_standardized,
    )
    from .fire_products_comparison.fire_comparison import (
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
        rasterize_polygons_to_grid
    )
    from .fire_products_comparison.reference_sources import (
        REFERENCE_SOURCES,
        get_reference,
        list_references,
        load_reference,
        reference_event_table,
    )
    from .fire_products_comparison.validation import (
        confusion_scores,
        recall_matrix,
        summarize_validation,
        validate_event,
        validate_products,
    )
    from .fire_products_comparison.plotting import (
        plot_agreement_heatmap,
        plot_annual_series,
        plot_overlay_map,
        plot_product_scatter,
        plot_temporal_heatmap,
        plot_validation_heatmap,
        set_fire_style,
    )

    __all__ += [
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
        "rasterize_polygons_to_grid",
        "REFERENCE_SOURCES",
        "get_reference",
        "list_references",
        "load_reference",
        "reference_event_table",
        "confusion_scores",
        "recall_matrix",
        "summarize_validation",
        "validate_event",
        "validate_products",
        "plot_agreement_heatmap",
        "plot_annual_series",
        "plot_overlay_map",
        "plot_product_scatter",
        "plot_temporal_heatmap",
        "plot_validation_heatmap",
        "set_fire_style",
    ]
except ImportError:
    # Geospatial dependencies not installed; the data_loading helpers above
    # remain available.
    pass

try:  # optional modeling API (peat condition -> fire); see modeling_roadmap.md
    from .modeling import (
        COVARIATES,
        available_covariates,
        build_frame,
        covariate_on_grid,
        fit_logit_clustered,
        fit_mixed_logit,
        list_covariates,
        load_covariate,
        load_restoration_sites,
        odds_ratios,
    )

    __all__ += [
        "COVARIATES",
        "available_covariates",
        "build_frame",
        "covariate_on_grid",
        "fit_logit_clustered",
        "fit_mixed_logit",
        "list_covariates",
        "load_covariate",
        "load_restoration_sites",
        "odds_ratios",
    ]
except ImportError:
    # Modeling dependencies (geopandas/statsmodels) not installed.
    pass

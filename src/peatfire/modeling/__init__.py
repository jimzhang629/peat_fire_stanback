"""Modeling pipeline: peat condition -> fire.

Turns the chosen fire product (FireCCIS311) plus environmental covariates and
peat condition/restoration/management into a tidy pixel-year frame and fitted
models. See ``modeling_roadmap.md`` for the design.

Three layers, mirroring the fire-comparison toolkit:

* :mod:`covariates` -- a ``CovariateSpec`` registry + standardized loaders that
  warp each environmental layer onto the shared EPSG:5070 grid.
* :mod:`frame` -- assembles the tidy ``[unit, year, covariates..., burned]``
  table from an upstream-matched set of treated + control units.
* :mod:`models` -- cluster-robust / mixed logistic fits, reported as odds ratios.
* :mod:`did` -- staggered difference-in-differences (Callaway & Sant'Anna, as in
  Castro et al. 2026): match, then identify off the pre/post change so
  time-invariant confounders difference out. An alternative estimator on the same
  ``build_frame`` output.

The upstream **matching** step (choosing control pixels for each restoration
site) is deliberately left out of this package so the causal design stays
explicit; ``build_frame`` consumes its output.
"""

from .covariates import (
    COVARIATES,
    TEMPORAL_COVARIATES,
    CovariateSpec,
    available_covariates,
    available_temporal_covariates,
    covariate_on_grid,
    get_covariate,
    get_temporal_covariate,
    list_covariates,
    list_temporal_covariates,
    load_covariate,
    load_temporal_covariate,
    temporal_covariate_on_grid,
)
from .frame import (
    build_frame,
    load_completed_restoration_sites_in_analysis_crs,
)
from .models import (
    fit_logit_clustered,
    fit_mixed_logit,
    odds_ratios,
)
from .did import (  # staggered DiD alternative (Castro et al. 2026)
    add_fire_lags,
    aggregate_att,
    attach_cohort,
    avoided_area,
    build_panel,
    estimate_att,
)
from .climate import (  # GHCN station points -> gridded climate covariates
    build_annual_climate,
    build_climate_normals,
    idw_to_grid,
    load_ghcn_stations,
    station_normals,
    write_annual_climate,
    write_climate_normals,
)
from .soil import (  # SSURGO soil polygons -> gridded soil covariates
    build_soil_rasters,
    inspect_soil_columns,
    mukey_attribute,
)
from .plotting import (  # step-by-step data-inspection diagnostics
    plot_covariate_maps,
    plot_covariate_pairs,
    plot_covariate_space,
    plot_matched_pairs_covariate,
    plot_matched_pairs_geographic,
)
from .matching import (  # assignment scaffold (Part A stubs) + test harness (Part B)
    assemble_units,
    attach_covariates,
    balance_table,
    build_candidate_pool,
    check_balance,
    check_candidate_pool,
    check_covariates,
    check_matches,
    check_pixels,
    check_treated_units,
    load_treated_units,
    match_controls,
    pixelate,
    plot_balance,
    standardized_mean_diff,
)

__all__ = [
    "COVARIATES",
    "TEMPORAL_COVARIATES",
    "CovariateSpec",
    "available_covariates",
    "available_temporal_covariates",
    "covariate_on_grid",
    "get_covariate",
    "get_temporal_covariate",
    "list_covariates",
    "list_temporal_covariates",
    "load_covariate",
    "load_temporal_covariate",
    "temporal_covariate_on_grid",
    "build_frame",
    "load_completed_restoration_sites_in_analysis_crs",
    "build_annual_climate",
    "build_climate_normals",
    "idw_to_grid",
    "load_ghcn_stations",
    "station_normals",
    "write_annual_climate",
    "write_climate_normals",
    "build_soil_rasters",
    "inspect_soil_columns",
    "mukey_attribute",
    "plot_covariate_maps",
    "plot_covariate_pairs",
    "plot_covariate_space",
    "plot_matched_pairs_covariate",
    "plot_matched_pairs_geographic",
    "fit_logit_clustered",
    "fit_mixed_logit",
    "odds_ratios",
    "add_fire_lags",
    "aggregate_att",
    "attach_cohort",
    "avoided_area",
    "build_panel",
    "estimate_att",
    "assemble_units",
    "attach_covariates",
    "balance_table",
    "build_candidate_pool",
    "check_balance",
    "check_candidate_pool",
    "check_covariates",
    "check_matches",
    "check_pixels",
    "check_treated_units",
    "load_treated_units",
    "match_controls",
    "pixelate",
    "plot_balance",
    "standardized_mean_diff",
]

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

The upstream **matching** step (choosing control pixels for each restoration
site) is deliberately left out of this package so the causal design stays
explicit; ``build_frame`` consumes its output.
"""

from .covariates import (
    COVARIATES,
    CovariateSpec,
    available_covariates,
    covariate_on_grid,
    get_covariate,
    list_covariates,
    load_covariate,
)
from .frame import (
    build_frame,
    build_modeling_grid,
    load_restoration_sites,
)
from .models import (
    fit_logit_clustered,
    fit_mixed_logit,
    odds_ratios,
)

__all__ = [
    "COVARIATES",
    "CovariateSpec",
    "available_covariates",
    "covariate_on_grid",
    "get_covariate",
    "list_covariates",
    "load_covariate",
    "build_frame",
    "build_modeling_grid",
    "load_restoration_sites",
    "fit_logit_clustered",
    "fit_mixed_logit",
    "odds_ratios",
]

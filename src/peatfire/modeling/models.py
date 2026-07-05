"""Fit the peat-condition -> fire models on the tidy frame.

Consumes the table from :mod:`peatfire.modeling.frame` and fits the sequence in
``modeling_roadmap.md``, from the honest version of the naive pixel GLM up to a
mixed model:

1. :func:`fit_logit_clustered` -- logistic ``burned ~ treated + covariates`` with
   **standard errors clustered by site**, so pixels within a restoration site are
   not treated as independent (the effective N is sites, not pixels).
2. :func:`fit_mixed_logit` -- a logistic GLMM with site/year random intercepts
   (optional dependency; see the docstring).

Results are reported as **odds ratios with confidence intervals**
(:func:`odds_ratios`): ``exp(beta)``, the multiplicative effect on the odds of a
pixel-year burning. Because fire is rare here, an odds ratio ~ a risk ratio.

``statsmodels`` (and, for the mixed model, ``pymer4``/an R backend) are imported
*inside* the functions so importing this module stays cheap and dependency-free.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _formula(response: str, treatment: str, covariates: Sequence[str]) -> str:
    rhs = " + ".join([treatment, *covariates]) if covariates else treatment
    return f"{response} ~ {rhs}"


def fit_logit_clustered(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    response: str = "burned",
    treatment: str = "treated",
    cluster: str = "site_id",
    formula: Optional[str] = None,
):
    """Fit logistic ``response ~ treatment + covariates`` with cluster-robust SEs.

    This is the trustworthy version of "regress burned on the covariates and read
    the betas": same point estimates as an ordinary logit, but the standard
    errors are clustered on ``cluster`` (the matching stratum / restoration site),
    which corrects the wildly overstated significance you get from treating every
    250-300 m pixel as an independent observation.

    Parameters
    ----------
    frame : DataFrame
        Output of :func:`peatfire.modeling.build_frame`.
    covariates : sequence of str
        Adjustment covariates (elevation, histosol %, ... -- match what you have).
    response, treatment, cluster : str
        Column names for the 0/1 outcome, the treatment indicator, and the
        clustering key.
    formula : str, optional
        Override the auto-built ``response ~ treatment + covariates`` formula (e.g.
        to add an interaction like ``treated:precip``).

    Returns
    -------
    statsmodels results
        Call :func:`odds_ratios` on it for the interpretable table.
    """
    import statsmodels.formula.api as smf

    formula = formula or _formula(response, treatment, covariates)
    model = smf.logit(formula, data=frame)
    return model.fit(
        disp=False,
        cov_type="cluster",
        cov_kwds={"groups": frame[cluster]},
    )


def fit_mixed_logit(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    response: str = "burned",
    treatment: str = "treatment",
    groups: str = "site_id",
    formula: Optional[str] = None,
):
    """Fit a logistic GLMM with a random intercept per ``groups`` (site).

    Random intercepts absorb site-level autocorrelation directly rather than only
    correcting the SEs. Uses ``statsmodels``'
    :class:`~statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` (a
    variational-Bayes binomial mixed model), which is the pure-Python option;
    for a year random effect too, or lme4-style REML, fit through ``pymer4``
    (an R/lme4 bridge) with ``(1|site_id) + (1|year)`` instead.

    Returns the fitted results object; read fixed-effect odds ratios off its
    ``fe_mean`` / summary.
    """
    import statsmodels.formula.api as smf

    vc_formula = "0 + C(%s)" % groups  # random intercept per site
    formula = formula or _formula(response, treatment, covariates)
    model = smf.BinomialBayesMixedGLM.from_formula(
        formula, {"site": vc_formula}, data=frame
    )
    return model.fit_vb()


def odds_ratios(result, alpha: float = 0.05) -> pd.DataFrame:
    """Turn a fitted logit into an odds-ratio table (the headline output).

    Returns one row per term with ``odds_ratio = exp(beta)`` and its
    ``(1-alpha)`` confidence interval, plus the p-value. An odds ratio of 1 means
    no effect; ``treated`` below 1 means restoration lowers the odds of fire. The
    CI is what decides significance -- and it is only honest because the fit used
    clustered SEs / random effects (see :func:`fit_logit_clustered`).
    """
    params = result.params
    conf = result.conf_int(alpha=alpha)
    conf.columns = ["ci_low", "ci_high"]
    out = pd.DataFrame(
        {
            "beta": params,
            "odds_ratio": np.exp(params),
            "or_ci_low": np.exp(conf["ci_low"]),
            "or_ci_high": np.exp(conf["ci_high"]),
        }
    )
    if hasattr(result, "pvalues"):
        out["p_value"] = result.pvalues
    return out

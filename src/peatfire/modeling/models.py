"""Fit the peat-condition -> fire models on the tidy frame.

Consumes the table from :mod:`peatfire.modeling.frame` and fits the sequence in
``modeling_notebook_explained.md``, from the honest version of the naive pixel
GLM up to a mixed model:

1. :func:`fit_logit_clustered` -- logistic ``burned ~ treated + covariates`` with
   **standard errors clustered by site**, so pixels within a restoration site are
   not treated as independent (the effective N is sites, not pixels).
2. :func:`fit_ols_clustered` -- the **linear** counterpart (a linear probability
   model on the 0/1 outcome, or an OLS on a continuous burned fraction/area), same
   site-clustered SEs. Read on the probability scale, and the fit to use when
   swapping ``drainage`` in as the independent variable.
3. :func:`fit_mixed_logit` -- a logistic GLMM with site/year random intercepts
   (optional dependency; see the docstring).

:func:`fit_drainage_models` is a convenience that runs drainage as the independent
variable **alone** and **with treated**, and :func:`drainage_effect_table`
tabulates the drainage (or treated) estimate across those specifications.

Logistic results are reported as **odds ratios with confidence intervals**
(:func:`odds_ratios`): ``exp(beta)``, the multiplicative effect on the odds of a
pixel-year burning. Because fire is rare here, an odds ratio ~ a risk ratio. Linear
results are reported as **raw coefficients with CIs** (:func:`coefficients`) --
already on the outcome (probability) scale, so they are *not* exponentiated.

``statsmodels`` (and, for the mixed model, ``pymer4``/an R backend) are imported
*inside* the functions so importing this module stays cheap and dependency-free.
"""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def _formula(response: str, treatment: str, covariates: Sequence[str]) -> str:
    rhs = " + ".join([treatment, *covariates]) if covariates else treatment
    return f"{response} ~ {rhs}"


def _standardize_predictors(
    clean: pd.DataFrame,
    cols: Sequence[str],
    response: str,
    treatment: str,
    cluster: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Center-and-scale the continuous predictors (on a copy) for stability.

    The logit is fit by Newton/IRLS, whose Hessian is ``X'WX``; the fit inverts it
    (``np.linalg.inv(-Hessian)``). When the columns of ``X`` live on wildly
    different scales -- elevation in metres (~1e1), ``precip_normal`` in mm (~1e3),
    temperatures in degrees C -- that matrix is badly conditioned, its inverse
    blows up into a spurious ``LinAlgError: Singular matrix``, and the extreme
    linear predictor trips the ``overflow in exp`` / ``divide by zero in log``
    warnings you see just before it. Centering each continuous covariate to mean 0
    and scaling to SD 1 fixes the conditioning without changing the model: the
    ``treated`` odds ratio (the headline) is left exactly as-is because ``treated``
    stays raw 0/1, and each covariate simply reports a *per-standard-deviation*
    odds ratio. It does **not** cure genuine collinearity or a constant column --
    those are caught by :func:`_rank_deficiency_report` with an actionable error.

    Returns ``(scaled_copy, names_scaled)``. Raises when a covariate has zero
    variance over the rows being fit (a constant column carries no information and
    on its own makes the design singular).
    """
    skip = {response, treatment, cluster}
    out = clean.copy()
    scaled: list[str] = []
    constant: list[str] = []
    for c in cols:
        if c in skip or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        sd = out[c].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:  # zero-variance: a constant column
            constant.append(c)
            continue
        if out[c].nunique() <= 2:  # binary / indicator: leave on its native scale
            continue
        out[c] = (out[c] - out[c].mean()) / sd
        scaled.append(c)
    if constant:
        raise ValueError(
            f"Cannot fit: covariate(s) {constant} are constant over the "
            f"{len(out)} rows being fit, so they add a column with no variation, "
            "carry no information, and make the design matrix singular. This "
            "usually means the layer is flat over the matched set (e.g. the "
            "near-constant histosol % once you subset to peat) or is misaligned to "
            "the grid -- drop it from `covariates`."
        )
    return out, scaled


def _rank_deficiency_report(formula: str, clean: pd.DataFrame) -> Optional[str]:
    """Actionable message if the model's design matrix is rank-deficient, else None.

    Perfectly (or near-perfectly) collinear predictors -- e.g. ``tmax_normal`` and
    ``tmin_normal``, two tightly coupled climate normals -- leave ``X'X`` singular
    no matter how well the columns are scaled, so the fit either crashes with
    "Singular matrix" or, worse, silently returns meaningless coefficients. We
    catch it up front and name the redundant columns (with their closest partner)
    so the caller knows exactly which covariate to drop.
    """
    try:
        import patsy
        from scipy import linalg
    except Exception:  # pragma: no cover - diagnostics are best-effort
        return None
    try:
        _y, X = patsy.dmatrices(formula, clean, return_type="dataframe")
    except Exception:  # pragma: no cover - let the real fit surface formula errors
        return None
    Xv = np.asarray(X, dtype=float)
    rank = np.linalg.matrix_rank(Xv)
    if rank >= Xv.shape[1]:
        return None
    # QR with column pivoting orders columns by how much independent information
    # each adds; everything past `rank` is linearly explained by the earlier ones.
    _q, _r, piv = linalg.qr(Xv, mode="economic", pivoting=True)
    redundant = [str(X.columns[i]) for i in piv[rank:]]
    non_intercept = [c for c in X.columns if c != "Intercept"]
    corr = X[non_intercept].corr().abs() if len(non_intercept) > 1 else None
    notes = []
    for r in redundant:
        if corr is not None and r in corr.columns:
            partners = corr[r].drop(index=r, errors="ignore")
            if len(partners) and np.isfinite(partners.max()):
                j = partners.idxmax()
                notes.append(f"{r} (|r|={partners[j]:.3f} with {j})")
                continue
        notes.append(r)
    return (
        "the design matrix is rank-deficient: these column(s) are (near-)perfectly "
        f"collinear and one of each redundant set must be dropped -> {'; '.join(notes)}. "
        "Tightly coupled normals such as tmax_normal and tmin_normal are the usual "
        "culprit; keep one, or replace the pair with their mean/difference."
    )


def _separation_report(formula: str, clean: pd.DataFrame) -> Optional[str]:
    """Actionable message if a covariate perfectly separates the outcome, else None.

    Fire is rare here, which makes (quasi-)complete separation easy: if some
    covariate's values for the burned pixels never overlap the unburned ones, the
    corresponding logit coefficient runs off to +/-inf and the Hessian degenerates.
    We report which covariate does it so the caller can drop it or switch to a
    penalized (Firth/ridge) fit.
    """
    try:
        import patsy
    except Exception:  # pragma: no cover
        return None
    try:
        y, X = patsy.dmatrices(formula, clean, return_type="dataframe")
    except Exception:  # pragma: no cover
        return None
    yv = np.asarray(y, dtype=float).ravel()
    if not set(np.unique(yv)) <= {0.0, 1.0} or not 0 < yv.sum() < len(yv):
        return None
    separators = []
    for c in X.columns:
        if c == "Intercept":
            continue
        x = np.asarray(X[c], dtype=float)
        x0, x1 = x[yv == 0], x[yv == 1]
        if x0.size and x1.size and (x0.min() > x1.max() or x1.min() > x0.max()):
            separators.append(str(c))
    if not separators:
        return None
    return (
        f"covariate(s) {separators} perfectly separate burned from unburned pixels, "
        "so their logit coefficient diverges (fire is rare, which makes this easy). "
        "Drop the covariate or fit a penalized/Firth logit."
    )


def _model_columns(
    frame: pd.DataFrame,
    response: str,
    treatment: str,
    covariates: Sequence[str],
    cluster: str,
    formula: Optional[str],
) -> list[str]:
    """Frame columns the fit depends on (for NaN-checking and group alignment).

    For the auto-built formula this is exact; for a custom ``formula`` it is a
    best-effort match of frame columns named as whole-word tokens in the formula
    (enough to catch the NaN-drop / cluster-misalignment failure modes). The
    ``cluster`` column is always included since its groups must line up with the
    design matrix.
    """
    if formula is None:
        cols = [response, treatment, *covariates]
    else:
        tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula))
        cols = [c for c in frame.columns if c in tokens]
    # dedupe, preserve order, and always keep the clustering key
    return list(dict.fromkeys([*cols, cluster]))


def _prepare_design(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    response: str,
    treatment: str,
    cluster: str,
    formula: Optional[str],
    standardize: bool,
) -> tuple[pd.DataFrame, str]:
    """Shared pre-fit preparation for the clustered logit / OLS fits.

    Both :func:`fit_logit_clustered` and :func:`fit_ols_clustered` need the same
    guardrails before handing a design matrix to ``statsmodels``: build the
    ``response ~ treatment + covariates`` formula (unless overridden), verify the
    required columns exist, drop NaN rows *ourselves* (so the cluster groups stay
    aligned with the thinned design and we can raise an actionable error instead of
    numpy's opaque "zero-size array"), optionally center-and-scale the continuous
    predictors for numerical conditioning, and reject a rank-deficient design up
    front with a message naming the collinear columns. Returns ``(clean, formula)``
    ready to fit; the estimator-specific failure modes (logit separation) stay in
    the caller's ``try/except``.
    """
    formula = formula or _formula(response, treatment, covariates)

    cols = _model_columns(frame, response, treatment, covariates, cluster, formula)
    missing_cols = [c for c in cols if c not in frame.columns]
    if missing_cols:
        raise ValueError(
            f"frame is missing columns required by the fit: {missing_cols}. "
            f"Available columns: {list(frame.columns)}."
        )
    if frame.empty:
        raise ValueError(
            "frame has no rows to fit -- build_frame produced an empty table "
            "(no units, or the fire product had no coverage for these years)."
        )

    clean = frame.dropna(subset=cols)
    if clean.empty:
        all_nan = [c for c in cols if frame[c].isna().all()]
        detail = (
            f" Columns entirely NaN over the frame: {all_nan}."
            if all_nan
            else " No single column is all-NaN, so different rows are missing "
            "different fields; check the covariate alignment/coverage."
        )
        raise ValueError(
            f"No complete rows to fit {formula!r}: all {len(frame)} rows have a "
            f"NaN in one of {cols}." + detail + " A covariate that does not cover "
            "the study area (or is misaligned to the grid) will null every pixel "
            "and wipe out the fit -- drop it from `covariates` or fix its layer."
        )

    if standardize:
        clean, _ = _standardize_predictors(clean, cols, response, treatment, cluster)
    rank_problem = _rank_deficiency_report(formula, clean)
    if rank_problem is not None:
        raise ValueError(f"Cannot fit {formula!r}: {rank_problem}")
    return clean, formula


def fit_logit_clustered(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    response: str = "burned",
    treatment: str = "treated",
    cluster: str = "site_id",
    formula: Optional[str] = None,
    standardize: bool = True,
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
    standardize : bool, default True
        Center-and-scale the continuous covariates to mean 0 / SD 1 before fitting.
        This is a pure numerical-conditioning step: covariates on wildly different
        scales (elevation in metres vs ``precip_normal`` in mm) otherwise make the
        Hessian ill-conditioned and the fit dies with ``LinAlgError: Singular
        matrix`` (preceded by ``overflow in exp`` warnings). It does not change the
        ``treated`` odds ratio -- ``treated`` stays raw 0/1 -- but each adjusted
        covariate then reports a *per-standard-deviation* odds ratio. Turn it off
        only if you need coefficients in the covariates' native units.

    Returns
    -------
    statsmodels results
        Call :func:`odds_ratios` on it for the interpretable table.

    Raises
    ------
    ValueError
        With an actionable message when the model cannot be identified: a constant
        covariate, (near-)perfectly collinear covariates (e.g. ``tmax_normal`` /
        ``tmin_normal``), or a covariate that perfectly separates the outcome --
        the structural reasons behind an otherwise opaque "Singular matrix".
    """
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    # Build the formula, drop NaN rows ourselves (so the cluster groups stay
    # aligned and an empty design raises an actionable error rather than numpy's
    # opaque "zero-size array"), standardize for conditioning, and reject a
    # rank-deficient design up front. Shared with fit_ols_clustered.
    clean, formula = _prepare_design(
        frame, covariates, response, treatment, cluster, formula, standardize
    )

    model = smf.logit(formula, data=clean)
    try:
        return model.fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": clean[cluster]},
        )
    except (np.linalg.LinAlgError, PerfectSeparationError) as err:
        # The MLE / its Hessian does not exist. We already ruled out bad scaling
        # (standardized) and rank deficiency (checked above), so the remaining
        # structural cause is usually separation -- diagnose it if we can.
        sep = _separation_report(formula, clean)
        detail = f" Likely cause: {sep}" if sep else (
            " Standardizing and the rank check rule out scaling and collinearity, "
            "so this is a degenerate fit -- too few burned pixels for the number "
            "of covariates, or (quasi-)separation. Try fewer covariates."
        )
        raise ValueError(
            f"Clustered logit for {formula!r} failed with a singular/degenerate "
            f"Hessian ({type(err).__name__})." + detail
        ) from err


def fit_ols_clustered(
    frame: pd.DataFrame,
    covariates: Sequence[str],
    response: str = "burned",
    treatment: str = "treated",
    cluster: str = "site_id",
    formula: Optional[str] = None,
    standardize: bool = True,
):
    """Fit **linear** ``response ~ treatment + covariates`` with cluster-robust SEs.

    The linear-regression counterpart of :func:`fit_logit_clustered`, added so the
    peat-condition -> fire relationship can be read on the *probability* scale
    directly instead of the odds scale. With the 0/1 ``burned`` response this is a
    **linear probability model**: each coefficient is the change in the probability
    of a pixel-year burning per unit of the predictor (per SD for a standardized
    continuous covariate, or the raw 0/1 contrast for ``treated``), which is often
    the easier number to talk about than an odds ratio -- and, unlike the logit, it
    stays estimable when a covariate perfectly separates the (rare) fires. It is
    also the fit to use when the response is a **continuous** outcome (e.g. a
    per-unit burned *fraction* / area rather than the binary flag), the linear
    regression the project notes call for when swapping drainage in as the
    independent variable.

    Same cluster-robust story as the logit: point estimates are ordinary OLS, but
    the standard errors are clustered on ``cluster`` (the restoration site /
    matching stratum) so pixels within a site are not counted as independent. All
    the pre-fit guardrails are shared with :func:`fit_logit_clustered` via
    :func:`_prepare_design` (NaN handling, per-SD standardization for conditioning,
    and an actionable rank-deficiency error). ``treatment`` is just the primary
    regressor of interest and can be any column -- pass ``treatment="drainage"`` to
    read drainage's association with fire, with ``treated`` moved into
    ``covariates`` (or use :func:`fit_drainage_models`, which wires that up).

    Read the result with :func:`coefficients` (raw betas + CIs), **not**
    :func:`odds_ratios`: an OLS coefficient is already on the outcome scale and must
    not be exponentiated.

    Returns
    -------
    statsmodels results
        An OLS results object with cluster-robust (``cov_type="cluster"``) SEs.
    """
    import statsmodels.formula.api as smf

    clean, formula = _prepare_design(
        frame, covariates, response, treatment, cluster, formula, standardize
    )
    model = smf.ols(formula, data=clean)
    return model.fit(cov_type="cluster", cov_kwds={"groups": clean[cluster]})


def fit_drainage_models(
    frame: pd.DataFrame,
    covariates: Sequence[str] = (),
    response: str = "burned",
    drainage: str = "drainage",
    treatment: str = "treated",
    cluster: str = "site_id",
    kind: str = "ols",
    **fit_kwargs,
):
    """Fit drainage as the independent variable -- **alone** and **with treated**.

    A thin convenience over :func:`fit_ols_clustered` /
    :func:`fit_logit_clustered` for the project note "swap in drainage as the
    independent variable rather than treated". It fits two nested specifications and
    returns them together so the drainage effect can be read before and after
    adjusting for restoration:

    * ``"drainage_alone"`` -- ``response ~ drainage (+ covariates)``: drainage's raw
      association with fire, with no treatment term.
    * ``"drainage_and_treated"`` -- ``response ~ drainage + treated (+ covariates)``:
      the same, now holding restoration status fixed. Comparing the drainage
      coefficient across the two says how much of drainage's signal is distinct from
      the treatment itself (rewetting *is* a drainage intervention, so the two are
      entangled by design).

    In both, ``drainage`` is the ``treatment`` (primary regressor) passed to the
    underlying fitter, and any ``drainage``/``treatment`` names are stripped from
    ``covariates`` so they are never duplicated on the right-hand side.

    Parameters
    ----------
    covariates : sequence of str
        Adjustment covariates to include in *both* specifications (e.g. elevation,
        climate normals). ``drainage`` and ``treatment`` are removed if present.
    kind : {"ols", "logit"}
        ``"ols"`` fits the linear probability model (:func:`fit_ols_clustered`,
        read with :func:`coefficients`); ``"logit"`` fits the clustered logistic
        model (:func:`fit_logit_clustered`, read with :func:`odds_ratios`).
    **fit_kwargs
        Forwarded to the underlying fitter (e.g. ``standardize=False``).

    Returns
    -------
    dict[str, statsmodels results]
        ``{"drainage_alone": ..., "drainage_and_treated": ...}``. Pass either value
        (or the whole dict, via :func:`drainage_effect_table`) to the matching
        reporter for ``kind``.
    """
    fit = {"ols": fit_ols_clustered, "logit": fit_logit_clustered}.get(kind)
    if fit is None:
        raise ValueError(f"kind must be 'ols' or 'logit', got {kind!r}.")
    covs = [c for c in covariates if c not in (drainage, treatment)]
    return {
        "drainage_alone": fit(
            frame, covariates=covs, response=response, treatment=drainage,
            cluster=cluster, **fit_kwargs,
        ),
        "drainage_and_treated": fit(
            frame, covariates=[treatment, *covs], response=response,
            treatment=drainage, cluster=cluster, **fit_kwargs,
        ),
    }


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


def coefficients(result, alpha: float = 0.05) -> pd.DataFrame:
    """Turn a fitted linear model into a raw-coefficient table (the OLS headline).

    The linear-scale counterpart of :func:`odds_ratios`, for reading a
    :func:`fit_ols_clustered` fit. An OLS coefficient is already on the outcome
    scale -- with the 0/1 ``burned`` response, the change in the *probability* of a
    pixel-year burning per unit of the predictor -- so this reports ``beta`` and its
    ``(1-alpha)`` confidence interval **as-is**, without the exponentiation
    :func:`odds_ratios` applies. A coefficient of 0 (CI straddling 0) means no
    effect; a negative ``treated`` (or ``drainage``) coefficient means that
    predictor lowers the probability of fire. As with the odds-ratio table, the CI
    is only honest because the fit clustered its SEs on the restoration site.

    Returns one row per term with ``beta``, ``ci_low``, ``ci_high`` and (when the
    fit exposes them) the standard error and p-value.
    """
    params = result.params
    conf = result.conf_int(alpha=alpha)
    conf.columns = ["ci_low", "ci_high"]
    out = pd.DataFrame(
        {
            "beta": params,
            "ci_low": conf["ci_low"],
            "ci_high": conf["ci_high"],
        }
    )
    if hasattr(result, "bse"):
        out["std_error"] = result.bse
    if hasattr(result, "pvalues"):
        out["p_value"] = result.pvalues
    return out


def drainage_effect_table(
    models: dict,
    term: str = "drainage",
    kind: str = "ols",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Pull one term's estimate across the specifications from :func:`fit_drainage_models`.

    Given the ``{"drainage_alone": ..., "drainage_and_treated": ...}`` dict, this
    extracts a single ``term`` (``"drainage"`` by default, or ``"treated"`` to see
    the treatment coefficient in the joint spec) and stacks it into one tidy table,
    one row per specification -- the at-a-glance answer to "does drainage's effect
    survive adjusting for restoration?". ``kind`` selects the reporter: ``"ols"``
    reports raw ``beta`` + CI via :func:`coefficients`, ``"logit"`` reports the
    ``odds_ratio`` + CI via :func:`odds_ratios`; it must match the ``kind`` the
    models were fit with. Specifications lacking ``term`` (e.g. ``"treated"`` in the
    drainage-alone spec) are skipped.

    Returns a DataFrame indexed by specification name.
    """
    report = {"ols": coefficients, "logit": odds_ratios}.get(kind)
    if report is None:
        raise ValueError(f"kind must be 'ols' or 'logit', got {kind!r}.")
    rows = []
    for spec, result in models.items():
        table = report(result, alpha=alpha)
        if term not in table.index:
            continue
        row = table.loc[term].to_dict()
        row["spec"] = spec
        row["n_terms"] = int(table.shape[0])
        rows.append(row)
    if not rows:
        raise ValueError(
            f"term {term!r} appears in none of the specifications {list(models)}."
        )
    return pd.DataFrame(rows).set_index("spec")

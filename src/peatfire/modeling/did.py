"""Staggered difference-in-differences: the Castro et al. (2026) estimator.

An **alternative to** :mod:`peatfire.modeling.models`. Where ``models.py`` matches
then fits a cluster-robust / mixed logistic on *levels* (an odds ratio for
``treated``), this module implements the causal design of

    Castro et al. (2026), "Effective restoration can avoid peatland fires:
    Large scale counterfactual assessment in Kalimantan, Indonesia", iScience,
    https://doi.org/10.1016/j.isci.2026.116041

which is our study transplanted to Indonesia: **match first, then estimate a
staggered difference-in-differences** (Callaway & Sant'Anna 2021) on the balanced
panel. The DiD is the stronger design because it identifies off the *change* in
burning after each site's restoration relative to not-yet/never-restored controls,
so every **time-invariant** confounder (persistent drainage, soil, access
differences we never measured) differences out -- not just the covariates we can
name and match on.

The two modules share the front half of the pipeline:

    matching.py  --->  build_frame()  --->  [ models.py    : levels / odds ratio ]
    (balance)          (pixel-year)         [ did.py       : staggered ATT       ]

What this module adds on top of ``build_frame``'s output:

* :func:`attach_cohort` -- tag every unit with its **treatment group** ``g`` =
  first-treatment (restoration) year; never-treated controls get ``g = 0``. This
  is Castro's "construction vintage".
* :func:`add_fire_lags` -- the temporal lag (did this pixel burn last year?) and
  the 4-neighbour **spatial lag** (did an adjacent pixel burn this year?) that
  enter Castro's outcome equation (their Eq. 1).
* :func:`build_panel` -- reshape into the balanced ``(entity, year)`` panel the
  Callaway-Sant'Anna estimator expects.
* :func:`estimate_att` -- the **doubly-robust** group-time ATT, SEs clustered at
  the ``site`` (Castro: village) level.
* :func:`aggregate_att` -- collapse the group-time ATTs to an overall ATT and to a
  **dynamic / event-study** path (the pre-treatment coefficients are the
  parallel-trends check).
* :func:`tidy_att` / :func:`compare_cluster_levels` -- flatten an aggregation into
  plain columns, and stack the site- and pixel-clustered runs of the *same* ATT in
  one table so the design effect (how much the pixel-level SE was deflated) is a
  column rather than a claim.
* :func:`att_collapsed` -- the transparent, dependency-free cross-check: collapse
  each pixel to one pre/post change and regress it on ``treated``, with the same
  ``cluster_by`` toggle. Same estimand, arithmetic you can read off the page.
* :func:`avoided_area` -- Castro's headline: ATT x rewetted area = avoided
  burned area.

Clustering the standard errors
------------------------------
Treatment is assigned to a **restoration site**, not to a pixel: every pixel
inside one site shares the same canal blocks, the same weather, the same
hydrology, and so the same shocks. Treating the thousands of pixel-years as
independent draws understates the sampling variance by roughly the square root of
the design effect -- with a handful of sites the honest SE can be several times
the pixel-level one.

So :func:`estimate_att`, :func:`fit_att` and :func:`att_collapsed` all take
``cluster_by``:

* ``"site"`` (**default**) -- SEs clustered on ``cluster_col`` (``site_id``, the
  key :func:`peatfire.modeling.match_controls` stamps on each treated pixel *and*
  on the controls matched to it). The effective sample size is the number of
  sites.
* ``"pixel"`` -- the old behaviour: each pixel (panel entity) is its own cluster.
  Keep it only to *quantify* the deflation, not to report.

Both choices leave the point estimate untouched; only the variance changes.
Clustering a Callaway-Sant'Anna ATT above the entity level is only defined on the
**multiplier-bootstrap** path (the closed-form SEs take no cluster argument), so
``"site"`` costs a ``boot_iterations``-draw bootstrap where ``"pixel"`` is
instant. That is the whole price.

.. warning::
   Cluster-robust inference is asymptotic **in the number of clusters**. This
   study has ~6 restoration sites, far below the usual rule of thumb (30-50), so
   even site-clustered SEs are optimistic and the normal approximation is poor.
   :func:`att_collapsed` therefore reports a ``t(G-1)`` interval, and
   :func:`estimate_att` warns when the cluster count is small. Treat the interval
   as indicative and corroborate it with the event study.

Backend. Castro ran ``csdid`` in Stata. We stay Python-native with the
``differences`` package (a Callaway-Sant'Anna port); pass ``backend="rpy2"`` to
call the reference R ``did`` package instead. Both are optional and imported
*inside* :func:`estimate_att`, so importing this module stays cheap -- the same
convention ``models.py`` uses for ``statsmodels``/``pymer4``.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Castro et al. judge a match valid at |SMD| <= 0.2 and, crucially for DiD, define
# the treatment "group" as the year of canal-block construction. We mirror the
# never-treated sentinel the Callaway-Sant'Anna estimators expect.
NEVER_TREATED = 0

# Below this many clusters, cluster-robust asymptotics are unreliable (the usual
# applied rule of thumb is 30-50 clusters). We warn rather than refuse: with ~6
# restoration sites the site-clustered SE is still far closer to honest than the
# pixel-level one, the user just needs to know it is optimistic.
FEW_CLUSTERS = 30

CLUSTER_LEVELS = ("site", "pixel")

# Clustering a Callaway-Sant'Anna ATT above the entity level runs through the
# **multiplier bootstrap**: Rademacher weights are drawn once per cluster and
# applied to that cluster's influence functions. The closed-form ("analytic")
# standard errors have no cluster argument at all -- in `differences` and in R
# `did` alike -- so asking for site clustering forces a bootstrap. This is the
# default draw count (R `did` uses the same).
BOOT_ITERATIONS = 1000


# ---------------------------------------------------------------------------
# 0. Match first: restrict the panel to the matched pixels (Castro's two-step)
# ---------------------------------------------------------------------------
def restrict_panel_to_matched(
    panel: pd.DataFrame,
    matched: pd.DataFrame,
    coord_cols: Sequence[str] = ("x", "y"),
    site_col: str = "site_id",
    pair_col: str = "pair_id",
) -> pd.DataFrame:
    """Filter a pixel-year panel down to the pixels that survived matching.

    This is the **"match first, then DiD"** join that wires
    :func:`peatfire.modeling.match_controls` into this pipeline. Run it *before*
    :func:`attach_cohort` / :func:`build_panel` so the Callaway-Sant'Anna ATT is
    identified against the **matched** controls only -- each treated pixel plus its
    caliper-matched twin(s) -- rather than the full unrestored candidate pool. The
    matched set (from ``match_controls`` / :func:`assemble_units`) selects *which*
    control pixels enter; the DiD then differences over time within that set, so
    the design gets both the match's covariate overlap and the DiD's removal of
    time-invariant confounders.

    Parameters
    ----------
    panel : DataFrame
        Pixel-year panel (e.g. the ``pixels_with_covariates`` the notebook samples
        the response onto): one row per ``(x, y, year)`` carrying the response, the
        per-year ``treated`` flag, and the restoration year for :func:`attach_cohort`.
    matched : DataFrame
        Output of :func:`match_controls` / :func:`assemble_units` -- one row per
        matched pixel, carrying ``coord_cols`` and the ``site_col`` (restoration
        site, the cluster key) and ``pair_col`` (the finer matched stratum).
    coord_cols : sequence of str, default ``("x", "y")``
        The pixel-coordinate keys shared by both tables (same EPSG:5070 grid).
    site_col, pair_col : str
        Columns copied from ``matched`` onto the kept panel rows so the DiD can
        cluster on the restoration site (``estimate_att(cluster=site_col)`` via the
        ``rpy2`` backend) and, if wanted, inspect the matched pair.

    Returns
    -------
    DataFrame
        The subset of ``panel`` whose pixels are in ``matched``, with ``site_col``
        (and ``pair_col`` when present) attached from the match. Feed straight to
        :func:`attach_cohort`.

    Notes
    -----
    With ``replace=True`` matching a control pixel can serve several treated pixels
    and so appear under more than one ``site``/``pair``; a DiD entity must be one
    pixel, so the first is kept (with a warning). Match without replacement
    (the ``match_controls`` default) to avoid the ambiguity.
    """
    xy = list(coord_cols)
    missing = [c for c in xy if c not in matched.columns]
    if missing:
        raise ValueError(f"`matched` is missing coordinate column(s) {missing}.")
    missing_panel = [c for c in xy if c not in panel.columns]
    if missing_panel:
        raise ValueError(f"`panel` is missing coordinate column(s) {missing_panel}.")

    carry = [c for c in (site_col, pair_col) if c in matched.columns]
    if matched.duplicated(subset=xy).any():
        n = int(matched.duplicated(subset=xy).sum())
        warnings.warn(
            f"{n} matched pixel(s) appear under more than one treated partner "
            "(matching with replacement); keeping the first "
            f"{'/'.join(carry) or 'match'} for each in the DiD panel.",
            stacklevel=2,
        )
    lookup = matched[xy + carry].drop_duplicates(subset=xy)

    # Let the match's site/pair win cleanly if the panel already carried them.
    drop = [c for c in carry if c in panel.columns]
    base = panel.drop(columns=drop) if drop else panel
    out = base.merge(lookup, on=xy, how="inner")
    if out.empty:
        raise ValueError(
            "no panel rows survived the match restriction -- do `panel` and "
            f"`matched` share the same {xy} grid coordinates (same res_m/CRS)?"
        )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Treatment cohort  (Castro's "construction vintage")
# ---------------------------------------------------------------------------
def attach_cohort(
    frame: pd.DataFrame,
    cohort_by: Union[Mapping, pd.Series],
    key: str = "site_id",
    cohort_col: str = "g",
    treated_col: str = "treated",
) -> pd.DataFrame:
    """Add the first-treatment-year column ``g`` that defines each DiD group.

    In Callaway & Sant'Anna, units are grouped by the period they are *first*
    treated; controls are the **never-treated** (``g = 0``) or, period by period,
    the **not-yet-treated**. This is exactly Castro's stratification by canal-block
    construction vintage.

    Parameters
    ----------
    frame : DataFrame
        Output of :func:`peatfire.modeling.build_frame` (one row per unit-year).
    cohort_by : mapping or Series
        Maps each ``key`` value (default ``site_id``) to the site's first
        treatment year -- i.e. the restoration year (``End_Yr``) from Stage 1 of
        the matching assignment (:func:`peatfire.modeling.load_treated_units`).
        Control strata should map to ``0`` / ``NaN`` (both read as never-treated).
    key : str
        Column used to look up the cohort (``site_id`` if you assigned each matched
        stratum a shared id; ``unit_id`` if cohorts are per pixel).
    cohort_col : str
        Name of the created column (``"g"`` by convention).

    Returns
    -------
    DataFrame
        ``frame`` with an integer ``cohort_col``: the treatment year for treated
        units, ``0`` for never-treated controls.

    Notes
    -----
    A treated unit whose ``g`` is later than every year in ``frame`` contributes
    only *pre-treatment* rows -- that is fine and expected; the estimator simply
    uses it as a not-yet-treated control until its year arrives.
    """
    lookup = pd.Series(cohort_by) if not isinstance(cohort_by, pd.Series) else cohort_by
    g = frame[key].map(lookup)

    # Controls (or unmapped strata) are never-treated. Treated rows must resolve to
    # a real year, or the DiD group is undefined -- fail loudly rather than silently
    # dropping treated units into the control pool.
    is_treated = frame[treated_col].astype(float) == 1
    missing_treated = is_treated & g.isna()
    if missing_treated.any():
        bad = frame.loc[missing_treated, key].unique()[:5]
        raise ValueError(
            f"{int(missing_treated.sum())} treated rows have no cohort year in "
            f"`cohort_by` (e.g. {key}={list(bad)}). Every treated site needs its "
            f"restoration year (End_Yr); only controls may be missing."
        )

    out = frame.copy()
    out[cohort_col] = g.fillna(NEVER_TREATED).astype(int)
    return out


def prepare_panel(
    frame: pd.DataFrame,
    restoration_yr_col: str = "End_Yr",
    site_col: str = "Proj_Name",
    treated_col: str = "treated",
    entity: str = "unit_id",
    x: str = "x",
    y: str = "y",
    cluster_col: Optional[str] = "site_id",
) -> pd.DataFrame:
    """Add the DiD bookkeeping a pixel-year panel needs: entity ids, cohort ``g``,
    and the SE clustering key.

    Wraps the steps every DiD run repeats once the response is attached
    (see :func:`peatfire.modeling.attach_fire_response`):

    1. one stable ``entity`` id per distinct ``(x, y)`` pixel -- the panel's
       entity dimension;
    2. :func:`attach_cohort` with the site -> restoration-year map read off the
       frame itself. Control-pool pixels carry no ``site_col``/restoration year
       and fall through to ``g = 0`` (never-treated);
    3. ``cluster_col`` -- the key the standard errors cluster on downstream
       (see the module docstring). Left alone if the frame already carries it
       (:func:`restrict_panel_to_matched` stamps ``site_id`` on treated pixels
       *and* on their matched controls); otherwise built from ``site_col``.

    Parameters
    ----------
    frame : DataFrame
        Pixel-year panel with ``x``/``y``, the per-year ``treated_col``, and --
        on restoration-site pixels -- ``site_col`` and ``restoration_yr_col``
        (as produced by ``get_treated_and_control_pixels``).
    cluster_col : str or None, default ``"site_id"``
        Name of the clustering key to ensure exists. Pass ``None`` to skip.

    Returns
    -------
    DataFrame
        ``frame`` plus ``entity``, the integer cohort column ``g``, and (unless
        disabled) ``cluster_col``. Feed to :func:`fit_att` (or :func:`build_panel`
        directly).

    Notes
    -----
    Pixels with no site -- the unmatched control pool -- have no restoration site
    to cluster with, so each is given its **own singleton cluster**
    (``"_pixel_<entity>"``). A singleton contributes no within-cluster correlation,
    which is the conservative reading of "we know nothing tying these controls
    together"; a warning reports how many were assigned this way. On the *matched*
    panel every control inherits its treated partner's ``site_id`` and none of this
    applies.
    """
    out = frame.copy()
    out[entity] = out.groupby([x, y]).ngroup()
    cohort_by = (
        out.dropna(subset=[restoration_yr_col])
        .groupby(site_col)[restoration_yr_col]
        .first()
    )
    out = attach_cohort(out, cohort_by=cohort_by, key=site_col, treated_col=treated_col)

    if cluster_col is not None:
        if cluster_col not in out.columns:
            out[cluster_col] = (
                out[site_col] if site_col in out.columns else np.nan
            )
        out[cluster_col] = out[cluster_col].astype("object")
        unassigned = out[cluster_col].isna()
        if unassigned.any():
            n_pix = int(out.loc[unassigned, entity].nunique())
            warnings.warn(
                f"{n_pix} pixel(s) have no {site_col!r} and so no restoration site "
                f"to cluster with; each gets its own singleton {cluster_col!r}. "
                "Match first (restrict_panel_to_matched) if you want controls to "
                "join their treated partner's site cluster.",
                stacklevel=2,
            )
            out.loc[unassigned, cluster_col] = (
                "_pixel_" + out.loc[unassigned, entity].astype(str)
            )
    return out


# ---------------------------------------------------------------------------
# 2. Fire history: temporal + spatial lags (Castro Eq. 1)
# ---------------------------------------------------------------------------
def add_fire_lags(
    frame: pd.DataFrame,
    response: str = "burned",
    entity: str = "unit_id",
    time: str = "year",
    res_m: Optional[float] = None,
    x: str = "x",
    y: str = "y",
    time_lag_col: str = "fire_tm1",
    space_lag_col: str = "fire_neighbors",
) -> pd.DataFrame:
    """Add the fire-history predictors from Castro's outcome equation (Eq. 1).

    Castro model current fire as a function of, among other terms:

    * ``y_{i,t-1}`` -- a **temporal lag**: did pixel *i* burn one year ago?
    * ``sum_l y_{i-l,t}`` -- a **spatial lag**: did the 4 rook-adjacent neighbours
      (N/S/E/W) burn *this* year? Fire spreads, so a neighbour burning raises a
      pixel's own risk. Castro sum the four neighbours into one count.

    Including these in the DiD outcome regression is what lets the estimator model
    fire contagion and pixel-level fire history instead of pretending pixel-years
    are independent draws.

    Parameters
    ----------
    frame : DataFrame
        Pixel-year table with an entity id, a year, the response, and cell
        coordinates ``x``/``y`` (all produced by :func:`build_frame`).
    res_m : float, optional
        Grid resolution in metres. Needed to find neighbours (a neighbour sits one
        ``res_m`` step away in x or y). If ``None``, it is inferred from the
        smallest positive gap between unique ``x`` coordinates.

    Returns
    -------
    DataFrame
        ``frame`` plus ``time_lag_col`` (0/1, NaN in each unit's first year) and
        ``space_lag_col`` (0-4 count; neighbours off-grid contribute 0).
    """
    out = frame.sort_values([entity, time]).copy()

    # --- temporal lag: previous-year burn for the same pixel ---
    out[time_lag_col] = out.groupby(entity)[response].shift(1)

    # --- spatial lag: sum of the four rook neighbours' burn in the same year ---
    if res_m is None:
        xs = np.sort(out[x].unique())
        diffs = np.diff(xs)
        res_m = float(diffs[diffs > 0].min()) if (diffs > 0).any() else 1.0

    # Snap coords to integer grid indices so neighbour joins are exact (floats from
    # the raster transform won't compare equal otherwise).
    ix = np.rint((out[x] - out[x].min()) / res_m).astype(int)
    iy = np.rint((out[y] - out[y].min()) / res_m).astype(int)
    out["_ix"], out["_iy"] = ix, iy

    burn = out.set_index([time, "_ix", "_iy"])[response]
    burn = burn[~burn.index.duplicated()]  # one burn value per cell-year

    def _neighbor_sum(row) -> float:
        t, cx, cy = row[time], row["_ix"], row["_iy"]
        total = 0.0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            total += float(burn.get((t, cx + dx, cy + dy), 0.0) or 0.0)
        return total

    out[space_lag_col] = out.apply(_neighbor_sum, axis=1)
    return out.drop(columns=["_ix", "_iy"])


# ---------------------------------------------------------------------------
# 3. Reshape to the Callaway-Sant'Anna panel
# ---------------------------------------------------------------------------
def build_panel(
    frame: pd.DataFrame,
    cohort_col: str = "g",
    entity: str = "unit_id",
    time: str = "year",
    response: str = "burned",
    covariates: Sequence[str] = (),
    cluster_col: Optional[str] = "site_id",
) -> pd.DataFrame:
    """Shape the pixel-year frame into the balanced panel the DiD estimator wants.

    The Callaway-Sant'Anna backends expect a **panel indexed by (entity, time)**
    carrying the outcome, the cohort column ``g``, and any covariates used for
    *conditional* parallel trends. This function validates and lays that out; it
    does no estimation.

    Requirements it enforces (each maps to a DiD assumption):

    * the outcome has no NaNs on the kept rows (drop coverage gaps first),
    * ``g`` is present (run :func:`attach_cohort` first),
    * there are **both** treated (``g > 0``) and never/not-yet-treated units --
      otherwise there is no comparison group,
    * the panel is unique on ``(entity, time)``.

    Parameters
    ----------
    cluster_col : str or None, default ``"site_id"``
        Carried through onto the panel (it is *not* a regressor) so
        :func:`estimate_att` can cluster the standard errors on it. Silently
        skipped when the frame does not have it -- ``estimate_att`` then raises an
        actionable error if site clustering was asked for. Pass ``None`` to drop it.

    Returns
    -------
    DataFrame
        Indexed by ``[entity, time]``, columns ``[response, cohort_col,
        *covariates]`` plus ``cluster_col`` when available. Feed straight to
        :func:`estimate_att`.
    """
    if cohort_col not in frame:
        raise ValueError(
            f"{cohort_col!r} missing; call attach_cohort() before build_panel()."
        )
    missing = [c for c in covariates if c not in frame]
    if missing:
        raise ValueError(f"covariates not in frame: {missing}")

    carry = (
        [cluster_col]
        if cluster_col is not None
        and cluster_col in frame.columns
        and cluster_col not in (entity, time, response, cohort_col, *covariates)
        else []
    )
    keep = [entity, time, response, cohort_col, *covariates, *carry]
    panel = frame.loc[:, keep].dropna(subset=[response, cohort_col]).copy()

    dup = panel.duplicated([entity, time]).sum()
    if dup:
        raise ValueError(
            f"{dup} duplicate (entity, time) rows -- the panel must be unique on "
            f"({entity}, {time}). Aggregate to one row per unit-year first."
        )
    treated_groups = panel.loc[panel[cohort_col] > 0, cohort_col].nunique()
    controls = int((panel[cohort_col] == NEVER_TREATED).any())
    if treated_groups == 0 or not controls:
        raise ValueError(
            "Need both treated cohorts (g>0) and never-treated controls (g=0); "
            f"found {treated_groups} treated cohort(s), controls present={bool(controls)}."
        )

    if carry:
        # The clustering key identifies a *unit*, so it must not move over time --
        # both backends reject (or silently mis-group on) a time-varying cluster.
        varying = panel.groupby(entity)[cluster_col].nunique(dropna=False) > 1
        if varying.any():
            raise ValueError(
                f"{int(varying.sum())} entity(ies) have more than one "
                f"{cluster_col!r} across years; the clustering key must be "
                "time-invariant. This usually means matching with replacement put "
                "one pixel in two sites -- see restrict_panel_to_matched()."
            )
        if panel[cluster_col].isna().any():
            n = int(panel.loc[panel[cluster_col].isna(), entity].nunique())
            raise ValueError(
                f"{n} pixel(s) have a null {cluster_col!r} and so cannot be "
                "assigned to a cluster. Run prepare_panel(cluster_col=...) (it "
                "gives unmatched controls singleton clusters) or "
                "restrict_panel_to_matched() first."
            )

    return panel.set_index([entity, time]).sort_index()


# ---------------------------------------------------------------------------
# 4. The doubly-robust group-time ATT (Callaway & Sant'Anna 2021)
# ---------------------------------------------------------------------------
def _resolve_cluster(
    panel: pd.DataFrame,
    cluster_by: str,
    cluster_col: str,
    warn_few: bool = True,
) -> Optional[str]:
    """Turn the ``cluster_by`` toggle into the column to cluster on (or ``None``).

    ``"pixel"`` -> ``None`` (the panel entity is each observation's own cluster,
    which is what both backends do by default). ``"site"`` -> ``cluster_col``,
    after checking it survived :func:`build_panel` and warning when there are too
    few clusters for cluster-robust asymptotics to mean much.
    """
    if cluster_by not in CLUSTER_LEVELS:
        raise ValueError(
            f"unknown cluster_by={cluster_by!r}; use one of {CLUSTER_LEVELS}."
        )
    if cluster_by == "pixel":
        return None

    entity = panel.index.names[0]
    if cluster_col not in panel.columns:
        raise ValueError(
            f"cluster_by='site' needs the {cluster_col!r} column on the panel, but "
            f"it is not there (columns: {list(panel.columns)}). Build the panel "
            f"with build_panel(..., cluster_col={cluster_col!r}) after "
            "prepare_panel()/restrict_panel_to_matched() have stamped it, or pass "
            "cluster_by='pixel' to keep the (deflated) pixel-level SEs."
        )
    if cluster_col == entity:
        return None  # clustering on the entity *is* pixel clustering

    n_clusters = panel[cluster_col].nunique()
    if warn_few and n_clusters < FEW_CLUSTERS:
        warnings.warn(
            f"clustering SEs on {cluster_col!r} with only {n_clusters} cluster(s). "
            f"Cluster-robust inference is asymptotic in the number of clusters "
            f"(rule of thumb: {FEW_CLUSTERS}+), so these SEs are still optimistic "
            "and their normal-approximation p-values are anti-conservative. They "
            "are nonetheless far more honest than pixel-level SEs; corroborate "
            "with att_collapsed() (t with G-1 df) and the event study.",
            stacklevel=3,
        )
    return cluster_col


def _attach_cluster_column(att, cluster_var: str) -> None:
    """Make ``cluster_var`` visible to ``differences``' aggregation step.

    ``ATTgt.aggregate(cluster_var=...)`` reads the column off the *fitted* design
    matrix, and the only supported way to get it there is ``ATTgt.fit(cluster_var=...)``
    -- which is broken in 0.3.0: ``fit`` slices the matrix with a bare string, hands
    ``differences`` a Series where its time-invariance check wants a DataFrame, and
    dies with ``'Series' object has no attribute 'columns'``. Passing a list instead
    trips a different failure one step earlier. So we do by hand, after the fit, the
    one line ``fit`` would have done, and let ``aggregate`` (whose own cluster
    handling *is* correct) take it from there.

    Raises a pointed error rather than corrupting the fit if a future
    ``differences`` reshapes these internals.
    """
    try:
        matrix = att._data_matrix
        att._data_matrix[cluster_var] = att.data[cluster_var].loc[
            lambda s: s.index.isin(matrix.index)
        ]
    except (AttributeError, KeyError, TypeError) as exc:  # pragma: no cover
        raise RuntimeError(
            f"could not attach the clustering key {cluster_var!r} to the fitted "
            f"`differences` model ({type(exc).__name__}: {exc}). This shim targets "
            "differences 0.3.0; if you have upgraded, try passing cluster_var "
            "straight to ATTgt.fit, or use backend='rpy2' / att_collapsed() for "
            "site-clustered standard errors."
        ) from exc
    if att._data_matrix[cluster_var].isna().any():  # pragma: no cover
        raise RuntimeError(
            f"{cluster_var!r} came out partly null on the fitted design matrix, so "
            "some units would land in no cluster. Check that every panel row "
            "carries the site key (build_panel validates this)."
        )


def estimate_att(
    panel: pd.DataFrame,
    response: str = "burned",
    cohort_col: str = "g",
    covariates: Sequence[str] = (),
    est_method: str = "dr",
    cluster_by: str = "site",
    cluster_col: str = "site_id",
    boot_iterations: Optional[int] = None,
    random_state: Optional[int] = None,
    cluster: Optional[str] = None,
    backend: str = "differences",
):
    """Fit the staggered, doubly-robust difference-in-differences.

    This is Castro's identification step: the Callaway & Sant'Anna (2021)
    estimator for multiple periods with staggered adoption, in its **doubly
    robust** form -- it fits both an outcome (fire-probability) regression and a
    propensity model for treatment, and is consistent if *either* is correct.
    Parallel trends need only hold **conditional on** ``covariates``.

    Parameters
    ----------
    panel : DataFrame
        Output of :func:`build_panel` (MultiIndex ``(entity, time)``).
    covariates : sequence of str
        Columns entering the conditional-parallel-trends regression -- Castro's
        distances (roads/rivers/land use), climate (precip, temp), night-lights
        population proxy, and the fire lags from :func:`add_fire_lags`.
    est_method : {"dr", "ipw", "reg"}
        Doubly-robust (default, Castro's choice), inverse-probability weighting, or
        outcome regression only.
    cluster_by : {"site", "pixel"}, default ``"site"``
        **Level the standard errors cluster at.** Restoration is assigned to a
        site, so pixels within one site share shocks and are not independent
        draws; ``"site"`` (the default) clusters on ``cluster_col`` and makes the
        number of *sites* the effective sample size. ``"pixel"`` reverts to the
        panel entity -- each pixel its own cluster -- which is what both backends
        do out of the box and which materially understates the SE. Use it to
        measure that deflation, not to report. The point estimate is identical
        either way; only the variance changes. See the module docstring.
    cluster_col : str, default ``"site_id"``
        The site key ``cluster_by="site"`` clusters on. Must be a column on
        ``panel`` (see :func:`build_panel`) and time-invariant within a pixel.
    boot_iterations : int, optional
        Multiplier-bootstrap draws. Clustering above the entity level *only*
        exists on the bootstrap path (see :data:`BOOT_ITERATIONS`), so this
        defaults to :data:`BOOT_ITERATIONS` under ``cluster_by="site"`` and to
        ``0`` (fast closed-form SEs) under ``cluster_by="pixel"``. Setting it to
        ``0`` while asking for site clustering is an error rather than a silent
        downgrade -- that combination is exactly the deflated SE you are trying
        to get away from.
    random_state : int, optional
        Seed for the bootstrap draws, so a reported SE is reproducible.
    cluster : str, optional
        Deprecated alias kept for older call sites: passing a column name is the
        same as ``cluster_by="site", cluster_col=<name>`` (or ``cluster_by="pixel"``
        when it names the entity index).
    backend : {"differences", "rpy2"}
        ``"differences"`` uses the pure-Python Callaway-Sant'Anna port;
        ``"rpy2"`` calls the reference R ``did::att_gt`` (what maps most directly
        onto Castro's Stata ``csdid``).

    Returns
    -------
    The fitted estimator object (backend-specific). For the ``differences``
    backend it carries the resolved clustering choice on
    ``.peatfire_cluster_var`` / ``.peatfire_boot_iterations`` /
    ``.peatfire_random_state``, so :func:`aggregate_att` clusters the *aggregated*
    ATT the same way without being told twice. Pass it to :func:`aggregate_att`
    for the overall ATT and the event-study path.

    Notes
    -----
    Both backends implement site clustering the same, correct way: the estimator's
    per-unit **influence functions** are averaged within each site, and the
    multiplier bootstrap draws one weight per site. That is *not* the same as
    averaging the ATT point estimates by site -- it keeps the propensity/outcome
    nuisance estimation inside the variance. :func:`att_collapsed` is the by-hand
    cross-check.

    With the ``differences`` backend the clustering applies to the **aggregated**
    ATTs (:func:`aggregate_att` -- the overall number and the event study, i.e.
    everything you report). The raw group-time table left on the fitted object by
    ``.fit()`` keeps entity-level analytic SEs, because ``differences`` 0.3.0
    cannot accept a cluster column at fit time (its ``fit(cluster_var=...)``
    mis-slices the data matrix and raises). Read your SEs off
    :func:`aggregate_att`, not off the group-time table.
    """
    xformla = " + ".join(covariates) if covariates else None
    if cluster is not None:
        # Legacy signature: `cluster=<column>` meant "cluster here if you can".
        cluster_col = cluster
        cluster_by = "pixel" if cluster == panel.index.names[0] else "site"
    cluster_var = _resolve_cluster(panel, cluster_by, cluster_col)

    if boot_iterations is None:
        boot_iterations = BOOT_ITERATIONS if cluster_var else 0
    elif cluster_var and not boot_iterations:
        raise ValueError(
            "cluster_by='site' needs boot_iterations > 0: cluster-robust SEs for "
            "a Callaway-Sant'Anna ATT are only defined on the multiplier-bootstrap "
            "path, so boot_iterations=0 would silently hand back the pixel-level "
            "SEs. Leave boot_iterations unset for the default "
            f"({BOOT_ITERATIONS}), or pass cluster_by='pixel' if you really want "
            "the fast analytic ones."
        )

    if backend == "differences":
        # pip install differences
        from differences import ATTgt

        # `differences` marks never-treated units with NaN in the cohort column,
        # whereas our panel (and the R `did` backend) uses NEVER_TREATED == 0.
        # Translate locally so we don't disturb the shared 0-convention.
        panel = panel.copy()
        panel[cohort_col] = panel[cohort_col].replace(NEVER_TREATED, np.nan)

        att = ATTgt(data=panel, cohort_column=cohort_col)
        formula = response if not xformla else f"{response} ~ {xformla}"
        att.fit(formula=formula, est_method=est_method)
        if cluster_var:
            _attach_cluster_column(att, cluster_var)
        att.peatfire_cluster_var = cluster_var
        att.peatfire_boot_iterations = int(boot_iterations)
        att.peatfire_random_state = random_state
        return att

    if backend == "rpy2":
        # The reference implementation Castro's Stata csdid mirrors.
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        did = importr("did")
        pandas2ri.activate()
        flat = panel.reset_index()
        entity, time = panel.index.names
        # R `did` documents clustervars as requiring the multiplier bootstrap for
        # the same reason `differences` does, so mirror the rule and set bstrap
        # explicitly rather than relying on att_gt's default.
        return did.att_gt(
            yname=response,
            tname=time,
            idname=entity,
            gname=cohort_col,
            xformla=ro.Formula(f"~ {xformla}") if xformla else ro.NULL,
            data=flat,
            est_method=est_method,
            clustervars=cluster_var or entity,
            bstrap=bool(boot_iterations),
            biters=int(boot_iterations) if boot_iterations else 1000,
            control_group="notyettreated",
        )

    raise ValueError(f"unknown backend {backend!r}; use 'differences' or 'rpy2'.")


# ---------------------------------------------------------------------------
# 5. Aggregate: overall ATT + event study (the parallel-trends check)
# ---------------------------------------------------------------------------
def aggregate_att(
    att,
    kind: str = "simple",
    backend: str = "differences",
    cluster_var: Optional[str] = None,
    boot_iterations: Optional[int] = None,
    random_state: Optional[int] = None,
):
    """Collapse group-time ATTs into a reported effect.

    ``kind``:

    * ``"simple"`` / ``"group"`` -- one overall ATT (Castro's headline
      "% reduction in burning inside the treated area").
    * ``"event"`` / ``"dynamic"`` -- ATT by time-since-treatment. The
      **pre-treatment** points (event time < 0) are the parallel-trends test:
      they should be indistinguishable from zero. This is the plot to eyeball
      before trusting the effect.
    * ``"calendar"`` -- ATT by year (dry El Nino years vs wet years).

    Parameters
    ----------
    cluster_var, boot_iterations, random_state : optional
        How to cluster the *aggregated* standard errors. Leave them ``None`` to
        reuse what :func:`estimate_att` chose (it records the choice on the fitted
        object) -- which is what you want, and what :func:`fit_att` relies on. They
        need restating here at all because ``differences`` re-derives the standard
        errors during aggregation from the stored influence functions; without
        them the headline ATT would quietly revert to pixel-level SEs. Only used
        by the ``differences`` backend -- R ``did``'s ``aggte`` inherits
        ``att_gt``'s clustering.

    Returns the backend's aggregation object/DataFrame; for ``differences`` it is a
    tidy table of estimates and CIs ready to plot. Its ``std_error`` column is
    labelled ``bootstrap`` when clustered and ``analytic`` when not, which is the
    quickest way to confirm which SEs you are looking at.
    """
    if backend == "differences":
        alias = {"dynamic": "event", "group": "simple"}
        if cluster_var is None:
            cluster_var = getattr(att, "peatfire_cluster_var", None)
        if boot_iterations is None:
            boot_iterations = getattr(
                att, "peatfire_boot_iterations", BOOT_ITERATIONS if cluster_var else 0
            )
        if random_state is None:
            random_state = getattr(att, "peatfire_random_state", None)
        if cluster_var and not boot_iterations:
            raise ValueError(
                f"clustering on {cluster_var!r} needs boot_iterations > 0; "
                "the analytic standard errors have no cluster argument."
            )
        return att.aggregate(
            alias.get(kind, kind),
            cluster_var=cluster_var,
            boot_iterations=int(boot_iterations),
            random_state=random_state,
        )
    if backend == "rpy2":
        from rpy2.robjects.packages import importr

        did = importr("did")
        alias = {"simple": "simple", "event": "dynamic", "dynamic": "dynamic",
                 "group": "group", "calendar": "calendar"}
        return did.aggte(att, type=alias.get(kind, kind))
    raise ValueError(f"unknown backend {backend!r}.")


# ---------------------------------------------------------------------------
# 5a. Reading the aggregations: tidy tables + the site-vs-pixel contrast
# ---------------------------------------------------------------------------
# `differences` returns its aggregations under a MultiIndex column header whose
# middle level names the SE flavour -- "bootstrap" when clustered, "analytic"
# when not. That label is the quickest confirmation of which standard errors you
# are looking at, so the tidier keeps it rather than flattening it away.
_SE_KINDS = ("bootstrap", "analytic")
_ATT_CANDIDATES = ("att", "estimate", "coefficient", "coef", "point_estimate")
_SE_CANDIDATES = ("std_error", "std.error", "standard_error", "stderr", "se")
_LOW_CANDIDATES = ("ci_low", "lower", "conf_low", "conf.low", "cband_lower")
_HIGH_CANDIDATES = ("ci_high", "upper", "conf_high", "conf.high", "cband_upper")


def _first_col(columns, candidates: Sequence[str]) -> Optional[str]:
    """First column matching a candidate name (exact match wins, then substring)."""
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    for cand in candidates:
        for lc, orig in lower.items():
            if cand in lc:
                return orig
    return None


def tidy_att(aggregation) -> pd.DataFrame:
    """Flatten an :func:`aggregate_att` result into plain, comparable columns.

    The backends disagree on shape -- ``differences`` hands back a three-level
    MultiIndex column header, R ``did`` something else again -- which makes two
    aggregations awkward to put side by side. This normalises one into
    ``att`` / ``std_error`` / ``ci_low`` / ``ci_high`` / ``se_kind``, keeping the
    original index (``0`` for the ``"simple"`` aggregation, ``relative_period``
    for the event study) so rows still line up across clustering levels.

    ``ci_low`` / ``ci_high`` are whatever interval the backend reported next to the
    estimate -- for ``differences`` that is its **simultaneous** confidence band
    across the aggregation's cells, which is wider than a pointwise 95% interval
    by design. ``se_kind`` is the backend's own label for the standard errors --
    ``"bootstrap"`` when they were clustered, ``"analytic"`` when they are the
    closed-form pixel-level ones (see the module docstring for why those are the
    same thing). It is carried through untouched, so it is evidence rather than
    a restatement of what was requested.
    """
    if isinstance(aggregation, pd.DataFrame):
        df = aggregation.copy()
    else:
        try:
            df = pd.DataFrame(aggregation)
        except Exception as exc:  # pragma: no cover - backend-specific shapes
            raise TypeError(
                "tidy_att expects the tabular aggregation returned by "
                f"aggregate_att (backend='differences'), got {type(aggregation)!r}."
            ) from exc

    se_kind = None
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            hits = [
                str(v).lower()
                for v in df.columns.get_level_values(level)
                if str(v).lower() in _SE_KINDS
            ]
            if hits:
                se_kind = hits[0]
                break
        df.columns = [
            "_".join(str(part) for part in tup if str(part) != "").strip("_")
            for tup in df.columns
        ]

    att_col = _first_col(df.columns, _ATT_CANDIDATES)
    if att_col is None:
        raise ValueError(
            "could not find the ATT column on the aggregation "
            f"(columns: {list(df.columns)})."
        )
    out = pd.DataFrame(index=df.index)
    out["att"] = pd.to_numeric(df[att_col], errors="coerce")
    for name, candidates in (
        ("std_error", _SE_CANDIDATES),
        ("ci_low", _LOW_CANDIDATES),
        ("ci_high", _HIGH_CANDIDATES),
    ):
        col = _first_col(df.columns, candidates)
        if col is not None:
            out[name] = pd.to_numeric(df[col], errors="coerce")
    out["se_kind"] = se_kind if se_kind is not None else "unknown"
    return out


def compare_cluster_levels(
    aggregations: Mapping[str, object],
    baseline: Optional[str] = "pixel",
) -> pd.DataFrame:
    """Stack the same ATT estimated at different clustering levels, side by side.

    The point of the table: ``cluster_by`` never moves the point estimate, only
    the variance, so putting the two runs in one frame makes the *only* thing
    that changed impossible to miss -- and turns the "how much was the pixel-level
    SE deflated?" question into a column.

    Parameters
    ----------
    aggregations : mapping
        ``{level_name: aggregation}``, e.g.
        ``{"site": overall_site, "pixel": overall_pixel}`` where each value is an
        :func:`aggregate_att` result **of the same kind** (both ``"simple"``, or
        both ``"event"``) fitted on the same panel.
    baseline : str or None, default ``"pixel"``
        Key whose standard error the ``design_effect`` column divides by. With the
        default, ``design_effect`` reads directly as "how many times wider the
        honest interval is". Pass ``None`` to omit the column.

    Returns
    -------
    DataFrame
        :func:`tidy_att`'s columns plus ``design_effect``, indexed by
        ``(cluster_by, <the aggregation's own index>)``.

    Warns
    -----
    If the point estimates disagree across levels -- they should be identical, so
    a mismatch means the aggregations came from different fits and the design
    effect compares two unrelated numbers.
    """
    if not aggregations:
        raise ValueError("compare_cluster_levels needs at least one aggregation.")
    tidied = {str(level): tidy_att(agg) for level, agg in aggregations.items()}

    atts = [t["att"].to_numpy(dtype=float) for t in tidied.values()]
    if len({a.shape for a in atts}) == 1:
        spread = float(np.nanmax(np.abs(np.array(atts) - atts[0])))
        if spread > 1e-8:
            warnings.warn(
                f"the point estimates differ by up to {spread:.3g} across "
                "clustering levels, but cluster_by only changes the variance. "
                "These aggregations are not from the same fit/panel, so the "
                "design effect below compares unrelated estimates.",
                stacklevel=2,
            )

    out = pd.concat(tidied)
    out.index = out.index.set_names("cluster_by", level=0)
    if baseline is not None and str(baseline) in tidied and "std_error" in out:
        ref = tidied[str(baseline)]["std_error"]
        out["design_effect"] = out["std_error"].to_numpy() / ref.reindex(
            out.index.get_level_values(-1)
        ).to_numpy()
    return out


# ---------------------------------------------------------------------------
# 5b. One-call convenience: panel -> ATT -> aggregates
# ---------------------------------------------------------------------------
def fit_att(
    frame: pd.DataFrame,
    covariates: Sequence[str] = (),
    response: str = "burned",
    entity: str = "unit_id",
    time: str = "year",
    est_method: str = "dr",
    cluster_by: str = "site",
    cluster_col: str = "site_id",
    boot_iterations: Optional[int] = None,
    random_state: Optional[int] = None,
    cluster: Optional[str] = None,
    backend: str = "differences",
):
    """Run the whole staggered DiD in one call and return its two summaries.

    Chains :func:`build_panel` -> :func:`estimate_att` -> :func:`aggregate_att`
    (``"simple"`` and ``"event"``) -- the sequence the notebook repeats for the
    unmatched and the matched panel. ``frame`` must already carry the cohort
    column ``g`` and the entity id (see :func:`prepare_panel`); rows with a NaN
    response are dropped inside :func:`build_panel`.

    ``cluster_by`` / ``cluster_col`` are forwarded to :func:`estimate_att`, which
    records them on the fitted object so both aggregations below inherit the same
    clustering -- the overall ATT and the event study are then clustered alike.
    The default is ``"site"``; see the module docstring for why. That default runs
    a ``boot_iterations``-draw multiplier bootstrap, so it is slower than the old
    analytic path; ``cluster_by="pixel"`` restores the fast (deflated) SEs.

    Returns
    -------
    (att, overall, event_study)
        The fitted backend estimator, the overall ATT aggregation (Castro's
        headline number), and the event-study table (its pre-treatment rows are
        the parallel-trends check -- plot with ``plot_event_study``).
    """
    covariates = list(covariates)
    panel = build_panel(
        frame,
        entity=entity,
        time=time,
        response=response,
        covariates=covariates,
        cluster_col=cluster_col,
    )
    att = estimate_att(
        panel,
        response=response,
        covariates=covariates,
        est_method=est_method,
        cluster_by=cluster_by,
        cluster_col=cluster_col,
        boot_iterations=boot_iterations,
        random_state=random_state,
        cluster=cluster,
        backend=backend,
    )
    overall = aggregate_att(att, kind="simple", backend=backend)
    event_study = aggregate_att(att, kind="event", backend=backend)
    return att, overall, event_study


# ---------------------------------------------------------------------------
# 5c. The by-hand cross-check: collapse to one pre/post change per pixel
# ---------------------------------------------------------------------------
def att_collapsed(
    frame: pd.DataFrame,
    response: str = "burned",
    cohort_col: str = "g",
    cluster_col: str = "site_id",
    entity: str = "unit_id",
    time: str = "year",
    cluster_by: str = "site",
    weights: str = "pixels",
    alpha: float = 0.05,
) -> dict:
    """Site-level DiD you can check by hand, with the same ``cluster_by`` toggle.

    The Callaway-Sant'Anna machinery in :func:`estimate_att` is the estimator to
    report; this is the arithmetic that shows *why* the clustering level matters,
    and it runs with no optional backend. It is the standard "collapse to one
    observation per cluster" design (Bertrand, Duflo & Mullainathan 2004): a
    many-period panel offers no more independent information about a site than the
    site itself provides, so throw the pixel-years away and work with the sites.

    The arithmetic, per restoration site *s* (its cohort year ``g_s`` read off the
    treated pixels in the cluster; pre = years ``< g_s``, post = years ``>= g_s``):

    .. code-block:: text

        delta_i    = mean(y_i | post) - mean(y_i | pre)        for each pixel i
        theta_s    = mean(delta_i | treated, s) - mean(delta_i | control, s)
        ATT        = sum_s w_s theta_s / sum_s w_s

    ``theta_s`` is that site's difference-in-differences -- one number per site,
    exactly the "single value per site" summary -- and the ATT is their weighted
    average. Only the **variance** then depends on ``cluster_by``:

    * ``"site"`` (default) -- the ``theta_s`` are the independent observations:
      ``SE^2 = G/(G-1) * sum_s w_s^2 (theta_s - ATT)^2 / (sum_s w_s)^2``, with
      ``t(G-1)`` critical values. Under equal weights this collapses to the
      familiar ``sd(theta) / sqrt(G)``.
    * ``"pixel"`` -- the ``delta_i`` are treated as independent draws instead, so
      ``SE`` is built from the within-site variance of ``delta_i`` over thousands
      of pixels. This is the deflated number; it is reported alongside either way
      as ``std_error_pixel`` so you can quote the design effect.

    Parameters
    ----------
    frame : DataFrame
        Pixel-year frame carrying ``response``, ``cohort_col`` (``g``), ``entity``,
        ``time`` and ``cluster_col`` -- i.e. the output of
        :func:`restrict_panel_to_matched` -> :func:`prepare_panel`.
    weights : {"pixels", "equal"}, default ``"pixels"``
        ``"pixels"`` weights each site by its treated-pixel count, so the estimand
        is the ATT over treated *area* (what :func:`avoided_area` wants and what
        the Callaway-Sant'Anna ``"simple"`` aggregation targets). ``"equal"``
        weights each site the same, giving the ATT for the average *site*, which
        is less sensitive to one large site dominating.
    alpha : float, default ``0.05``
        Two-sided level for the reported interval.

    Returns
    -------
    dict
        ``att``, ``std_error``, ``t_stat``, ``p_value``, ``ci_low``, ``ci_high``,
        ``df``, ``cluster_by``, ``n_clusters``, ``n_pixels``, plus
        ``std_error_site`` / ``std_error_pixel`` (both levels, always) and
        ``design_effect`` = their ratio. ``by_site`` is a DataFrame of the per-site
        ``theta_s`` with pre/post means and pixel counts; ``deltas`` is the
        per-pixel table it was built from.

    Raises
    ------
    ValueError
        If any cluster holds no treated pixel -- then that cluster has no ``g_s``
        and no within-site contrast. Run :func:`restrict_panel_to_matched` first so
        each control sits in its treated partner's site cluster; on the *unmatched*
        panel the controls belong to no site and only :func:`estimate_att` (which
        pools them) applies.

    Notes
    -----
    With a handful of sites this interval is wide and its ``t(G-1)`` reference
    distribution is itself an approximation -- but a wide honest interval is the
    finding. Do not read a collapsed estimate that disagrees with
    :func:`estimate_att` as a bug: this design uses a single pre/post contrast and
    no covariate adjustment, while the Callaway-Sant'Anna estimator is
    doubly-robust and period-by-period.
    """
    from scipy import stats

    if cluster_by not in CLUSTER_LEVELS:
        raise ValueError(
            f"unknown cluster_by={cluster_by!r}; use one of {CLUSTER_LEVELS}."
        )
    if weights not in ("pixels", "equal"):
        raise ValueError(f"unknown weights={weights!r}; use 'pixels' or 'equal'.")
    required = [response, cohort_col, cluster_col, entity, time]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing column(s) {missing} needed to collapse.")

    df = frame.loc[:, required].dropna(subset=[response, cohort_col, cluster_col])
    df = df.copy()
    df[cohort_col] = df[cohort_col].astype(int)

    # Each cluster's reference year comes from its treated pixels; the matched
    # controls in that cluster are split pre/post on the *same* year, which is what
    # makes theta_s a difference-in-differences rather than two unrelated trends.
    treated_pix = df[df[cohort_col] > NEVER_TREATED]
    g_by_cluster = treated_pix.groupby(cluster_col)[cohort_col].min()
    orphan = set(df[cluster_col].unique()) - set(g_by_cluster.index)
    if orphan:
        n_pix = int(df.loc[df[cluster_col].isin(orphan), entity].nunique())
        raise ValueError(
            f"{len(orphan)} cluster(s) covering {n_pix} pixel(s) hold no treated "
            f"pixel, so they have no restoration year to split pre/post on. "
            "att_collapsed() needs matched clusters -- run "
            "restrict_panel_to_matched() so each control joins its treated "
            "partner's site -- or use estimate_att(), which pools unattached "
            "controls as never-treated."
        )

    df["_g_site"] = df[cluster_col].map(g_by_cluster)
    df["_post"] = df[time] >= df["_g_site"]
    df["_treated"] = df[cohort_col] > NEVER_TREATED

    # --- pixel level: one pre/post change per pixel ---
    cell = df.groupby([cluster_col, entity, "_treated", "_post"], observed=True)[
        response
    ].mean()
    wide = cell.unstack("_post")
    wide = wide.reindex(columns=[False, True])
    n_incomplete = int(wide.isna().any(axis=1).sum())
    wide = wide.dropna()
    if wide.empty:
        raise ValueError(
            "no pixel has both a pre- and a post-restoration observation of "
            f"{response!r}; check the fire product's year coverage against the "
            "restoration years."
        )
    deltas = (
        (wide[True] - wide[False])
        .rename("delta")
        .reset_index()
        .assign(pre=wide[False].to_numpy(), post=wide[True].to_numpy())
    )

    # --- site level: theta_s = treated change - control change ---
    rows = []
    for site, block in deltas.groupby(cluster_col, observed=True):
        t_side = block.loc[block["_treated"], "delta"].to_numpy()
        c_side = block.loc[~block["_treated"], "delta"].to_numpy()
        if t_side.size == 0 or c_side.size == 0:
            warnings.warn(
                f"site {site!r} kept {t_side.size} treated and {c_side.size} "
                "control pixel(s) after requiring both a pre- and a post-period "
                "observation; it cannot form a DiD and is dropped.",
                stacklevel=2,
            )
            continue
        rows.append(
            {
                cluster_col: site,
                "g": int(g_by_cluster[site]),
                "theta": float(t_side.mean() - c_side.mean()),
                "treated_change": float(t_side.mean()),
                "control_change": float(c_side.mean()),
                "n_treated": int(t_side.size),
                "n_control": int(c_side.size),
                # Variance of this site's theta if pixels *were* independent --
                # the ingredient of the pixel-clustered SE.
                "var_pixel": float(
                    (t_side.var(ddof=1) / t_side.size if t_side.size > 1 else 0.0)
                    + (c_side.var(ddof=1) / c_side.size if c_side.size > 1 else 0.0)
                ),
            }
        )
    by_site = pd.DataFrame(rows)
    n_clusters = len(by_site)
    if n_clusters < 2:
        raise ValueError(
            f"only {n_clusters} site(s) yield a difference-in-differences; the "
            "across-site standard error needs at least 2 (and is meaningless "
            "below a handful). Fall back to cluster_by='pixel' only to describe "
            "the point estimate, never to test it."
        )

    w = (
        by_site["n_treated"].to_numpy(dtype=float)
        if weights == "pixels"
        else np.ones(n_clusters)
    )
    theta = by_site["theta"].to_numpy(dtype=float)
    w_sum = w.sum()
    att = float((w * theta).sum() / w_sum)

    # Site-clustered: the between-site spread of theta_s is the whole variance,
    # with the usual G/(G-1) finite-cluster correction. Equal weights reduce this
    # to sd(theta)/sqrt(G).
    se_site = float(
        np.sqrt(n_clusters / (n_clusters - 1) * (w**2 * (theta - att) ** 2).sum())
        / w_sum
    )
    # Pixel-clustered: within-site sampling variance only, pixels independent.
    se_pixel = float(
        np.sqrt((w**2 * by_site["var_pixel"].to_numpy(dtype=float)).sum()) / w_sum
    )

    n_pixels = int(len(deltas))
    if cluster_by == "site":
        se, dof = se_site, n_clusters - 1
        if n_clusters < FEW_CLUSTERS:
            warnings.warn(
                f"site-clustered SE from only {n_clusters} site(s); the t({dof}) "
                "interval is indicative rather than exact. This is a property of "
                "the study design (few restoration sites), not of the code.",
                stacklevel=2,
            )
    else:
        se, dof = se_pixel, max(n_pixels - 2 * n_clusters, 1)

    t_stat = att / se if se > 0 else np.nan
    crit = float(stats.t.ppf(1 - alpha / 2, dof))
    p_value = float(2 * stats.t.sf(abs(t_stat), dof)) if se > 0 else np.nan

    return {
        "att": att,
        "std_error": se,
        "t_stat": float(t_stat),
        "p_value": p_value,
        "ci_low": att - crit * se,
        "ci_high": att + crit * se,
        "df": int(dof),
        "alpha": alpha,
        "cluster_by": cluster_by,
        "weights": weights,
        "n_clusters": n_clusters,
        "n_pixels": n_pixels,
        "n_pixels_incomplete": n_incomplete,
        "std_error_site": se_site,
        "std_error_pixel": se_pixel,
        "design_effect": (se_site / se_pixel) if se_pixel > 0 else np.nan,
        "by_site": by_site,
        "deltas": deltas,
    }


# ---------------------------------------------------------------------------
# 6. Castro's headline output: avoided burned area
# ---------------------------------------------------------------------------
def avoided_area(att_estimate: float, rewet_area_ha: float) -> float:
    """Translate an ATT into avoided burned area (Castro's "Calculating avoided fires").

    With a binary fire outcome and a linear-probability outcome model, the ATT is
    the change in burn *probability* attributable to restoration, i.e. the fraction
    of the treated area whose burning was avoided. Multiplying by the treated
    (rewetted) area gives the avoided burned area. A protective effect has
    ``att_estimate < 0``; the avoided area is then positive.

    Parameters
    ----------
    att_estimate : float
        Overall ATT from :func:`aggregate_att` (change in P(burn); negative =
        fewer fires).
    rewet_area_ha : float
        Restored/treated area these blocks cover, in hectares.

    Returns
    -------
    float
        Avoided burned area in hectares (``-att_estimate * rewet_area_ha``);
        positive when restoration reduced fire.
    """
    return -att_estimate * rewet_area_ha

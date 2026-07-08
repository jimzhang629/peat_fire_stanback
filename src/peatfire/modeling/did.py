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
* :func:`avoided_area` -- Castro's headline: ATT x rewetted area = avoided
  burned area.

Backend. Castro ran ``csdid`` in Stata. We stay Python-native with the
``differences`` package (a Callaway-Sant'Anna port); pass ``backend="rpy2"`` to
call the reference R ``did`` package instead. Both are optional and imported
*inside* :func:`estimate_att`, so importing this module stays cheap -- the same
convention ``models.py`` uses for ``statsmodels``/``pymer4``.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Castro et al. judge a match valid at |SMD| <= 0.2 and, crucially for DiD, define
# the treatment "group" as the year of canal-block construction. We mirror the
# never-treated sentinel the Callaway-Sant'Anna estimators expect.
NEVER_TREATED = 0


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

    Returns
    -------
    DataFrame
        Indexed by ``[entity, time]``, columns ``[response, cohort_col,
        *covariates]``. Feed straight to :func:`estimate_att`.
    """
    if cohort_col not in frame:
        raise ValueError(
            f"{cohort_col!r} missing; call attach_cohort() before build_panel()."
        )
    missing = [c for c in covariates if c not in frame]
    if missing:
        raise ValueError(f"covariates not in frame: {missing}")

    keep = [entity, time, response, cohort_col, *covariates]
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

    return panel.set_index([entity, time]).sort_index()


# ---------------------------------------------------------------------------
# 4. The doubly-robust group-time ATT (Callaway & Sant'Anna 2021)
# ---------------------------------------------------------------------------
def estimate_att(
    panel: pd.DataFrame,
    response: str = "burned",
    cohort_col: str = "g",
    covariates: Sequence[str] = (),
    est_method: str = "dr",
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
    cluster : str, optional
        Entity-level column to cluster SEs on (Castro cluster at the village level;
        here pass your ``site_id``/matching-stratum name). Defaults to the panel's
        entity index.
    backend : {"differences", "rpy2"}
        ``"differences"`` uses the pure-Python Callaway-Sant'Anna port;
        ``"rpy2"`` calls the reference R ``did::att_gt`` (what maps most directly
        onto Castro's Stata ``csdid``).

    Returns
    -------
    The fitted estimator object (backend-specific). Pass it to
    :func:`aggregate_att` for the overall ATT and the event-study path.
    """
    xformla = " + ".join(covariates) if covariates else None

    if backend == "differences":
        # pip install differences
        from differences import ATTgt

        att = ATTgt(data=panel, cohort_name=cohort_col)
        formula = response if not xformla else f"{response} ~ {xformla}"
        att.fit(formula=formula, est_method=est_method)
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
        return did.att_gt(
            yname=response,
            tname=time,
            idname=entity,
            gname=cohort_col,
            xformla=ro.Formula(f"~ {xformla}") if xformla else ro.NULL,
            data=flat,
            est_method=est_method,
            clustervars=cluster or entity,
            control_group="notyettreated",
        )

    raise ValueError(f"unknown backend {backend!r}; use 'differences' or 'rpy2'.")


# ---------------------------------------------------------------------------
# 5. Aggregate: overall ATT + event study (the parallel-trends check)
# ---------------------------------------------------------------------------
def aggregate_att(att, kind: str = "simple", backend: str = "differences"):
    """Collapse group-time ATTs into a reported effect.

    ``kind``:

    * ``"simple"`` / ``"group"`` -- one overall ATT (Castro's headline
      "% reduction in burning inside the treated area").
    * ``"event"`` / ``"dynamic"`` -- ATT by time-since-treatment. The
      **pre-treatment** points (event time < 0) are the parallel-trends test:
      they should be indistinguishable from zero. This is the plot to eyeball
      before trusting the effect.
    * ``"calendar"`` -- ATT by year (dry El Nino years vs wet years).

    Returns the backend's aggregation object/DataFrame; for ``differences`` it is a
    tidy table of estimates and CIs ready to plot.
    """
    if backend == "differences":
        alias = {"dynamic": "event", "group": "simple"}
        return att.aggregate(alias.get(kind, kind))
    if backend == "rpy2":
        from rpy2.robjects.packages import importr

        did = importr("did")
        alias = {"simple": "simple", "event": "dynamic", "dynamic": "dynamic",
                 "group": "group", "calendar": "calendar"}
        return did.aggte(att, type=alias.get(kind, kind))
    raise ValueError(f"unknown backend {backend!r}.")


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

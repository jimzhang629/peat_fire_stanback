"""Design diagnostics: could this study ever have detected the effect it looked for?

A null ATT has two very different readings, and the modeling pipeline alone cannot
tell them apart:

1. **restoration does not reduce burning** -- the scientific finding; or
2. **this design could not have detected a reduction of any plausible size** --
   a statement about the *design*, not about peatlands.

This module answers (2) directly, so the null can be reported as a bounded claim
("we can rule out reductions larger than X") instead of an absence. It is the
tooling behind the TNC roadmap's own contingency: *"It may be that treatment
effects cannot be identified -- if this is the case, document limitations."*

Why the pixel-level SEs are not the answer
------------------------------------------
The DiD reports SEs on the order of 0.001 because it counts ~300k pixel-years.
But treatment is assigned to a **restoration site**, and NC peat fire arrives as a
handful of **large, landscape-scale events** -- one fire paints thousands of
contiguous pixels burned in a single draw. The independent replicate is therefore
closer to a *site-year* (and, for the fire process itself, closer to a *fire
event*) than to a pixel-year. With six sites, two usable cohorts, and a 2019-2024
outcome window, the honest denominator is small enough that it has to be
confronted numerically rather than argued about.

So everything here works on the **site-year panel**: collapse the pixel-year
frame to one row per (restoration site, year), carrying the burned *fraction* of
that site's pixels. That is the level at which treatment varies, and the level at
which the standard errors were always meant to be clustered.

What is here
------------
* :func:`site_year_panel` -- collapse the pixel-year DiD panel to site-years.
* :func:`design_summary` -- the denominator, stated plainly: usable cohorts,
  pre/post site-years, and how many site-years actually contain fire. Usually
  enough on its own to explain a null.
* :func:`did_site_year` -- two-way fixed-effects DiD on the site-year panel with
  cluster-robust SEs and a ``G - 1`` t reference distribution. The transparent
  cross-check on the Callaway-Sant'Anna fit, and the estimator the power
  simulation re-runs thousands of times.
* :func:`randomization_inference` -- an exact-in-design p-value from permuting
  *which* sites were restored and when. Valid with six clusters, and it does not
  produce the ``NaN`` bootstrap intervals the site-clustered run currently hits
  when a site-year contains no fire at all.
* :func:`DesignSpec` / :func:`design_from_panel` -- the data-generating process
  for the simulation, either read off the real panel or written by hand to ask
  "what if we had 20 sites and 20 years?".
* :func:`simulate_power`, :func:`power_curve`, :func:`minimum_detectable_effect`
  -- parametric bootstrap: inject a known reduction, re-run the estimator,
  count rejections. The MDE is the headline: *the smallest true reduction this
  design would have caught 80% of the time.*
* :func:`sample_size_curve` -- the same machinery run forward, to answer "how
  many site-years would we need?" for the monitoring recommendation.

Only numpy/pandas are required; the module is deliberately dependency-free so it
runs even when the optional ``differences``/rpy2 DiD backends are not installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "DesignSpec",
    "design_from_panel",
    "design_summary",
    "did_site_year",
    "minimum_detectable_effect",
    "power_curve",
    "randomization_inference",
    "sample_size_curve",
    "simulate_power",
    "site_year_panel",
]


# ---------------------------------------------------------------------------
# 1. The site-year panel: the level at which treatment actually varies
# ---------------------------------------------------------------------------
def site_year_panel(
    panel: pd.DataFrame,
    response: str = "burned",
    site_col: str = "site_id",
    year_col: str = "year",
    cohort_col: str = "g",
    entity_col: str = "entity",
) -> pd.DataFrame:
    """Collapse a pixel-year DiD panel to one row per (site, year).

    Treatment is assigned to a restoration site, so this is the coarsest -- and
    most honest -- panel that still carries all the treatment variation. Nothing
    identifying is lost: the site-year burned *fraction* keeps the intensity
    information the 0/1 pixel flag carried, and the pixel count is retained so
    fractions can be weighted if wanted.

    Parameters
    ----------
    panel : DataFrame
        Pixel-year panel from :func:`peatfire.modeling.prepare_panel` -- needs
        ``site_col``, ``year_col``, the cohort column ``g`` (restoration year; 0
        for never-treated) and the 0/1 ``response``.
    response : str, default ``"burned"``
        Per-pixel-year 0/1 burn flag.

    Returns
    -------
    DataFrame
        Columns ``[site_id, year, g, n_pixels, n_burned, burn_fraction,
        any_burn, post, treated]`` where ``post`` is ``year >= g`` for treated
        cohorts (always 0 for ``g == 0``) and ``treated`` is the DiD regressor.
    """
    for col in (site_col, year_col, cohort_col, response):
        if col not in panel.columns:
            raise KeyError(
                f"site_year_panel: column {col!r} missing from panel; run "
                "prepare_panel (and attach_fire_response) first."
            )

    agg = {response: ["size", "sum"]}
    if entity_col in panel.columns:
        agg[entity_col] = "nunique"

    grouped = panel.groupby([site_col, year_col, cohort_col], dropna=False).agg(agg)
    grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
    out = grouped.reset_index().rename(
        columns={
            f"{response}_size": "n_pixels",
            f"{response}_sum": "n_burned",
            site_col: "site_id",
            year_col: "year",
            cohort_col: "g",
        }
    )
    out["n_burned"] = out["n_burned"].fillna(0.0)
    out["burn_fraction"] = np.where(
        out["n_pixels"] > 0, out["n_burned"] / out["n_pixels"], np.nan
    )
    out["any_burn"] = (out["n_burned"] > 0).astype(int)
    out["g"] = pd.to_numeric(out["g"], errors="coerce").fillna(0).astype(int)
    out["post"] = ((out["g"] > 0) & (out["year"] >= out["g"])).astype(int)
    out["treated"] = out["post"]
    return out.sort_values(["site_id", "year"]).reset_index(drop=True)


def design_summary(
    site_years: pd.DataFrame,
    response: str = "burn_fraction",
) -> pd.DataFrame:
    """The denominator, stated plainly -- usually the whole explanation of a null.

    Returns one row per restoration site plus a ``TOTAL`` row, reporting how many
    pre- and post-restoration years that site contributes and how many of those
    site-years contain any fire at all. A site with no pre-period (restored before
    the outcome window opens) or no post-period (restored after it closes)
    contributes **nothing** to a DiD; a site-year with no fire anywhere contributes
    no information about a *reduction* in fire.

    Read the ``TOTAL`` row as the study's real sample size. If ``post_years_with_fire``
    is 0 or 1, no estimator can rescue the design and the rest of this module will
    simply quantify by how much.
    """
    sy = site_years.copy()
    sy["_pre"] = ((sy["g"] > 0) & (sy["year"] < sy["g"])).astype(int)
    sy["_post"] = sy["post"].astype(int)
    sy["_fire"] = (sy[response].fillna(0) > 0).astype(int)

    rows = []
    for site, block in sy.groupby("site_id", dropna=False):
        g = int(block["g"].iloc[0])
        rows.append(
            {
                "site_id": site,
                "restoration_year": g if g > 0 else pd.NA,
                "site_years": len(block),
                "pre_years": int(block["_pre"].sum()),
                "post_years": int(block["_post"].sum()),
                "usable_for_did": bool(
                    g > 0 and block["_pre"].sum() > 0 and block["_post"].sum() > 0
                ),
                "site_years_with_fire": int(block["_fire"].sum()),
                "pre_years_with_fire": int((block["_pre"] & block["_fire"]).sum()),
                "post_years_with_fire": int((block["_post"] & block["_fire"]).sum()),
                "mean_burn_fraction": float(block[response].mean(skipna=True)),
            }
        )
    table = pd.DataFrame(rows)
    treated = table[table["restoration_year"].notna()]
    total = {
        "site_id": "TOTAL",
        "restoration_year": pd.NA,
        "site_years": int(table["site_years"].sum()),
        "pre_years": int(table["pre_years"].sum()),
        "post_years": int(table["post_years"].sum()),
        "usable_for_did": int(table["usable_for_did"].sum()),
        "site_years_with_fire": int(table["site_years_with_fire"].sum()),
        "pre_years_with_fire": int(table["pre_years_with_fire"].sum()),
        "post_years_with_fire": int(table["post_years_with_fire"].sum()),
        "mean_burn_fraction": float(table["mean_burn_fraction"].mean()),
    }
    total["usable_for_did"] = f"{int(treated['usable_for_did'].sum())}/{len(treated)} sites"
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


# ---------------------------------------------------------------------------
# 2. The estimator the simulation re-runs: two-way FE DiD, clustered by site
# ---------------------------------------------------------------------------
def _twoway_design(
    sy: pd.DataFrame, response: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build ``[treated, site dummies, year dummies]`` and drop unusable rows."""
    clean = sy.dropna(subset=[response]).copy()
    site_d = pd.get_dummies(clean["site_id"].astype(str), drop_first=True, dtype=float)
    year_d = pd.get_dummies(clean["year"].astype(int), drop_first=True, dtype=float)
    X = np.column_stack(
        [
            np.ones(len(clean)),
            clean["treated"].to_numpy(dtype=float),
            site_d.to_numpy(dtype=float),
            year_d.to_numpy(dtype=float),
        ]
    )
    y = clean[response].to_numpy(dtype=float)
    groups = clean["site_id"].astype(str).to_numpy()
    return X, y, groups


def did_site_year(
    site_years: pd.DataFrame,
    response: str = "burn_fraction",
    weights: Optional[str] = None,
) -> dict:
    """Two-way fixed-effects DiD on the site-year panel, clustered by site.

    Fits ``y_st = alpha_s + gamma_t + beta * treated_st + e_st`` by OLS and reports
    ``beta`` -- the change in the burned fraction of a restored site's peat,
    relative to the same-year change at not-yet/never-restored sites. Site fixed
    effects absorb every time-invariant difference between sites (drainage
    history, soil, access); year fixed effects absorb the shared weather that
    makes 2011 and 2008 look nothing like 2022.

    The standard errors are cluster-robust on ``site_id`` with the usual finite-
    sample correction, and the p-value uses a **t distribution with ``G - 1``
    degrees of freedom** rather than the normal -- with six clusters that
    difference is not cosmetic (t(5) puts the 95% critical value at 2.57, not
    1.96). Even so, cluster-robust inference is asymptotic in ``G``; with a
    handful of sites treat this p-value as optimistic and read
    :func:`randomization_inference` alongside it.

    Note this is a **linear** DiD on a fraction, which is the estimand the
    carbon accounting actually needs (avoided burned area = beta x restored
    area), not an odds ratio.

    Returns
    -------
    dict
        ``{"estimate", "std_error", "t", "p_value", "ci_low", "ci_high",
        "n_obs", "n_clusters", "df"}``.
    """
    X, y, groups = _twoway_design(site_years, response)
    if weights is not None:
        w = site_years.dropna(subset=[response])[weights].to_numpy(dtype=float)
        rw = np.sqrt(w / w.mean())
        X, y = X * rw[:, None], y * rw

    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    uniq = np.unique(groups)
    meat = np.zeros((k, k))
    for g in uniq:
        m = groups == g
        xg_u = X[m].T @ resid[m]
        meat += np.outer(xg_u, xg_u)

    G = len(uniq)
    correction = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    vcov = xtx_inv @ (correction * meat) @ xtx_inv
    se = float(np.sqrt(max(vcov[1, 1], 0.0)))

    df = max(G - 1, 1)
    t_stat = float(beta[1] / se) if se > 0 else np.nan
    try:  # scipy is in the env, but keep the module importable without it
        from scipy import stats

        p = float(2 * stats.t.sf(abs(t_stat), df)) if np.isfinite(t_stat) else np.nan
        crit = float(stats.t.ppf(0.975, df))
    except Exception:  # pragma: no cover - fallback only
        p = np.nan
        crit = 1.96

    return {
        "estimate": float(beta[1]),
        "std_error": se,
        "t": t_stat,
        "p_value": p,
        "ci_low": float(beta[1] - crit * se),
        "ci_high": float(beta[1] + crit * se),
        "n_obs": int(n),
        "n_clusters": int(G),
        "df": int(df),
    }


def randomization_inference(
    site_years: pd.DataFrame,
    response: str = "burn_fraction",
    n_permutations: int = 2000,
    random_state: Optional[int] = 0,
) -> dict:
    """Permutation p-value: reshuffle *which* sites were restored, and when.

    The cluster bootstrap breaks here -- a resample that draws only sites with no
    fire yields a degenerate replicate, which is where the ``NaN`` bootstrap
    intervals in the site-clustered DiD come from. Randomization inference has no
    such failure mode: it holds the observed fire history completely fixed and
    only permutes the *treatment assignment* (the vector of restoration years)
    across sites, which is exactly the thing whose randomness the null hypothesis
    is about.

    Under the sharp null "restoration changed nothing at any site", every
    assignment of restoration years to sites was equally likely to produce the
    data we saw, so the p-value is the share of permutations whose |beta| matches
    or exceeds the observed |beta|. Valid with six clusters; valid with three.

    Returns
    -------
    dict
        ``{"estimate", "p_value", "n_permutations", "null_quantiles"}`` where
        ``null_quantiles`` gives the 2.5/50/97.5 percentiles of the permutation
        distribution -- the honest picture of how much this design bounces around
        when nothing at all is going on.
    """
    rng = np.random.default_rng(random_state)
    observed = did_site_year(site_years, response=response)["estimate"]

    sites = site_years["site_id"].drop_duplicates().to_numpy()
    cohorts = (
        site_years.drop_duplicates("site_id").set_index("site_id")["g"].reindex(sites)
    ).to_numpy()

    draws = np.empty(n_permutations)
    draws[:] = np.nan
    for i in range(n_permutations):
        permuted = dict(zip(sites, rng.permutation(cohorts)))
        sy = site_years.copy()
        sy["g"] = sy["site_id"].map(permuted).astype(int)
        sy["post"] = ((sy["g"] > 0) & (sy["year"] >= sy["g"])).astype(int)
        sy["treated"] = sy["post"]
        if sy["treated"].nunique() < 2:  # no variation -> uninformative draw
            continue
        draws[i] = did_site_year(sy, response=response)["estimate"]

    valid = draws[np.isfinite(draws)]
    p = (
        float((np.abs(valid) >= abs(observed) - 1e-15).mean())
        if valid.size
        else float("nan")
    )
    return {
        "estimate": float(observed),
        "p_value": p,
        "n_permutations": int(valid.size),
        "null_quantiles": {
            "2.5%": float(np.percentile(valid, 2.5)) if valid.size else np.nan,
            "50%": float(np.percentile(valid, 50)) if valid.size else np.nan,
            "97.5%": float(np.percentile(valid, 97.5)) if valid.size else np.nan,
        },
    }


# ---------------------------------------------------------------------------
# 3. The data-generating process behind the power simulation
# ---------------------------------------------------------------------------
@dataclass
class DesignSpec:
    """How fire arrives on this landscape -- the DGP the power simulation draws from.

    NC peat fire is *episodic*, not a steady drizzle: in most years nothing burns
    anywhere, and then one dry year paints tens of thousands of hectares. A power
    calculation that models fire as an i.i.d. per-pixel coin flip will therefore
    be wildly optimistic. This spec encodes the two-stage structure that matters:

    1. **Does a fire reach this site this year?** ``site_fire_prob`` -- either one
       number, or a per-year mapping so the simulation reproduces the observed
       clustering of fire into a few bad years (which is what actually destroys
       the power).
    2. **If it does, how much of the site burns?** ``burn_fraction_draws``, an
       empirical pool of positive site-year burned fractions to resample.

    Restoration is then modelled as reducing (1), the probability the site burns
    at all, by a factor ``reduction``; set ``reduce_extent=True`` to have it also
    shrink (2), the fraction that burns when a fire does arrive.

    Attributes
    ----------
    sites : sequence
        Site labels. Length is the number of clusters -- the quantity power is
        most sensitive to.
    years : sequence of int
        Outcome-window years.
    cohorts : mapping
        ``site -> restoration year`` (0 or absent = never treated). Sites whose
        restoration year falls outside ``years`` contribute no DiD information,
        exactly as in the real panel.
    site_fire_prob : float or mapping
        P(any fire at a site in a year). A mapping ``{year: p}`` reproduces
        episodic years.
    burn_fraction_draws : ndarray
        Pool of positive burned fractions to resample when a fire hits.
    """

    sites: Sequence
    years: Sequence[int]
    cohorts: Mapping = field(default_factory=dict)
    site_fire_prob: float | Mapping[int, float] = 0.05
    burn_fraction_draws: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.15, 0.4])
    )

    def year_prob(self, year: int) -> float:
        if isinstance(self.site_fire_prob, Mapping):
            return float(self.site_fire_prob.get(year, 0.0))
        return float(self.site_fire_prob)

    def n_treated_sites(self) -> int:
        return sum(
            1
            for s in self.sites
            if int(self.cohorts.get(s, 0)) in {int(y) for y in self.years}
        )


def design_from_panel(
    site_years: pd.DataFrame,
    response: str = "burn_fraction",
    control_only: bool = True,
) -> DesignSpec:
    """Read the DGP off the observed panel: how often sites burn, and how much.

    ``site_fire_prob`` is estimated **per year** as the share of sites with any
    fire that year, and the burned-fraction pool is every observed positive
    site-year fraction. Estimating both from *untreated* site-years only
    (``control_only``, the default) keeps the simulation's baseline free of any
    real treatment effect, so the injected reduction is the only effect present.

    Falls back to pooling all site-years when there are too few untreated
    positives to resample from.
    """
    sy = site_years
    pool_src = sy[sy["treated"] == 0] if control_only else sy
    positives = pool_src.loc[pool_src[response].fillna(0) > 0, response].to_numpy()
    if positives.size < 3:
        positives = sy.loc[sy[response].fillna(0) > 0, response].to_numpy()
    if positives.size == 0:
        positives = np.array([0.05])

    per_year = (
        pool_src.assign(_fire=(pool_src[response].fillna(0) > 0).astype(float))
        .groupby("year")["_fire"]
        .mean()
    )
    return DesignSpec(
        sites=list(sy["site_id"].drop_duplicates()),
        years=sorted(int(y) for y in sy["year"].unique()),
        cohorts=sy.drop_duplicates("site_id").set_index("site_id")["g"].astype(int).to_dict(),
        site_fire_prob={int(k): float(v) for k, v in per_year.items()},
        burn_fraction_draws=positives,
    )


def _simulate_panel(
    spec: DesignSpec,
    reduction: float,
    rng: np.random.Generator,
    reduce_extent: bool = False,
) -> pd.DataFrame:
    """One synthetic site-year panel under a *known* proportional reduction."""
    rows = []
    for site in spec.sites:
        g = int(spec.cohorts.get(site, 0))
        for year in spec.years:
            post = int(g > 0 and year >= g)
            p = spec.year_prob(int(year))
            if post:
                p *= 1.0 - reduction
            frac = 0.0
            if rng.random() < p:
                frac = float(rng.choice(spec.burn_fraction_draws))
                if post and reduce_extent:
                    frac *= 1.0 - reduction
            rows.append(
                {
                    "site_id": site,
                    "year": int(year),
                    "g": g,
                    "post": post,
                    "treated": post,
                    "burn_fraction": frac,
                }
            )
    return pd.DataFrame(rows)


def simulate_power(
    spec: DesignSpec,
    reduction: float,
    n_sims: int = 500,
    alpha: float = 0.05,
    method: str = "cluster",
    n_permutations: int = 300,
    reduce_extent: bool = False,
    random_state: Optional[int] = 0,
) -> dict:
    """Power at one effect size: simulate, estimate, count rejections.

    Draws ``n_sims`` synthetic site-year panels in which restoration truly cuts
    the probability a site burns by ``reduction`` (0.5 = a 50% reduction), runs
    the same estimator the real analysis uses, and reports the share of runs that
    reject at ``alpha``. That share **is** the power of this design against that
    effect.

    ``method="cluster"`` uses :func:`did_site_year`'s cluster-robust t-test;
    ``method="randomization"`` uses :func:`randomization_inference` (slower, but
    the honest test at six clusters -- and the one to quote if the two disagree).

    Returns
    -------
    dict
        ``{"reduction", "power", "mean_estimate", "n_sims", "n_informative",
        "method"}``. ``n_informative`` counts simulations in which any fire
        occurred at all -- when it is far below ``n_sims``, the design is
        failing for lack of fire, not for lack of precision.
    """
    rng = np.random.default_rng(random_state)
    rejects, estimates, informative = 0, [], 0

    for _ in range(n_sims):
        sim = _simulate_panel(spec, reduction, rng, reduce_extent=reduce_extent)
        if sim["burn_fraction"].sum() <= 0 or sim["treated"].nunique() < 2:
            estimates.append(0.0)
            continue
        informative += 1
        if method == "randomization":
            res = randomization_inference(
                sim,
                n_permutations=n_permutations,
                random_state=int(rng.integers(0, 2**31 - 1)),
            )
            p, est = res["p_value"], res["estimate"]
        else:
            res = did_site_year(sim)
            p, est = res["p_value"], res["estimate"]
        estimates.append(est)
        if np.isfinite(p) and p < alpha:
            rejects += 1

    return {
        "reduction": float(reduction),
        "power": rejects / n_sims if n_sims else np.nan,
        "mean_estimate": float(np.nanmean(estimates)) if estimates else np.nan,
        "n_sims": int(n_sims),
        "n_informative": int(informative),
        "method": method,
    }


def power_curve(
    spec: DesignSpec,
    reductions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    **kwargs,
) -> pd.DataFrame:
    """Power across a grid of true reductions -- one row per effect size.

    ``reduction = 1.0`` is the ceiling case: restoration prevents fire entirely.
    If power is still below 0.8 there, **no** true effect is detectable by this
    design, and that single fact is the cleanest way to report the null.
    """
    return pd.DataFrame([simulate_power(spec, r, **kwargs) for r in reductions])


def minimum_detectable_effect(
    spec: DesignSpec,
    target_power: float = 0.8,
    reductions: Sequence[float] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    **kwargs,
) -> dict:
    """The headline: the smallest true reduction this design would usually catch.

    Returns ``mde = None`` when even a 100% effective treatment fails to reach
    ``target_power`` -- the case in which the null result carries no information
    about peatlands at all. Report it as: *"this design has ``X``% power against
    complete fire prevention; it was never capable of detecting the effect
    sizes at stake."*

    Returns
    -------
    dict
        ``{"mde", "target_power", "power_at_full_prevention", "curve"}``.
    """
    curve = power_curve(spec, reductions=reductions, **kwargs)
    hit = curve[curve["power"] >= target_power]
    return {
        "mde": float(hit["reduction"].min()) if len(hit) else None,
        "target_power": float(target_power),
        "power_at_full_prevention": float(curve["power"].iloc[-1]),
        "curve": curve,
    }


def sample_size_curve(
    spec: DesignSpec,
    reduction: float = 0.5,
    site_counts: Sequence[int] = (6, 10, 20, 40, 80),
    year_counts: Sequence[int] = (6, 12, 24),
    treat_share: float = 0.5,
    **kwargs,
) -> pd.DataFrame:
    """Run the design forward: how many sites x years would suffice?

    Rebuilds ``spec`` at each ``(n_sites, n_years)`` -- keeping its fire
    probabilities and burned-fraction pool, staggering restoration years across
    the middle of the window so every treated site has both a pre- and a
    post-period -- and reports power against a ``reduction`` effect.

    This is the input to a prospective monitoring recommendation: it converts
    "we could not detect it" into "detecting a 50% reduction requires roughly N
    site-years, so here is what to instrument now."
    """
    rows = []
    base_year = int(min(spec.years))
    for n_years in year_counts:
        years = list(range(base_year, base_year + n_years))
        probs = (
            {y: spec.year_prob(y) for y in years}
            if isinstance(spec.site_fire_prob, Mapping)
            else spec.site_fire_prob
        )
        if isinstance(probs, Mapping):
            observed = [spec.year_prob(y) for y in spec.years]
            fill = float(np.mean(observed)) if observed else 0.0
            probs = {y: (probs.get(y) or fill) for y in years}

        for n_sites in site_counts:
            sites = [f"site_{i}" for i in range(n_sites)]
            n_treated = max(1, int(round(n_sites * treat_share)))
            # Stagger restoration through the middle half of the window so each
            # treated site keeps a real pre-period and a real post-period.
            lo, hi = base_year + n_years // 4, base_year + (3 * n_years) // 4
            span = max(hi - lo, 1)
            cohorts = {
                s: lo + (i % span) for i, s in enumerate(sites[:n_treated])
            }
            sub = DesignSpec(
                sites=sites,
                years=years,
                cohorts=cohorts,
                site_fire_prob=probs,
                burn_fraction_draws=spec.burn_fraction_draws,
            )
            res = simulate_power(sub, reduction, **kwargs)
            rows.append(
                {
                    "n_sites": n_sites,
                    "n_years": n_years,
                    "n_treated_sites": n_treated,
                    "site_years": n_sites * n_years,
                    "reduction": reduction,
                    "power": res["power"],
                }
            )
    return pd.DataFrame(rows)

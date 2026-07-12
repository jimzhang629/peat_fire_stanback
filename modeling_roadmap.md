# Modeling roadmap: peat condition → fire

How we go from "FireCCIS311 is the dependent variable" to fitted models with
interpretable effects. This is the plan behind the design decisions logged in
`decisions.md` (see the **Modeling** section). The dependent variable (DV) is
deliberately kept swappable so the same pipeline runs for burned *occurrence*,
burned *area*, or *severity* later.

The scientific question is causal, not predictive: **does peatland restoration
(and, more generally, peat hydrological condition) reduce fire?** Peat condition
is the treatment of interest; everything else (climate, elevation, distance to
coast, land cover, histosol %, management) is a confounder we adjust or match
for. That framing drives every choice below.

---

## 0. The instinct, and where it needs sharpening

The natural first move — "stack every pixel, regress `burned ~ peat_condition +
covariates + interactions`, read off the betas" — is the right *shape* but has
three problems that would make the betas untrustworthy. The roadmap below is
mostly about fixing these.

1. **Pixels are not independent observations.** Pixels inside one restoration
   site (or one fire) share unmeasured conditions. Treating millions of 30–250 m
   pixels as independent shrinks the standard errors toward zero, so *everything*
   looks significant. The effective sample size is closer to the number of
   independent **sites × years**, not pixels. → cluster/aggregate, or use a mixed
   model with a site random effect.

2. **Restored peat is not located like drained peat.** Restoration sites differ
   systematically from the rest of the landscape in elevation, coast distance,
   and land cover. A plain GLM silently extrapolates across covariate space where
   there is no comparable control, and the "treatment" beta absorbs that
   imbalance. → **match controls to treated units first**, then model. (This is
   exactly the control-pixel design already sketched in the project notes.)

3. **Fire is rare.** Burned pixels are a tiny fraction of peatland-years, and
   burned *area* per unit is zero-inflated. Ordinary logistic/linear fits are
   biased and over-confident in this regime. → rare-event-aware models
   (penalized logistic / mixed logistic; Tweedie or hurdle for area).

---

## 1. Sample frame — *where* we sample

- **Peat extent (80% histosol threshold).** Restrict the universe to
  high-confidence peat: gSSURGO pixels with `H% ≥ 80`. (The download notebook now
  builds this mask at `PEAT_THRESHOLD = 80`.) This is the population the question
  is about and keeps non-peat noise out of the covariate distributions.

- **Treated units = restoration polygons** (`data/.../peat_restoration`), each
  carrying a restoration **date**. The date matters: only fire *after* restoration
  is "post-treatment," and a unit can contribute pre- and post-restoration years.

- **Control units = matched unrestored peat.** Clip the 80% peat mask, drop the
  restoration polygons (plus a buffer), and draw the **nearest** unrestored peat
  to each restoration site as the candidate control pool, then **match** on:
  distance to coast, elevation, land cover (LandFire EVT), and histosol %. Matching
  (nearest-neighbour or coarsened-exact, e.g. via `pymatch`/manual Mahalanobis, or
  just stratified nearest-neighbour on those four axes) buys covariate **overlap**
  so the restoration effect isn't confounded by geography. Check balance with a
  before/after love plot — this is a required EDA output, not optional.

---

## 2. Unit of analysis — *what one row is*

Recommended unit: **pixel-year** on the shared analysis grid, tagged with its
`site_id` (restoration polygon or matched control cluster). Reuse the existing
`build_common_grid(aoi, res_m=..., crs="EPSG:5070")` from the fire toolkit so the
modeling grid is the *same* equal-area grid the product comparison already uses —
no new CRS decisions, area stays meaningful. FireCCIS311 is ~250 m and 2019–2024,
so 6 years × the peat-AOI cells is a comfortable table.

Keep `site_id` and `year` on every row: they become the random-effect / clustering
keys that fix problem #1, and `year` absorbs the fact that a single dry year drives
fire everywhere.

If pixel-year proves too autocorrelated or heavy, aggregate one level up to
**site-year** (burned fraction of the site that year) and model that — fewer rows,
each closer to independent, at the cost of within-site resolution.

---

## 3. Building the analysis frame (the data layer)

One function builds the table; one column is the response. Mirror the existing
`ProductSpec` registry pattern (`fire_products.py`) with a parallel **covariate
registry** so each predictor is "register a spec, no loader changes":

| role | layer | source | how it enters the model |
|------|-------|--------|--------------------------|
| **DV (swappable)** | FireCCIS311 burned mask / area; later severity | `load_standardized(product, year, aoi)` | `y` |
| **treatment** | restoration status + time-since-restoration | restoration polygons | key beta |
| **treatment** | drainage / wetness | NASA HAND 30 m | **area-weighted mean** of HAND per unit |
| covariate | histosol % | gSSURGO | continuous |
| covariate | elevation | Copernicus GLO-30 DEM | continuous |
| covariate | distance to coast | derived from NC boundary / coastline | continuous |
| covariate | land cover | LandFire EVT | categorical (grouped) |
| covariate | climate (precip, temp, VPD, …) | PRISM/Daymet via the R script | continuous, by year |
| covariate | management | SEUS Forest Mgmt raster | binary/categorical |

Notes:
- **Drainage is area-weighted**, per the project notes: a unit's drainage value is
  the area-weighted mean of HAND over its pixels, not a centroid sample.
- **Plug-and-play DV:** the response is produced by `load_standardized(product, …)`,
  which already returns a boolean burned mask (`burned_area` family) or a
  continuous grid (`severity` family). Swapping `product="FireCCIS311"` for a
  severity product changes the `y` column and the model family — nothing upstream.
- Resample every raster covariate onto the common grid with `to_common_grid(...,
  how=...)`: `mean` for continuous, `mode` for categorical — the same helper the
  fire stack already uses.

Deliverable of this stage: a tidy `frame.parquet` with one row per unit-year and
columns `[unit_id, site_id, year, treated, years_since_restoration, drainage,
histosol_pct, elev, dist_coast, lc_class, precip, temp, managed, burned, burned_area]`.

## 4. EDA — quick plots (yes, do these first)

1. Maps of each covariate over the peat AOI (sanity / coverage gaps).
2. Covariate **distributions split by treated vs control**, before and after
   matching (the love plot) — this is what tells you the matching worked.
3. Raw burned rate by treatment and by year (the unadjusted signal).
4. Pairwise covariate correlation (collinearity check before the GLM; drop/merge
   anything with VIF that blows up, e.g. elevation vs distance-to-coast may be
   redundant on the NC coastal plain).

Reuse `set_fire_style()` and the Okabe-Ito palette so figures match the rest of
the deck. The "prettier table coloring" the notes ask for lives here: render the
summary/agreement tables with a sequential colormap (`DataFrame.style.background_
gradient`) and export to the slides.

## 5. Models — start simple, add structure

Fit in this order, reporting each so the effect of adding structure is visible:

1. **Baseline GLM (cluster-robust).** `burned ~ treated + drainage + covariates`,
   logistic, with standard errors clustered by `site_id`. The honest version of
   the original instinct — same betas, trustworthy SEs.

2. **Mixed logistic (GLMM).** `burned ~ treated + drainage + covariates +
   (1|site_id) + (1|year)`. Random intercepts soak up site- and year-level
   autocorrelation (problem #1) directly. `statsmodels` BinomialBayesMixedGLM /
   `pymer4` (lme4) / `bambi`.

3. **Treatment × climate interaction.** Add `treated:precip` (or `:VPD`): *does
   restoration help more in dry years?* — the scientifically interesting term.
   Keep interactions few and pre-registered; with rare events, don't fish.

4. **Rare-event / area variants.** Firth-penalized logistic if separation
   appears; for burned *area* as the DV, a **Tweedie GLM** or a **hurdle**
   (occurrence × conditional-area) instead of plain Gaussian.

Report **odds ratios with 95% CIs** (not just raw betas), the treatment effect
first, then interactions. Treat peat condition (`treated` + `drainage`) as the
headline; everything else is an adjusted covariate.

## 6. Plug-and-play and reproducibility

- Keep the whole thing parameterized by `product` so severity is a one-line swap,
  per the project notes ("make the modeling pipeline plug and play for the
  dependent variable").
- Model code lives in `src/peatfire/modeling/` (scaffolded):
  - `covariates.py` -- a `CovariateSpec` registry mirroring `ProductSpec`, with
    loaders that warp each layer onto the shared grid (`elevation`, `histosol_pct`
    are live; `land_cover`, `drainage`, `management` registered and skip until
    downloaded).
  - `frame.py` -- `build_frame(units, product, years, ...)` assembles the tidy
    pixel-year table. It **consumes** an upstream-matched `units` set (treated
    restoration polygons + matched controls); the matching itself is not in the
    package, keeping the causal design explicit.
  - `models.py` -- `fit_logit_clustered` / `fit_mixed_logit` -> `odds_ratios`.
  Drive it from a `notebooks/modeling.ipynb`.
- Log every choice that has a defensible alternative in `decisions.md`.

## 7. Alternative estimator — staggered difference-in-differences

Everything above matches on observed covariates and then reads the treatment
effect off the *levels* (an odds ratio for `treated`). That is only as good as the
covariates we measured: an unmeasured, persistent difference between restored and
unrestored peat (drainage legacy, soil, access) still biases the effect. The
peatland-fire literature's stronger design — **match first, then estimate a
staggered difference-in-differences (DiD)** — removes every *time-invariant*
confounder by identifying off the **change** in burning after each site's
restoration, relative to not-yet/never-restored controls. This is exactly what
**Castro et al. (2026)** do for canal-block rewetting in Kalimantan, and our data
already has the one ingredient it needs: restoration **dates** (`pivot_year`) and
planned-but-not-yet-restored sites (natural not-yet-treated controls).

What Castro et al. do, and how it maps onto our pipeline:

| Castro et al. (2026), Kalimantan | our analogue |
|----------------------------------|--------------|
| Treated = 250 m upstream semicircle of each canal block; control = rest of a 2 km buffer | treated = restoration polygons; control = matched unrestored peat (matching.py) |
| Match 1:1, Mahalanobis, no replacement, **exact on subdistrict + peat depth**, |SMD| ≤ 0.2 | our Stage 5 matching (add exact-match keys; tighten caliper) |
| Match on **propensity + prognostic scores** + pre-treatment fire history (t-1, t-2, 2015 drought) | two options: `add_matching_scores` (collapsed scalar `pscore`+`phat`, simple) **or** the faithful series -- `add_prognostic_score_series` (per-year `phat_<yr>`), `add_propensity_score_series` (per-vintage `psm_<g>`), matched by `match_controls_event_time` on the event-time trajectory vector (fire lags + `phat_<t>` + `psm_<g>`) |
| Outcome = binary fire, 50 m pixel-year (MODIS MCD64A1, ≤10% uncertainty) | `build_frame` pixel-year `burned` (swappable DV) |
| Estimator = **Callaway & Sant'Anna (2021)** staggered DiD, **doubly robust**, `csdid` in Stata | `modeling/did.py` (`estimate_att`, `differences` pkg or R `did`) |
| Outcome eq. (their Eq. 1): distances, climate, night-lights, **temporal lag** (fire t-1) + **spatial lag** (4 neighbours) | `add_fire_lags` builds both lags; pass as DiD covariates |
| Group = canal-block **construction vintage**; controls = never-/not-yet-treated | `attach_cohort` sets `g = pivot_year`, controls `g = 0` |
| SEs clustered at **village**; report only results with 80% power | cluster on `site_id`; check power at our (smaller) N |
| Headline = ATT × rewetted area = avoided burned area | `avoided_area(att, area_ha)` |

Where it differs from Castro et al. and matters for us: **N**. They have 11.3M
pixel-years and estimate a separate counterfactual per subdistrict × vintage ×
block type. Our handful of NC restoration sites will not support that slicing —
expect one pooled ATT plus, at most, an event-study path, and treat sub-group
effects as underpowered. Also: our within-site control is *matched landscape
peat*, not a within-buffer donut, so the conditional-parallel-trends assumption
leans harder on the match quality (Stage 6 balance) than theirs does.

Implementation lives in `src/peatfire/modeling/did.py` and reuses `build_frame`:

```python
from peatfire.modeling import (
    add_matching_scores, match_controls, restrict_panel_to_matched,
    attach_cohort, add_fire_lags, build_panel,
    estimate_att, aggregate_att, avoided_area,
)
# Match first (Castro's two-step): score, then pair each treated pixel to its twin.
scored = add_matching_scores(panel, continuous=covs, categorical=cats)  # -> pscore, phat
matched = match_controls(scored, continuous=["pscore", "phat"],         # bijective match
                         carry=covs)                                     # keep raw covs for balance
# Restrict the DiD to the matched controls only, THEN identify off the change.
panel  = restrict_panel_to_matched(panel, matched)           # matched pixels only
panel  = attach_cohort(panel, cohort_by=pivot_year_by_site)  # g = restoration yr, 0 = control
panel  = add_fire_lags(panel, res_m=300)                     # Castro Eq. 1 lags
panel  = build_panel(panel, covariates=["fire_neighbors", "fire_tm1", "elev", "histosol_pct"])
att    = estimate_att(panel, est_method="dr", cluster="site_id")   # doubly-robust CS
overall = aggregate_att(att, "simple")   # headline ATT
events  = aggregate_att(att, "event")    # event study -> pre-trends check
```

Match on **propensity + prognostic scores** (`add_matching_scores`) to reproduce
Castro's score-based match instead of raw-covariate Mahalanobis, then
`restrict_panel_to_matched` wires that matched set into the DiD so the ATT is
identified against each treated pixel's matched control(s), not the full
candidate pool — the "match first, then estimate" design in full.

The event-study pre-treatment coefficients are the **parallel-trends test**;
Castro's planned-but-unbuilt placebo is our negative control (§ matching_assignment
"negative control"). Keep the matched design either way — it is what makes
conditional parallel trends credible.

---

## References

**Methods (matched design + DiD)**
- Stuart, E. A. (2010). *Matching Methods for Causal Inference: A Review and a Look
  Forward.* Statistical Science 25(1), 1–21. — the readable review behind
  `matching.py` (distances, calipers, SMD/love plots).
- Ho, Imai, King & Stuart (2007). *Matching as Nonparametric Preprocessing for
  Reducing Model Dependence in Parametric Causal Inference.* Political Analysis
  15(3), 199–236. — the "match first, then model" two-step; the `MatchIt` package.
- Callaway, B. & Sant'Anna, P. H. C. (2021). *Difference-in-Differences with
  Multiple Time Periods.* Journal of Econometrics 225(2), 200–230. — the staggered,
  doubly-robust ATT estimator in `did.py`.

**Domain (matching / DiD applied to peat + fire)**
- Castro et al. (2026). *Effective restoration can avoid peatland fires: Large
  scale counterfactual assessment in Kalimantan, Indonesia.* iScience.
  doi:10.1016/j.isci.2026.116041 — the study `did.py` implements.
- *The Impact of Rewetting Peatland on Fire Hazard in Riau, Indonesia* (2023).
  Sustainability 15(3), 2169. — propensity-score matching, peat + fire.
- Nguyen Huy, Adjognon & Van Soest (2023). *Combatting Forest Fires in the
  Drylands of Sub-Saharan Africa: Quasi-Experimental evidence.* — matching + DiD
  on fire (Castro's methodological cite).

---

### TL;DR

Match first (treated restoration vs nearest comparable unrestored peat on the
80% histosol frame), build a tidy **pixel-year** table on the existing EPSG:5070
common grid with an **area-weighted** drainage covariate and a **swappable**
FireCCIS311 response, plot covariate balance, then fit **`burned ~ treated +
drainage + covariates + (1|site) + (1|year)`** with a treatment×climate
interaction — reading effects off as odds ratios, peat condition first.

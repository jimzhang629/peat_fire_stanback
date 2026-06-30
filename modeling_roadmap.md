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
- Put model code in `src/peatfire/modeling/` (a `covariates.py` registry + a
  `frame.py` table-builder + a `models.py` fitter), tested the same way as the
  fire toolkit, and drive it from a `notebooks/modeling.ipynb`.
- Log every choice that has a defensible alternative in `decisions.md`.

---

### TL;DR

Match first (treated restoration vs nearest comparable unrestored peat on the
80% histosol frame), build a tidy **pixel-year** table on the existing EPSG:5070
common grid with an **area-weighted** drainage covariate and a **swappable**
FireCCIS311 response, plot covariate balance, then fit **`burned ~ treated +
drainage + covariates + (1|site) + (1|year)`** with a treatment×climate
interaction — reading effects off as odds ratios, peat condition first.

# Methods: estimating the effect of peatland restoration on wildfire in North Carolina

Jim Zhang · Duke University / TNC North Carolina Chapter
Prepared for C. Chamberlain and K. Austin · August 2026

---

## 1. Objective and estimand

The question is causal, not predictive: **does peatland restoration (canal
blocking and re-wetting) reduce the probability that a given patch of peat
burns?** Restoration re-wets drained peat, so restored pixels should burn less
often than otherwise-comparable unrestored pixels.

The target parameter is the **average treatment effect on the treated (ATT)** —
the change in annual burn probability at restored sites attributable to
restoration, rather than to where restoration happens to have been sited:

$$\mathrm{ATT} = \mathbb{E}\!\left[\,Y(1) - Y(0) \mid D = 1\,\right]$$

where $Y$ is a 0/1 "did this pixel burn this year" indicator and $D = 1$ marks
restored pixels. The observable half, $\mathbb{E}[Y(1)\mid D=1]$, is just the
burn rate at restored sites; the entire methodological effort goes into
estimating the counterfactual $\mathbb{E}[Y(0)\mid D=1]$ — how often restored
sites *would* have burned had they not been restored.

Two independent strategies are used for that counterfactual, under different
identifying assumptions, so that agreement between them is informative:

| Route | Strategy | Identifying assumption | Reported quantity |
|---|---|---|---|
| **A. Changes** | Staggered difference-in-differences (Callaway & Sant'Anna 2021), following Castro et al. (2026) | Conditional parallel trends | ATT: absolute change in $\Pr(\text{burn})$ per pixel-year |
| **B. Levels** | 1:1 matched pixels + cluster-robust logistic regression | Selection on observables | Odds ratio on `treated` |

Route A is the primary specification. Route B is a robustness check on a
different assumption set; the two are on different scales and should be compared
qualitatively, not numerically.

---

## 2. Study area and sample frame

The analysis frame is the North Carolina Coastal Plain peatland extent,
restricted to cells mapped as **≥ 80 % major-component histosol** in gSSURGO
(`nc_peatlands_80_histosol_aoi.gpkg`; threshold applied in
`notebooks/download_and_clip_data.ipynb`). The 80 % threshold selects
high-confidence peat and keeps non-peat pixels from polluting the covariate
distributions used for matching.

**Common geometry.** Every layer — fire products, treatment polygons, covariate
rasters — is warped onto one shared grid: **EPSG:5070 (NAD83 / CONUS Albers,
equal-area, metres)** at **300 m** resolution. This makes areas meaningful,
distances and buffers metric, and every layer aligned cell-for-cell. Binary
layers are aggregated to the common grid with `max` ("any burned sub-cell lights
the cell"), continuous layers with an area-weighted mean, and categorical layers
with the majority class.

---

## 3. Treatment definition

Treatment is membership in a **completed TNC restoration project polygon**.

- Sites are read from the TNC restoration layer and reprojected to EPSG:5070.
- A site's **completion year (`End_Yr`)** — not its start year — is the moment it
  flips from unrestored to restored, and defines the DiD cohort. Records with
  `End_Yr = 0` (the placeholder for "not finished") are dropped: "completed" is
  *defined* as having a real end year.
- A pixel inside a restoration polygon is coded `treated = 1` from its site's
  `End_Yr` forward and `treated = 0` before it. Pre-completion site-years are
  therefore retained as **not-yet-treated controls**, which is exactly what the
  staggered DiD estimator wants.
- **Control candidate pool:** all peat pixels in the frame that fall outside every
  restoration polygon (`gpd.overlay(..., how="difference")`). A spillover halo
  around treated sites is supported (`spillover_m`) but is set to **0 m**, on
  TNC's assessment that the projects do not hydrologically spill into
  neighbouring parcels.

Two distinct notions of "treated" coexist in the panel and are kept separate
throughout:

- the **per-year flag** (`treated`), which switches on at `End_Yr` — used by the
  DiD; and
- **site membership** (`End_Yr` is not null), constant for a pixel — used by
  matching and scoring, because a restoration pixel is a treated *unit*
  regardless of year.

---

## 4. Outcome variable

The response is a binary **burned / not burned flag per pixel-year**, sampled at
each pixel's grid location from a satellite burned-area product.

| Product | Native res. | Record used | Role |
|---|---|---|---|
| **MODIS MCD64A1** | 500 m (analysed at 300 m common grid) | 2001–2026 | **Primary** — the long record; recovers the 2008 Evans Road and 2011 Pains Bay fires |
| **FireCCI S3.1.1** | 300 m | 2019–2026 | Secondary / robustness |

The product is a single configuration switch (`FIRE_PRODUCT`) at the top of
`notebooks/modeling.ipynb`; every downstream grid, response, score, match and
model reads from it, so the whole analysis re-runs on a different outcome without
code changes. Because 300 m and 500 m intermediate objects must not be mixed, the
kernel is restarted and the notebook re-run from the configuration cell when the
product is switched.

Product choice was not assumed. A separate comparison and validation exercise
(`src/peatfire/fire_products_comparison/`, `notebooks/run_fire_comparison.ipynb`,
`notebooks/validate_against_reference.ipynb`) benchmarked the candidate products
against each other on the common grid (Jaccard/IoU, Cohen's κ, total
least-squares regression of annual totals) and against independent reference
perimeters (NIFC incident perimeters, TNC preserve burn records, NCWRC
prescribed-fire records), reporting **recall (1 − omission) as the headline
metric** and precision only conditionally, since a perimeter is an outer boundary
with unburned islands rather than a spatially exhaustive census.

Years in which a product has no coverage at a pixel are left missing and dropped,
not coded as zero.

---

## 5. Covariates

Covariates are managed through a registry (`CovariateSpec` in
`modeling/covariates.py`) that isolates per-layer detail (path, band, nodata,
aggregation rule) behind one interface. Layers not yet built are registered but
skipped with a warning, so the pipeline degrades gracefully. Covariates split
along two axes that matter for how they are used.

### 5.1 Static (site-characteristic) layers — used for matching

| Covariate | Source | Role |
|---|---|---|
| `elevation` | Copernicus GLO-30 DEM | continuous |
| `histosol_pct` | gSSURGO major-histosol % | continuous |
| `gdd_normal` | GHCN stations → growing-degree-days (base 5 °C), IDW-interpolated | continuous |
| `soil_organic_matter` | SSURGO `om_r` (fuel-load proxy) | continuous |
| `soil_awc` | SSURGO `awc_r` (moisture-retention proxy) | continuous |
| `soil_site_index` | SSURGO-derived forest productivity | continuous |
| `soil_water_table_depth` | SSURGO April–June minimum, cm | continuous — **excluded**, see §5.4 |
| `soil_drainage_class` | SSURGO drainage class | categorical → exact-match key |
| `land_cover` | NLCD | categorical → exact-match key |
| `drainage` | HAND, area-weighted per unit | continuous |

Categorical layers are **never** used as numeric distance axes — class codes are
labels, not magnitudes — but as **exact-match strata**: a treated pixel may only
be matched to a control of the same class.

### 5.2 Temporal (per-year weather) layers — used in the outcome stage

`pdsi` — the **self-calibrating Palmer Drought Severity Index (scPDSI)**,
computed per station-month in R (`src/get_climate&soil_data_updated.R`) from GHCN
station precipitation and Hargreaves potential evapotranspiration
(`SPEI::hargreaves`, `scPDSI::pdsi(sc = TRUE)`), then interpolated to one raster
per calendar year. scPDSI is centred so ~0 is normal and negative values are
drought. Because it is self-calibrating, its long-run normal is ~0 everywhere —
it exists *only* as a per-year covariate and has no static counterpart.

### 5.3 Climate data construction

Climate arrives as **station points, not rasters**: GHCN daily records for the
Coastal Plain counties, pulled in R with `FedData::get_ghcn_daily` (1926–2026),
unit-converted, and read into Python from the resulting `.Rds` / GeoPackage
exports. Two products are derived from the same station records:

- a **long-run normal** per station (1991–2020 baseline), interpolated to the
  grid as a *static* site characteristic (this is how `gdd_normal` is built); and
- **one grid per calendar year**, as a *temporal* covariate (this is how `pdsi`
  is built).

Interpolation is **inverse-distance weighting** onto the 300 m grid from the $k$
nearest stations, with weights $w_i = d_i^{-p}$, $p = 2$, distances computed in
EPSG:5070 metres via a KD-tree. IDW was chosen over kriging deliberately: it is
transparent, dependency-light, and — because every prediction is a convex
combination of observed station values — it **never extrapolates** outside the
observed range. That is all a match-only covariate requires. Kriging can be
substituted later without changing the covariate contract (a GeoTIFF on the
grid) if uncertainty ever needs to be propagated.

### 5.4 Covariate-set decision: PDSI and GDD only

Per TNC's request (C. Chamberlain, 27 Aug 2026), the reported specification
retains **exactly two climate predictors — scPDSI and growing-degree days — and
drops all other temperature and precipitation metrics** (`precip_normal`,
`tmax_normal`, `tmin_normal`, and the per-year `precip` / `tmax` / `tmin`
layers). The rationale is that PDSI and GDD are the two mechanistically
interpretable summaries of the relevant climate signal — moisture deficit and
growing-season warmth — whereas raw temperature and precipitation normals are
strongly collinear with them and with each other, which inflates the effective
dimensionality of the matching distance without adding independent information.
In the code this is a single set at the top of the notebook:

```python
excluded_covariates = {"soil_water_table_depth",
                       "precip_normal", "tmax_normal", "tmin_normal",
                       "precip", "tmax", "tmin"}
```

`soil_water_table_depth` is excluded for a separate, empirical reason: it caused
a **positivity failure** — control pixels received near-zero propensity for the
later restoration cohorts, so the overlap the doubly-robust estimator needs was
not present.

---

## 6. Unit of analysis and panel construction

The unit is the **pixel-year**: one row per 300 m grid cell per calendar year.
Construction (`modeling/matching.py`, `modeling/frame.py`):

1. **Pixelate.** Lay the common grid over the AOI, take each cell centre as a
   point, spatially join to the treatment and candidate-pool polygons, and carry
   `End_Yr` and `Proj_Name` onto each pixel so a treated pixel remembers which
   site it belongs to.
2. **Stack across years.** Repeat the (expensive, one-time) pixel geometry once
   per calendar year to form the panel, adding `treated`, the cohort year, and
   event time `years_after_restoration = year − End_Yr`.
3. **Attach covariates.** Static layers vary only in space, so each raster is
   sampled **once per unique pixel** and broadcast across that pixel's years —
   identical result, no redundant raster reads, and no risk of a pixel's yearly
   rows being multiply counted in a static computation. Per-year layers are
   sampled per year and joined on `(x, y, year)`.
4. **Attach the fire response** at each pixel's location, per year.

**Cohort support restriction.** Before any scoring or matching,
`restrict_to_supported_cohorts` keeps never-treated controls and only those
restoration cohorts $g$ satisfying

$$\text{first outcome year} < g \le \text{last outcome year},$$

i.e. cohorts that have at least one pre-restoration and one post-restoration year
inside the fire record. Cohorts with no pre-period or no post-period are dropped
with a warning. **This restriction changes the estimand**: the reported ATT is
the effect for the *supported* cohorts only, and which cohorts those are depends
on the fire product. The longer MCD64A1 record supports strictly more cohorts
than FireCCI S3.1.1, which is the main reason it is the primary outcome.

---

## 7. Estimation

### 7.1 Route A — staggered difference-in-differences (primary)

NC sites were restored in different years, so there is no single pre/post split.
A pooled two-way fixed-effects regression is known to be biased under staggered
adoption with heterogeneous effects, because it implicitly uses *already-treated*
units as controls for later cohorts. The estimator used instead never pools:
**Callaway & Sant'Anna (2021) group-time ATTs**, one clean 2×2-style comparison
per cohort × year.

With $g$ a pixel's cohort (its site's restoration year; $g = 0$ for
never-treated), the building block is

$$\mathrm{ATT}(g, t) = \mathbb{E}\!\left[Y_t - Y_{g-1} \mid G = g\right] - \mathbb{E}\!\left[Y_t - Y_{g-1} \mid \text{control at } t\right],$$

comparing each cohort's outcome change since its last untreated year $g-1$
against the same change among clean controls — never-treated pixels **and**
not-yet-treated cohorts ($g' > t$), which are valid controls until their own
restoration year arrives.

Each $(g,t)$ cell is estimated by the **doubly robust** estimator of Sant'Anna &
Zhao (2020) (`est_method="dr"`), which combines an outcome-regression model for
the untreated change with an inverse-probability-weighted control comparison. It
is consistent if **either** the outcome model or the propensity model is
correctly specified. Conditioning on covariates relaxes plain parallel trends to
**conditional** parallel trends: trends need only be parallel among pixels with
similar covariates, which matters here because low-lying wet peat plausibly
responds to a drought year differently from higher ground.

The group-time family is aggregated two ways:

- **Simple / overall ATT** — the cohort-size-weighted average of all
  post-treatment cells. This is the headline number.
- **Event study** — the average effect at each event time $e = t - g$. The
  negative-$e$ points are pseudo-effects estimated entirely pre-treatment: under
  parallel trends they should straddle zero, so **the left half of the event
  study is the identifying-assumption check** and the right half is the dynamic
  path of the effect.

Backend: the pure-Python `differences` package, with the reference R `did`
package available through `rpy2` as a cross-check (the direct analogue of
Castro's Stata `csdid`).

**From ATT to physical impact.** Because the outcome is 0/1 and the outcome model
is linear in probability, the ATT is an *absolute* change in burn probability per
pixel-year, so avoided burned area follows directly:
$\text{avoided area} = -\widehat{\mathrm{ATT}} \times \text{restored area (ha)}$
per year (`did.avoided_area`).

### 7.2 Route A′ — Castro-style match-first DiD

Matching and DiD are complements, not substitutes: matching buys covariate
overlap, DiD differences out time-invariant confounders. Following Castro et al.
(2026), the primary specification does both, and matches on a **trajectory of
predicted risk** rather than a single collapsed covariate vector.

1. **Per-year prognostic scores** (`add_prognostic_score_series`) → `phat_<year>`.
   A separate regularized logistic fire-risk model is fit **for each outcome
   year, on never-treated control pixels only**, then predicted for every pixel.
   Because coefficients differ by year, the series encodes how baseline risk
   shifts through drought and wet years — something a single collapsed score
   cannot represent. Fit-on-controls / predict-for-all is the **no-leakage rule**:
   a treated pixel's score is a pure function of its covariates and never sees its
   own post-restoration burns.
2. **Per-vintage propensity scores** (`add_propensity_score_series`) → `psm_<g>`.
   One logistic per restoration cohort, with cohort-$g$ pixels as positives and
   never-treated pixels as negatives; other vintages are excluded from training
   (they would contaminate the contrast) but are still scored, so they remain
   eligible as candidate controls for cohort $g$. Rationale: siting criteria
   plausibly differed between program years. (Castro et al. fit per province ×
   vintage; NC has no province analogue, so this is per-vintage.)
   Matching on **both** scores is doubly robust matching (Leacy & Stuart 2014) —
   the same either-one-suffices logic as the DR estimator.
3. **Event-time 1:1 Mahalanobis match** (`match_controls_event_time`). Per
   cohort, the matching vector is
   $\big(\text{fire}_{g-1},\ \text{fire}_{g-2},\ \text{fire}_{2015},\ \hat p_g,\ \hat p_{g+1},\ \dots,\ \text{psm}_g\big)$
   — pre-construction fire history (including the 2015 drought year as a
   benchmark) plus the forward predicted-risk path, one coordinate per year.
   Matching is done **per vintage** (controls have no event time of their own, so
   the event window is translated to calendar years cohort by cohort), **without
   replacement** within a vintage, **exact-matched on land cover**, with a
   **caliper of 1.0** whitened SD. Components outside the fire record are omitted
   rather than causing failure.
4. **Match-first DiD.** `restrict_panel_to_matched` performs an inner join on
   `(x, y)`, so the DiD control pool becomes *exactly* the matched controls, and
   the Callaway–Sant'Anna estimator of §7.1 is refit on that panel.

**Matching distance.** All matching uses Mahalanobis distance,
$d_M(x_i,x_j) = \sqrt{(x_i-x_j)^\top \Sigma^{-1}(x_i-x_j)}$, which corrects both
for scale (elevation in metres vs. histosol in percent) and for correlation
between covariates. It is implemented by **whitening once** —
$z = \Sigma^{-1/2}(x-\mu)$, with a small ridge $\Sigma + 10^{-6}I$ so
near-collinear covariates still invert — after which ordinary Euclidean distance
among the $z$'s equals Mahalanobis distance among the $x$'s and fast KD-tree
nearest-neighbour search applies unchanged. A treated pixel with no control
inside the caliper is **dropped rather than matched to a poor twin**; the drop
count is reported as a design diagnostic.

**Balance.** Match quality is assessed with the **standardized mean difference**
per covariate, before vs. after matching:

$$\mathrm{SMD} = \frac{\bar{x}_{\text{treated}} - \bar{x}_{\text{control}}}{\sqrt{(s^2_{\text{treated}} + s^2_{\text{control}})/2}}$$

reported as a love plot (`balance_love.png`) with $|\mathrm{SMD}| < 0.1$ as the
acceptance threshold (Castro et al. accept ≤ 0.2). Automated contracts
(`check_matches`, `check_balance`) assert that every matched control is inside
the caliper and that $|\mathrm{SMD}|$ shrank for every covariate.

**Common support / positivity** is checked *before* matching with score-overlap
histograms (`plot_score_overlap`): a region of treated mass with no controls
underneath it is a positivity failure, and is where matching must either drop
treated pixels or stretch to poor twins. This check is what identified and
removed `soil_water_table_depth` (§5.4).

### 7.3 Route B — matched logistic model in levels

As a check under a different assumption (selection on observables rather than
parallel trends):

1. **1:1 nearest-neighbour match** on the whitened static covariates, within
   exact-match categorical strata, caliper 1.0 whitened SD, no replacement.
2. Fit

   $$\log\frac{p}{1-p} = \beta_0 + \beta_T\,\text{treated} + \beta^\top X$$

   on the matched pixel-year frame, with continuous predictors centred and
   scaled, and report $e^{\beta}$ with 95 % CIs. $\mathrm{OR}_T < 1$ with a CI
   below 1 means restoration lowers the odds of burning.
3. **Drought interaction.** Refit as
   `burned ~ treated * pdsi + covariates`, so the treatment log-odds effect
   becomes $\beta_T + \beta_{T\times \text{pdsi}}\cdot \text{pdsi}$ — a treatment
   effect that varies with that year's drought conditions. Since low PDSI means
   drought, a *positive* interaction coefficient indicates the protective effect
   is strongest in drought years, which is the mechanistically expected sign.
   Interactions are kept few and specified in advance; with rare events, testing
   many is fishing.

The model detects and reports rank deficiency and quasi-separation rather than
returning a silently degenerate fit. A Firth-penalized logistic is the documented
fallback for the rare-event case, and a mixed logistic with a site random
intercept (`fit_mixed_logit`) is available as a cross-check on the SEs.

---

## 8. Inference: why standard errors are clustered on the restoration site

This is the single most consequential inference decision in the analysis.

Restoration is assigned to a **site**, not a pixel. Every pixel inside one site
shares its canal blocks, its water table, its management, and every fire that
crosses it, so thousands of pixel-years are not thousands of independent draws.
Treating them as independent is pseudo-replication: it lets the standard error
shrink like $1/\sqrt{n_{\text{pixels}}}$ indefinitely, which is arithmetic, not
evidence, and makes essentially everything look significant.

All reported standard errors are therefore **clustered on the restoration site**
(`site_id`). Matched control pixels **inherit their matched treated pixel's
site**, so treated and control halves of a pair land in the same cluster.

- In Route B this is the cluster-robust (sandwich) variance estimator, which
  leaves the coefficients unchanged and allows arbitrary correlation *within* a
  cluster while assuming independence *across* clusters.
- In Route A, clustering above the entity level exists only on the
  **multiplier-bootstrap** path (neither `differences` nor R `did` accepts a
  cluster argument for the closed-form influence-function SEs). Site-clustered
  SEs are therefore obtained by averaging the estimator's per-unit influence
  functions within each site and drawing one Rademacher weight per site, with
  **1,000 bootstrap iterations** and a fixed seed. The point estimate is
  identical either way; only the variance moves. Requesting site clustering with
  zero bootstrap iterations raises an error rather than silently returning the
  deflated pixel-level SE.
- `att_collapsed` provides a dependency-free by-hand cross-check: collapse each
  pixel to one pre/post change, difference treated against control *within* each
  site to get one $\theta_s$ per site, average those, and take the SE from their
  spread with $t(G-1)$ — the textbook Bertrand–Duflo–Mullainathan collapse. It
  reports both SEs and their ratio, so the deflation is a number that can be
  quoted rather than asserted.

**Caveat, stated up front.** Cluster-robust inference is asymptotic in the number
of *clusters*, and this design has roughly six restoration sites — far below the
usual 30–50 rule of thumb. The site-clustered SE is far more honest than the
pixel-clustered one but is still optimistic. This motivates §9.

---

## 9. Design diagnostics and power analysis

Because the number of sites is small, the design's ability to detect an effect
was quantified explicitly (`src/peatfire/modeling/power.py`) rather than
inferred after the fact from a p-value. This should be read **before** any ATT,
and especially before interpreting a null.

1. **Collapse to the level treatment actually varies at.** `site_year_panel`
   reduces the pixel-year panel to a site-year panel. NC peat fire arrives as a
   handful of large, landscape-scale events — one fire paints thousands of
   contiguous pixels burned in a single draw — so the independent replicate is a
   site-year, not a pixel-year.
2. **`design_summary`** reports the real denominator: sites × pre/post years ×
   years containing any fire. If `post_years_with_fire` is 0 or 1, that single
   number explains the result and everything else is confirmation.
3. **`did_site_year`** fits a transparent two-way fixed-effects DiD on the
   site-year panel with $t(G-1)$ inference, as a legible cross-check on the
   Callaway–Sant'Anna fit.
4. **`randomization_inference`** holds the fire history fixed and permutes *which
   sites were restored and when* — which is precisely what the null hypothesis
   asserts. Unlike the cluster bootstrap, it is valid at six clusters, valid at
   three, and has no degenerate failure mode when a site-year contains no fire.
   This is the recommended p-value for this design.
5. **`minimum_detectable_effect` / `simulate_power` / `sample_size_curve`**
   simulate the observed design under a known true effect, to report the smallest
   effect this design could have detected and what a design powered at 80 % would
   require (in sites × years).

These diagnostics are reported alongside the ATT, not instead of it. Their role
is to convert an ambiguous "we found nothing" into a quantified "here is what
this design could and could not have detected, and here is what would be
required."

Supporting design checks:

- **Pre-trends**, from the negative-event-time half of the event study (§7.1).
- **Jackknife by site** — re-run leaving each site out; a stable ATT / odds ratio
  shows the estimate is not driven by one project.
- **Negative control** — run the full pipeline on *planned but not-yet-restored*
  sites at their scheduled years. Any "effect" found there is upstream
  confounding, not restoration.

---

## 10. Software, reproducibility, and quality control

- **Data acquisition (R):** `src/get_climate&soil_data.R` and
  `src/get_climate&soil_data_updated.R` — GHCN daily climate via
  `FedData::get_ghcn_daily`, growing-degree days, Hargreaves PET via `SPEI`,
  self-calibrating PDSI via `scPDSI`, and SSURGO soils via `FedData` (with
  patched download helpers for the upstream URL changes).
- **Analysis (Python 3.11):** an importable `peatfire` package
  (`pip install -e .`) so notebooks import it from anywhere without path hacks.
  Key dependencies: `geopandas` / `rioxarray` / `xarray` / `rasterio`
  (geospatial), `scikit-learn` (scores, KD-tree matching), `statsmodels`
  (cluster-robust GLM), `differences` (Callaway–Sant'Anna DiD), optionally
  `rpy2` for the R `did` cross-check. Environment pinned in `environment.yml`.
- **Structure:** `covariates.py` (registry + loaders), `climate.py` (station →
  raster), `soil.py` (SSURGO → raster), `matching.py` (pixel sets, scores,
  matching, balance), `frame.py` (tidy panel + fire response), `did.py`
  (staggered DiD), `models.py` (cluster-robust fits, odds ratios), `power.py`
  (design diagnostics), `plotting.py` (all diagnostic figures).
- **Staged contracts.** The pipeline is built in seven stages
  (load treated units → build candidate pool → pixelate → attach covariates →
  match → balance → assemble units), each with a `check_*` function asserting its
  invariants (treated pixels fall inside restoration polygons; candidate pool has
  zero overlap with treated; no all-NaN covariate column; every match inside the
  caliper; SMD improved). These are run after every change and serve as the
  regression-test suite.
- **Diagnostic figure at every step.** Geospatial errors are obvious on a map and
  invisible in a dataframe, so every stage saves a figure to
  `outputs/figures/modeling/`: covariate maps (a visually flat panel is a
  covariate with no spatial signal to match on), covariate scatter matrix,
  candidate/control pixel map, matched pairs in covariate space and in geography
  (long connecting lines flag controls drawn from far away — possible spatial
  confounding), love plot, score-overlap histograms, score maps, event study, and
  raw unadjusted burn rates by calendar year and by event time.
- **Descriptive layer.** Independently of any treatment contrast,
  `build_mask_frame` plus the burned-area-vs-covariate views summarise how burned
  hectares distribute across covariate values and years over the whole peat AOI
  (covariates binned into equal-count bins, since a 0/1 flag has no useful raw
  scatter). These are explicitly descriptive and are used to motivate what the
  causal models then try to identify.

---

## 11. Summary of design decisions

| Decision | Choice | Why |
|---|---|---|
| Sample frame | ≥ 80 % histosol | High-confidence peat; keeps non-peat out of the covariate distributions |
| Analysis CRS / grid | EPSG:5070, 300 m | Equal-area (areas meaningful), metric buffers, all layers aligned |
| Unit | Pixel-year | Native resolution of the outcome; falls back to site-year for inference |
| Treatment timing | `End_Yr` (completion), not start | Completion is when the site is actually re-wetted |
| Spillover buffer | 0 m | TNC assessment that projects do not spill hydrologically |
| Primary outcome | MCD64A1 (2001–2026) | Long record supports more cohorts and captures the 2008/2011 fires |
| Design | Match first, then DiD | Matching buys overlap; DiD removes time-invariant confounders |
| DiD estimator | Callaway–Sant'Anna, doubly robust | Staggered adoption makes two-way FE biased; DR is consistent if either nuisance model is right |
| Control group | Never-treated **and** not-yet-treated | Uses the pre-restoration years of later cohorts |
| Matching distance | Mahalanobis (whitened), caliper 1.0, 1:1, no replacement | Corrects scale and correlation; caliper drops rather than force-matches |
| Categorical covariates | Exact-match strata | Class codes are labels, not magnitudes |
| Climate predictors | scPDSI + GDD only | Mechanistically interpretable; other temp/precip metrics are collinear (TNC request, Aug 2026) |
| Interpolation | IDW ($p=2$), not kriging | Transparent, never extrapolates; adequate for a match-only covariate |
| Clustering | Restoration site, 1,000-draw multiplier bootstrap | Treatment is assigned at the site; pixel-level SEs are pseudo-replication |
| Power | Reported explicitly, before the ATT | With ~6 sites, a null is uninformative unless the design's sensitivity is quantified |

---

## 12. Key assumptions and limitations

1. **Few clusters.** ~6 restoration sites; cluster-robust inference is asymptotic
   in the number of clusters. Randomization inference and the event study should
   carry more weight than the bootstrap interval.
2. **Conditional parallel trends is untestable directly.** Only its
   pre-treatment proxy is testable. If sites were selected for restoration partly
   *because* they had recently burned or were visibly degraded, that is an
   Ashenfelter dip, and adding covariates does not repair it. (Note that
   post-fire rehabilitation is a real pattern in this landscape.)
3. **Control geometry differs from the source design.** Castro et al. draw
   controls from a within-buffer donut around each canal block; here they are
   matched landscape peat, so conditional parallel trends leans harder on match
   quality — which is why the balance diagnostics are treated as the acceptance
   test of the design.
4. **Small control pool.** Per-vintage matching + no replacement + a land-cover
   exact-match constrains candidates hard. Per-cohort drop counts are monitored.
5. **Positivity.** `soil_water_table_depth` had to be excluded; adding further
   covariates would worsen rather than improve overlap.
6. **Outcome–mechanism gap.** This is the most important limitation.
   A 300 m burned-area product measures burned *extent*. The literature
   (Richardson et al. 2022; Reardon et al. 2007; Flanagan et al. 2020) indicates
   that restored pocosins **still burn at the surface** — surface fire is natural
   in these systems at 20–80 year intervals — and that what re-wetting changes is
   **how deep the fire burns into the peat**. Burn depth is precisely what a
   burned-area product cannot see. Any extent-based estimate is therefore testing
   a weaker version of the mechanism than the one the carbon accounting depends
   on.
7. **Spatial dependence within a year** is handled defensively through the
   site-clustered SEs rather than modelled explicitly; fire-history lags
   (temporal and 4-neighbour spatial, following Castro et al.'s Eq. 1) are
   implemented in `add_fire_lags` and can be added as DiD covariates if fire
   contagion should be modelled directly.

---

## References

**Matching / causal inference**

- Rosenbaum, P. & Rubin, D. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika* 70(1), 41–55.
- Hansen, B. B. (2008). The prognostic analogue of the propensity score. *Biometrika* 95(2), 481–488.
- Leacy, F. P. & Stuart, E. A. (2014). On the joint use of propensity and prognostic scores in estimation of the ATT. *Statistics in Medicine* 33(20), 3488–3508.
- Stuart, E. A. (2010). Matching methods for causal inference: a review and a look forward. *Statistical Science* 25(1), 1–21.

**Difference-in-differences**

- Callaway, B. & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. *Journal of Econometrics* 225(2), 200–230.
- Sant'Anna, P. H. C. & Zhao, J. (2020). Doubly robust difference-in-differences estimators. *Journal of Econometrics* 219(1), 101–122.
- Bertrand, M., Duflo, E. & Mullainathan, S. (2004). How much should we trust differences-in-differences estimates? *QJE* 119(1), 249–275.

**Peat, restoration, and fire**

- Castro, A. et al. (2026). Effective restoration can avoid peatland fires: large-scale counterfactual assessment in Kalimantan, Indonesia. *iScience.* doi:10.1016/j.isci.2026.116041 — the design this pipeline transplants.
- Richardson, C. J. et al. (2022). Annual carbon sequestration and loss rates under altered hydrology and fire regimes in southeastern USA pocosin peatlands. *Global Change Biology* 28, 6370–6384.
- Reardon, J. et al. (2007). Peat consumption vs. water-table position, Green Swamp, NC.
- Flanagan, N. E. et al. (2020). Prescribed-fire peat loss at Pocosin Lakes NWR.
- Mickler, R. A. et al. (2017). Deep peat fire consumption, Pocosin Lakes.
- Poulter, B. et al. (2006). Pocosin fire return intervals and emissions.
- Humber, M. L. et al. (2019). Spatial and temporal intercomparison of four global burned-area products. *International Journal of Digital Earth* 12(4), 460–484.

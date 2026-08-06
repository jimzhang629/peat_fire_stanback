# Peat condition → fire: the master modeling guide

This is the single reference document for the modeling pipeline. It explains
`notebooks/modeling.ipynb` at **nested levels of abstraction** — the outermost
bullets are the big ideas, each level of indentation steps down one level, ending
at what the individual function calls actually do — and then goes one level
deeper still, into **the math and statistics** each step implements.

It absorbs and replaces four earlier documents:

- `modeling_roadmap.md` — the design rationale (now Parts I, V, VI, VIII)
- `decisions.md` — the decision log (modeling decisions now in Part V; the
  fire-comparison and validation-toolkit decisions preserved in Appendix A)
- `score_matching_and_did.md` — the score-matching / match-first-DiD
  implementation walkthrough (now folded into Parts III and IV)
- `matching_assignment.md` — the staged matching assignment (now Part VII)

Contents:

- **[Part I — The big picture](#part-i--the-big-picture)**
- **[Part II — Code map](#part-ii--code-map)**
- **[Part III — Cell-by-cell walkthrough](#part-iii--cell-by-cell-walkthrough)**
- **[Part IV — The math and statistics](#part-iv--the-math-and-statistics)**
  (potential outcomes, Mahalanobis matching, **propensity scores**, prognostic
  scores, logistic/odds ratios/clustered SEs, **the ATT and staggered DiD**, and
  **how time-varying covariates are handled**)
- **[Part V — Design-decision log](#part-v--design-decision-log-modeling)**
- **[Part VI — The Castro et al. (2026) blueprint and NC caveats](#part-vi--the-castro-et-al-2026-blueprint-and-nc-caveats)**
- **[Part VII — The matching stages, and how to test them](#part-vii--the-matching-stages-and-how-to-test-them)**
- **[Part VIII — Extensions and next steps](#part-viii--extensions-and-next-steps)**
- **[References](#references)**
- **[Appendix A — Fire-product comparison & validation toolkit decisions](#appendix-a--fire-product-comparison--validation-toolkit-decisions)**

---

## Part I — The big picture

- **The question is causal, not predictive:** does peatland restoration (and,
  more generally, peat hydrological condition) reduce fire? Restoration re-wets
  drained peat, so restored pixels should burn less than comparable unrestored
  pixels. Peat condition is the treatment; everything else (climate, elevation,
  land cover, soil, histosol %) is a confounder to adjust or match for.
  - **Treatment** = pixels inside a *completed* restoration site, "on" from that
    site's restoration year (`End_Yr`) forward.
  - **Control** = peat pixels (≥80% histosol) outside every restoration site and
    its spillover halo.
  - **Response** = a 0/1 "did this pixel burn this year" flag sampled from a
    swappable satellite fire product (FireCCIS311, ~300 m, 2019–2024). Swapping
    `product=` re-runs everything for e.g. burn severity.
- **Why not just one big regression?** The natural first move — stack every
  pixel, fit `burned ~ treated + covariates`, read off the betas — is the right
  *shape* but untrustworthy for three reasons; the whole pipeline exists to fix
  them:
  1. **Pixels are not independent.** Pixels inside one restoration site (or one
     fire) share unmeasured conditions. Treating millions of pixels as
     independent shrinks standard errors toward zero, so *everything* looks
     significant. The effective sample size is closer to sites × years.
     → cluster-robust SEs / mixed models ([M6](#m6--the-logistic-model-odds-ratios-and-cluster-robust-standard-errors)).
  2. **Restored peat is not located like drained peat.** Restoration sites
     differ systematically in elevation, coast distance, land cover. A plain GLM
     silently extrapolates across covariate space with no comparable control,
     and the treatment beta absorbs the imbalance. → match controls first
     ([M3](#m3--balance-and-the-standardized-mean-difference)–[M5](#m5--the-prognostic-score)).
  3. **Fire is rare.** Burned pixels are a tiny fraction of peatland-years, so
     ordinary logistic fits are biased and over-confident. → penalized fits,
     and design (matching) rather than model complexity doing the work.
- **Two estimation routes, one shared front end**, answering the same question
  under different assumptions:
  - **Route 1 — changes (staggered DiD, §2b/2c):** compare the *change* in
    burning after each site's restoration against not-yet/never-restored
    controls (Callaway & Sant'Anna 2021, following Castro et al. 2026). All
    *time-invariant* confounders — even unmeasured ones — difference out.
    Headline number: the **ATT**, the change in burn probability
    ([M7](#m7--difference-in-differences-and-the-att)).
  - **Route 2 — levels (matched logistic, §3–4):** match each treated pixel to a
    covariate twin, then fit `burned ~ treated + covariates` with SEs clustered
    by restoration site. Headline number: the **odds ratio** for `treated`
    (< 1 ⇒ restoration lowers fire odds) ([M6](#m6--the-logistic-model-odds-ratios-and-cluster-robust-standard-errors)).
  - Agreement between the two (and between the collapsed-score match and the
    trajectory match) is the strongest sign of a design-robust effect.
- **One shared geometry:** everything — fire products, covariates, pixels — is
  warped onto the same EPSG:5070 (metres, equal-area) grid at 300 m via
  `build_common_grid`/`to_common_grid`, so every layer aligns cell-for-cell,
  areas are meaningful, and buffers/distances are in metres.

## Part II — Code map

- `notebooks/modeling.ipynb` — orchestration + diagnostics; Part III below.
- `src/peatfire/modeling/`
  - `covariates.py` — a registry (`CovariateSpec`) of every environmental layer
    on disk + generic loaders that clip/standardize/warp them. Static and
    per-year (temporal) covariates are separate registries.
  - `climate.py` — GHCN station records → interpolated climate rasters
    ([M2](#m2--interpolating-station-climate-inverse-distance-weighting)).
  - `soil.py` — SSURGO soil polygons → rasterized soil covariates.
    `build_soil_rasters` aggregates the *raw* relational SSURGO export
    (`nc_soil_ssurgo.gpkg`); `build_soil_database_rasters` adds the two extra
    layers carried on Cat's pre-aggregated `soil_database.gpkg`
    (`soil_site_index`, `soil_water_table_depth`).
  - `matching.py` — pixel sets, covariate sampling, scores, and all matching
    ([M3](#m3--balance-and-the-standardized-mean-difference)–[M5](#m5--the-prognostic-score)).
  - `frame.py` — restoration-site loaders, `build_frame` (tidy pixel-year
    table), `attach_fire_response` (fire response at pixel points).
  - `did.py` — staggered difference-in-differences: panel prep (`prepare_panel`),
    ATT fit (`fit_att` → `estimate_att`), aggregation ([M7](#m7--difference-in-differences-and-the-att)).
  - `models.py` — cluster-robust logistic/OLS fits + odds-ratio tables
    ([M6](#m6--the-logistic-model-odds-ratios-and-cluster-robust-standard-errors)).
  - `plotting.py` — every diagnostic figure the notebook saves.

---

## Part III — Cell-by-cell walkthrough

### Setup (imports cell)

- Imports the toolkit, calls `set_fire_style()` (shared matplotlib style,
  Okabe-Ito colour-blind-safe palette), and creates
  `FIG_DIR = outputs/figures/modeling/` — **every figure is saved there**.
- Prints `available_covariates()`: registered covariates whose files exist on
  disk right now. Downstream cells select against this list, so the notebook
  degrades gracefully when a layer hasn't been built yet.

### §1 — Load treatment + peat frame

- `load_completed_restoration_sites_in_analysis_crs(restoration_yr_col='End_Yr')`
  → `peat_restoration`, the **treatment polygons**.
  - Reads the TNC restoration shapefile, reprojects to EPSG:5070.
  - Replaces `End_Yr == 0` (placeholder for "not finished") with NaN and drops
    those rows — "completed" is *defined* as "has a real end year". The end
    year, not the start year, is the moment a site flips from unrestored to
    restored; everything downstream (per-year `treated`, DiD cohort `g`) keys
    off it.
- `gpd.read_file(...nc_peatlands_80_histosol_aoi.gpkg)` → `aoi_nc_peat_80_histosol`,
  the **sample frame**: the part of NC that is ≥80% histosol (high-confidence
  peat; the 80% threshold keeps non-peat noise out of the covariate
  distributions). Everything — grid extent, candidate controls, covariate
  clipping — happens inside this AOI.

### §1b — Build the climate + soil covariates, then map every layer

- Sets the two constants reused everywhere: `RES_M = 300` (fire-product native
  resolution) and `YEARS = range(2019, 2025)` (fire-product coverage).
- **Climate (from GHCN weather stations — points, not rasters):**
  - `climate.load_ghcn_stations(...)` reads each `.Rds` long table (one row per
    station-day, exported by `src/get_climate&soil_data.R`, already
    unit-converted) into a GeoDataFrame of station points in EPSG:5070.
  - `build_climate_normals(...)` → **static** covariates for *matching*
    (`precip_normal`, `tmax_normal`, `tmin_normal`).
    - Internally: `station_normals` collapses each station's daily record over
      the 1991–2020 baseline (annual precip total / mean temperature), then
      `idw_to_grid` interpolates the station values onto the 300 m grid
      (inverse-distance weighting; math in [M2](#m2--interpolating-station-climate-inverse-distance-weighting)).
    - Why a *normal*: long-run climate is a stable site characteristic — the
      right thing to match treated and control pixels on. A single year's
      weather is noise from the match's point of view (see
      [M8](#m8--where-does-time-go-static-vs-per-year-covariates)).
  - The **growing-degree-days normal** (`gdd_normal`) is built the same way —
    `build_climate_normals` with `climate.DEFAULT_GDD_ELEMENTS` — but off Cat's
    monthly export `clim_monthly.gpkg` (`get_climate&soil_data_updated.R`)
    instead of the daily frames. That file already carries a per-station-year
    `totalGDD` (base-5 °C growing-season warmth, repeated across the year's 12
    month rows), so the within-year reduce is a mean over duplicates and the
    across-years mean is the normal. It has no `STATION` column and a lowercase
    `year`, so the call passes `year_col="year"`. Being a stable site
    characteristic, it too is a **static** match covariate.
  - `build_annual_climate(...)` → **temporal** covariates for the *outcome
    stage* (`precip`, `tmax`, `tmin`, one raster per year).
    - Same station → IDW path, but per calendar year: this year-specific
      weather is what a `treated:precip` dry-year interaction needs, joined per
      `(x, y, year)` later by `build_frame`.
  - The **scPDSI drought index** (`pdsi`) reuses the same per-year path from its
    own long table. It is self-calibrating (~0 = normal, negative = drought), so
    its long-run normal is ~0 everywhere — it exists *only* as a temporal
    covariate, entering §4 as a `treated:pdsi` interaction.
  - `write_climate_normals` / `write_annual_climate` save GeoTIFFs under
    `data/processed/climate/` where the covariate registry picks them up.
- **Soil (from the SSURGO GeoPackage — polygons, not rasters):**
  - `inspect_soil_columns(...)` prints the attribute columns so you can verify
    names before rasterizing.
  - `build_soil_rasters(...)` aggregates attributes to one value per soil map
    unit (`mukey_attribute`, depth-weighted where relevant) and burns them onto
    the grid: continuous layers (organic matter `om_r` — a fuel-load proxy;
    available water capacity `awc_r` — a moisture-retention proxy) and
    categorical ones (drainage class, factorized to codes and used as an
    exact-match key, never as a numeric distance axis). These come from the
    *raw* relational SSURGO export `nc_soil_ssurgo.gpkg`
    (`get_climate&soil_data.R`).
  - `build_soil_database_rasters(...)` then **augments** those with two more
    continuous layers off Cat's *pre-aggregated* `soil_database.gpkg`
    (`get_climate&soil_data_updated.R`): `soil_site_index` (forest
    productivity, from `industrial`) and `soil_water_table_depth` (April–June
    minimum, cm, from `wtdepaprjunmin`). These sit directly on the polygons, so
    it is a thin wrapper over `build_soil_rasters` that needs no MUKEY
    aggregation — run it *in addition to*, not instead of, `build_soil_rasters`.
- Each build is wrapped in `try/except` so a missing input file skips that block
  instead of killing the notebook.
- `plot_covariate_maps(...)` draws every static covariate over the AOI with
  restoration sites outlined. **Reading it:** a visually flat panel (e.g.
  histosol %, pinned ~90 across the 80% frame) has no spatial signal to match
  on — the reason matching on elevation alone was degenerate and climate/soil
  were added.
- Two inspection cells follow: the raw GHCN long tables (row counts, station
  counts, year spans, value distributions) and the soil columns.

### §2 — Build the treated/control pixel-year panel

- Selects the matching axes from what's on disk:
  - `covariates` = every registered **continuous** covariate present
    (elevation, histosol %, climate normals including the GDD normal, soil
    organic matter/AWC, site index, water-table depth) — these become distance
    axes.
  - `categorical` = every registered **categorical** covariate present
    (drainage class, land cover) — these become *exact-match* keys (class codes
    are labels, not magnitudes, so they don't belong in a distance; a treated
    pixel may only match controls of the same class).
- `get_treated_and_control_pixels(...)` → `pixels`, one row per pixel per
  calendar year. Internally:
  - `build_candidate_pool` — `gpd.overlay(peat_aoi, buffered_sites,
    how="difference")`: peat minus the restoration sites and a `spillover_m`
    halo around them (here 0: TNC prevents spillover into neighboring
    communities). What's left is the control candidate pool.
  - `pixelate` — lays the common 300 m grid over the AOI, takes each cell
    centre as a point, keeps points inside the polygons (a spatial join), and
    carries polygon attributes (`End_Yr`, `Proj_Name`) onto each pixel so a
    treated pixel remembers *which* site (and restoration year) it belongs to.
  - `_stack_across_years` — repeats the (expensive, one-time) pixel geometry
    once per calendar year to form the panel.
  - Per year, per pixel: `treated = 1` only once the pixel's site has been
    restored (`year ≥ End_Yr`); before that it is a **not-yet-treated control**
    (exactly what the staggered DiD wants); `years_after_restoration = year −
    End_Yr` is the event time. Candidate-pool pixels are `treated = 0` in every
    year with no restoration year.
  - Two different notions of "treated" now live in the panel — keeping them
    straight is essential:
    - `treated` (per-year flag) — flips at the restoration year. Used by the
      **DiD** (a pixel is genuinely untreated before restoration).
    - restoration-site **membership** (`End_Yr.notna()`) — constant per pixel.
      Used by **matching and scoring** (a restoration pixel is a treated *unit*
      regardless of year; we match on its static geography).
- `attach_covariates(...)` → `pixels_with_covariates`.
  - Static layers vary only in space, so each raster is sampled **once per
    unique pixel** (nearest grid cell) and broadcast across that pixel's years —
    identical result, no redundant raster reads. This *collapse-then-broadcast*
    pattern (`drop_duplicates(["x","y"])` → work → merge back on `(x, y)`)
    recurs throughout `matching.py`; it prevents a pixel's six yearly rows from
    being triple-counted in any static computation.
  - Per-year layers (`precip`, `tmax`, `tmin`, `pdsi`), when built, are sampled
    per calendar year and joined on `(x, y, year)` — each pixel-year gets *that
    year's* value.
- Diagnostics: `plot_covariate_space` (treated vs control on two axes) and
  `plot_covariate_pairs` (full scatter matrix). A flat axis = nothing to match
  on there.

### §2b — Event-time panel → staggered DiD

- `attach_fire_response(pixels_with_covariates, aoi, product="FireCCIS311",
  res_m=RES_M)` → `panel` with a `burned` column.
  - For each year: load the standardized fire raster, warp it onto the common
    grid (`how="max"`: any burned sub-cell lights the cell), then read the value
    at each pixel's `(x, y)`. Years without coverage stay NaN.
- `did.prepare_panel(panel, restoration_yr_col, site_col="Proj_Name")` adds the
  three bookkeeping columns the estimator needs:
  - `unit_id` — one stable entity id per distinct `(x, y)` pixel.
  - `g` — the Callaway–Sant'Anna **cohort**: each site's first-treatment
    (restoration) year, looked up from the panel itself via `did.attach_cohort`;
    control pixels have no site and get `g = 0` (never-treated). It errors
    loudly if a *treated* row has no cohort year.
  - `site_id` — the key the standard errors cluster on (see M7). Left alone if
    the frame already carries it (the matched panel does, with each control
    inheriting its treated partner's site); otherwise built from `Proj_Name`,
    with unmatched control pixels — which belong to no site — given their own
    singleton cluster and a warning saying how many.
- `did.fit_att(panel, covariates=covs, response="burned")` runs the whole
  estimator and returns `(att, overall, event_study)`:
  - `build_panel` — validates and reshapes to a `(unit_id, year)`-indexed
    panel: drops NaN-response rows, requires uniqueness on (entity, time) and
    the presence of *both* treated cohorts and never-treated controls.
  - `estimate_att` — the doubly-robust group-time ATT; the full math is
    [M7](#m7--difference-in-differences-and-the-att). Backend: the pure-Python
    `differences` package, or `backend="rpy2"` for the reference R `did`
    package (the direct analogue of Castro's Stata `csdid`).
  - `aggregate_att(kind="simple")` — one overall ATT (the headline);
    `aggregate_att(kind="event")` — ATT by time-since-restoration.
  - Wrapped in `try/except` because `differences` is an optional dependency —
    the levels route (§3–4) still runs without it.
- `plot_event_study(event_study)` — ATT vs event time. **Reading it:** the
  pre-treatment points (left of the onset line) are the parallel-trends check
  and should straddle zero; the post-treatment points trace how the effect
  evolves after restoration.
- `plot_raw_burn_rate_by_year` / `plot_raw_burn_rate_by_event_time` — the
  *unadjusted* burn rate for treated vs control, by calendar year (shared
  dry-year swings appear in both groups) and re-aligned to event time. This is
  the model-free signal the DiD and logit are built to explain.

### Burned area vs the covariates, across years (descriptive, mask-wide)

- No treatment contrast here at all: the question is how the fire product's
  **burned hectares** distribute over the peat AOI's covariates and years.
- `build_mask_frame(aoi, product=..., years=..., res_m=...)` — the same tidy
  pixel-year table as `build_frame`, but over every cell in a plain area **mask**
  instead of matched units, plus a constant `pixel_area_ha` so burned area is a
  weighted sum.
- `plot_burned_area_vs_covariate` — covariate on x, burned area on y, one line
  per year (the covariate is binned into equal-count bins first, because a 0/1
  `burned` flag has no useful raw scatter).
- `plot_burned_area_covariate_heatmap` — year on x, covariate on y, burned area
  as a hot colour scale. **Reading it:** a bright band that *drifts* across
  columns means the covariate level where fire concentrates moves year to year;
  a stable bright row means the covariate–fire relation holds across years.
- `plot_burned_area_and_covariate_by_year` — annual burned-area bars with the
  mask-mean covariate overlaid, the "do the big fire years line up with the dry
  years?" picture (per-year covariates only; a static layer is flat by
  construction). Its annotated correlation is descriptive — with a handful of
  panel years it is far too noisy to be inference.
- `plot_annual_burned_area_vs_covariate` — the same two annual numbers plotted
  against *each other*: mask-aggregate covariate on x, mask burned area on y, one
  labelled point per year. The quickest glance, and the most limited: one point
  per panel year, and a static covariate puts every point on a single vertical
  line, because collapsing the mask to one number per year discards exactly the
  spatial variation those layers carry.
- **The split that matters:** the first two views bin *within* each year, so they
  keep the spatial variation (~20k pixels per year) and work for static layers;
  the last two collapse the mask to one number per year, so they show the
  year-to-year driver and go degenerate on anything static.
- All three take `metric`: `"area_ha"` (absolute hectares — a count, so a bin
  covering more of the landscape can lead just by being bigger), `"rate"` (burned
  fraction of the bin's pixels, exposure-free), `"share"` (percent of *that
  year's* burned area, which stops one big fire year from dominating a colour
  scale). `plot_burned_area_vs_covariates_over_years` drives the lot over every
  continuous covariate and saves them per covariate.

### Match controls + balance (Stage 5–7) → `units`

- The causal-design heart of the *levels* route: pair every treated pixel with a
  covariate twin, then *prove the pairing worked*. (The staged assignment behind
  these functions, with its testing recipes, is Part VII.)
- `balance_table(pixels_with_covariates, ...)` → `before`: the standardized mean
  difference per covariate between treated and control **before** matching
  (formula in [M3](#m3--balance-and-the-standardized-mean-difference)).
- `match_controls(pixels_with_covariates, continuous=covariates,
  categorical=categorical, caliper=1.0, k=1, replace=False, ...)` → `matched`.
  Internally:
  - Collapses the year panel to unique pixels; the treated *group* is
    restoration-site membership, not the per-year flag.
  - **Whitens** the covariate matrix so plain Euclidean distance equals
    **Mahalanobis** distance (math and why in [M4](#m4--matching-distance-standardization-whitening-mahalanobis)),
    with a small ridge on the covariance so near-collinear covariates still
    invert.
  - **Exact-match strata:** the nearest-neighbour search runs *within* each
    categorical class (drainage class / land cover) via a KD-tree.
  - **Caliper** (1.0 whitened SD): a treated pixel with no control that close is
    *dropped* rather than matched to a poor twin (the drop count is printed — a
    design diagnostic, not noise).
  - Optional **geographic caliper** `max_dist_m` (metres): controls must also be
    physically near their treated pixel; eligibility is applied
    geography-first (KD-tree radius query), *then* candidates are ranked by
    covariate distance.
  - `replace=False`: each control serves at most one treated pixel.
  - `site_id` on the output is the restoration site (`Proj_Name`); a control
    *inherits its matched treated pixel's site*, so downstream standard errors
    cluster on the handful of sites, not thousands of correlated pixels.
- `balance_table(matched)` → `after`; then the automated contracts:
  `check_matches` (every control within the caliper) and `check_balance`
  (|SMD| shrank per covariate; |SMD| < 0.1 is "balanced").
- `plot_balance(before, after)` — the **love plot** (`balance_love.png`), the
  figure that justifies the design.
- `assemble_units(matched)` → `units` (`unit_id`, `site_id`, `treated`,
  geometry), what `build_frame` consumes in §3.
- The matched *control* pixels are saved to
  `processed/peat_restoration/matched_controls/matched_controls.gpkg` for
  inspection in the fire-comparison notebook.
- Per-pair diagnostics:
  - `plot_matched_pairs_covariate` — pairs connected in covariate space (short
    segments = close twins).
  - `plot_matched_pairs_geographic` — pairs on the map (long lines flag controls
    drawn far away — possible spatial confounding from unmeasured differences).
  - `plot_candidate_control_pixels` — full candidate pool (grey) vs selected
    controls (blue) vs treated (orange) over NC.

### §2c — Faithful Castro (2026) matching: score *trajectories*, then match-first DiD

- Motivation: the §2 match compares pixels on a single collapsed covariate
  vector. Castro et al. instead match on a **time series of predicted risk**, so
  pixels are twins not just on average but through the dry-year spikes. (This is
  the "faithful track"; `add_matching_scores` — one pooled `pscore` + one
  collapsed `phat` per pixel — is the simpler "scalar track" alternative that
  collapses time away, and its `.max()`-over-years baseline is exactly the
  limitation the per-year series fixes.)
- Preparation: `panel` is promoted to a GeoDataFrame (`scored`) and any
  categorical land-cover layer is sampled onto it for exact matching.
- `add_prognostic_score_series(...)` → one `phat_<year>` column per outcome
  year.
  - A *separate* logistic fire-risk model per year — fit **only on never-treated
    control pixels** (`End_Yr.isna()`; Castro's "control pixels that never had a
    block") — then predicted for everyone. Because coefficients differ by year,
    the series encodes how baseline risk shifts across dry/El Niño years, which
    a single collapsed score cannot. Fit-on-controls, predict-for-all is the
    **no-leakage rule**: a treated pixel's score never sees its own
    post-restoration burns ([M5](#m5--the-prognostic-score)).
- `add_propensity_score_series(...)` → one `psm_<g>` column per restoration
  **vintage** `g`.
  - One logistic per cohort: pixels restored in year `g` as positives,
    never-treated controls as negatives, *other vintages excluded from training
    but still scored* — the coefficients capture each vintage's siting
    decisions, and every pixel gets a score so it can enter cohort-`g` matching
    as a candidate. (Castro fit per *province × vintage*; NC has no province
    layer, so this is per-vintage.) Deep dive on propensity scores:
    [M4b](#m4b--the-propensity-score).
- Score diagnostics: `plot_prognostic_trajectory` (predicted risk over time,
  IQR band, observed never-treated burn rate overlaid as validation — the
  spikes are the dry years) and `plot_score_overlap` (the **common support /
  positivity** check: treated mass where no controls sit is where matching must
  drop or stretch — run it *before* matching to judge feasibility).
- `match_controls_event_time(scored, ...)` → `matched_et`: per-vintage 1:1
  Mahalanobis match on the **trajectory vector**. It is an *orchestrator*: per
  cohort it assembles the event-time-anchored column list and delegates the
  actual pairing to `match_controls`, so the whitening, caliper, exact-match and
  no-replacement logic are reused, not re-implemented. Per vintage `g` the
  vector contains:
  - pre-construction fire-history lags: the *actual* burn at `g−1`, `g−2`, and
    the 2015 drought benchmark — each its own coordinate, when in coverage
    (out-of-coverage years are silently omitted rather than crashing);
  - the forward prognostic path `phat_t` for `t ≥ g`;
  - the vintage propensity `psm_g`.
  - Why per-cohort? Controls have no event time of their own — "one year after
    restoration" only means something relative to a specific `g`. So the event
    window is translated to calendar years per vintage, every candidate control
    is evaluated at those same calendar years, and matching repeats per vintage
    (controls offered afresh to each vintage; no-replacement applies within a
    vintage — mirroring Castro).
  - Exact-matched on land cover; deliberately **no** `site_id` exact-match (too
    few NC sites; SEs cluster on site downstream instead). `pair_id` is kept
    globally unique across cohorts; `cohort` records the vintage.
- `plot_score_map` + the pair plots — scores per pixel on the map with each
  treated pixel joined to its matched control (shape = treatment, colour =
  score, shared scale; same colour at both ends of a line = close score match).
  Scores are looked up from `scored`, since the match output carries pixel ids,
  not the ragged per-cohort score columns.
- Match-first DiD: `restrict_panel_to_matched(scored, matched_et)` filters the
  panel to the surviving pixels — an **inner** join on `(x, y)` (the
  load-bearing word: every unmatched candidate is dropped, so the DiD's control
  pool becomes *exactly* the matched controls), carrying the match's
  `site_id`/`pair_id` — then the same `did.prepare_panel` → `did.fit_att`
  pipeline as §2b. The ATT is now identified against each treated pixel's
  matched control(s): covariate overlap from the match *plus* differencing-out
  of time-invariant confounders from the DiD.

### §3 — Build the tidy pixel-year frame (levels route)

- `build_frame(units, product="FireCCIS311", years=YEARS,
  site_id_col='site_id')` → `frame`, one row per (unit-pixel × year).
  Internally:
  - Rasterizes `unit_id` / `treated` / `site_id` onto the common grid
    (`all_touched=True` so small units survive the coarse grid).
  - Attaches static covariates: values already sampled upstream by the match are
    burned from the units table; anything else is read from its raster.
  - Per year: loads the standardized fire product (`burned`, via `max` warping)
    and joins the per-year weather (`precip`, `tmax`, `tmin`, `pdsi`) on
    `(x, y, year)`.
  - Drops cell-years where the product had no coverage.
- Prints the raw burn rate by treatment as a first look.

### §4 — Fit + odds ratios

- `fit_logit_clustered(frame, covariates=covs)` fits
  `burned ~ treated + elevation + precip_normal + tmax_normal + gdd_normal +
  soil_organic_matter + soil_awc + soil_site_index + soil_water_table_depth`
  (whichever are actually in the frame — selected against `frame.columns`, so a
  layer that wasn't built in §1b is simply skipped).
  Internally:
  - standardizes continuous predictors for numerical stability,
  - reports actionable diagnoses for rank-deficiency or perfect separation,
  - clusters standard errors on `site_id` ([M6](#m6--the-logistic-model-odds-ratios-and-cluster-robust-standard-errors)).
- `odds_ratios(result)` → `or_table`: `exp(β)` with 95% CIs. **Reading it:**
  `treated` OR < 1 with CI below 1 ⇒ restoration lowers the odds of burning.
- **Weather interactions** (one loop over `precip` and `pdsi`): refits with
  `burned ~ treated * <weather> + covariates`.
  - `treated:precip > 0` ⇒ the fire-lowering effect weakens as annual rain
    rises, i.e. restoration does most of its work in dry years.
  - `treated:pdsi`: low PDSI = drought, so a *positive* coefficient likewise
    means the effect is strongest in drought years.
  - Each runs only if its per-year rasters were built in §1b.
- Presentation: a colour-graded styled table for slides, and the headline forest
  plot `odds_ratios.png` — one point + CI per term with the dashed no-effect
  line at OR = 1.

---

## Part IV — The math and statistics

This part explains what the pipeline *computes*, from first principles. Sections
are numbered M1–M9 and cross-referenced from the walkthrough above.

### M1 — Potential outcomes and the estimand

- For each pixel $i$ (in year $t$), imagine two **potential outcomes**:
  $Y_i(1)$ = would it burn if its site were restored, and $Y_i(0)$ = would it
  burn if not. The causal effect for pixel $i$ is $Y_i(1) - Y_i(0)$ — and it is
  fundamentally unobservable, because each pixel is only ever one of the two.
- The pipeline's target is the **average treatment effect on the treated**:

$$\mathrm{ATT} = \mathbb{E}\left[\,Y(1) - Y(0) \mid D = 1\,\right]$$

  where $D=1$ marks treated (restored) pixels. $\mathbb{E}[Y(1) \mid D=1]$ is
  just the observed burn rate of restored pixels; the entire difficulty is the
  **counterfactual** $\mathbb{E}[Y(0) \mid D=1]$ — how often restored pixels
  *would* have burned had they not been restored.
- Every method in the pipeline is a different strategy for estimating that
  missing counterfactual:
  - **Matching** (M3–M5): find unrestored pixels so similar that their observed
    burning stands in for the counterfactual. Assumes *selection on
    observables*: given the matched covariates, treatment is as-if random.
  - **DiD** (M7): use treated pixels' own *pre-restoration* burning, corrected
    by the contemporaneous change among controls. Assumes *parallel trends*
    instead — a different, often weaker assumption, robust to any confounder
    that doesn't change over time.
- Confounding operates through exactly **two channels**, and each score in M4b/M5
  closes one of them: a covariate can bias the comparison only if it (a) affects
  *who gets treated* (selection — closed by the propensity score) and/or
  (b) affects *the outcome* (prognosis — closed by the prognostic score).

### M2 — Interpolating station climate: inverse-distance weighting

- GHCN gives climate at scattered station points; the model needs it on every
  grid cell. `idw_to_grid` predicts at grid location $s$ from the $k$ nearest
  stations:

$$\hat{z}(s) = \frac{\sum_{i=1}^{k} w_i\, z_i}{\sum_{i=1}^{k} w_i}, \qquad w_i = \frac{1}{d(s, s_i)^{p}}$$

  with $d$ the distance to station $i$ (EPSG:5070 metres, found via a KD-tree)
  and power $p$ (default 2). Closer stations dominate; a station exactly at $s$
  gets all the weight.
- Properties that matter here: the prediction is always a **convex combination**
  of station values, so IDW never extrapolates outside the observed station
  range — safe for a covariate whose only job is matching. It provides no
  uncertainty estimate (kriging would), but a match-only covariate doesn't need
  one; kriging can be swapped in later without changing the covariate contract
  (a GeoTIFF on the grid).
- Upstream of the interpolation, `station_normals` reduces each station's daily
  record to one number: annual total (precip) or annual mean (temperature) per
  year, then the mean over the 1991–2020 baseline — a climatological **normal**.

### M3 — Balance and the standardized mean difference

- Matching succeeds when the treated and matched-control groups have the same
  covariate distributions. The per-covariate summary is the **standardized mean
  difference**:

$$\mathrm{SMD} = \frac{\bar{x}_{\text{treated}} - \bar{x}_{\text{control}}} {\sqrt{\left(s_{\text{treated}}^2 + s_{\text{control}}^2\right)/2}}$$

  — the group mean gap in **pooled standard deviation units**, so it is
  comparable across covariates with different scales and unaffected by sample
  size (unlike a t-test, which conflates imbalance with N).
- Convention: $|\mathrm{SMD}| < 0.1$ is "balanced" (Castro et al. accept
  $\le 0.2$). The **love plot** shows $|\mathrm{SMD}|$ per covariate before vs
  after matching; `check_balance` asserts the after values shrank and clear the
  threshold. If balance doesn't improve, the matching has a bug or no good
  controls exist — either way, stop and look.

### M4 — Matching distance: standardization, whitening, Mahalanobis

- Raw Euclidean distance across covariates is meaningless: elevation (~0–50 m)
  and histosol % (0–100) are on different scales, so whichever has the bigger
  numeric spread dominates. Standardizing fixes scale but not **correlation**:
  if elevation and coast distance are correlated, plain Euclidean distance
  counts their shared signal twice.
- The fix is **Mahalanobis distance**, which measures separation in units of the
  covariate cloud's own shape:

$$d_M(x_i, x_j) = \sqrt{(x_i - x_j)^\top \Sigma^{-1} (x_i - x_j)}$$

  with $\Sigma$ the covariate covariance matrix. Directions in which the data
  vary a lot (or which merely repeat another covariate) count for less.
- Implementation: rather than computing $d_M$ pairwise, `match_controls`
  **whitens** once — $z = \Sigma^{-1/2}(x - \mu)$ — after which *ordinary*
  Euclidean distance among the $z$'s equals Mahalanobis distance among the
  $x$'s, and fast KD-tree nearest-neighbour search applies unchanged. A small
  ridge ($\Sigma + 10^{-6} I$) keeps the inverse defined when covariates are
  nearly collinear.
- **Caliper**: matches farther than 1.0 whitened (≈ SD) units are refused; a
  treated pixel with no control inside the caliper is dropped. This trades
  sample for quality — better to lose a treated pixel than to "match" it to a
  pixel unlike it and let the model extrapolate the difference.
- **Exact matching** for categoricals: the continuous distance is computed only
  *within* each class (land cover, drainage class), because class codes are
  labels — |class 3 − class 1| is not "twice as different" as |class 2 − class 1|.

### M4b — The propensity score

*The deep dive: what it is, why a single scalar can stand in for the whole
covariate vector, and exactly how the pipeline estimates it.*

- **Definition.** The propensity score is the probability of being treated given
  the covariates:

$$e(X) = \Pr(D = 1 \mid X)$$

  Here $D$ is restoration-site **membership** (the pixel-level constant, not the
  per-year flag), and $X$ is the static covariate vector (elevation, climate
  normals, soil, one-hot land cover).
- **The problem it solves.** Matching directly on many covariates gets
  exponentially harder as covariates are added (with 8 axes, almost no pixel has
  a close twin on *all* of them — the curse of dimensionality). Rosenbaum &
  Rubin (1983) proved you don't have to: $e(X)$ is a **balancing score**,

$$D \;\perp\; X \;\bigm|\; e(X)$$

  — among pixels with the *same propensity score*, the covariate distribution is
  identical for treated and control. Intuition for the proof: conditional on
  $e(X)$, the probability of treatment is $e(X)$ itself, a constant — so within
  a thin slice of the score, treatment assignment carries no further information
  about $X$; it is as-if random. Matching on the **one-dimensional** $e(X)$
  therefore balances the *entire* covariate vector in expectation.
- **What it does and does not buy.** If (i) treatment is unconfounded given $X$
  (no relevant *unobserved* covariates) and (ii) **positivity** holds
  ($0 < e(X) < 1$: every treated pixel has some chance of a control twin), then
  comparing outcomes within propensity-matched pairs estimates the ATT. The
  score does nothing about covariates you didn't measure — which is exactly why
  the pipeline also has the prognostic score (M5) and, more fundamentally, the
  DiD (M7).
- **Positivity is checkable**: `plot_score_overlap` histograms treated vs
  control scores. A region of treated mass with no controls under it is a
  positivity failure — matching there must either drop treated pixels (caliper)
  or stretch to poor twins.
- **How the pipeline estimates it.** A regularized logistic regression
  (`sklearn.LogisticRegression`):

$$\log\frac{e(X)}{1 - e(X)} = \beta_0 + \beta^\top X$$

  fit by penalized maximum likelihood — minimizing
  $-\sum_i \big[ D_i \log e(X_i) + (1-D_i)\log(1-e(X_i)) \big] + \tfrac{1}{2C}\lVert\beta\rVert^2$.
  The L2 penalty (small `C` = strong penalty; default `C=1`) matters because
  fire/treatment are rare and covariates can (quasi-)separate the classes, which
  would push unpenalized coefficients to ±∞. Continuous covariates are
  standardized and categoricals one-hot encoded, in a design matrix built
  **once over all pixels** so training (a subset) and prediction (everyone) use
  identical scaling and dummy columns. Computation happens on unique pixels
  (collapse-then-broadcast).
- **The per-vintage refinement** (`add_propensity_score_series`, §2c). Instead of
  one pooled score, one model per restoration cohort $g$:

$$e_g(X) = \Pr(\text{restored in year } g \mid X, \; \text{restored in } g \text{ or never restored})$$

  trained with vintage-$g$ pixels as positives and never-treated pixels as
  negatives — *other* vintages are excluded from training (they'd contaminate
  the contrast) but still receive a predicted $\text{psm}_g$, so they can serve
  as candidate controls for cohort $g$. Rationale: siting criteria plausibly
  changed between vintages (different program years target different
  landscapes), so the coefficients should be allowed to differ by cohort.
- **Where it is used.**
  - In the *scalar track*, `pscore` is a matching axis alongside `phat`.
  - In the *faithful track*, $\text{psm}_g$ is one coordinate of cohort $g$'s
    trajectory-matching vector.
  - Inside the DR DiD estimator (M7), a propensity model appears again — there
    it *reweights* controls rather than matching them.

### M5 — The prognostic score

- The propensity score's mirror image (Hansen 2008): instead of modelling
  *treatment*, model the *untreated outcome*:

$$\Psi(X) = \mathbb{E}\left[\,Y(0) \mid X\,\right]$$

  — a pixel's **baseline fire risk** absent treatment. Matching on $\Psi(X)$
  balances precisely the covariates that matter for the outcome (a covariate
  that predicts neither treatment nor fire is harmless and needs no balancing).
- Estimated as a (regularized) logistic of "did this pixel burn in an
  **untreated** pixel-year" on the covariates, **fit on control observations
  only, then predicted for every pixel**. The training rows are exactly the
  $Y(0)$ observations: all years for never-treated pixels, pre-restoration years
  for treated ones.
- Fit-on-controls / predict-for-all is the **no-leakage rule**: a treated
  pixel's prognostic score is a pure function of its covariates — it never sees
  that pixel's own post-restoration burns, so the score cannot absorb the very
  treatment effect we're estimating.
- Matching on **both** scores (propensity + prognostic) is *doubly robust
  matching* (Leacy & Stuart 2014): the comparison is valid if **either** score
  model is correctly specified — the same logic as the DR estimator in M7.
- The per-year series `phat_2019 … phat_2024` (§2c) fits a *separate* model per
  outcome year on never-treated controls. Because each year's coefficients
  differ, the series encodes how baseline risk shifts through dry and wet years
  — the time-resolved version of $\Psi$ (see M8).

### M6 — The logistic model, odds ratios, and cluster-robust standard errors

- **The model** (Route 2's estimator). With $p = \Pr(\text{burned}=1)$:

$$\log\frac{p}{1-p} = \beta_0 + \beta_T\,\text{treated} + \beta^\top X \quad\Longleftrightarrow\quad p = \frac{1}{1 + e^{-\eta}}$$

  fit by maximum likelihood on the matched pixel-year frame.
- **Odds ratios.** Exponentiating a coefficient gives a multiplicative effect on
  the *odds* $p/(1-p)$: $\mathrm{OR}_T = e^{\beta_T}$ is the factor by which
  restoration multiplies the odds of burning, holding covariates fixed.
  $\mathrm{OR} = 0.6$ ⇒ 40% lower odds. Confidence interval:
  $\exp(\hat\beta \pm 1.96\,\widehat{\mathrm{SE}})$, which is why OR CIs are
  asymmetric around the point estimate.
- **Interactions.** `burned ~ treated * precip + …` adds
  $\beta_{T\times w}\,(\text{treated} \times \text{precip})$, so the treatment
  log-odds effect becomes $\beta_T + \beta_{T\times w}\,\text{precip}$ — a
  treatment effect that *varies with that year's weather*. Sign readings are in
  the §4 walkthrough.
- **Why cluster-robust SEs — the heart of "problem 1".** Ordinary logistic SEs
  assume independent observations. But all pixels in one restoration site share
  drainage history, management, and every fire that crosses the site; their
  errors are correlated. Ignoring that is *pseudo-replication*: the model thinks
  it has ~10⁵ independent data points when the design really has ~10 sites.
  The **cluster-robust (sandwich) estimator** fixes the variance without
  changing the coefficients:

$$\widehat{\mathrm{Var}}(\hat\beta) = A^{-1} \left( \sum_{g=1}^{G} s_g s_g^\top \right) A^{-1}, \qquad s_g = \sum_{i \in \text{cluster } g} \frac{\partial \ell_i}{\partial \beta}$$

  where $A$ is the usual information matrix and $s_g$ sums each cluster's score
  contributions. Errors may correlate arbitrarily *within* a cluster; only
  independence *across* clusters is assumed. The effective sample size becomes
  the number of clusters $G$ (restoration sites — which is why matched controls
  inherit their treated partner's `site_id`), so with few sites, treat inference
  cautiously.
- **Rare events.** Fire is rare, so quasi-separation is a real risk;
  `fit_logit_clustered` detects and reports it, and the roadmap's fallback is a
  Firth-penalized logistic (for burned *area* as the response: Tweedie or hurdle
  models). `fit_mixed_logit` (random intercept per site) is the GLMM
  cross-check: instead of correcting the SEs after the fact, it models the
  site-level correlation directly with a site effect $u_g \sim N(0, \sigma^2)$.

### M7 — Difference-in-differences and the ATT

*The deep dive: from the 2×2 table to the staggered doubly-robust estimator the
pipeline actually runs.*

- **The canonical 2×2.** One treated group, one control group, one pre and one
  post period:

$$\widehat{\mathrm{ATT}} = \underbrace{\left(\bar{Y}^{\text{treat}}_{\text{post}} - \bar{Y}^{\text{treat}}_{\text{pre}}\right)}_{\text{change in treated}} - \underbrace{\left(\bar{Y}^{\text{ctrl}}_{\text{post}} - \bar{Y}^{\text{ctrl}}_{\text{pre}}\right)}_{\text{change in controls}}$$

  Worked example: treated pixels burn at 10% before restoration and 4% after;
  controls burn at 8% then 6% over the same years (a wetter period everywhere).
  The naive post-only comparison says 4% vs 6%; the DiD says
  $(0.04-0.10) - (0.06-0.08) = -0.04$: restoration cut burn probability by 4
  percentage points beyond the shared wet-period decline.
- **Why differencing is powerful.** The first difference (within group, over
  time) cancels *anything constant about the group* — persistent drainage
  legacy, soil, access, every unmeasured time-invariant confounder. The second
  difference (across groups) cancels *anything common to the period* — a dry
  year, a policy change, sensor drift. What survives is the treatment effect,
  **provided** the **parallel-trends assumption** holds: absent treatment, the
  treated group's outcome would have moved like the controls'. That
  counterfactual claim is untestable directly, but its natural proxy is
  testable: did the two groups move in parallel *before* treatment? That is the
  event-study pre-trends check.
- **Staggered adoption.** NC sites were restored in different years, so there is
  no single pre/post split. Naively pooling with a two-way fixed-effects
  regression is now known to be biased when effects vary over time, because it
  implicitly uses *already-treated* units as controls for later cohorts.
  Callaway & Sant'Anna (2021) instead never pool: they estimate a whole family
  of clean 2×2-style effects, one per **cohort × year**.
- **The group-time building block.** Let $g$ be a pixel's cohort (its site's
  restoration year; $g=0$ for never-treated). The target is

$$\mathrm{ATT}(g, t) = \mathbb{E}\left[\,Y_t(g) - Y_t(0) \mid G = g\,\right]$$

  — the effect in calendar year $t$ for the cohort first treated in $g$. Under
  conditional parallel trends it is identified by comparing each cohort's
  outcome *change* since its last untreated year ($g-1$, the base period)
  against the same change among clean controls:

$$\mathrm{ATT}(g, t) = \mathbb{E}\left[\,Y_t - Y_{g-1} \mid G = g\,\right] - \mathbb{E}\left[\,Y_t - Y_{g-1} \mid \text{control at } t\,\right]$$

  where "control at $t$" means never-treated pixels (and, configurably,
  not-yet-treated cohorts with $g' > t$ — pixels that will be restored later
  are valid controls *until* their year arrives, which is why the panel keeps
  pre-restoration site pixels).
- **Conditional parallel trends and the covariates.** Plain parallel trends may
  fail if, say, low-lying wet peat responds differently to a drought year than
  higher ground — a *covariate-dependent* trend. The fix: require trends to be
  parallel only among pixels with the same $X$ (elevation, climate normals,
  soil). The estimator therefore needs a way to compare like with like, which
  is where its internal outcome and propensity models come in.
- **The doubly-robust estimator** (`est_method="dr"`; Sant'Anna & Zhao 2020).
  For each $(g, t)$ cell, with $\Delta Y = Y_t - Y_{g-1}$, cohort indicator
  $G_g$, control indicator $C$, an outcome-regression model
  $\hat m_{g,t}(X) = \widehat{\mathbb{E}}[\Delta Y \mid X, C=1]$ and a
  propensity model $\hat e_g(X) = \widehat{\Pr}(G_g = 1 \mid X)$:

$$\widehat{\mathrm{ATT}}(g,t) = \mathbb{E}_n\!\left[ \left( \underbrace{\frac{G_g}{\mathbb{E}_n[G_g]}}_{\text{treated weight}} - \underbrace{\frac{\dfrac{\hat e_g(X)\,C}{1 - \hat e_g(X)}}{\mathbb{E}_n\!\left[\dfrac{\hat e_g(X)\,C}{1-\hat e_g(X)}\right]}}_{\text{reweighted controls}} \right) \left(\Delta Y - \hat m_{g,t}(X)\right) \right]$$

  In words, three ideas stacked:
  1. **Outcome regression** alone would predict each treated pixel's
     counterfactual change from the control-fitted model and average
     $\Delta Y - \hat m_{g,t}(X)$ over the treated — right if the outcome model
     is right.
  2. **Inverse-probability weighting** alone would reweight controls by
     $\hat e_g(X)/(1-\hat e_g(X))$ — up-weighting controls that *look like*
     treated pixels — and take a weighted difference of raw changes — right if
     the propensity model is right.
  3. The **DR combination** does both: it compares treated and reweighted
     controls on the *residuals* from the outcome model. If the outcome model is
     right, the residuals are clean regardless of the weights; if the weights
     are right, the residual comparison is unbiased regardless of the outcome
     model. Consistent if **either** model is correctly specified — the same
     either-one-suffices logic as matching on propensity *plus* prognostic
     scores.
- **Aggregation** (`aggregate_att`). The $(g,t)$ family is collapsed to
  readable summaries:
  - `simple` — one overall ATT: the weighted average of all post-treatment
    cells, weights proportional to cohort size,
    $\theta = \sum_{g} \sum_{t \ge g} w(g,t)\,\mathrm{ATT}(g,t)$. This is the
    headline number.
  - `event` — the **event study**: average effect at each event time
    $e = t - g$ across cohorts, $\theta(e) = \sum_g w_g\, \mathrm{ATT}(g, g+e)$.
    Negative $e$ values are **pseudo-effects estimated purely pre-treatment**:
    under parallel trends they should be ≈ 0, so the left half of the event
    study *is* the assumption check, and the right half is the dynamic path of
    the effect (does the protection grow as the water table recovers? fade?).
  - `calendar` — ATT by calendar year (is the effect concentrated in dry
    years?).
- **Inference.** SEs come from the influence function of each
  $\mathrm{ATT}(g,t)$, and **the level they are clustered at is a parameter**:
  `cluster_by="site"` (the default) or `cluster_by="pixel"`, on `estimate_att`,
  `fit_att` and `att_collapsed` alike. The point estimate is identical either
  way; only the variance moves.
  - *Why site.* Restoration is assigned to a **site**, not a pixel. Every pixel
    inside one site shares its canal blocks, its weather, its water table — so
    the thousands of pixel-years are not thousands of independent draws.
    Clustering at the pixel level lets the SE shrink like $1/\sqrt{n_{\text{pix}}}$
    forever, which is arithmetic, not evidence. The site-clustered SE instead
    plateaus at whatever the ~6 sites can tell you. Castro cluster at the
    *village* level for the same reason; the restoration site is our analogue.
  - *How.* Clustering above the entity level exists **only on the multiplier-
    bootstrap path** — the closed-form influence-function SEs take no cluster
    argument in `differences` or in R `did`. So `cluster_by="site"` draws one
    Rademacher weight per site and reruns the bootstrap
    (`boot_iterations`, default 1000); `cluster_by="pixel"` keeps the instant
    analytic SEs. Check the `std_error` column header — `bootstrap` vs
    `analytic` — to see which you got.
  - *Cross-check.* `att_collapsed` does the same thing by hand and
    dependency-free: collapse each pixel to one pre/post change, difference
    treated against control **within each site** to get one $\theta_s$ per site,
    average those, and take the SE from their spread with $t(G-1)$ — the
    textbook Bertrand–Duflo–Mullainathan collapse. It returns the per-site
    $\theta_s$ table and reports both SEs plus their ratio (`design_effect`), so
    the deflation is a number you can quote.
  - *Caveat.* Cluster-robust inference is asymptotic in the number of
    **clusters**, and ~6 sites is far below the usual 30–50 rule of thumb. The
    site-clustered SE is much more honest than the pixel one but still
    optimistic; read it with the small-G caution of M6 and lean on the event
    study.
  - *Full derivation.* `clustered_standard_errors_explained.md` builds this up
    from a sample mean: what an influence function is (with a leave-one-out check
    against the real doubly-robust estimator), how it becomes a standard error,
    what a multiplier bootstrap does, and why clustering can only live on the
    bootstrap path. Every number in it is reproducible output.
- **Interpretation.** The outcome is 0/1 and the outcome model linear-in-
  probability, so the ATT is an absolute change in $\Pr(\text{burn})$ per
  pixel-year (e.g. −0.03 = 3 percentage points fewer burns). That makes the
  conversion to impact trivial (M9) — and note it is a *different scale* from
  Route 2's odds ratio; don't compare the two numbers directly, compare their
  qualitative story.

### M8 — Where does time go? Static vs per-year covariates

*Answering directly: per-year covariates are neither discarded nor "flattened"
into the model — but they are used completely differently in each of the four
stages, and one deliberate flattening happens upstream.*

- **The data layout is always long, never wide.** Every table is a panel with
  one row per (pixel, year). Static covariates (elevation, soil, climate
  *normals*) repeat down a pixel's rows; temporal covariates (`precip`, `tmax`,
  `tmin`, `pdsi`) take a different value on each row, joined on `(x, y, year)`.
  Nothing is averaged over years to make the model fit.
- **Stage by stage:**

  | stage | how the time dimension of covariates is used |
  |---|---|
  | geographic match (§2) | **not at all — by design.** Matching collapses to unique pixels and uses only static covariates. |
  | trajectory match (§2c) | **as a literal time series**: each year is one coordinate of the matching vector. |
  | staggered DiD (§2b) | covariates enter as static; **the estimator itself owns the time dimension** (differencing). |
  | logistic model (§4) | **per-row values**: each pixel-year row carries that year's weather; interactions let the effect vary with it. |

- **§2 geographic match — static only, deliberately.** The match asks "which
  unrestored pixel is *the same kind of place*?" — a question about stable site
  character. Matching on 2020's rainfall would match on noise, and worse, on a
  variable that cannot differ between a pixel and its own counterfactual. So the
  match uses the 30-year **normals**. That normal is the one true "flattening"
  in the pipeline: a station's daily record is reduced to a long-run mean — but
  upstream, on purpose, and producing a *different variable*
  (`precip_normal` ≠ `precip`; both exist side by side).
- **§2c trajectory match — the time series used literally.** Cohort $g$'s
  matching vector is
  $\big(\text{fire}_{g-1},\, \text{fire}_{g-2},\, \text{fire}_{2015},\, \hat p_{g},\, \hat p_{g+1},\, \dots,\, \hat p_{2024},\, \text{psm}_g\big)$ —
  pre-treatment burn history plus the pixel's predicted-risk path, one
  coordinate per year. Mahalanobis distance over this vector compares **whole
  trajectories**: two pixels match only if their risk moved through the dry and
  wet years together. This is exactly the Castro fix for the scalar track's
  limitation (a single collapsed `phat` throws the year-to-year shifts away).
- **§2b DiD — time handled by the estimator, not the covariate list.** The
  covariates passed to `estimate_att` are the static ones; Callaway–Sant'Anna
  treats them as fixed pixel characteristics for the conditional-parallel-trends
  adjustment. Where did the weather go? A dry year that raises fire risk
  *everywhere equally* is a common period shock — the second difference cancels
  it by construction, no covariate needed. Weather only threatens the design if
  it hits treated and control *differently*, and the match/conditioning on
  static covariates (similar places respond similarly to the same weather) is
  what guards against that.
- **§4 logistic — per-year values as ordinary regressors.** A 2020 row carries
  2020's precip at that pixel; the model pools all years but the regressor
  varies by row, and `treated:precip` lets the *treatment effect itself* be a
  function of that year's weather. The time series is not modeled as a series
  (no lags, no autocorrelation structure in this fit) — year-to-year dependence
  is instead absorbed defensively by the site-clustered SEs, and Castro's
  fire-history lags (`add_fire_lags`: did this pixel burn last year, did a
  rook-neighbour burn this year) are available as explicit DiD covariates when
  fire contagion should be modeled.

### M9 — From ATT to avoided area

- Because the ATT is a change in burn probability, scaling by the treated area
  gives Castro's headline physical quantity:

$$\text{avoided burned area} = -\,\widehat{\mathrm{ATT}} \times \text{restored area (ha)}$$

  (`did.avoided_area`; a protective effect has ATT < 0, so avoided area comes
  out positive). Per-year, since the ATT is per pixel-year.

---

## Part V — Design-decision log (modeling)

Each choice below had a defensible alternative; this is why the pipeline is the
way it is.

- **Dependent variable = FireCCIS311, kept swappable.** The response comes from
  `load_standardized(product, ...)` (boolean mask for burned-area products,
  continuous grid for severity), so switching the DV changes only the `y`
  column and, for severity, the model family — nothing upstream.
- **Peat extent uses an 80% histosol threshold, not 0.** The sample frame is
  high-confidence peat (gSSURGO `H% ≥ 80`), keeping non-peat noise out of the
  covariate distributions (built at `PEAT_THRESHOLD = 80` in
  `download_and_clip_data.ipynb`).
- **Matched case-control design, not a whole-landscape GLM.** Matching buys
  covariate overlap so the restoration effect isn't confounded by where
  restoration happens; the naive pixel-GLM survives only as the cluster-robust
  baseline (Part I's three problems).
- **Unit of analysis = pixel-year on the existing EPSG:5070 common grid**,
  tagged with `site_id` and `year` — the clustering/random-effect keys. If
  pixel-year proves too autocorrelated or heavy, the documented fallback is
  aggregating to site-year (burned fraction per site per year).
- **Match on climate + soil, not elevation alone.** Histosol % is ~constant on
  the 80% frame, so `[elevation, histosol_pct]` was effectively an
  elevation-only match with trivially small distances balancing nothing. The
  matchable set is every continuous layer on disk — climate normals (precip,
  tmax, tmin, GDD), soil organic matter/AWC, forest site index, water-table
  depth; drainage class and land cover are exact-match keys. The GDD normal and
  the soil site-index / water-table-depth layers come from Cat's analysis script
  (`get_climate&soil_data_updated.R`) and were wired in alongside the raw-SSURGO
  and daily-GHCN layers.
- **Climate enters as a static long-run normal, IDW-interpolated from GHCN
  points** (M2); year-specific weather is a separate temporal covariate for the
  outcome stage (M8). IDW over kriging: transparent, dependency-light, never
  extrapolates — all a match-only covariate needs; kriging can swap in later
  without changing the covariate contract.
- **Ingest formats match what the R script writes:** climate as GHCN `.Rds`
  long tables (via `pyreadr`), soil as SSURGO polygons rasterized
  attribute-by-attribute; GeoPackages read through a resilient reader that
  copies to a writable temp file on SQLite "readonly database" errors.
- **Drainage is an area-weighted mean per unit** (HAND), not a centroid sample.
- **Spillover buffer set to 0 m** in the notebook — TNC prevents spillover into
  neighboring communities; the machinery (`spillover_m`) is there when a halo
  is warranted.
- **Step-by-step diagnostic plots at every stage** (`plotting.py`), sharing the
  fire-comparison style: covariate maps (flat panel = no signal), covariate
  space + scatter matrix (degeneracy visible), matched pairs in covariate space
  and geography (spatial-confounding flag), love plot, score overlap/maps,
  event study, raw burn rates.
- **Models build up a ladder** — cluster-robust GLM → mixed logistic GLMM
  (`(1|site)`; year effects) → treatment × climate interaction — reported as
  odds ratios with CIs, peat condition first. Rare-event fallbacks: Firth
  logistic; Tweedie/hurdle for burned area.
- **Staggered DiD offered as the stronger estimator** alongside the levels
  GLM/GLMM, following Castro et al. (2026): match first, then Callaway–Sant'Anna
  doubly-robust ATT; Castro's Eq. 1 fire history available via `add_fire_lags`
  (temporal lag + 4-neighbour spatial lag); SEs clustered on site (their
  village). Backend: `differences` (Python) or R `did` via rpy2 (their Stata
  `csdid`).

## Part VI — The Castro et al. (2026) blueprint and NC caveats

Castro et al. evaluate canal-block rewetting → peat fire in Kalimantan; the
§2b/2c design transplants it to NC.

| Castro et al. (2026), Kalimantan | our analogue |
|---|---|
| Treated = 250 m upstream semicircle of each canal block; control = rest of a 2 km buffer | treated = restoration polygons; control = matched unrestored peat |
| Match 1:1 Mahalanobis, no replacement, exact on subdistrict + peat depth, \|SMD\| ≤ 0.2 | `match_controls` / `match_controls_event_time` (exact on land cover; caliper 1.0) |
| Match on propensity + prognostic scores + pre-treatment fire history (t−1, t−2, 2015 drought) | scalar track (`add_matching_scores`) or faithful track (`add_prognostic_score_series` + `add_propensity_score_series` + `match_controls_event_time`) |
| Outcome = binary fire, 50 m pixel-year (MODIS MCD64A1) | pixel-year `burned` from FireCCIS311 (swappable) |
| Estimator = Callaway & Sant'Anna staggered DiD, doubly robust (`csdid`, Stata) | `did.fit_att` (`differences` or R `did` via rpy2) |
| Outcome eq. covariates incl. temporal + 4-neighbour spatial fire lags | `add_fire_lags` |
| Group = construction vintage; controls = never-/not-yet-treated | `attach_cohort`: `g` = restoration year, 0 = never |
| SEs clustered at village | `cluster_by="site"` (default) → cluster on `site_id` |
| Headline = ATT × rewetted area = avoided burned area | `avoided_area(att, area_ha)` |

Caveats where NC differs, and it matters:

- **N.** Castro have 11.3M pixel-years and estimate per-subdistrict × vintage
  counterfactuals; a handful of NC sites supports one pooled ATT plus an
  event-study check at most — treat sub-group effects as underpowered.
- **Control geometry.** Our controls are matched landscape peat, not a
  within-buffer donut, so conditional parallel trends leans harder on match
  quality (the Stage 6 balance checks) than theirs does.
- **Fire-history coverage and the estimand.** The notebook now configures the
  product, years, and native-scale analysis grid together. Before scores or
  matching, `restrict_to_supported_cohorts` keeps controls and only treatment
  vintages satisfying `first outcome year < g <= last outcome year`. With
  FireCCIS311 (2019–2024), the current data therefore estimate the ATT for the
  supported 2021/2023 cohorts and exclude 2019 (no pre-period) and 2026 (no
  post-period). Selecting MCD64A1 (configured as 2001–2024 at 500 m) recovers the
  2019 cohort but still excludes 2026. Switching products requires restarting
  the kernel and rerunning from the configuration cell so grids, responses,
  scores, and matches are rebuilt rather than mixing 300 m and 500 m objects.
- **Small control pool.** Per-vintage + no-replacement + land-cover exact-match
  constrains candidates hard; watch the per-cohort drop counts and loosen the
  caliper if too many treated pixels go unmatched (the same effect behind
  Castro's failed 4 km robustness check).
- **Few site clusters.** Both backends now support the requested site-level
  clustering, but only a handful of restoration sites remain. Cluster-robust
  inference is asymptotic in the number of sites, so report the event study and
  `att_collapsed()` cross-check alongside the bootstrap interval.
- **Negative control.** Castro's planned-but-unbuilt placebo maps to running the
  pipeline on *planned but not-yet-restored* sites at their scheduled years:
  any "effect" found there flags upstream confounding.

## Part VII — The matching stages, and how to test them

`matching.py` grew out of a staged assignment; the stages remain the cleanest
mental model of the front half of the pipeline, and each has a `check_*`
contract you can run after any change.

- **Stage 1 — `load_treated_units`**: completed restoration polygons, EPSG:5070,
  non-null restoration year. → `check_treated_units`.
- **Stage 2 — `build_candidate_pool`**: peat minus treated + spillover halo
  (`gpd.overlay(..., how="difference")` — the inverse of a clip). →
  `check_candidate_pool` (zero overlap, positive area). Plot the donut.
- **Stage 3 — `pixelate`**: grid-cell-centre points inside polygons, ~area/res²
  of them, carrying site attributes. → `check_pixels` (treated pixels fall
  inside restoration polygons).
- **Stage 4 — `attach_covariates`**: one column per covariate, sampled at each
  pixel. → `check_covariates` (no all-NaN column for layers on disk; sane
  ranges).
- **Stage 5 — `match_controls`**: whiten → within-class KD-tree
  nearest-neighbour → caliper → (no) replacement. → `check_matches` (every
  control within the caliper; the printed drop count matters).
- **Stage 6 — `balance_table` + `plot_balance`**: SMD before vs after; the love
  plot is the acceptance test of the whole design. → `check_balance`.
- **Stage 7 — `assemble_units`**: package into the `units` GeoDataFrame →
  `build_frame` → fit.

How to test each piece (beyond "it runs"):

- **Toy inputs first.** Feed each function a hand-built case where the answer is
  known (3 fake squares + a 5×5 raster); real data hides logic bugs.
- **Invariants as regression tests.** The `check_*` functions are the test
  suite — run them after every change.
- **Visual checks.** Geospatial bugs are obvious on a map and invisible in a
  dataframe; plot after Stages 2, 3, 6.
- **Known-answer probes.** A pixel inside a restoration site must come out
  `treated == 1`; one 50 km inland must never be a candidate.

How to know the *design* is good, not just the code correct:

- Enough matched controls survived without blowing the caliper open.
- Balance improved (Stage 6) — the real acceptance test.
- **Not driven by one site**: jackknife — re-run leaving each site out; the
  odds ratio / ATT should be stable.
- **The negative control passes** (Part VI): no effect on
  planned-but-unrestored sites.

## Part VIII — Extensions and next steps

- **Severity**: swap `product=` for a severity layer; the frame/DiD rerun
  unchanged, the levels model family changes (continuous outcome → OLS/Tweedie).
- **Mixed logit** (`fit_mixed_logit`, site random intercept) to cross-check the
  cluster-robust SEs.
- **Fire-history lags** (`add_fire_lags`) as DiD covariates to model contagion
  (Castro Eq. 1) instead of leaving spatial dependence to the SEs.
- **rpy2 backend** for site-level clustering in the DiD.
- **Burned area as DV**: Tweedie or hurdle (occurrence × conditional area).
- **Kriging** in place of IDW if climate uncertainty ever needs to propagate.
- Keep interactions few and pre-registered — with rare events, don't fish.

---

## References

**Methods — matching**

- Rosenbaum, P. & Rubin, D. (1983). *The Central Role of the Propensity Score in
  Observational Studies for Causal Effects.* Biometrika 70(1), 41–55.
- Hansen, B. B. (2008). *The prognostic analogue of the propensity score.*
  Biometrika 95(2), 481–488.
- Leacy, F. P. & Stuart, E. A. (2014). *On the joint use of propensity and
  prognostic scores in estimation of the average treatment effect on the
  treated.* Statistics in Medicine 33(20), 3488–3508.
- Stuart, E. A. (2010). *Matching Methods for Causal Inference: A Review and a
  Look Forward.* Statistical Science 25(1), 1–21.
- Ho, Imai, King & Stuart (2007). *Matching as Nonparametric Preprocessing for
  Reducing Model Dependence in Parametric Causal Inference.* Political Analysis
  15(3), 199–236.

**Methods — difference-in-differences**

- Callaway, B. & Sant'Anna, P. H. C. (2021). *Difference-in-Differences with
  Multiple Time Periods.* Journal of Econometrics 225(2), 200–230.
- Sant'Anna, P. H. C. & Zhao, J. (2020). *Doubly robust
  difference-in-differences estimators.* Journal of Econometrics 219(1),
  101–122.

**Domain — peat, restoration, fire**

- Castro et al. (2026). *Effective restoration can avoid peatland fires: Large
  scale counterfactual assessment in Kalimantan, Indonesia.* iScience.
  doi:10.1016/j.isci.2026.116041 — the study `did.py` implements.
- *The Impact of Rewetting Peatland on Fire Hazard in Riau, Indonesia* (2023).
  Sustainability 15(3), 2169.
- Nguyen Huy, Adjognon & Van Soest (2023). *Combatting Forest Fires in the
  Drylands of Sub-Saharan Africa: Quasi-Experimental evidence.*

---

## Appendix A — Fire-product comparison & validation toolkit decisions

Preserved from the original decision log; these concern
`src/peatfire/fire_products_comparison/` and the validation toolkit, not the
modeling notebook.

**Fire-product comparison** (`fire_products.py`, `fire_comparison.py`,
`plotting.py`):

- **Analysis CRS = EPSG:5070** (NAD83 / CONUS Albers, equal-area), fixed for all
  area math and grid comparisons; the AOI's own CRS is only a clip mask. Area is
  only meaningful equal-area; mirrors Humber et al. (2019) cosine-weighting
  their geographic products.
- **Common grid with "max" aggregation** ("any sub-cell burn lights the cell"),
  default 500 m (the coarsest product), removing spatial resolution as a
  confound — as in Humber et al.'s shared ~6 km grid. Every product is matched
  onto one product-independent grid (the earlier sandbox privileged MODIS's
  native grid).
- **Resolution confound is only partially fixable:** coarse pixels inflate area
  per detection (mixed-pixel commission) but also omit small fires entirely; in
  small-fire landscapes like NC pocosins omission usually dominates (cf.
  Vetrita et al. 2021). Hence annual burned area is reported both at native
  resolution and on the common grid, and all agreement is computed on the
  common grid; totals also as % of AOI for cross-AOI comparability.
- **Agreement metrics:** binary maps get binary metrics (Jaccard/IoU, Cohen's
  kappa, % agreement — Pearson on binary data is only the phi coefficient); a
  separate temporal matrix (Pearson/Spearman of annual totals) captures
  year-to-year co-variation; severity products use Spearman on the common grid
  (CBI vs dNBR vs MTBS classes aren't unit-comparable). VIIRS joins as per-cell
  presence (binary) / detection count (correlation).
- **Monthly products** (MCD64A1, FireCCI51, MOSEV) are OR'd/max-aggregated into
  annual layers; within-year timing is preserved upstream for a future
  seasonality analysis.
- **Total least squares + RMSE for pairwise scatter** (per Humber et al.): TLS
  (orthogonal regression) rather than OLS because both products carry error
  (errors-in-variables — OLS would bias the slope by assuming x error-free);
  RMSE reported against y=x. Each scatter point is a common-grid cell (the
  analogue of Humber's per-TSA points).
- **Equal-area CRS vs aggregation units are different roles:** EPSG:5070 is the
  analogue of Humber's latitude/area correction; the common-grid cells play the
  role of their TSA polygons.
- **Plot style** (`set_fire_style`): products encoded by colour only
  (Okabe-Ito, colour-blind-safe), top/right spines removed, frameless legend.

**Ground-truth validation** (`reference_sources.py`, `validation.py`):

- **References are a separate registry from products** (`REFERENCE_SOURCES` vs
  `FIRE_PRODUCTS`): a reference (NIFC perimeters, TNC preserve burns, NCWRC Rx)
  is truth *within its footprint* but not a spatially-exhaustive census, so it
  can't play a product's symmetric role.
- **Recall (1 − omission) is the headline; precision is conditional.** Recall =
  fraction of a perimeter's burned cells the product also maps. Precision is
  reported only per-event inside a buffered window (default 5 km) and flagged
  conditional, because (1) a product burn outside an incomplete reference may be
  a real small fire, and (2) a perimeter is an outer boundary with unburned
  islands — an upper bound that depresses precision and inflates omission.
- **Time-matched, per-event — never a timeless union.** Each incident is scored
  against the product layer for its own year (`validate_event`); events stay
  distinct across years. `summarize_validation` reports both cell-weighted
  `recall_pooled` (dominated by big fires) and `recall_mean_event` (surfaces
  small-fire omission).
- **Occurrence products judged at the event level:** VIIRS detects a burning
  front, not burned area, so `detected` (any overlap) is the meaningful metric;
  VIIRS is 2012+, so older incidents are skipped, not scored 0.
- **Perimeters rasterized `all_touched=True`**, matching the products'
  `how="max"` convention, so small reference fires survive the 500 m grid.
- **Severity is not validated against perimeters** (they carry no severity; no
  field CBI). Perimeters instead restrict the severity cross-comparison
  (SE FireMap vs MOSEV vs MTBS) to known-burned cells; MTBS straddles
  product/reference.
- **Two NIFC sources, not pooled:** `NIFC_IFPH` is the default;
  `GEOMAC` (2000–2018) overlaps it and serves as a cross-check — pooling would
  double-count the same fires. Schema differences handled by per-role
  candidate-column lists.

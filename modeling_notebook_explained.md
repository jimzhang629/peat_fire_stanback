# `notebooks/modeling.ipynb`, explained at nested levels of abstraction

How to read this document: the **outermost bullets are the big ideas**; each
level of indentation steps down one level of abstraction, ending at what the
individual function calls actually do. The notebook itself is thin
orchestration — nearly all machinery lives in `src/peatfire/modeling/`, so the
deepest bullets describe those package functions.

---

## The big picture

- **The question:** did TNC's pocosin (peatland) restoration in North Carolina
  reduce wildfire? Restoration re-wets drained peat, so restored pixels should
  burn less than comparable unrestored pixels.
  - **Treatment** = pixels inside a *completed* restoration site, "on" from
    that site's restoration year (`End_Yr`) forward.
  - **Control** = peat pixels (≥80% histosol) outside every restoration site
    and its spillover halo.
  - **Response** = a 0/1 "did this pixel burn this year" flag sampled from a
    swappable satellite fire product (FireCCIS311, ~300 m, 2019–2024). Swapping
    `product=` re-runs the whole analysis for e.g. burn severity.
- **Two independent estimation routes**, answering the same question with
  different identification assumptions:
  - **Route 1 — changes (staggered DiD, §2b/2c):** compare the *change* in
    burning after each site's restoration against not-yet/never-restored
    controls (Callaway & Sant'Anna 2021, following Castro et al. 2026). All
    *time-invariant* confounders difference out, even unmeasured ones.
    Headline number: the **ATT** (change in burn probability).
  - **Route 2 — levels (matched logistic, §3–4):** match each treated pixel to
    a covariate twin, then fit `burned ~ treated + covariates` with standard
    errors clustered by restoration site. Headline number: the **odds ratio**
    for `treated` (< 1 ⇒ restoration lowers fire odds).
- **One shared geometry:** everything — fire products, covariates, pixels — is
  warped onto the same EPSG:5070 (metres) grid at 300 m via
  `build_common_grid`/`to_common_grid`, so every layer aligns cell-for-cell and
  distances/buffers are in metres.

## Code map (where each piece lives)

- `notebooks/modeling.ipynb` — orchestration + diagnostics; each section below.
- `src/peatfire/modeling/`
  - `covariates.py` — a registry (`CovariateSpec`) of every environmental
    layer on disk + generic loaders that clip/standardize/warp them.
  - `climate.py` — GHCN station records → interpolated climate rasters.
  - `soil.py` — SSURGO soil polygons → rasterized soil covariates.
  - `matching.py` — pixel sets, covariate sampling, scores, and all matching.
  - `frame.py` — restoration-site loaders, `build_frame` (tidy pixel-year
    table), `attach_fire_response` (fire response at pixel points).
  - `did.py` — staggered difference-in-differences (panel prep, ATT fit,
    aggregation).
  - `models.py` — cluster-robust logistic/OLS fits + odds-ratio tables.
  - `plotting.py` — every diagnostic figure the notebook saves.

---

## Cell-by-cell walkthrough

### Setup (imports cell)

- Imports the toolkit, calls `set_fire_style()` (shared matplotlib style), and
  creates `FIG_DIR = outputs/figures/modeling/` — **every figure in the
  notebook is saved there**.
- Prints `available_covariates()`: the registered covariates whose files
  actually exist on disk right now. Cells downstream select against this list,
  so the notebook degrades gracefully when a layer hasn't been built yet.

### §1 — Load treatment + peat frame

- `load_completed_restoration_sites_in_analysis_crs(restoration_yr_col='End_Yr')`
  → `peat_restoration`, the **treatment polygons**.
  - Reads the TNC restoration shapefile, reprojects to EPSG:5070.
  - Replaces `End_Yr == 0` (placeholder for "not finished") with NaN and drops
    those rows — "completed" is *defined* as "has a real end year".
- `gpd.read_file(...nc_peatlands_80_histosol_aoi.gpkg)` → `aoi_nc_peat_80_histosol`,
  the **sample frame**: the part of NC that is ≥80% histosol (peat soil).
  Everything (grid extent, candidate controls, covariate clipping) happens
  inside this AOI.

### §1b — Build the climate + soil covariates, then map every layer

- Sets the two constants reused everywhere: `RES_M = 300` (fire-product native
  resolution) and `YEARS = range(2019, 2025)` (fire-product coverage).
- **Climate (from GHCN weather stations — points, not rasters):**
  - `climate.load_ghcn_stations(...)` reads each `.Rds` long table (one row per
    station-day, exported by `src/get_climate&soil_data.R`) into a GeoDataFrame
    of station points in EPSG:5070.
  - `build_climate_normals(...)` → **static** covariates for *matching*
    (`precip_normal`, `tmax_normal`, `tmin_normal`).
    - Internally: `station_normals` collapses each station's daily record over
      the 1991–2020 baseline (annual precip total / mean temperature), then
      `idw_to_grid` inverse-distance-weight-interpolates the station values
      onto the 300 m grid.
    - Why a *normal*: long-run climate is a stable site characteristic — the
      right thing to match treated and control pixels on.
  - `build_annual_climate(...)` → **temporal** covariates for the *outcome
    stage* (`precip`, `tmax`, `tmin`, one raster per year).
    - Same station → IDW path, but per calendar year: this year-specific
      weather is what a `treated:precip` dry-year interaction needs, joined per
      `(x, y, year)` later by `build_frame`.
  - The **scPDSI drought index** (`pdsi`) reuses the same per-year path from
    its own long table. It is self-calibrating (~0 = normal, negative =
    drought), so its long-run normal is ~0 everywhere — it exists *only* as a
    temporal covariate, entering §4 as a `treated:pdsi` interaction.
  - `write_climate_normals` / `write_annual_climate` save GeoTIFFs under
    `data/processed/climate/` where the covariate registry picks them up.
- **Soil (from the SSURGO GeoPackage — polygons, not rasters):**
  - `inspect_soil_columns(...)` prints the attribute columns so you can verify
    the names before rasterizing.
  - `build_soil_rasters(...)` aggregates attributes to one value per soil map
    unit (`mukey_attribute`, depth-weighted where relevant) and burns them onto
    the grid: continuous layers (organic matter `om_r`, available water
    capacity `awc_r`) and categorical ones (drainage class).
- Each build is wrapped in `try/except` so a missing input file skips that
  block instead of killing the notebook.
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
    (elevation, histosol %, climate normals, soil organic matter/AWC) — these
    become distance axes.
  - `categorical` = every registered **categorical** covariate present
    (drainage class, land cover) — these become *exact-match* keys (a treated
    pixel may only match controls of the same class; class codes are labels,
    not magnitudes, so they don't belong in a numeric distance).
- `get_treated_and_control_pixels(...)` → `pixels`, one row per pixel per
  calendar year. Internally:
  - `build_candidate_pool` — `gpd.overlay(peat_aoi, buffered_sites,
    how="difference")`: peat minus the restoration sites and a `spillover_m`
    halo around them (here 0: TNC prevents spillover). What's left is the
    control candidate pool.
  - `pixelate` — lays the common 300 m grid over the AOI, takes each cell
    centre as a point, keeps points inside the polygons (a spatial join), and
    carries polygon attributes (`End_Yr`, `Proj_Name`) onto each pixel so a
    treated pixel remembers *which* site (and restoration year) it belongs to.
  - `_stack_across_years` — repeats the (expensive, one-time) pixel geometry
    once per calendar year to form the panel.
  - Per year, per pixel: `treated = 1` only once the pixel's site has been
    restored (`year ≥ End_Yr`); before that it is a **not-yet-treated
    control** (exactly what the staggered DiD wants);
    `years_after_restoration = year − End_Yr` is the event time. Candidate-pool
    pixels are `treated = 0` in every year with no restoration year.
- `attach_covariates(...)` → `pixels_with_covariates`.
  - Static layers vary only in space, so each raster is sampled **once per
    unique pixel** (nearest grid cell) and broadcast across that pixel's years
    — identical result, no redundant raster reads.
- Diagnostics: `plot_covariate_space` (treated vs control on two axes) and
  `plot_covariate_pairs` (full scatter matrix). A flat axis = nothing to match
  on there.

### §2b — Event-time panel → staggered DiD

- `attach_fire_response(pixels_with_covariates, aoi, product="FireCCIS311",
  res_m=RES_M)` → `panel` with a `burned` column.
  - For each year: load the standardized fire raster, warp it onto the common
    grid (`how="max"`: any burned sub-cell lights the cell), then read the
    value at each pixel's `(x, y)`. Years without coverage stay NaN.
  - This is `build_frame`'s response step re-expressed for pixel *points*
    (the panel) instead of rasterized polygons.
- `did.prepare_panel(panel, restoration_yr_col, site_col="Proj_Name")` adds the
  two bookkeeping columns the estimator needs:
  - `unit_id` — one stable entity id per distinct `(x, y)` pixel.
  - `g` — the Callaway–Sant'Anna **cohort**: each site's first-treatment
    (restoration) year, looked up from the panel itself via
    `did.attach_cohort`; control pixels have no site and get `g = 0`
    (never-treated). It errors loudly if a *treated* row has no cohort year.
- `did.fit_att(panel, covariates=covs, response="burned")` runs the whole
  estimator and returns `(att, overall, event_study)`:
  - `build_panel` — validates and reshapes to a `(unit_id, year)`-indexed
    panel: drops NaN-response rows, requires uniqueness on (entity, time) and
    the presence of *both* treated cohorts and never-treated controls.
  - `estimate_att` — the doubly-robust group-time ATT (via the `differences`
    package; `backend="rpy2"` calls the reference R `did` package instead).
    "Doubly robust" = fits both an outcome model and a treatment-propensity
    model and is consistent if *either* is right; parallel trends need only
    hold conditional on the covariates.
  - `aggregate_att(kind="simple")` — one overall ATT (the headline);
    `aggregate_att(kind="event")` — ATT by time-since-restoration.
  - The whole call is wrapped in `try/except` because `differences` is an
    optional dependency — the levels route (§3–4) still runs without it.
- `plot_event_study(event_study)` — ATT vs event time. **Reading it:** the
  pre-treatment points (left of the onset line) are the parallel-trends check
  and should straddle zero; the post-treatment points trace how the effect
  evolves after restoration.
- `plot_raw_burn_rate_by_year` / `plot_raw_burn_rate_by_event_time` — the
  *unadjusted* burn rate for treated vs control, by calendar year (shared
  dry-year swings appear in both groups) and re-aligned to event time. This is
  the model-free signal the DiD and logit are built to explain.

### Match controls + balance (Stage 5–7) → `units`

- The causal-design heart of the *levels* route: pair every treated pixel with
  a covariate twin, then *prove the pairing worked*.
- `balance_table(pixels_with_covariates, ...)` → `before`: the standardized
  mean difference (SMD = mean difference / pooled SD) per covariate between
  treated and control **before** matching.
- `match_controls(pixels_with_covariates, continuous=covariates,
  categorical=categorical, caliper=1.0, k=1, replace=False, ...)` → `matched`.
  Internally:
  - Collapses the year panel to unique pixels; the treated *group* is
    restoration-site membership (`End_Yr` non-null), not the per-year flag.
  - **Whitens** the covariate matrix by `cov⁻¹ᐟ²` so plain Euclidean distance
    in the whitened space equals **Mahalanobis** distance — every covariate
    contributes on an equal footing and correlated covariates aren't
    double-counted.
  - **Exact-match strata:** nearest-neighbour search runs *within* each
    categorical class (drainage class / land cover) via a KD-tree.
  - **Caliper** (1.0 whitened SD): a treated pixel with no control that close
    is *dropped* rather than matched to a poor twin (the drop count is a
    design diagnostic).
  - `replace=False`: each control serves at most one treated pixel.
  - `site_id` on the output is the restoration site (`Proj_Name`); a control
    *inherits its matched treated pixel's site*, so downstream standard errors
    cluster on the handful of sites, not thousands of correlated pixels.
- `balance_table(matched)` → `after`; then the automated contracts:
  `check_matches` (every control within the caliper) and `check_balance`
  (|SMD| shrank per covariate; balanced means |SMD| < 0.1).
- `plot_balance(before, after)` — the **love plot** (`balance_love.png`), the
  figure that justifies the design: before-vs-after |SMD| per covariate.
- `assemble_units(matched)` → `units`, the GeoDataFrame (`unit_id`, `site_id`,
  `treated`, geometry) that `build_frame` consumes in §3.
- The matched *control* pixels are saved to
  `processed/peat_restoration/matched_controls/matched_controls.gpkg` for
  inspection in the fire-comparison notebook.
- Per-pair diagnostics:
  - `plot_matched_pairs_covariate` — pairs connected in covariate space
    (short segments = close twins).
  - `plot_matched_pairs_geographic` — pairs on the map (long lines flag
    controls drawn far away — possible spatial confounding).
  - `plot_candidate_control_pixels` — full candidate pool (grey) vs selected
    controls (blue) vs treated (orange) over NC.

### §2c — Faithful Castro (2026) matching: score *trajectories*, then match-first DiD

- Motivation: the §2 match compares pixels on a single collapsed covariate
  vector. Castro et al. instead match on a **time series of predicted risk**,
  so that pixels are twins not just on average but through the dry-year spikes.
- Preparation: `panel` is promoted to a GeoDataFrame (`scored`) and any
  categorical land-cover layer is sampled onto it for exact matching.
- `add_prognostic_score_series(...)` → one `phat_<year>` column per outcome
  year.
  - A *separate* logistic fire-risk model per year — fit **only on
    never-treated control pixels** (so a treated pixel's score is a pure
    covariate prediction, no leakage from its own post-restoration burns) —
    then predicted for everyone. Because coefficients differ by year, the
    series encodes how baseline risk shifts across dry/El Niño years, which a
    single collapsed score cannot.
- `add_propensity_score_series(...)` → one `psm_<g>` column per restoration
  **vintage** `g`.
  - One logistic per cohort: pixels restored in year `g` as positives,
    never-treated controls as negatives, projected onto every pixel — it
    captures the *siting decisions* of each construction vintage.
- Score diagnostics: `plot_prognostic_trajectory` (predicted risk over time
  with the observed control burn rate overlaid as a sanity check) and
  `plot_score_overlap` (common support: treated mass where no controls sit is
  where matching must drop or stretch).
- `match_controls_event_time(scored, ...)` → `matched_et`: per-vintage 1:1
  Mahalanobis match on the **trajectory vector**:
  - pre-construction fire-history lags (actual burn at `g−1`, `g−2`, and the
    2015 drought benchmark, when those years are in coverage),
  - the forward prognostic path `phat_t` for `t ≥ g`,
  - the vintage propensity `psm_g`;
  - exact-matched on land cover, no replacement within a vintage, no
    `site_id` exact-match (SEs cluster on site downstream instead).
  - Delegates the actual pairing to `match_controls` with these derived
    columns as the continuous axes; out-of-coverage components are simply
    omitted from that vintage's vector.
  - Balance is re-checked with the same `balance_table` before/after pattern.
- `plot_score_map` + the pair plots — scores per pixel on the map with each
  treated pixel joined to its matched control (same colour at both ends of a
  line = close score match).
- Match-first DiD: `restrict_panel_to_matched(scored, matched_et)` filters the
  panel to the surviving pixels (an inner join on `(x, y)`, carrying the
  match's `site_id`/`pair_id`), then the same `did.prepare_panel` →
  `did.fit_att` pipeline as §2b. The ATT is now identified against each
  treated pixel's matched control(s) instead of the full candidate pool —
  covariate overlap from the match *plus* differencing-out of time-invariant
  confounders from the DiD.

### §3 — Build the tidy pixel-year frame (levels route)

- `build_frame(units, product="FireCCIS311", years=YEARS,
  site_id_col='site_id')` → `frame`, one row per (unit-pixel × year).
  Internally:
  - Rasterizes `unit_id` / `treated` / `site_id` onto the common grid
    (`all_touched=True` so small units survive the coarse grid).
  - Attaches static covariates: values already sampled upstream by the match
    are burned from the units table; anything else is read from its raster.
  - Per year: loads the standardized fire product (`burned`, via `max`
    warping) and joins the per-year weather (`precip`, `tmax`, `tmin`,
    `pdsi`) on `(x, y, year)`.
  - Drops cell-years where the product had no coverage.
- Prints the raw burn rate by treatment as a first look.

### §4 — Fit + odds ratios

- `fit_logit_clustered(frame, covariates=covs)` fits
  `burned ~ treated + elevation + precip_normal + tmax_normal +
  soil_organic_matter + soil_awc` (whichever of those are actually in the
  frame). Internally:
  - standardizes continuous predictors for numerical stability,
  - reports actionable diagnoses for rank-deficiency or perfect separation,
  - clusters standard errors on `site_id` — the effective sample size is the
    handful of restoration sites, not the thousands of correlated pixels.
- `odds_ratios(result)` → `or_table`: `exp(β)` with 95% CIs.
  **Reading it:** `treated` OR < 1 with CI below 1 ⇒ restoration lowers the
  odds of burning.
- **Weather interactions** (one loop over `precip` and `pdsi`): refits with
  `burned ~ treated * <weather> + covariates`.
  - `treated:precip > 0` ⇒ the fire-lowering effect weakens as annual rain
    rises, i.e. restoration does most of its work in dry years.
  - `treated:pdsi`: low PDSI = drought, so a *positive* coefficient likewise
    means the effect is strongest in drought years.
  - Each runs only if its per-year rasters were built in §1b.
- Presentation: a colour-graded styled table for slides, and the headline
  forest plot `odds_ratios.png` — one point + CI per term with the dashed
  no-effect line at OR = 1.

---

## How the two answers relate

- The **DiD ATT** is an absolute change in burn *probability* (Castro's
  headline; `did.avoided_area` converts it to avoided burned hectares), robust
  to any time-invariant confounder but reliant on (conditional) parallel
  pre-trends — which the event-study plot lets you check.
- The **odds ratio** is a relative effect on burn *odds* in levels, robust to
  year-to-year shocks only through the covariates/interactions you include,
  but simpler and estimable without the `differences` backend.
- Agreement between the two (and between the §2 collapsed-score match and the
  §2c trajectory match) is the strongest sign the finding is a design-robust
  effect rather than an artifact of one specification.

## Note on the 2026-07 refactor

- Duplicated cells were removed (the panel-build/DiD-fit pair, the Stage 5–7
  matching block, and §3 each appeared twice; the surviving matching cell is
  the newer one that exact-matches on categorical layers).
- Repeated inline logic moved into the package:
  - `peatfire.modeling.attach_fire_response` (was an inline per-year sampling
    loop, twice),
  - `peatfire.modeling.did.prepare_panel` (was inline `unit_id` + cohort
    bookkeeping, twice),
  - `peatfire.modeling.did.fit_att` (was the inline `build_panel` →
    `estimate_att` → `aggregate_att` chain, three times).
- The two §4 weather-interaction fits were collapsed into one loop, the §2c
  local imports were consolidated into the setup cell, and `RES_M`/`YEARS`
  are now defined once in §1b and reused everywhere.

# Score-based matching + match-first staggered DiD — implementation walkthrough

This document explains the causal-inference machinery added to
`src/peatfire/modeling/` for matching restoration (treated) pixels to comparable
control pixels using **propensity** and **prognostic** scores, and for feeding the
matched set into the staggered **difference-in-differences** (DiD) estimator.

It is the *implementation* companion to two other docs:

- **`modeling_roadmap.md`** — the design and the scientific rationale (why match
  first, why DiD, what Castro et al. 2026 did). Read that for the *why-at-a-level*.
- **`matching_assignment.md`** — the staged matching scaffold (`match_controls`
  and friends).

Here we go function-by-function through the *code*: what each step takes in, what it
returns, how it works, and the justification for each design choice — including the
subtle implementation traps that aren't obvious from the signatures.

Everything is keyed to two modules:

- `peatfire/modeling/matching.py` — scores, matching, plots' data.
- `peatfire/modeling/did.py` — cohorts, the panel, the ATT, and the match-first join.
- `peatfire/modeling/plotting.py` — the diagnostic figures.

---

## The two tracks

There are two matching strategies, sharing the front half of the pipeline:

| | **Scalar track** (simple) | **Faithful track** (Castro 2026) |
|---|---|---|
| Prognostic | one collapsed `phat` per pixel | **per-year** `phat_<year>` series |
| Propensity | one pooled `pscore` per pixel | **per-vintage** `psm_<g>` series |
| Match on | the two scalars `[pscore, phat]` | the **event-time trajectory vector** (fire lags + `phat_<t>` + `psm_<g>`) |
| Builder | `add_matching_scores` | `add_prognostic_score_series` + `add_propensity_score_series` |
| Matcher | `match_controls(continuous=[...])` | `match_controls_event_time(...)` |

The scalar track is the quick, legible option. The faithful track reproduces the
Castro design, which keeps the **time dimension** in both the scores and the match
(different fire-risk functions per year; a treated pixel matched on its whole risk
*trajectory*, not a single number).

Both tracks then flow into the same match-first DiD via `restrict_panel_to_matched`.

---

## Pipeline overview

```
pixels_with_covariates            one row per (pixel, year):
   (the panel)                    x, y, year, <covariates>, End_Yr, Proj_Name, burned
        │
        ├── SCORES ───────────────────────────────────────────────────────────────
        │     scalar   : add_matching_scores            → pscore, phat
        │     faithful : add_prognostic_score_series     → phat_2019 … phat_2024
        │                add_propensity_score_series      → psm_2020, psm_2021, …
        │
        ├── MATCH ─────────────────────────────────────────────────────────────────
        │     scalar   : match_controls(continuous=["pscore","phat"], carry=covs)
        │     faithful : match_controls_event_time(...)   ← per-vintage trajectory match
        │                                                    (exact-match on land cover)
        │
        ├── MATCH-FIRST DiD ───────────────────────────────────────────────────────
        │     restrict_panel_to_matched(panel, matched)   ← keep matched pixels only
        │        → attach_cohort → build_panel → estimate_att → aggregate_att
        │
        └── PLOTS ─────────────────────────────────────────────────────────────────
              plot_score_map / plot_score_overlap / plot_prognostic_trajectory
```

---

## 0. Data prerequisite: the panel shape

Every function below consumes the **pixel-year panel** produced upstream by
`get_treated_and_control_pixels(...)` + `attach_covariates(...)`
(`matching.py`) — one row per `(pixel, year)` with, at minimum:

- `x`, `y` — the pixel-centroid coordinates on the shared EPSG:5070 grid (the join
  key throughout);
- `year` — calendar year;
- the covariate columns (`elevation`, `histosol_pct`, climate normals, soil, …);
- `End_Yr` — the site's restoration year for treated pixels, `NaN` for controls
  (this is `restoration_yr_col`);
- `Proj_Name` — the restoration site id (`site_col`), `NaN` for controls;
- `treated` — the **per-year** 0/1 flag (1 once a pixel's site has been restored);
- `burned` — the 0/1 fire response for that pixel-year (sampled from the product).

A recurring idea: **covariates are static** (they don't change year to year), so most
functions collapse the panel to **unique pixels** with
`drop_duplicates(["x", "y"])`, do their work, and broadcast the per-pixel result back
across the years with an `(x, y)` merge. This avoids triple-counting a pixel that
appears in six yearly rows.

Two different notions of "treated" live in the panel, and picking the right one is
essential:

- `treated` (per-year) — flips at the restoration year. Used by the **DiD** (a pixel
  is genuinely untreated before its restoration).
- restoration-site **membership** (`End_Yr.notna()`) — constant per pixel. Used by
  **matching and scoring** (a restoration pixel is a "treated unit" regardless of
  year; we match on its static geography).

---

## 1. The scores

### 1a. `add_matching_scores` — the scalar propensity + prognostic score

**Purpose.** Collapse the covariate vector into the two scalars that summarise the two
channels a covariate can bias the comparison through: selection into treatment
(propensity) and baseline outcome (prognostic).

**Signature.**
```python
add_matching_scores(pixels, continuous, categorical=(), response="burned",
                    treated_col="treated", restoration_yr_col="End_Yr",
                    propensity=True, prognostic=True,
                    pscore_col="pscore", phat_col="phat", C=1.0)
```

**How it works.**

1. Collapse to unique pixels; the treated *group* is site membership, not the per-year
   flag:
   ```python
   unique = df.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
   grp = unique[restoration_yr_col].notna().to_numpy()      # treated GROUP
   ```
2. Build the design matrix — standardised continuous + one-hot categorical, computed
   **once** over all pixels so scaling and dummy columns are identical for training and
   prediction:
   ```python
   Xc = StandardScaler().fit_transform(use[cont].to_numpy(float))
   Xd = pd.get_dummies(use[cat].astype("category"), drop_first=True)
   X  = np.hstack([Xc, Xd.to_numpy(float)])
   ```
3. **Propensity** `e(X) = P(treated | X)` — a regularised logistic of group membership:
   ```python
   clf = LogisticRegression(max_iter=1000, C=C).fit(X, grp_use.astype(int))
   use[pscore_col] = clf.predict_proba(X)[:, 1]
   ```
4. **Prognostic** `Ψ(X) = E[burn | X, untreated]` — the target is "did this pixel burn
   in an **untreated** year", fit on **controls only**, predicted for everyone:
   ```python
   untreated = df.loc[df[treated_col] == 0, ["x","y",response]]   # the Y(0) rows
   base = untreated.groupby(["x","y"])[response].max()            # ever-burned-untreated
   ...
   train = (grp_use == 0) & use["_baseline"].notna()             # controls, defined baseline
   pmodel.fit(X[train], yb)
   use[phat_col] = pmodel.predict_proba(X)[:, 1]                  # predict all
   ```
5. Broadcast `pscore`/`phat` back onto every pixel-year via an `(x, y)` merge.

**Justification.**

- *Why scores at all?* A confounder biases the comparison only through those two
  channels. Matching on `pscore` closes the "affects treatment" channel; matching on
  `phat` closes the "affects outcome" channel. Matching on **both** is doubly-robust
  (comparable if *either* model is right) — the same logic as `est_method="dr"` in the
  DiD.
- *Why `treated_col == 0` for the prognostic target?* Those are precisely the **Y(0)
  observations** — all years for controls, pre-restoration years for treated pixels.
  Training on them estimates the *untreated* potential outcome.
- *Why fit on controls but predict for all?* A treated pixel's prognostic score is then
  a pure covariate prediction — it never sees that pixel's post-restoration burns. **No
  leakage** into the treatment effect.
- *Why `C` (L2 penalty)?* Fire is rare here, so covariates easily (quasi-)separate the
  classes; the penalty keeps the logistic coefficients finite.

**Gotcha.** `base = ...max()` **collapses all years into one bit** — time is gone. That
is the limitation that motivated the per-year series below.

---

### 1b. `add_prognostic_score_series` — per-year prognostic (faithful)

**Purpose.** Keep the time dimension: a *separate* fire-risk model per year, so the
scores encode how baseline risk shifts across dry / El Niño years.

**Signature.**
```python
add_prognostic_score_series(pixels, continuous, categorical=(), response="burned",
                            treated_col="treated", restoration_yr_col="End_Yr",
                            year_col="year", years=None, prefix="phat_", C=1.0)
```

**How it works.** The design matrix `X` is built once over all pixels; the year loop
only swaps the *target*, which comes from a pixel×year table of burns:
```python
bw = _burned_wide(df, response, year_col)          # pixel × year grid of 0/1 burns
for yr in years:
    tgt   = <burned in year yr, per pixel>
    train = never_use & ~np.isnan(tgt)             # NEVER-treated controls, this year
    model = LogisticRegression(max_iter=1000, C=C).fit(X[train], tgt[train])
    use[f"{prefix}{yr}"] = model.predict_proba(X)[:, 1]   # phat_2019, phat_2020, ...
```
`_burned_wide` just pivots the long panel so "burned in year Y" is a column lookup.

**Justification.**

- *Why per-year?* The fitted coefficients differ each year, so `phat_2019` (a dry year)
  comes out systematically higher than `phat_2021`. The *series* represents the
  year-to-year risk shift a single collapsed score cannot.
- *Why `never_use` (never-treated), not all Y(0) rows?* Castro calibrates on "control
  pixels that never had a block" — a stricter, cleaner training set than the scalar
  version's not-yet-treated rows. `never_use = End_Yr.isna()`.

**Output.** One `phat_<year>` column per fitted year, broadcast across the pixel-years.
A year with no control-pixel fire variation falls back to that year's base rate (with a
warning).

---

### 1c. `add_propensity_score_series` — per-vintage propensity (faithful)

**Purpose.** One propensity model per construction **vintage** `g`, capturing
cohort-specific siting, projected onto every pixel.

**Signature.**
```python
add_propensity_score_series(pixels, continuous, categorical=(), treated_col="treated",
                            restoration_yr_col="End_Yr", prefix="psm_", C=1.0)
```

**How it works.** A three-way target per vintage:
```python
for g in vintages:
    target = np.where(ry_use == g, 1.0,           # vintage g = positive
              np.where(never_use, 0.0, np.nan))    # never-treated = negative; others excluded
    train = ~np.isnan(target)
    model.fit(X[train], target[train])
    use[f"{prefix}{g}"] = model.predict_proba(X)[:, 1]    # projected onto ALL pixels
```

**Justification.**

- *Why exclude other vintages from training but still predict for them?* This is
  Castro's "fit per vintage, project onto the entire dataset": the coefficients are
  cohort-specific, but every pixel receives a `psm_<g>` so it can enter cohort-`g`
  matching as a candidate control.
- *Why per-vintage and not per-year?* With **static** covariates a vintage propensity is
  one number per pixel; the year-indexing in the matching vector comes from the
  *prognostic* series, not this. (Castro fit per *province × vintage*; NC has no province
  layer, so this is per-vintage — extend by fitting within region groups if one appears.)

---

## 2. The matching

### 2a. `carry=` on `match_controls`

**Purpose.** When you match on **derived axes** (the scores), the raw covariates aren't
match axes and would be dropped from the output — losing the love plot and the columns
`build_frame` needs. `carry=` widens the kept-column set.

**How it works (the whole change).**
```python
extra = [c for c in (restoration_yr_col, *carry) if c in cross_w.columns]
keep_cols = list(dict.fromkeys(["x","y","geometry", *cont, *categorical, *extra]))
```
`dict.fromkeys` dedupes while preserving order. Additive and backward-compatible.

**Usage.**
```python
match_controls(scored, continuous=["pscore", "phat"], carry=["elevation", "histosol_pct"])
```
so `balance_table(matched, continuous=["elevation","histosol_pct"])` still works on the
raw covariates the match balanced *indirectly*.

---

### 2b. `match_controls_event_time` — the trajectory match (faithful)

**Purpose.** Match a treated pixel to the control whose **whole time-series of predicted
risk** looks like its own, rather than a single collapsed score. Faithful to Castro.

**Signature.**
```python
match_controls_event_time(pixels, response="burned", categorical=(),
                          restoration_yr_col="End_Yr", site_col="Proj_Name",
                          year_col="year", treated_col="treated",
                          phat_prefix="phat_", psm_prefix="psm_",
                          pre_lags=(1, 2), drought_year=2015, post_horizon=None,
                          caliper=1.0, k=1, max_dist_m=None, carry=(), lag_prefix="fire_y")
```

**Key idea.** This is an **orchestrator**. It assembles the event-time vector *per
cohort* and hands each cohort to the existing `match_controls` engine — so the whitening
(Mahalanobis), caliper, exact-match on categoricals, and no-replacement logic are all
reused, not re-implemented.

**How it works.**

1. Attach burned-by-year columns for the fire-history lags:
   ```python
   bw = _burned_wide(df, response, year_col)
   unique = unique.merge(bw.reset_index(), on=["x","y"], how="left")   # int-named year cols
   ```
2. **Per vintage `g`**, assemble the event-time-anchored column list (only what the panel
   actually covers):
   ```python
   for g in vintages:
       cols = []
       for ly in [g-1, g-2, drought_year]:            # pre-construction fire + drought
           if ly in avail: cols.append(f"fire_y{ly}")
       cols += [f"phat_{py}" for py in sorted(avail) if py >= g]   # forward prognostic path
       cols.append(f"psm_{g}")                        # this vintage's propensity
   ```
3. Match this vintage's treated against **never-treated controls only**, reusing the
   engine (and exact-matching on land cover via `categorical=`):
   ```python
   sub = unique[(unique[restoration_yr_col] == g) | (unique[restoration_yr_col].isna())]
   m = match_controls(sub, continuous=cols, categorical=categorical,
                      caliper=caliper, k=k, replace=False,
                      restoration_yr_col=restoration_yr_col, site_col=site_col, carry=carry)
   m["cohort"] = g
   m["pair_id"] = m["pair_id"] + pair_offset          # keep pair ids unique across cohorts
   ```
4. Concatenate all cohorts; assign a fresh `unit_id`.

**Justification.**

- *Why per-cohort?* Controls have **no event time of their own** — event offset `+1` only
  means something relative to a specific treated pixel's `g`. So you fix `g`, translate
  its event window to calendar years, evaluate every candidate control at those same
  calendar years, and match. Repeat per vintage. Controls are offered fresh to each
  vintage (no-replacement is within-cohort) — Castro's per-vintage matching.
- *Why `if ly in avail`?* Robustness to NC coverage. FireCCIS311 is 2019–2024, so a 2015
  drought column or a `g-2` lag before 2019 simply isn't included, rather than crashing.
- *Why never-treated as the control pool?* Castro matches to never-block controls (not
  not-yet-treated). Other vintages' pixels are excluded from a cohort's candidate pool.
- *Why reuse `match_controls`?* The Mahalanobis whitening, caliper, exact-match, and
  no-replacement are exactly what we want on the assembled vector — no reason to
  duplicate them. The novelty is *which columns* form the vector.

**Output.** Treated + matched controls across all vintages, with `unit_id`, `site_id`
(the restoration site — controls inherit their partner's), `pair_id` (unique across
vintages), `cohort`, `treated`, `match_distance`, `x`, `y`, `geometry`, plus any `carry`
columns. Ready for `assemble_units` / `restrict_panel_to_matched`.

**Note on exact-matching.** Pass `categorical=["land_cover"]` to force controls to share
their treated partner's land-cover class. We deliberately do **not** exact-match on
`site_id` (few NC sites; cluster the SEs on site downstream instead).

---

## 3. Match-first DiD: `restrict_panel_to_matched`

**Purpose.** The join that finally wires matching into the DiD — "match first, then
estimate", so the Callaway–Sant'Anna ATT is identified against the **matched** controls,
not the full candidate pool.

**Signature.**
```python
restrict_panel_to_matched(panel, matched, coord_cols=("x","y"),
                          site_col="site_id", pair_col="pair_id")   # in did.py
```

**How it works.**
```python
lookup = matched[["x","y","site_id","pair_id"]].drop_duplicates(subset=["x","y"])
base   = panel.drop(columns=[c for c in ("site_id","pair_id") if c in panel.columns])
out    = base.merge(lookup, on=["x","y"], how="inner")   # inner = drop unmatched pixels
```

**Justification.**

- *Why `how="inner"`?* The load-bearing word: every candidate pixel not in the matched
  set is dropped, so the DiD's control pool becomes *exactly the matched controls*.
- *Why attach `site_id` from the match?* Controls inherit their treated partner's site,
  so you can `estimate_att(cluster="site_id")` downstream (via the `rpy2` backend) — the
  effective N is the handful of sites, not the thousands of pixels.
- *Replacement warning.* With `replace=True`, a control can appear under two treated
  partners (two `site_id`s); a DiD entity must be one pixel, so the first is kept with a
  warning. The default no-replacement avoids this.

**Downstream.** Run the normal DiD on the restricted panel:
```python
panel_m = restrict_panel_to_matched(scored, matched_et)
panel_m["unit_id"] = panel_m.groupby(["x","y"]).ngroup()
panel_m = attach_cohort(panel_m, cohort_by=cohort_by, key="Proj_Name")   # g = restoration yr
cs      = build_panel(panel_m.dropna(subset=["burned"]), entity="unit_id", time="year",
                      response="burned", covariates=covs)
att     = estimate_att(cs, response="burned", covariates=covs)
overall = aggregate_att(att, "simple")    # headline ATT
events  = aggregate_att(att, "event")     # event study (pre-trends check + dynamics)
```

---

## 4. Diagnostic plots

All three live in `plotting.py`, share `set_fire_style()` + the Okabe-Ito palette, and
return the matplotlib `Figure`.

### `plot_score_map(matched, score_col, scores=…)`

Matched pixels on the map, filled by a propensity/prognostic score, with each treated
pixel joined to its matched control by a segment. **Shape encodes treatment (square =
treated, circle = control); colour encodes the score**, on a shared scale so the two are
comparable.

- *Reading it:* a segment whose two ends are the **same colour** = a control with
  matching predicted risk = a close match; different colours = a stretched match.
- *The `scores=` lookup:* the event-time match output carries pixel ids but not the
  (ragged, per-cohort) score columns, so the score is joined from a per-pixel frame:
  ```python
  if score_col not in m.columns:
      lut = scores.drop_duplicates(["x","y"])[["x","y",score_col]]
      m = m.merge(lut, on=["x","y"], how="left")
  ```
  Works for `pscore`, `phat`, a vintage `psm_2020`, or a per-year `phat_2021`.

### `plot_score_overlap(pixels, score_col)`

Treated-vs-control histogram of one score — the **common-support / positivity** check.
Healthy overlap means a control with a matching score exists for most treated pixels; a
treated mass where no controls sit (shaded) is where matching drops or stretches. Run it
on the **pre-match** scored panel to judge feasibility before matching.

### `plot_prognostic_trajectory(pixels)`

Mean per-year prognostic score (`phat_<year>` columns) for treated vs control across the
study years, with an IQR band and — if `response` is present — the observed never-treated
burn rate overlaid (dashed). The **spikes are the dry / El Niño years** the per-year model
captures; the dashed line is the validation that the scores track the real fire signal.

---

## 5. Two end-to-end recipes

### Scalar track (simple)
```python
scored  = add_matching_scores(panel, continuous=covs, categorical=cats)   # pscore, phat
matched = match_controls(scored, continuous=["pscore","phat"], carry=covs)
panel_m = restrict_panel_to_matched(scored, matched)
# → attach_cohort → build_panel → estimate_att → aggregate_att
```

### Faithful track (Castro 2026)
```python
scored  = add_prognostic_score_series(panel, continuous=covs, categorical=cats)  # phat_<yr>
scored  = add_propensity_score_series(scored, continuous=covs, categorical=cats) # psm_<g>
matched = match_controls_event_time(scored, categorical=["land_cover"],          # exact-match
                                    pre_lags=(1,2), drought_year=2015, carry=covs)
panel_m = restrict_panel_to_matched(scored, matched)
# → attach_cohort → build_panel → estimate_att → aggregate_att
```

Both are wired, with diagnostic plots, in `notebooks/modeling.ipynb` (§2c).

---

## 6. Cross-cutting implementation patterns

These recur throughout — recognising them makes the code read quickly:

- **Collapse-then-broadcast.** Anything static (covariates, scores) is computed on
  `drop_duplicates(["x","y"])` and merged back on `(x, y)`. Avoids triple-counting a
  pixel's yearly rows.
- **Two "treated"s.** Matching/scoring use restoration-site membership
  (`End_Yr.notna()`, constant per pixel); the DiD uses the per-year `treated` flag.
- **Fit on controls, predict for all.** The recurring shape of both the prognostic score
  and the doubly-robust logic — it's what prevents outcome leakage.
- **Lazy `sklearn` imports.** Imported inside the functions (like `statsmodels` in
  `models.py`), so `import peatfire` stays cheap.
- **Left-merges preserve row order**, which is why a design matrix `X` built before a
  merge stays aligned with the frame after it.
- **Reuse the engine.** `match_controls_event_time` assembles columns and delegates the
  actual matching maths to `match_controls`.

---

## 7. Caveats for the NC data

- **Prognostic lags need pre-treatment fire coverage.** FireCCIS311 is 2019–2024, so for
  a 2019/2020 restoration the `g-1`/`g-2` lags and the 2015 drought fall outside coverage
  and are silently dropped — those vintages match on the prognostic/propensity trajectory
  alone. Extending fire history earlier restores them.
- **Small control pool.** Per-vintage, no-replacement, land-cover exact-match constrains
  the candidate pool hard given how few NC sites there are. Watch the drop counts
  `match_controls` prints per cohort; loosen the caliper if too many treated pixels go
  unmatched. (This is the same effect behind Castro's failed 4 km robustness check.)
- **Clustering at site level needs the `rpy2` backend.** The pure-Python `differences`
  backend clusters at the entity (pixel) level; `restrict_panel_to_matched` attaches
  `site_id` so `estimate_att(cluster="site_id")` works once you're on `rpy2`.
- **Land-cover exact-match only fires if the raster is on disk.** The notebook attaches
  the categorical layer defensively and prints whether exact-match keys are actually in
  use.

---

## References

- Callaway, B. & Sant'Anna, P. H. C. (2021). *Difference-in-Differences with Multiple
  Time Periods.* Journal of Econometrics 225(2), 200–230.
- Castro et al. (2026). *Effective restoration can avoid peatland fires: Large scale
  counterfactual assessment in Kalimantan, Indonesia.* iScience. doi:10.1016/j.isci.2026.116041
- Rosenbaum, P. & Rubin, D. (1983). *The Central Role of the Propensity Score in
  Observational Studies for Causal Effects.* Biometrika 70(1), 41–55.
- Hansen, B. B. (2008). *The prognostic analogue of the propensity score.* Biometrika
  95(2), 481–488.
- Leacy, F. P. & Stuart, E. A. (2014). *On the joint use of propensity and prognostic
  scores in estimation of the average treatment effect on the treated.* Statistics in
  Medicine 33(20), 3488–3508.
- Stuart, E. A. (2010). *Matching Methods for Causal Inference: A Review and a Look
  Forward.* Statistical Science 25(1), 1–21.

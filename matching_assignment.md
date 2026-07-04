# Assignment: control-pixel matching

Build the matched treated/control pixel set that feeds `peatfire.build_frame`.
This is the causal-design heart of the study: a **matched observational study**.

Work stage by stage. Each stage is *one small, independently testable function*
in `src/peatfire/modeling/matching.py` (stubs already there, bodies raise
`NotImplementedError`). Don't wire the whole pipeline together until each piece
passes its own `check_*`.

**Getting hints:** each stage has a one-line *nudge* below. When stuck, ask
Claude **"hint for stage N"** — hints escalate gradually (concept → tool/signature
→ worked micro-example on toy data), stopping wherever you say. The solution body
stays yours unless you ask for it outright.

**The test harness is provided** (`matching.py` Part B): `standardized_mean_diff`
and every `check_*` are written for you. Call them on your output:

```python
from peatfire.modeling import matching as m
treated = m.load_treated_units()
m.check_treated_units(treated)     # raises on the first broken invariant, else prints "[Stage 1 OK]"
```

---

## Stage 0 — Load & smell-test
- [ ] **Goal:** get the three real layers open and confirm they line up.
- **Figure out:** What CRS is each in? Do the restoration sites sit inside the
  histosol/DEM extent? Are units metres or degrees?
- **Nudge:** Plot all three on one axes after `.to_crs(5070)`; eyeball overlap.
- **Done when:** one map shows restoration polygons on top of the peat/DEM, and
  you've printed each layer's CRS and feature/pixel count.

## Stage 1 — `load_treated_units()`
- [ ] **Goal:** *completed* restoration polygons in EPSG:5070, each with a pivot year.
- **Figure out:** Which column encodes status? Your rule for "completed"? What do
  you do with a missing end year (impute / drop / fall back to start)? Which year
  is the pivot (you chose end year)?
- **Nudge:** Pure pandas filtering on the attribute table — inspect `.columns`
  and `.value_counts()` first. `load_restoration_sites()` in `frame.py` is the raw loader.
- **Done when:** `check_treated_units(treated)` passes — N matches your read of
  Cat's database, every row has a non-null `pivot_year`, CRS is 5070.

## Stage 2 — `build_candidate_pool(peat_aoi, treated)`
- [ ] **Goal:** peat area that is neither treated nor in the spillover halo (the donut).
- **Figure out:** How do you get a peat polygon from the histosol raster (≥80)?
  How do you *subtract* shapes instead of keeping them? What buffer distance
  represents plausible rewetting spillover?
- **Nudge:** `gpd.overlay(peat, buffered_treated, how="difference")` is the
  inverse of a clip; buffer in metres (you're in 5070).
- **Done when:** `check_candidate_pool(candidates, treated)` passes — zero overlap
  with the treated + halo, positive area. Also *plot the donut* and look at it.

## Stage 3 — `pixelate(polygons, res_m)`
- [ ] **Goal:** pixel-centroid points covering an area, tagged `treated` (1/0).
- **Figure out:** What resolution (match the fire product, ~300 m)? How do you
  turn a grid into centroid points? How do you keep only centroids inside `polygons`?
- **Nudge:** Reuse `build_modeling_grid`; a cell centre is `(x+res/2, y-res/2)`;
  call once for treated area, once for the candidate pool, add `treated`, concat.
- **Done when:** `check_pixels(points, treated_polys)` passes — `n ≈ area/res²`
  and every treated pixel falls inside a restoration polygon.

## Stage 4 — `attach_covariates(points, names)`
- [ ] **Goal:** add one column per covariate, sampled at each pixel.
- **Figure out:** How do you read a raster value at a point? Which covariates are
  continuous vs categorical (don't average land cover)? Your rule for pixels that
  are NaN in a covariate?
- **Nudge:** `covariate_on_grid(name, grid, aoi)` warps a layer onto the grid;
  then index the cell each point sits in.
- **Done when:** `check_covariates(points, names)` passes — no all-NaN column
  (for layers on disk: elevation, histosol_pct), ranges sane.

## Stage 5 — `match_controls(pixels, continuous, categorical, caliper, k)`
- [ ] **Goal:** pair each treated pixel with its nearest control(s).
- **Figure out:** Why standardize before distances? Euclidean vs Mahalanobis
  (what does Mahalanobis fix, given elevation ↔ distance-to-coast are correlated)?
  How do you *exact-match* land cover instead of putting it in the distance? What
  is a caliper, in what units? With/without replacement, 1:1 or 1:k?
- **Nudge:** `sklearn.neighbors.NearestNeighbors` on z-scored continuous columns,
  run **within** each land-cover class; drop matches beyond the caliper.
- **Done when:** `check_matches(matched, caliper)` passes — every retained treated
  pixel has a control within the caliper, controls disjoint from treated, and you
  can print how many treated pixels got dropped for lack of a match (that number matters).

## Stage 6 — `balance_table` + `plot_balance` (the payoff)
- [ ] **Goal:** show matching worked.
- **Figure out:** What is a standardized mean difference (SMD) and what threshold
  means "balanced" (|SMD| < 0.1)? What does *before* vs *after* prove?
- **Nudge:** `standardized_mean_diff` is provided — apply it per covariate on the
  unmatched pool and again on the matched set; the love plot is those numbers as dots.
- **Done when:** `check_balance(before, after)` passes — post-match |SMD| shrinks
  toward 0 and the overlaid distributions visibly overlap. *If balance doesn't
  improve, the matching has a bug or no good controls exist.*

## Stage 7 — `assemble_units` → frame → fit
- [ ] **Goal:** package into the `units` GeoDataFrame `build_frame` expects, then fit.
- **Figure out:** What are `unit_id`, `site_id`, `treated` in your matched output
  (site_id = the matched stratum / pair)?
- **Nudge:** select/rename columns; then `build_frame(units)` →
  `fit_logit_clustered` → `odds_ratios`.
- **Done when:** the frame has both classes and multiple years, the model fits
  without separation errors, and the OR table is interpretable (read cautiously at this N).

---

## How to test each piece (not just "does it run")

1. **Toy inputs first.** Before real data, feed each function a hand-built tiny
   case where you know the answer: 3 fake restoration squares + a 5×5 numpy
   raster. For `build_candidate_pool` you can work out by hand which cells
   survive. This catches logic bugs real data hides.
2. **Invariants as assertions.** The `check_*` functions *are* your regression
   tests — run them after every change.
3. **Visual checks.** Geospatial bugs are obvious on a map and invisible in a
   dataframe. Plot after Stages 2, 3, and 6.
4. **Known-answer probes.** Pick a pixel you *know* is inside a restoration site;
   assert it comes out `treated == 1`. Pick one 50 km inland; assert it's never a candidate.

## How to know it's *good*, not just correct
- **Enough matched controls survived** — you didn't have to blow the caliper open.
- **Balance improved** (Stage 6) — the real acceptance test for the design.
- **Not driven by one site** — jackknife: re-run leaving each site out and check
  the odds ratio is stable. Given the "a couple of sites drive it" worry, this
  matters most.
- **Negative control passes** — run the whole pipeline on the *planned* (not-yet-
  restored) sites' scheduled years; you should see *no* effect. If you do,
  something upstream is confounded.

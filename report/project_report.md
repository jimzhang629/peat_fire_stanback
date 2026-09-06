# Wildfire risk in North Carolina peatlands and the effect of peatland restoration on burned area

*Prepared for The Nature Conservancy · September 2026*

---

## Summary

This project did two things. First, it compared eight satellite fire products —
covering fire occurrence, burned area, and burn severity — and validated them
against fire-perimeter reference datasets over North Carolina and over North
Carolina peatlands specifically. Second, it built a matched, staggered
difference-in-differences design to test whether rewetting-based restoration at
six TNC peatland sites reduced burned area relative to ecologically similar
unrestored peat.

**Recommended products.** VIIRS active fire for occurrence; FireCCIS311 for
spatial accuracy in burned area; MODIS MCD64A1 wherever a long pre-treatment
record is needed. No burn severity product is usable over NC peat.

**Effect of restoration.** Not identifiable with the data available. The pixel-level
estimator returns a negative average treatment effect on the treated (ATT) of
−0.024 in annual burn probability with a confidence band excluding zero, but that
estimate cannot be read as a restoration effect: the parallel-trends assumption
fails a year *before* treatment begins, all of the fire in the matched panel
falls in three calendar years at two sites, and the two sites that contribute a
usable pre/post contrast point in opposite directions and nearly cancel.

**What predicts fire.** Elevation is the only covariate with a stable, tightly
bounded association with burning (odds ratio 0.11 per standard deviation, 95% CI
0.08–0.14). Restoration status is not a usable predictor; the levels model that
includes it is quasi-separated and its treatment coefficient is not interpretable.

**The central finding is about the outcome variable, not the sites.** Restored
pocosins still burn at the surface. What rewetting is expected to change is how
*deep* a fire burns — peat consumption — which is precisely what a 300 m
burned-area product cannot see. Burned extent was never the quantity the
mechanism acts on, so a null on burned extent is close to the expected result
rather than an anomaly. Sections 4.2 and 5 develop this and set out what to do
instead.

---

## 1. Introduction

This project compared the results from several fire products in assessing the
occurrence, burn severity, and burned area of fires in North Carolina peatlands.
It then examined whether peatland restoration via rewetting reduced burned area
in six peatland sites restored by TNC compared to ecologically similar
non-rewetted areas. The small number of restoration sites and their relatively
recent restoration years precluded conclusive claims about the impact of
restoration on burned area.

Peatlands hold a disproportionate share of North Carolina's terrestrial carbon,
and drained pocosin peat is vulnerable to deep, smouldering ground fire that
consumes carbon accumulated over centuries. Rewetting raises the water table.
The mechanism by which it is expected to reduce emissions is by limiting how deep
a fire burns, not by preventing fire from occurring or from spreading. That
distinction between burn *extent* and burn *depth* runs through this report and
is its main conclusion.

**[FIGURE 1 — North Carolina peat extent, with the completed TNC restoration
sites marked. Use the peat-extent figure already in the draft.]**

*Figure 1. North Carolina peat extent. Peat is derived from the gSSURGO
major-histosol percentage layer of Lilleskov et al. (2025). The analysis frame
used for matching and modelling is high-confidence peat, histosol % ≥ 80.*

---

## 2. Methods

### 2.1 Data

All datasets compiled for this project, including layers not used in the analyses
below, are catalogued in `data_inventory.csv` with source, native resolution,
temporal coverage, units, and Box path. Acquisition and clipping are handled in
`notebooks/download_and_clip_data.ipynb`.

**Fire products.** Occurrence: VIIRS active fire (375 m, 2012–present). Burned
area: MODIS MCD64A1 (500 m, Nov 2000–present), FireCCI51 (250 m, 2001–2020),
FireCCIS311 (300 m, 2019–present), GABAM (30 m, 1985–2021). Severity: MODIS
MOSEV (500 m, 2000–present), MTBS (30 m, 1984–present), SE FireMap (30 m,
2000–2022).

**Reference perimeters.** NIFC Interagency Wildland Fire Perimeter History
(1980–present), GEOMAC Historic Perimeters Combined (2000–2018), TNC Coastal
Plain burn history (2025), and TNC Sandhills burn history (2025). Fires under 100
acres were excluded from NIFC_IFPH and GEOMAC. TNC Sandhills has no overlap with
the ≥80% histosol mask and so contributes only to the statewide validation.

**Covariates.** Elevation (Copernicus GLO-30); histosol % (gSSURGO, Lilleskov et
al. 2025); mean annual precipitation, mean daily maximum and minimum temperature,
and growing degree days as 1991–2020 normals interpolated from GHCN station
records; PDSI drought index by year; HAND drainage; and the SSURGO soil layers
(organic matter, available water capacity, site index, water-table depth,
drainage class). LANDFIRE land cover and soil drainage class are used as
exact-match keys rather than as continuous matching dimensions.

### 2.2 Common grid and mask preprocessing

Every layer was clipped to North Carolina and reprojected to EPSG:5070 (NAD83 /
CONUS Albers), an equal-area projection in metres, so that areas are meaningful
and all distances and buffers are in metres. Layers were then warped onto a
single product-independent grid: **300 m for the modelling** and **500 m for the
fire-product comparison**, the latter set by the coarsest product compared so
that spatial resolution is not itself a confound in the agreement metrics.

Continuous layers are resampled by area-weighted average; categorical layers take
the majority class within the cell; fire products are aggregated with a "max"
rule, so any sub-cell burn lights the whole cell. Monthly products (MCD64A1,
FireCCI51, MOSEV) are OR'd into annual layers.

Peat extent comes from the gSSURGO major-histosol percentage layer. A binary
extent mask (histosol % > 0, with adjacent peat pixels merged into contiguous
blobs) is used for the descriptive product comparison; a high-confidence frame
(histosol % ≥ 80) is used for all matching and modelling, which keeps non-peat
noise out of the covariate distributions.

### 2.3 Product comparison and validation

Products were compared pairwise on the common grid using binary metrics —
Jaccard index, Cohen's κ, percent agreement — rather than Pearson correlation,
which on binary maps is only the phi coefficient. Pairwise scatter of per-cell
burned area was fitted by total least squares rather than OLS, because both
products carry measurement error and OLS would bias the slope by treating one
axis as error-free.

Reference perimeters were rasterized to the same grid with `all_touched=True`, so
small reference fires survive the 500 m cells. Each fire event is scored against
the product layer **for its own year**; events are never pooled into a timeless
union. For each event, true positives are burned cells inside the perimeter that
the product detected and false negatives are cells inside the perimeter that it
missed, giving:

- **Recall** = tp / (tp + fn), the fraction of a known fire's extent recovered.
  This is 1 − omission error and is the headline metric.
- **Pooled recall**, summed across all events — cell-weighted, so dominated by
  large fires.
- **Mean per-event recall**, the unweighted mean across events — counts every
  fire equally, so it surfaces small-fire omission, the failure mode that matters
  most in a small-fire landscape like the pocosins.
- **Detection rate**, the fraction of events for which the product mapped any
  burned cell inside the perimeter. This is the meaningful metric for VIIRS,
  which detects a burning front rather than an area.

Precision is reported only conditionally, per event within a 5 km buffer, for two
reasons: a product burn outside an incomplete reference may be a real fire the
reference never recorded, and a perimeter is an outer boundary containing
unburned islands, so it overstates true burned area and thereby depresses
precision. Recall is what the recommendations rest on. Severity was not validated
against perimeters, which carry no severity information.

### 2.4 Restoration analysis

The design follows Castro et al. (2026), who evaluated canal-block rewetting and
peat fire in Kalimantan.

**Estimand.** The average treatment effect on the treated: for each restored
pixel in each year, the difference in burn probability between the world in which
it was restored and the world in which it was not, averaged over restored pixels.
Treatment is a pixel inside a completed restoration site, switched on from that
site's restoration year forward; controls are peat pixels (≥80% histosol) outside
every restoration site; the response is a 0/1 "did this pixel burn this year"
flag; the unit is a pixel-year on the 300 m grid, tagged with site and year.

**Matching.** Treated and candidate control pixels were matched 1:1 without
replacement by minimum Mahalanobis distance in covariate space, with the
covariate matrix whitened so that distance is unaffected by covariate units or
correlation. Land cover and soil drainage class are matched exactly. A caliper of
1 standard deviation caps the acceptable distance, and treated pixels with no
control inside it are dropped and counted. Match quality is judged by the
standardized mean difference (SMD) per covariate before and after matching.

**Estimator.** The canonical two-period difference-in-differences is biased under
staggered adoption because it reuses already-treated units as controls. The
analysis instead uses the Callaway & Sant'Anna (2021) doubly robust estimator
(Python `differences` package), which uses only never-treated and not-yet-treated
pixels as controls and, for each cohort and year, fits both an outcome regression
and a propensity model, so that parallel trends need only hold conditional on
covariates. Matching also uses propensity and prognostic scores — respectively
the probability of being treated given covariates, and baseline fire risk absent
treatment fit on controls only — together with temporal and spatial fire lags.

Only cohorts with both a pre- and a post-treatment year can contribute. Under
FireCCIS311 (2019–2024) this retains the 2021 and 2023 cohorts and excludes 2019
(27,424 pixels, no pre-period) and 2026 (no post-period).

`soil_water_table_depth` was dropped from the difference-in-differences
covariates: including it alongside drainage, precipitation, and the other soil
layers drove control pixels to near-zero estimated treatment probability for the
later cohorts, a positivity failure that destabilises the inverse-probability
weights the doubly robust estimator depends on.

**Second route.** In parallel, a matched logistic regression was fit in levels —
`burned ~ treated + covariates` on the matched pixel-years, with standard errors
clustered on restoration site — and reported as odds ratios. Agreement between
the two routes would be the strongest evidence of a design-robust effect.

---

## 3. Results

### 3.1 Product comparison

Products were assessed on spatial resolution, length and currency of temporal
coverage, update frequency, and sensitivity to the small fires that dominate the
southeastern fire regime (see the Fire Product Comparison and Recommendation
Notes tabs in `fire_product_comparison.xlsx`). On those criteria the most
suitable datasets for occurrence, burned area, and severity were VIIRS Active
Fire, GABAM, and SE FireMap respectively.

Inspection of the severity products over North Carolina, however, showed all
three to be far too sparse over peat to support analysis, and severity was
dropped as an outcome.

**[FIGURE 2 — burn severity products over North Carolina: a, MTBS; b, SE
FireMap; c, MOSEV. Use the severity figure already in the draft.]**

*Figure 2. Burn severity products overlaid over North Carolina. **a**, MTBS.
**b**, SE FireMap. **c**, MOSEV. None provides usable coverage over peat, so
burn severity was dropped as an outcome.*

### 3.2 Validation against reference perimeters

Product accuracy was then measured against the reference perimeter datasets over
North Carolina peatlands.

**[FIGURE 3 — recall by product against the reference perimeters. Saved as
`outputs/figures/fire/validation_recall_nifc_ifph_<aoi>.png` and
`outputs/figures/fire/nifc_ifph_burned_area_validation_summary.png`.]**

*Figure 3. Burned-area product recall against fire-perimeter references within
North Carolina peatlands. Pooled recall is cell-weighted and dominated by large
fires; mean per-event recall counts each fire equally and is sensitive to
small-fire omission; detection rate is the fraction of events the product caught
at all. Higher is better on all three. A product with high pooled recall but low
mean-event recall is finding the big fires and missing the small ones.*

The per-event results show the pattern that drives the recommendation. On the
large fires every product does reasonably well — on the 2012 Dad fire, recall was
0.89 for MCD64A1, 0.88 for FireCCI51, and 0.67 for GABAM. On small fires the
products diverge sharply and most fail outright: of the small NIFC events scored,
several (Gum Branch 2010, Deep Creek 2012, Deep Bend 2013) were missed entirely
by all three burned-area products. VIIRS, being an occurrence product, detected
some of these but recovers only a fraction of any perimeter's extent, as expected
from a front detector.

**Recommendation.** For **occurrence**, VIIRS active fire — it is the only
product with sub-daily temporal resolution and the only one that could support a
smouldering-persistence analysis (§5). For **burned area**, the choice depends on
the question: **FireCCIS311** (300 m, 2019–present) has the best combination of
small- and large-fire detection in peat and is the right default where the
analysis window falls inside its record; **MODIS MCD64A1** (500 m, 2000–present)
is the right choice wherever a long pre-treatment baseline matters, which
includes every before-and-after restoration comparison. For **severity**, no
product is recommended over NC peat; this should be recorded as a data gap.

This creates a genuine tension for the restoration analysis. FireCCIS311 is the
more accurate product but begins in 2019, which leaves only the 2021 and 2023
restoration cohorts with any pre-treatment data. The results below are from
FireCCIS311; re-running them on MCD64A1, which recovers the 2019 cohort and the
large 2008 and 2011 fires, remains outstanding.

### 3.3 Effect of restoration on burned area

**Matching.** Of 3,786 treated pixels, **3,623 were matched 1:1 to controls** and
163 were dropped for having no control inside the caliper; the largest accepted
match distance was 0.998, just inside the caliper of 1.0. Matching substantially
improved balance on every covariate (Table 1): the largest absolute standardized
mean difference fell from **0.46 to 0.048**, comfortably inside the conventional
0.2 threshold.

| Covariate | \|SMD\| before | \|SMD\| after |
|---|---|---|
| soil_water_table_depth | 0.465 | 0.000 |
| soil_organic_matter | 0.420 | 0.000 |
| gdd_normal | 0.412 | 0.035 |
| histosol_pct | 0.400 | 0.002 |
| elevation | 0.316 | 0.048 |
| tmin_normal | 0.301 | 0.047 |
| tmax_normal | 0.286 | 0.016 |
| soil_awc | 0.217 | 0.000 |
| precip_normal | 0.109 | 0.031 |
| soil_site_index | 0.011 | 0.000 |

*Table 1. Standardized mean difference between treated and control pixels, before
and after covariate matching. Values closer to zero indicate better balance.*

**[FIGURE 4 — covariate balance love plot, `outputs/figures/modeling/balance_love.png`.]**

*Figure 4. Standardized mean difference per covariate before and after matching.*

**Treatment effect.** The pooled estimator returns an ATT of **−0.024** in annual
burn probability (analytic SE 0.0013; band −0.027 to −0.021), and the event study
returns a coefficient at every event time whose band excludes zero: −0.011 at one
year *before* restoration, +0.003 in the restoration year, −0.008 at one year
after, −0.096 at two years, and +0.005 at three.

**[FIGURE 5 — difference-in-differences event study,
`outputs/figures/modeling/FireCCIS311_pixel_level_did_event_study_tidy.png`.]**

*Figure 5. ATT on annual burn probability by year relative to each site's
restoration year, FireCCIS311, 2019–2024. Estimates at negative event times test
the parallel-trends assumption and should be indistinguishable from zero.*

None of these numbers can be read as a restoration effect, for three reasons that
are visible in the estimate itself.

**The parallel-trends assumption fails before treatment begins.** The estimate at
one year pre-restoration is −0.011 with a band excluding zero. Treated and
control pixels were already diverging before any restoration occurred, so the
post-treatment coefficients are not identified as treatment effects regardless of
their p-values.

**Almost all the fire is in two sites and three years.** In the matched panel,
burned pixel-years occur only in 2020 (3), 2022 (43), and 2023 (33) — 79 in total
— and they fall in just two restoration sites' neighbourhoods. Six sites enter
the matching; three survive into the matched event-time panel; only two
contribute a non-zero pre/post contrast. The pooled estimator, by contrast, is
computed over roughly 326,000 pixel-years, which is where its apparent precision
comes from.

**Those two sites point in opposite directions and nearly cancel** (Table 2).
Treated pixels did not burn at all in the matched panel. All the fire is in
control pixels, and whether that shows up as a benefit or a penalty depends
entirely on whether the control fire happened to fall before or after that site's
restoration year.

| Site | Control burn rate, pre | Control burn rate, post | Site contribution |
|---|---|---|---|
| Angola Bay Restoration Area | 0.0227 | 0.0000 | **+0.0227** |
| PLNWR Flood Resilience | 0.0000 | 0.0212 | **−0.0212** |
| GDSNWR Pasquotank Headwaters | 0.0000 | 0.0000 | 0.0000 |

*Table 2. Site-level pre/post burn rates in the matched panel, FireCCIS311.
Treated pixels recorded no fire at any site in any year, so each site's
contribution is set by its control pixels alone. The two non-zero contributions
are of opposite sign and similar magnitude.*

The narrow band on the pooled ATT is therefore an artifact of counting pixels
rather than sites. Clustered on restoration site, the standard error is 0.0148;
computed across pixels, it is 0.0018 — a factor of eight. The pixel-level
interval is the one the estimator prints, and it is the wrong one.

Finally, the largest event-study coefficient, −0.096 at two years post, is driven
by a single fire in a single year and should not be reported as an effect.

**Levels route.** The matched logistic regression does not corroborate the
difference-in-differences result; it contradicts it. In the matched levels frame
(43,452 pixel-years) the raw burn rate is 0.0047 in control pixels and 0.0339 in
treated pixels — restored pixels burned *more* — and the fitted model returns a
treatment odds ratio of 9.2 (95% CI 2.8–30.3). That estimate is not usable: the
fit reports possible quasi-separation with 18% of observations perfectly
predicted, and the climate normals return intervals spanning nine orders of
magnitude, both signs of a badly conditioned model. The disagreement between the
two routes, on the same data, is itself the finding: the design does not support
a stable treatment estimate.

### 3.4 Which factors best predict fire

Across both model specifications, the only covariate with a stable and tightly
bounded association with burning is **elevation**: odds ratio 0.11 per standard
deviation (95% CI 0.08–0.14, p < 10⁻⁵⁰). Lower-lying peat burns more. This is the
one coefficient in the levels model that is unaffected by the conditioning
problems described above, and it is consistent with the hydrological mechanism —
elevation in this landscape is a proxy for depth to water table.

Everything else is weak or unidentified. Growing degree days (OR 61, 95% CI
0.5–7,465) and, in the specification carrying a treatment × PDSI interaction,
PDSI itself (OR 0.35, 95% CI 0.10–1.24) have the expected signs — warmer and
drier means more fire — but intervals far too wide to support a claim. The
soil covariates (organic matter, available water capacity, site index,
water-table depth) are all indistinguishable from no effect. The precipitation
and maximum-temperature normals return uninterpretable intervals.

**Restoration status is not a usable predictor of fire.** The difference-in-
differences and levels routes disagree in sign, and the levels model that
produces the larger effect is quasi-separated.

Two caveats on this section, both material. First, these results are from
FireCCIS311, which covers 2019–2024 and therefore contains only three fire years;
the MCD64A1 fit over 2001–2024 would be far better identified and has not been
run. Second — and directly responsive to a question raised in review — the fitted
model currently carries PDSI, growing degree days, precipitation normal, **and**
maximum-temperature normal. Dropping the precipitation and temperature normals
and retaining only PDSI and GDD, as requested, is a one-line change to the
covariate list and has not yet been run; given how wide the current climate
intervals are, the reduced specification is likely to be better conditioned.

Note that HAND drainage — the covariate closest to the restoration mechanism —
was compiled but does not appear in the fitted models, because it enters as a
matching variable rather than an outcome predictor. Section 5 argues this is
backwards.

---

## 4. Discussion

### 4.1 What the analysis can and cannot support

The restoration effect on burned area is not identifiable from these data. That
conclusion does not rest on a single diagnostic but on the convergence of
several: parallel trends fails a year before treatment, the fire signal reduces
to 79 burned pixel-years in two sites, the two usable sites cancel, the two
estimation routes disagree in sign, and the surviving levels model is
quasi-separated.

It is worth being precise about why. Treatment is assigned to a *restoration
site*, and North Carolina peat fire arrives as a handful of large,
landscape-scale events, each of which paints thousands of contiguous pixels
burned in a single draw. The independent replicate is therefore a site-year, not
a pixel-year, and there are at most 36 of them (six sites over six years) — of
which, under FireCCIS311, far fewer are usable. Counting 325,740 pixel-years
produces standard errors that look precise, but that precision is arithmetic, not
evidence.

There is also a plausible reason the pre-treatment trends diverge that no
covariate adjustment can fix: if sites were selected for restoration partly
*because* they had recently burned or were visibly degraded, treatment is
endogenous to fire history. Restoration in this landscape does follow fire. This
should be checked against the TNC restoration records; if it holds,
difference-in-differences on these sites is not salvageable.

This was an anticipated outcome, not a surprise — the project roadmap allowed
that treatment effects might not be identifiable and asked that limitations be
documented if so.

### 4.2 Burned area is the wrong outcome for the rewetting mechanism

The more useful finding is that burned extent was never the quantity restoration
acts on.

Surface fire in undrained pocosin is common and occurs naturally, at fire return
intervals on the order of 20–80 years, and most surface fires are low severity
with respect to peat combustion. Restored pocosins still burn at the surface.
What rewetting changes is how deep the fire goes — and burn depth is a strong,
well-documented function of water table position:

| Water table depth | Vertical peat loss in fire | Source |
|---|---|---|
| 66 cm (drained) | 12.5–24 cm | Reardon et al. (2007), Green Swamp |
| 30–38 cm | little or none | Reardon et al. (2007) |
| ~30 cm (PLNWR prescribed burn, 2015) | 1–2 cm | Flanagan et al. (2020) |
| 8 cm (restored block, PLNWR) | none observed | Richardson et al. (2022) |
| drained, severe ground fire | ~40 cm | Mickler et al. (2017) |

*Table 3. Vertical peat loss in fire as a function of water table depth. Figures
compiled from the pocosin literature; verify against the primary sources before
external use.*

A 300 m burned-area product cannot see any of this. It records that a cell
burned, not how much peat was consumed. So a null on burned extent is close to
the expected result, and reporting it that way — restoration did not reduce
burned *extent*, consistent with the literature, because extent was never the
mechanism — is a substantive contribution rather than a consolation.

This has a direct bearing on the carbon accounting. An estimate that assumes
restored sites experience no wildfire emissions is inconsistent with what these
data show: restored sites do burn. What they plausibly do not do is lose peat.
The lower alternative already contemplated in the CPRG memo — shallow burn in
restored peat versus deep burn in drained peat — is the assumption consistent
with both the literature and this analysis. **This result supports the revised,
lower estimate**, and it argues for treating avoided fire as a durability or
buffer-pool term rather than a credited annual flux. The specific numbers should
be taken from the CPRG memo directly.

### 4.3 Decisions made, and why

| Decision | Alternative | Rationale |
|---|---|---|
| Peat frame at histosol % ≥ 80 | > 0% | Keeps non-peat noise out of the covariate distributions; the descriptive extent map still uses > 0%. |
| Matched case-control design | Whole-landscape GLM | Buys covariate overlap, so the restoration effect is not confounded by where restoration happens. |
| Match on the full climate + soil set | Elevation + histosol % | On the ≥80% frame histosol % is nearly constant, so the original pair was effectively an elevation-only match balancing nothing. |
| Climate as IDW-interpolated normals | Kriging | Transparent, dependency-light, and never extrapolates — all a matching covariate needs. |
| EPSG:5070 for all analysis | AOI native CRS | Equal-area, so area arithmetic and distances are meaningful. |
| Product-independent common grid, "max" aggregation | Each product's native grid | Removes spatial resolution as a confound in the agreement metrics. |
| Total least squares for product scatter | OLS | Both products carry error; OLS would bias the slope by treating one axis as error-free. |
| Recall as headline metric, precision conditional | Symmetric precision/recall | Perimeters are outer boundaries containing unburned islands and are not a spatially exhaustive census. |
| Callaway–Sant'Anna staggered DiD | Canonical two-period DiD | Canonical DiD reuses already-treated units as controls and is biased under staggered adoption. |
| Standard errors clustered on site | Pixel clustering | Treatment is assigned at the site; pixel clustering understates the SE by a factor of eight here. |
| `soil_water_table_depth` dropped from the DiD | Retained | Positivity failure — control pixels received near-zero treatment probability for later cohorts. |
| Severity dropped as an outcome | Retained | All three severity products are too sparse over NC peat. |

*Table 4. Design decisions and their rationale.*

### 4.4 Limitations

- Six restoration sites, of which three survive into the matched panel and two
  contribute a usable pre/post contrast.
- Results are from FireCCIS311 (2019–2024, three fire years). The MCD64A1
  long-record run, which would recover the 2019 cohort and the large 2008 and
  2011 fires, has not been completed.
- Parallel trends fails before treatment, plausibly because restoration follows
  fire; this has not been checked against the restoration records.
- Controls are matched landscape peat rather than a within-buffer donut, so
  conditional parallel trends leans harder on match quality than in the design
  this one is adapted from.
- The planned negative control — running the pipeline on planned-but-unrestored
  sites at their scheduled years, where any apparent effect would flag
  confounding — was not run.
- Belowground smouldering, the mechanism of interest, is unmeasured by every
  product used.

---

## 5. Next steps

Ordered by value per unit of effort.

**1. Finish the two runs that are one configuration change away.** Re-run the
models on MCD64A1 over 2001–2024, and re-run the levels model with PDSI and
growing degree days as the only climate covariates, dropping the precipitation
and temperature normals. Both are small changes, both would materially improve
§3.4, and the second answers a question already raised in review.

**2. Switch the treatment variable from a six-site binary to a landscape-wide
hydrologic gradient.** This is the largest available gain. The binding constraint
is six sites, but the mechanism — water table depth — varies continuously across
the whole peat AOI, and the layers are already built (HAND drainage, soil
water-table depth, soil drainage class). Estimating `burned ~ f(drainage) +
covariates` across the full peat area and the full MODIS record uses on the order
of two million pixel-years and every large fire since 2001, instead of a handful
of site-years. It is not a clean quasi-experiment — drainage is not randomly
assigned — but it targets the same causal link and can be pushed with
ditch-density gradients, a distance-to-canal discontinuity at drainage-district
boundaries, and matched drained/undrained pairs. Note that drainage is currently
used only as a matching variable; this proposal makes it the treatment.

**3. Change the outcome to something that reflects burn depth.** In order of
feasibility:

  - **VIIRS detection persistence.** Smouldering peat fires produce repeated
    active-fire detections at the same location over days to weeks; surface fires
    flash and stop. Detections per pixel per event, or fire duration, is a free
    proxy for smouldering using data already downloaded. This is the cheapest
    real move.
  - **InSAR or repeat-lidar subsidence.** Peat surface elevation loss after a
    fire is a direct measurement of burn depth. Repeat lidar exists for coastal
    North Carolina; differencing pre- and post-fire DEMs over the 2008 and 2011
    fires would produce actual burn-depth numbers for NC, which currently do not
    exist.

**4. A bounding analysis for the carbon estimate.** A Monte Carlo over the
parameters that actually drive the wildfire emissions term — fire return
interval, burn depth under drained versus restored conditions, peat versus
aboveground carbon contribution, and restoration effectiveness — produces a
distribution and a variance decomposition rather than a point estimate. Its
advantage over any further empirical work is that it cannot return null: it
produces a number and an uncertainty range by construction. The unestimable
restoration effect enters as a parameter with a wide prior rather than as an
obstacle. The expectation, stated in advance so it can be checked, is that burn
depth and fire return interval will dominate the variance and restoration
effectiveness will contribute least — in which case the parameter this project
spent its effort trying to estimate was never the one that mattered.

**5. A prospective monitoring design.** If a future study is to answer this
question, the instrumentation has to be in place before the next fire: water
table wells at restored *and* paired control sites, peat depth pins or
rod-surface-elevation tables to measure burn depth directly, and a
pre-registered analysis. Pair this with the negative control described in §4.4.

**What not to do.** Three things would look like progress and are not. Adding
more covariates will not rescue the treatment estimate — the constraint is the
number of sites and a mismatched outcome, not omitted variables, and more
covariates worsen the positivity failure that already forced out
`soil_water_table_depth`. Loosening the caliper or reverting to pixel-level
clustering would recover apparent significance, but that apparent precision is
the artifact, not the result. And the −0.096 event-study point should not be
reported as an effect; one fire in one year drives it.

---

## References

- Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-differences with
  multiple time periods. *Journal of Econometrics* 225(2), 200–230.
- Castro et al. (2026). [Kalimantan canal-block rewetting and peat fire — complete
  citation.]
- Flanagan, N.E. et al. (2020). [Prescribed fire peat loss at Pocosin Lakes NWR —
  complete citation.]
- Lilleskov, E. et al. (2025). [gSSURGO major-histosol percentage — complete
  citation.]
- Mickler, R.A. et al. (2017). [Deep peat fire consumption, Pocosin Lakes —
  complete citation.]
- Poulter, B. et al. (2006). [Pocosin fire return intervals and emissions —
  complete citation.]
- Reardon, J. et al. (2007). [Peat consumption vs. water table position, Green
  Swamp NC — complete citation.]
- Richardson, C.J. et al. (2022). Annual carbon sequestration and loss rates under
  altered hydrology and fire regimes in southeastern USA pocosin peatlands.
  *Global Change Biology* 28, 6370–6384.
- Rosenbaum, P. & Rubin, D. (1983). The central role of the propensity score in
  observational studies for causal effects. *Biometrika* 70(1), 41–55.

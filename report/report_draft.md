# Wildfire risk in North Carolina peatlands and the effect of peatland restoration on burned area

**Draft scaffold.** Every `[[...]]` is a hole to fill; nothing else is meant to be
rewritten from scratch. Marker key:

| Marker | Means |
|---|---|
| `[[NUM: ...]]` | a number/statistic to paste in from the notebook |
| `[[FIG n: ...]]` | a figure slot — see `figure_table_manifest.md` for where it comes from |
| `[[TAB n: ...]]` | a table slot |
| `[[TODO: ...]]` | a sentence or paragraph only you can write (a judgment call) |
| `[[CHECK: ...]]` | a claim I drafted from the repo docs / deck that you should confirm before it ships |
| `[[CUT?]]` | candidate to delete if you run out of time |

Prose that is **not** in a marker is drafted boilerplate — read it, adjust the
voice, and keep it. It is written from `modeling_notebook_explained.md`,
`null_results_and_next_steps.md`, and `cat_kemen_meetings.pptx`, so it should
already be roughly true; `[[CHECK]]` flags the spots where I inferred.

---

## Executive summary

> Write this **last**, but keep the slot — Cat's email asks for exactly these
> four things and a reader who only reads one page should get all four here.

This report covers work on wildfire risk in North Carolina peatlands carried out
between June and September 2026, in support of TNC's pocosin restoration carbon
accounting. It does four things:

1. **What was done.** Eight satellite fire products (occurrence, burned area,
   and burn severity) were compared and validated against fire-perimeter
   reference datasets over North Carolina and, separately, over North Carolina
   peatlands. A matched, staggered difference-in-differences design was then
   built to test whether rewetting-based restoration at six TNC sites reduced
   burned area relative to ecologically similar unrestored peat.
2. **What was decided, and why.** [[TODO: one-sentence version of §3.2 — the
   recommended products and the reason each was chosen.]] The headline
   recommendation is **VIIRS active fire for occurrence, FireCCIS311 for spatial
   accuracy in burned area, and MODIS MCD64A1 where a long pre-treatment record
   matters**; no burn severity product was usable over NC peat.
3. **What predicts fire.** [[TODO: fill from §4.4 once the covariate models are
   run. Expect drought (PDSI) and drainage to lead; restoration status does
   not.]]
4. **What comes next.** The restoration effect on burned area is **not
   identifiable at six sites**. The design has [[NUM: 0.2–3%]] power against a
   treatment that prevents fire entirely, so the null describes the study, not
   the peatlands. The recommended path forward is a bounding analysis for the
   CPRG carbon estimate plus a shift of outcome from burned *extent* to burn
   *depth*; see §5.

---

## 1. Introduction

This project compared the results from several fire products in assessing the
occurrence, burn severity, and burned area of fires in North Carolina peatlands.
It then examined whether peatland restoration via rewetting reduced burned area
in six peatland sites restored by TNC compared to ecologically similar
non-rewetted areas.

However, the small number of restoration sites and their relatively recent
restoration years precluded conclusive claims about the impact of peatland
restoration on burned area, due to this limited spatial and temporal extent of
data on restored sites.

> The two paragraphs above are yours, unchanged. The three below are the added
> connective tissue — they set up why the null is a planned deliverable rather
> than a failure, so the Discussion does not arrive as a surprise.

Peatlands are a disproportionate share of North Carolina's terrestrial carbon
stock, and drained pocosin peat is vulnerable to deep, smoldering ground fire
that consumes carbon accumulated over centuries. Rewetting raises the water
table, and the mechanism by which it is expected to reduce emissions is by
limiting how deep a fire burns rather than whether or how widely one burns
[[CHECK: Richardson et al. 2022; this framing is the crux of §4.2 — see
`null_results_and_next_steps.md` §2]]. This distinction between burn *extent* and
burn *depth* recurs throughout the report and is the main scientific finding.

Restored pocosins still burn at the surface. Surface fire in undrained pocosin
wetlands is common and occurs naturally at fire return intervals on the order of
20–80 years, and most such fires are low severity with respect to peat
combustion [[CHECK: Richardson et al. 2022 §3.7 — quote is paraphrased from
`null_results_and_next_steps.md`; confirm against the PDF before quoting]].
Burned area, the quantity a 300 m satellite product can measure, is therefore
not the quantity restoration is expected to change.

[[FIG 1: North Carolina peat extent — the peatland mask overlaid on NC, with
completed TNC restoration sites marked. You already have this figure; drop it in
as-is.]]

*Figure 1. North Carolina peat extent. Peat is defined as gSSURGO major-histosol
percentage [[CHECK: > 0% for the extent map, ≥ 80% for the analysis frame — say
which this figure shows]], from Lilleskov et al. (2025). [[TODO: describe what
the restoration site markers show.]]*

[[TAB 1: Completed TNC restoration sites — site name, area (ha), restoration
completion year (`End_Yr`), and whether the site has both pre- and
post-restoration years under each fire product. This replaces the "put in a
figure of sites with their years" idea from your comment and is much less work
than a map. Source: `NC_Pocosin_Restoration_Sites_2026.shp` via
`load_treated_units`; the pre/post columns come from `design_summary`.]]

*Table 1. Completed peatland restoration sites used as treated units. Sites
without both a pre-restoration and a post-restoration year under a given fire
product cannot contribute to the difference-in-differences estimate and are
listed for completeness. [[NUM: under FireCCIS311 (2019–2024) only 3 of 6 sites
are usable; under MCD64A1 (2001–2024) all 6 are.]]*

---

## 2. Methods

### 2.1 Data sources

All datasets compiled for this project — including those not used in the
analyses reported here — are catalogued in `data_inventory.csv`, which records
each layer's source, native resolution, temporal coverage, units, and Box path.
The datasets that enter the analyses below are summarized in Table 2.

> Your comment 4 says "just point to data_inventory.csv entirely and get rid of
> this table." I'd keep a **short** version — a reader should not have to open a
> CSV to know what the analysis ran on — but cut it to the three groups below
> rather than every layer. That is ~20 rows instead of ~60.

[[TAB 2a: Fire products assessed. Columns: product, phenomenon (occurrence /
burned area / severity), native resolution, temporal coverage, update frequency,
citation. Rows: VIIRS active fire; MODIS MCD64A1; FireCCI51; FireCCIS311; GABAM;
MODIS MOSEV; MTBS; SE FireMap. This table already exists nearly verbatim on
slides 75/97/100 of the deck — copy it across.]]

[[TAB 2b: Reference ("ground truth") fire-perimeter datasets. Rows: NIFC_IFPH
(Interagency Wildland Fire Perimeter History, 1980–present); GEOMAC (Historic
Perimeters Combined, 2000–2018); TNC Coastal Plain burn history (2025); TNC
Sandhills burn history (2025). Note in the caption that TNC Sandhills has no
overlap with the ≥80% histosol peat mask and so is reported for NC-wide
validation only, and that fires smaller than 100 acres were excluded from
NIFC_IFPH and GEOMAC. Descriptions: pull from `data_inventory.csv`. Source:
slide 103.]]

[[TAB 2c: Covariate layers used in matching and modeling. Columns: variable,
source, native resolution, temporal extent, units, role (matching / outcome /
exact-match key). Rows: restoration status (`NC_Pocosin_Restoration_Sites_2026.shp`);
elevation (Copernicus GLO-30 DEM, 30 m); histosol % (gSSURGO, Lilleskov et al.
2025, 30 m); mean annual precipitation, mean daily Tmax, mean daily Tmin (GHCN
daily, IDW-interpolated, 1991–2020 normals); growing degree days; PDSI drought
index (per-year); HAND drainage; soil organic matter %; soil available water
capacity; soil drainage class (exact-match key); soil site index; soil water
table depth (**dropped** — see §2.4); LANDFIRE land cover (exact-match key).
Slide 59 has the first five rows already formatted; the soil/GDD layers come
from `get_climate&soil_data_updated.R`.]]

*Table 2. Datasets used in this report. See `data_inventory.csv` for the full
inventory, including layers compiled but not used.*

**Reference datasets are not a spatially exhaustive census.** A fire perimeter is
truth *within its own footprint* only: it records that a mapped fire happened
there, not that no fire happened elsewhere. Reference and product datasets are
therefore kept in separate registries and are not treated symmetrically — a
product detection outside a perimeter is not automatically a false positive.

### 2.2 Common grid and mask preprocessing

> Your comment 5 suggests renaming "Preprocessing." I'd call this section
> **"Common grid and mask preprocessing"** — it is specifically about getting
> every layer onto one CRS, resolution, and grid.

Every raster and vector layer was clipped to North Carolina and reprojected to
EPSG:5070 (NAD83 / CONUS Albers), an equal-area projection in metres, so that
areas are meaningful and all distances and buffers are in metres. Layers were
then warped onto a single product-independent common grid: **300 m for the
modeling analyses** and **500 m for the fire-product comparison**, the latter set
by the coarsest product compared (MODIS, 500 m) so that spatial resolution is not
a confound in the agreement metrics.

Warping rules depend on variable type. Continuous layers (e.g. elevation) are
resampled by area-weighted average, so a single 300 m cell takes the mean of the
30 m cells within its extent. Categorical layers (e.g. land cover) take the
majority class within the cell. Fire products are aggregated with a "max" rule —
any sub-cell burn lights the whole cell — and monthly products (MCD64A1,
FireCCI51, MOSEV) are OR'd into annual layers, with within-year timing preserved
upstream for a possible future seasonality analysis.

Peat extent was derived from the gSSURGO major-histosol percentage layer of
Lilleskov et al. (2025). Two masks were built: a **binary extent mask** where
histosol % > 0, with adjacent peat pixels merged into contiguous blobs, used for
the descriptive fire-product comparison; and an **analysis frame** of
high-confidence peat where histosol % ≥ 80, used for all matching and modeling.
The 80% threshold keeps non-peat noise out of the covariate distributions.
Comparing the two masks showed little difference in peat extent
[[CHECK: slide 73 — "not too much difference"; add the actual area numbers if
you have them]].

### 2.3 Fire-product comparison and validation against reference perimeters

> This section is your comment 12 — "add a methods section on what I did for
> characterizing fire occurrence, burned area, and severity across fire
> products." It is drafted here from Appendix A of `modeling_notebook_explained.md`
> and slides 77/107. Comments 8 and 14 (define the metrics, define precision)
> are folded in.

**Agreement between products.** Products were compared pairwise on the common
grid. Because burned-area products are binary maps, agreement is reported with
binary metrics — Jaccard index (intersection over union), Cohen's κ, and percent
agreement — rather than Pearson correlation, which on binary data is only the phi
coefficient. A separate temporal correlation matrix (Pearson and Spearman of
annual burned-area totals) captures year-to-year co-variation. Pairwise scatter
of per-cell burned area is fitted with total least squares rather than ordinary
least squares, because both products carry measurement error and OLS would bias
the slope by treating one axis as error-free; RMSE is reported against the 1:1
line. Annual burned area is reported both at native resolution and on the common
grid, and as a percentage of the AOI. Severity products, whose units are not
comparable across products (CBI vs dNBR vs MTBS classes), are compared with
Spearman rank correlation only.

**Validation against reference perimeters.** Reference fire perimeters were
rasterized to the same EPSG:5070 grid as the fire products, with
`all_touched=True` so that small reference fires survive the 500 m grid — this
matches the products' own "max" aggregation convention. Each fire event is scored
against the product layer **for its own year**; events are never pooled into a
timeless union. For each event:

- **tp** (true positive) = burned area inside the reference perimeter that the
  product detected;
- **fn** (false negative) = burned area inside the reference perimeter that the
  product did not detect;
- **Recall** = tp / (tp + fn), the fraction of a known fire's extent the product
  recovered. Recall is 1 − omission error, and is the headline metric.
- **Recall (pooled)** = Σtp / Σ(tp + fn) across all events. Cell-weighted, so
  dominated by large fires.
- **Recall (mean event)** = the unweighted mean of per-event recall. Counts every
  fire equally regardless of size, so it surfaces small-fire omission — the
  failure mode that matters most in a small-fire landscape like the NC pocosins.
- **Detection rate** = the fraction of reference fire events for which the
  product mapped *any* burned pixel inside the perimeter. This is the meaningful
  metric for occurrence products such as VIIRS, which detect a burning front
  rather than burned area.

**Precision is reported only conditionally.** Precision — the fraction of a
product's mapped burn that falls inside a reference perimeter — is computed per
event within a buffered window (default 5 km) and is explicitly flagged as
conditional, for two reasons. First, a product burn outside an incomplete
reference may be a real small fire the reference simply never recorded. Second, a
perimeter is an outer boundary that typically contains unburned islands, so it
overstates true burned area — which depresses precision and inflates apparent
omission. Recall is therefore the metric the recommendations rest on.

**Severity was not validated against perimeters**, because fire perimeters carry
no severity information and no field CBI data were available. Perimeters were
instead used to restrict the severity cross-comparison (SE FireMap vs MOSEV vs
MTBS) to known-burned cells.

Fires smaller than 100 acres were excluded from the NIFC_IFPH and GEOMAC
reference sets [[TODO: one clause saying why — small-perimeter geolocation
error at 500 m, or reference completeness? Whichever it was.]]. NIFC_IFPH and
GEOMAC overlap in time and record many of the same fires, so they are reported
separately as cross-checks rather than pooled, which would double-count.

### 2.4 Matched, staggered difference-in-differences design

The restoration analysis follows the design of Castro et al. (2026), who
evaluated canal-block rewetting and peat fire in Kalimantan, transplanted to the
North Carolina pocosins.

**Estimand.** The target is the average treatment effect on the treated (ATT):
for each restored pixel in each year, the difference in burn probability between
the world in which it was restored and the world in which it was not, averaged
over restored pixels. The counterfactual — how often restored pixels would have
burned had they not been restored — is supplied by matched, unrestored control
pixels.

- **Treatment** = pixels inside a *completed* restoration site, switched on from
  that site's restoration year (`End_Yr`) forward.
- **Control** = peat pixels (≥80% histosol) outside every restoration site and
  its spillover halo. The spillover buffer is set to 0 m, because TNC's
  restoration design prevents hydrological spillover into neighbouring
  properties [[CHECK: this is the stated reason in the decision log — confirm it
  is TNC's actual rationale before asserting it]]; the machinery for a non-zero
  halo exists if a future analysis wants one.
- **Response** = a 0/1 "did this pixel burn this year" flag sampled from a
  swappable satellite burned-area product.
- **Unit of analysis** = pixel-year on the shared 300 m EPSG:5070 grid, tagged
  with site ID and year.

**Why not one large regression.** Stacking every peat pixel and fitting
`burned ~ treated + covariates` is the right shape but is untrustworthy here for
three reasons, and the pipeline exists to address each. (i) Pixels are not
independent: pixels within one restoration site, or within one fire, share
unmeasured conditions, so treating millions of pixels as independent shrinks
standard errors toward zero and makes everything look significant — the effective
sample size is closer to sites × years. (ii) Restored peat is not located like
drained peat, so a plain GLM extrapolates across covariate space with no
comparable control and the treatment coefficient absorbs the imbalance. (iii)
Fire is rare, so ordinary logistic fits on very low event rates are biased and
over-confident.

**Matching.** Treated and candidate control pixels were matched 1:1 within each
year, without replacement, by minimum Mahalanobis distance in covariate space.
The covariate matrix is whitened — mean-centred and multiplied by the square root
of its inverse covariance matrix — so that Euclidean distance in the whitened
space is Mahalanobis distance in the original space, which removes the arbitrary
effect of covariate units and of correlation between covariates. Categorical
covariates (land cover, soil drainage class) are matched **exactly**: matching is
performed within category groups rather than treating a category code as a
continuous dimension. A caliper of 1 standard deviation caps the acceptable
distance; treated pixels with no control inside the caliper are dropped and
counted.

**Balance.** Match quality is assessed with the standardized mean difference
(SMD) per covariate — (mean treated − mean control) / pooled SD — before and
after matching, displayed as a love plot. Improvement in SMD is the acceptance
test for the whole design.

**Propensity and prognostic scores.** Rather than matching on raw covariates
alone, the design also uses two scalar summaries:

- the **propensity score**, Pr(treated in year *g* | covariates), estimated by
  logistic regression and fit only for years containing treated sites; and
- the **prognostic score**, a pixel's baseline fire risk absent treatment,
  Pr(burned | covariates), fit on control pixels only and then predicted for both
  treated and control pixels.

Matching on these two scores, together with a temporal fire lag (burn status in
the previous two years) and a spatial fire lag (summed burn status of the four
neighbouring pixels in the current year), collapses a high-dimensional matching
problem into a low-dimensional one and reproduces Castro et al.'s matching
variables.

**Estimator.** The canonical two-period ATT — (treated post − treated pre) −
(control post − control pre) — is biased under staggered treatment adoption
because it reuses already-treated units as controls. The analysis therefore uses
the Callaway & Sant'Anna (2021) staggered difference-in-differences estimator via
the Python `differences` package, which (i) uses only never-treated and
not-yet-treated pixels as controls, and (ii) is doubly robust: for each treatment
cohort *g* and calendar year *t* it fits both an outcome regression
m̂(X) = E[ΔY | X, control] and a propensity model ê(X) = Pr(cohort *g* | X),
estimating ATT(g,t) as the mean residual over treated pixels minus the
propensity-weighted mean residual over controls. Parallel trends then only has to
hold *conditional on* covariates X, and the counterfactual is estimated two
independent ways.

Only treatment cohorts satisfying `first outcome year < g ≤ last outcome year`
are retained, since a cohort with no pre-period or no post-period cannot
contribute. Under FireCCIS311 (2019–2024) this leaves the [[NUM: 2021 and 2023]]
cohorts; under MCD64A1 [[NUM: which cohorts?]].

**Standard errors** are clustered on restoration site, not on pixel, because
treatment is assigned at the site level and fire arrives as a small number of
large, landscape-scale events that paint thousands of contiguous pixels in a
single draw.

**Second estimation route.** In parallel, a matched logistic regression was fit
in levels: `burned ~ treated + covariates` on the covariate-matched pixel-years,
with cluster-robust standard errors on restoration site, reported as odds ratios
with 95% confidence intervals (continuous covariates standardized before
fitting). An odds ratio below 1 for `treated` would indicate that restoration
lowers the odds of fire. This route shares the entire matching front end with the
DiD route but does not use the propensity and prognostic scores. Agreement
between the two routes would be the strongest evidence of a design-robust effect.

**Covariate set.** Two covariate specifications were fit [[CHECK: this is your
comment 18 and Cat's email — confirm both were actually run before claiming
it]]:

- **Full set** — elevation, histosol %, precipitation normal, Tmax normal, Tmin
  normal, growing degree days, PDSI, HAND drainage, soil organic matter %, soil
  available water capacity, soil site index, with land cover and soil drainage
  class as exact-match keys.
- **Reduced set (Cat's request)** — PDSI and growing degree days as the only
  climate variables, with Tmax, Tmin, and precipitation removed, on the grounds
  that PDSI and GDD are the ecologically interpretable composites and the raw
  temperature and precipitation normals are largely redundant with them.

`soil_water_table_depth` was dropped from both specifications. Including it
alongside drainage, precipitation, and the other soil variables produced a
**positivity failure**: for the later treatment cohorts, control pixels received
near-zero estimated probability of treatment, so the inverse-probability weights
the doubly robust estimator depends on became unstable.

### 2.5 Design diagnostics

Because the number of treated clusters is small, the design was assessed
explicitly rather than assumed adequate (`src/peatfire/modeling/power.py`):

- **`design_summary`** collapses the pixel-year panel to the site-year level —
  the level at which treatment actually varies — and reports sites × pre/post
  years × years containing fire.
- **`minimum_detectable_effect`** simulates the observed design under a known
  true effect and reports power across a range of true reductions.
- **`randomization_inference`** holds the observed fire history fixed and
  permutes which sites were restored and when, producing a p-value that is valid
  at six clusters and has no NaN failure mode — unlike the cluster bootstrap,
  which degenerates when a site-year contains zero fire.
- **`did_site_year`** fits a transparent two-way fixed-effects DiD on the
  site-year panel with t(G−1) reference distribution, as a cross-check on the
  Callaway–Sant'Anna fit.

---

## 3. Results

### 3.1 Fire-product comparison

Prior to any analysis, a fire occurrence product and several burned area and
severity products were compared regarding their potential viability in
characterizing wildfire risk in North Carolina peatlands (see the Fire Product
Comparison and Recommendation Notes tabs in `fire_product_comparison.xlsx` for
details). Products were assessed on spatial resolution, length and currency of
temporal coverage, update frequency, and sensitivity to the small fires that
dominate the southeastern fire regime. From this comparison, the most suitable
datasets for assessing fire occurrence, burned area, and burn severity were
determined to be VIIRS Active Fire, GABAM, and SE FireMap, respectively.

[[FIG 2: Annual burned area over NC peat for the four burned-area products
compared (MCD64A1, FireCCI51, FireCCIS311, GABAM), on the common grid. Your own
note: "maybe do the burned area over nc peat for the four products that I
compared." This is the figure that makes the product comparison concrete —
worth the effort if you have the plotting code already.]] [[CUT? if time is
short, the recall figures below carry the argument on their own.]]

*Figure 2. Annual burned area within the North Carolina peatland mask, by
burned-area product, 2001–2024. [[TODO: one sentence on what the divergence
between products shows.]]*

However, upon inspecting the burn severity data overlaid over North Carolina
(Figure 3), it was determined that the burn severity products were too sparse
over North Carolina peatlands and none of these products were used in further
analyses.

[[FIG 3: Burn severity products overlaid over NC. Panels a, MTBS; b, SE FireMap;
c, MOSEV. Already in your draft.]]

*Figure 3. Burn severity products overlaid over North Carolina. **a**, MTBS.
**b**, SE FireMap. **c**, MOSEV. [[TODO: add a clause quantifying the sparsity —
e.g. "n peat pixels have any severity value across the record," or "only the
2008 and 2011 fires appear."]] Because no product provided usable severity over
peat, burn severity was dropped as an outcome.*

### 3.2 Validation against reference perimeters

The accuracy of the fire occurrence and burned area products was then obtained
through comparison with ground truth reference datasets over North Carolina
peatlands (Figure 4; see Methods §2.3).

[[FIG 4: Recall of each burned-area product against each reference dataset,
within NC peat. Panels: **a**, NIFC_IFPH; **b**, GEOMAC; **c**, TNC Coastal
Plain; **d**, TNC Sandhills. Your comment 13 specifies this panel order. Note
that TNC Sandhills has no overlap with the ≥80% histosol mask, so panel d is
NC-wide only — or drop panel d from the peat figure and keep it in an NC-wide
version.]]

*Figure 4. Burned-area product recall against fire-perimeter reference datasets
within North Carolina peatlands. **a**, NIFC_IFPH (fires ≥ 100 acres). **b**,
GEOMAC (fires ≥ 100 acres). **c**, TNC Coastal Plain. **d**, TNC Sandhills.
Bars show pooled recall (cell-weighted across all events, dominated by large
fires), mean per-event recall (each fire counted equally, sensitive to small-fire
omission), and detection rate (fraction of events for which the product mapped
any burned pixel). Higher is better on all three; a product with high pooled
recall but low mean-event recall is finding the big fires and missing the small
ones. [[TODO: one sentence naming which product wins on which metric.]]*

[[TAB 3: Recall table underlying Figure 4 — product × reference, with
recall_pooled, recall_mean_event, detection_rate, and n events. Your comment 13
asks for the recall tables; a table plus the figure is probably one item too
many, so pick whichever reads better and cut the other.]]

FireCCIS311 was determined to have the best spatial accuracy in North Carolina
peatlands, with the best combination of large-fire and small-fire detection
[[NUM: give the numbers — FireCCIS311 recall_pooled = ?, recall_mean_event = ?,
vs MCD64A1 = ?]]. It was therefore used initially for the modeling analyses.
Because this product only begins in 2019, however, it lacks pre-treatment data
for the [[NUM: 3 of 6]] restoration sites whose restoration years fall in or
after 2019, leaving only [[NUM: 3]] sites with both a pre- and a post-restoration
period. MODIS MCD64A1, which begins in November 2000, was therefore used for the
modeling results reported below, recovering all six sites and — importantly — the
2008 Evans Road and 2011 Pains Bay fires, the two large events that dominate the
recent NC peat fire record [[CHECK: that those two fires are in fact the dominant
events in the MCD64A1 peat record — worth confirming from the annual totals]].

**Recommended products.** [[TODO: this is Cat's ask #2 and should be stated
plainly and early. Draft:]] For fire **occurrence**, VIIRS active fire (375 m,
2012–present, near-real-time) is recommended: it is the only product with
sub-daily temporal resolution and the only one that can support a smoldering-
persistence analysis (§5.4). For **burned area**, the recommendation is
product-dependent on the question: FireCCIS311 (300 m, 2019–present) for spatial
accuracy where the analysis window falls inside its record, and MODIS MCD64A1
(500 m, 2000–present) wherever a long pre-treatment baseline is required — which
includes every before/after restoration comparison. For **burn severity**, no
product is recommended over NC peat; MTBS, SE FireMap, and MOSEV are all too
sparse there to support analysis, and severity should be treated as a data gap
rather than a variable.

### 3.3 Effect of restoration on burned area

**Matching and balance.**

[[FIG 5: Matching results. Suggested panels: **a**, treated and matched control
pixels in geographic space; **b**, treated vs control in covariate space (2D
projection); **c**, love plot of standardized mean difference per covariate,
before vs after matching. Slides 17, 65, 66 have versions of all three.]]

*Figure 5. Covariate matching of treated (restored) and control peat pixels.
**a**, Locations of treated and matched control pixels. **b**, Treated and
control pixels in covariate space [[TODO: name the two covariates shown]].
**c**, Standardized mean difference per covariate before and after matching;
values closer to zero indicate better balance, and |SMD| ≤ 0.2 is the
conventional threshold. [[NUM: how many treated pixels were dropped for having
no control within the caliper?]]*

Matching improved balance on [[NUM: n of m]] covariates, bringing all covariates
within |SMD| ≤ [[NUM]] [[TODO: or state honestly which covariates remained
imbalanced]].

**Difference-in-differences.**

[[FIG 6: DiD results. Panels: **a**, event study — ATT by event time, with
confidence bands and the treatment year marked; **b**, aggregated ATT. Slide 43
has a version, but note your own caveat there that the error bars were broken by
NaN bootstrap replicates — regenerate with `randomization_inference` before
using.]]

*Figure 6. Staggered difference-in-differences estimate of the effect of
peatland restoration on annual burn probability, MODIS MCD64A1, 2001–2024.
**a**, Event study: ATT by year relative to each site's restoration year.
Estimates at negative event times test the parallel-trends assumption and should
be indistinguishable from zero. **b**, Aggregated ATT. Standard errors are
clustered by restoration site. [[NUM: report the aggregated ATT and its interval,
and state the inference method actually used — cluster bootstrap or randomization
inference.]]*

The aggregated ATT was [[NUM: value, CI]], which is not distinguishable from
zero. Three diagnostics indicate that this null is uninformative about
restoration rather than evidence against it, and each is reported here rather
than in the discussion because each is a property of the estimate itself:

1. **Parallel trends fails before treatment.** The event study reports
   ATT(−1) = [[NUM: −0.011]] with a confidence band excluding zero. Because the
   pre-treatment trends already diverge, the post-treatment point estimate is not
   an ATT regardless of its p-value.
2. **Positivity fails when the full soil covariate set is used.**
   `soil_water_table_depth` had to be dropped (§2.4) because control pixels
   received near-zero treatment probability for the later cohorts. The covariate
   overlap the doubly robust estimator requires is marginal.
3. **The cluster bootstrap degenerates.** Some site-years contain zero fire in
   the treated or control group, producing NaN bootstrap replicates and unusable
   site-clustered standard errors. Randomization inference, which permutes
   restoration assignment rather than resampling clusters, is valid at this
   cluster count and is reported instead [[NUM: randomization p-value]].

**Matched logistic regression.**

[[FIG 7: Odds ratios with 95% CIs from the matched logistic model, covariates
sorted by effect size, with `treated` highlighted and a reference line at OR = 1.
Run this for both the full and the PDSI+GDD-only covariate sets — see below.]]

*Figure 7. Odds ratios and 95% confidence intervals from the matched logistic
regression of annual pixel burn status on restoration status and covariates,
with standard errors clustered by restoration site. Continuous covariates are
standardized, so each odds ratio is the multiplicative change in fire odds per
one standard deviation. OR = 1 indicates no effect; OR < 1 for `treated` would
indicate that restoration lowers fire odds. [[NUM: report the treated OR and
CI.]]*

### 3.4 Which factors best predict fire

> This is Cat's ask #3 and currently the thinnest part of the draft. It needs the
> logistic model coefficients — the DiD does not report per-covariate
> contributions to the propensity and prognostic scores, so the odds ratios from
> §3.3 are the source. Cat also specifically asked for the reduced PDSI + GDD
> specification, so report both.

[[TAB 4: Odds ratios from both covariate specifications side by side. Columns:
covariate | OR (full set), 95% CI | OR (PDSI + GDD only), 95% CI. A covariate
whose sign or magnitude flips between specifications is worth a sentence.]]

*Table 4. Odds ratios for annual pixel burn status under two covariate
specifications: the full covariate set, and the reduced set retaining PDSI and
growing degree days as the only climate variables. Standard errors clustered by
restoration site; continuous covariates standardized.*

[[FIG 8: Burn rate vs covariate value, per covariate, with pixels binned into 8
equal-count bins. You already have these for PDSI, Tmin, Tmax, and precipitation
(slides 21–34) for both FireCCIS311 at 300 m and MCD64A1 at 300 m and 500 m.
Pick one product and one resolution for the report and put the rest in an
appendix or leave them in the deck.]]

*Figure 8. Annual burn rate against each per-year covariate, MODIS MCD64A1 at
300 m. Pixels are divided into eight bins of equal count across each covariate's
value range. [[TODO: one sentence per panel, or one sentence for the figure —
which covariates show a monotone relationship with burn rate and which are
flat.]]*

[[TODO: the actual answer to "what predicts fire." Draft skeleton once you have
Table 4:

"Restoration status was not among the covariates predicting fire
(OR = [[NUM]], 95% CI [[NUM]]). The strongest predictors were [[NUM]] and
[[NUM]]. Drought (PDSI) [[TODO: direction — more negative PDSI = drier; expect
higher burn odds]] and [[TODO]]. Growing degree days [[TODO]]. Notably,
[[TODO: the interesting one — e.g. drainage, if it predicts fire in the
direction restoration is supposed to act, that is the single most useful result
in the report, because it is the mechanism showing up where the six-site design
could not see it]]."

That last point is worth chasing: if HAND drainage predicts burn probability
across the whole peat AOI, it is the same causal link the restoration analysis
was a badly underpowered proxy for, measured on ~2 million pixel-years instead of
18 site-years. See §5.3.]]

### 3.5 Design diagnostics: what the study could have detected

Simulating this exact design — six restoration sites, [[NUM: 3 or 6, depending on
product]] with both a pre- and a post-restoration year, over the product's
temporal window — gives [[NUM: 0.2–3%]] power against a treatment that prevents
fire completely.

[[TAB 5: Power table from `minimum_detectable_effect` / the scenario table in
`null_results_and_next_steps.md` §1a. Columns: scenario | P(site-year has fire) |
power vs 100% prevention | power vs 50% reduction. Rows: FireCCIS311 as run,
MCD64A1, Landsat-era. **Re-run this with the fire probability estimated from the
actual panel via `design_from_panel` before quoting any number — the values in
the memo are illustrative parameter choices, not fitted ones.**]]

*Table 5. Statistical power of the observed design against known true effects,
by fire product and assumed site-year fire probability. [[NUM: n simulations per
cell]].*

[[TAB 6: Sample-size curve from `sample_size_curve` — power to detect a 50%
reduction, by number of sites × number of years. Same caveat about re-running
with fitted parameters.]]

*Table 6. Power to detect a 50% reduction in burn probability, by design size.
[[NUM]] simulations per cell.*

The binding constraint is the **number of restoration sites, not the length of
the fire record**. An 80%-powered test of a 50% reduction requires on the order
of [[NUM: 4,000–5,000]] site-years; this study has [[NUM: 18–36]]. Extending the
fire record back to the Landsat era without adding sites does not close the gap:
at six sites, power against even complete fire prevention does not exceed
[[NUM: 0.22]] no matter how long the record.

---

## 4. Discussion

### 4.1 The null result is a statement about the design

[[TODO: 2–3 paragraphs. The argument is already made in
`null_results_and_next_steps.md` §1 — this section is the report-facing
compression of it. Points to hit, in order:

- The finding is "restored sites do not show significantly less burned area than
  matched controls," and before that is treated as a result about peatlands it
  has to survive three checks: power, identification, and outcome–mechanism
  match. It fails all three.
- The denominator is site-years, not pixel-years. The DiD reports standard errors
  around [[NUM: 0.0013]] because it counts ~[[NUM: 326,000]] pixel-years, but
  treatment is assigned at the site and NC peat fire arrives as a handful of
  large events that paint thousands of contiguous pixels in one draw. The
  independent replicate is a site-year, and there are at most 36 of them.
- Only [[NUM: 2]] cohorts survive the support restriction, and they are 1–3 years
  post-restoration. Rewetting takes time to raise water tables to the 15–30 cm
  range identified as the threshold, so a 1-year post window may be measuring the
  pre-effect period.
- TNC's own roadmap anticipated this: "It may be that treatment effects cannot be
  identified — if this is the case, document limitations." Documenting it
  rigorously is a planned deliverable.]]

### 4.2 Burned area is the wrong outcome for the rewetting mechanism

[[TODO: 2–3 paragraphs. This is the substantive scientific point and should be
the part of the discussion a reader remembers. Points to hit:

- Restored pocosins still burn at the surface; surface fire is natural and common
  in undrained pocosin at 20–80 year intervals, and most surface fires are low
  severity with respect to peat combustion.
- What restoration changes is how *deep* fire burns. Include the dose–response
  table below.
- Burn depth is exactly what a 300 m burned-area product cannot see. The null is
  therefore roughly the expected result, not an anomaly — and reporting it that
  way is a contribution, not a consolation.]]

[[TAB 7: Water table depth vs vertical peat loss in fire. Rows (all from
`null_results_and_next_steps.md` §2, sourced from the Richardson et al. 2022
discussion — **verify each against the primary sources before publishing**):
66 cm (drained) → 12.5–24 cm loss (Reardon et al. 2007, Green Swamp);
30–38 cm → little or none (Reardon et al. 2007);
~30 cm (PLNWR prescribed, 2015) → 1–2 cm (Flanagan et al. 2020);
8 cm (restored block, PLNWR) → none observed (Richardson et al. 2022);
drained, severe ground fire → ~40 cm average (Mickler et al. 2017).]]

*Table 7. Vertical peat loss in fire as a function of water table depth, compiled
from the pocosin literature. Restoration acts on this relationship, not on
whether or how widely a fire spreads.*

### 4.3 Implications for the CPRG carbon accounting

[[TODO: 1–2 paragraphs, and this is the part with direct client impact. The
argument:

- The CPRG grant's high estimate assumes restored sites experience no wildfire
  emissions, giving [[NUM: 22.6]] Mg CO2e/ac/yr of avoided emissions.
- These data say restored sites do burn. What they plausibly do not do is lose
  peat.
- The CPRG memo's own alternative — shallow burn in restored ([[NUM: 11.1]] Mg
  CO2e/ac) vs deep burn in drained ([[NUM: 112.1]] Mg CO2e/ac), giving
  [[NUM: 3.4]] Mg CO2e/ac/yr — is the assumption consistent with both the
  literature and this null. **The null supports the revised, lower estimate.**
- The memo's own decision rule — if fire return interval exceeds 25 years, treat
  wildfire emission reductions as a durability benefit rather than crediting them
  directly — combined with a 20–86 year (most 30–60) interval, nearly settles the
  question in favour of a buffer-pool treatment.

[[CHECK: every number in this section comes from the CPRG memo as summarized in
`null_results_and_next_steps.md`. Pull them from the memo itself before they go
in front of TNC.]]]]

### 4.4 Decisions made, and why

> Cat's ask #2. Most of the content is scattered through the Methods; this
> section is the one-place summary a reader can skim. Suggested as a table so it
> stays short.

[[TAB 8: Decision log. Columns: decision | alternative considered | why this one.
Rows to include, all drafted from Part V of `modeling_notebook_explained.md` and
Appendix A — trim to the 8–10 that a reader would actually question:

- Peat extent uses an 80% histosol threshold, not >0% | >0% threshold | keeps
  non-peat noise out of the covariate distributions; the analysis frame should be
  high-confidence peat even though the descriptive extent map uses >0%.
- Matched case-control design, not a whole-landscape GLM | plain pooled GLM |
  matching buys covariate overlap, so the restoration effect is not confounded by
  where restoration happens.
- Unit of analysis is pixel-year, clustered by site | site-year aggregate | keeps
  covariate variation; site-year remains the documented fallback and is what the
  power analysis uses.
- Match on the full climate + soil covariate set, not elevation alone | elevation
  + histosol % | on the 80% frame histosol % is nearly constant, so the original
  pair was effectively an elevation-only match balancing nothing.
- Climate enters as a long-run IDW-interpolated normal | kriging | IDW is
  transparent, dependency-light, and never extrapolates, which is all a
  match-only covariate needs; kriging can be swapped in later without changing
  the covariate contract.
- Analysis CRS is EPSG:5070 | the AOI's native CRS | equal-area, so area
  arithmetic and distances are meaningful.
- Product-independent common grid with "max" aggregation | each product's native
  grid | removes spatial resolution as a confound in agreement metrics.
- Total least squares for pairwise product scatter | OLS | both products carry
  error; OLS would bias the slope by treating one axis as error-free.
- Recall is the headline validation metric; precision is conditional | symmetric
  precision/recall | fire perimeters are an outer boundary containing unburned
  islands and are not a spatially exhaustive census.
- Callaway–Sant'Anna staggered DiD, not canonical two-period DiD | canonical
  DiD | canonical DiD reuses already-treated units as controls and is biased
  under staggered adoption.
- Standard errors clustered by site, not pixel | pixel clustering | treatment is
  assigned at the site; pixel clustering produces artificially precise estimates.
- `soil_water_table_depth` dropped | kept | positivity failure — control pixels
  received near-zero treatment probability for later cohorts.
- MCD64A1 used for the modeling despite FireCCIS311's better spatial accuracy |
  FireCCIS311 | FireCCIS311 begins in 2019 and leaves only 3 sites with a
  pre-period; the pre-treatment baseline dominates spatial accuracy for a
  before/after design.
- Severity dropped as an outcome | included | all three severity products are too
  sparse over NC peat.]]

### 4.5 Limitations

[[TODO: short and honest. Beyond the power and identification issues already
covered:

- Six restoration sites, of which [[NUM]] usable; restoration years cluster
  recently.
- Restoration site selection may be endogenous to fire history — if sites were
  restored partly *because* they had recently burned or were visibly degraded,
  that is an Ashenfelter dip and no amount of covariate adjustment repairs it.
  Worth checking the restoration records for post-fire rehabilitation projects;
  the 2008 fire lowered the peat surface at the PD site by 0.3 m, so restoration
  in this landscape genuinely does follow fire. [[CHECK: this, if true, is a
  real limitation and should be stated even if you cannot resolve it.]]
- Controls are matched landscape peat rather than a within-buffer donut, so
  conditional parallel trends leans harder on match quality than in Castro et
  al.'s design.
- No negative control was run. The planned placebo — running the pipeline on
  *planned but not-yet-restored* sites at their scheduled years, where any
  apparent effect flags confounding — was not completed.
- Belowground smoldering is unmeasured by every product used.]]

---

## 5. Next steps

> Cat's ask #4, and your own note says "check my meeting notes for the stuff I
> said I was gonna do next." I pulled them: the recurring next-steps items across
> the 6/4, 6/10, 6/24, 7/2, 7/9, 7/16, and 8/11 decks are listed in
> `REPORT_TODO.md` §D so you do not have to re-read the deck. The five below are
> the ones that survive the null result; the rest are superseded.

Ordered by value per unit effort.

**5.1 Report the null as a bounded claim** (~1 week; do this regardless).
Deliverable: a limitations memo carrying the power table, the minimum detectable
effect, the randomization p-value, and the design summary (sites × pre/post years
× years with fire). This converts "we found nothing" into "we can rule out
nothing, and here is what would have been required." All tooling exists in
`peatfire.modeling.power`.

**5.2 A bounding analysis for the CPRG carbon estimate** (~1 week, mostly
writing) — **the recommended path to closing the GHG deliverable.** A Monte Carlo
over the parameters that actually drive the wildfire term — fire return interval,
burn depth under drained vs restored conditions, peat vs aboveground carbon
contribution, emissions per fire event, peat stock, and restoration effectiveness
— produces a distribution for Mg CO2e/ac/yr and a variance decomposition showing
which parameter dominates the spread. A bounding analysis cannot return null; it
produces a number and an uncertainty range by construction, which is what makes
it the option with a guaranteed landing point. **The null is an input to this
analysis, not an obstacle to it**: restoration effectiveness enters as a
parameter with a wide prior, and §3.5's power result is what justifies the width.

The falsifiable expectation, stated in advance: **burn depth and fire return
interval will dominate the variance, and restoration effectiveness — the
parameter this project spent its effort trying to estimate — will contribute
least.** If that holds, the unestimable ATT was never the parameter that
mattered, and the cost of the null to TNC is much smaller than it appears.
Expected recommendation: treat avoided fire as a durability / buffer-pool term
sized from the distribution, rather than crediting [[NUM: 22.6]] Mg CO2e/ac/yr as
an annual flux.

**5.3 Switch the treatment variable from a six-site binary to a landscape-wide
hydrologic gradient** — the largest available power gain. The binding constraint
is six sites, but the mechanism — water table depth — varies continuously across
all [[NUM: ~83,000]] peat pixels, and the covariate layers are already built
(HAND drainage, soil water table depth, soil drainage class). Estimating
`burned ~ f(drainage) + covariates` across the whole peat AOI over the full MODIS
record uses [[NUM: ~2 million]] pixel-years and every large fire since 2001,
rather than 18 site-years. It is not a clean quasi-experiment — drainage is not
randomly assigned — but it targets the same causal link, and it can be pushed
with ditch-density gradients, a distance-to-canal regression discontinuity at
drainage-district boundaries, and matched drained/undrained pairs.

**5.4 Change the outcome to something that reflects burn depth** — the highest
scientific value, ranked by feasibility:

1. **VIIRS detection persistence.** Smoldering peat fires produce repeated
   active-fire detections at the same location over days to weeks; surface fires
   flash and stop. Detections per pixel per event, or fire duration, is a free
   proxy for smoldering, using data already downloaded (375 m, 2012–present).
   This is the cheapest real move.
2. **Severity within fire perimeters.** Conditioning on a fire having happened
   removes the occurrence-rarity problem entirely, because the sample becomes
   burned pixels rather than all pixels. [[CHECK: this is in tension with §3.1's
   finding that severity products are too sparse over peat. Reconcile — the
   conditional analysis may still be impossible, in which case say so here.]]
3. **InSAR / lidar subsidence.** Peat surface elevation loss after a fire is the
   direct measurement of burn depth. Repeat lidar exists for coastal NC via the
   NC Floodplain Mapping Program; pre/post-fire DEM differencing over the 2008
   and 2011 fires would produce actual burn-depth numbers for NC.

**5.5 A prospective monitoring design.** Use `sample_size_curve()` to state what
an 80%-powered test would require, then recommend the instrumentation now: water
table wells at restored *and* paired control sites, peat depth pins or
rod-surface-elevation tables to measure burn depth after the next fire, and
pre-registration of the analysis. Pair this with the negative control the
roadmap already calls for.

**What not to do.** Three things would look like progress and are not:
adding more covariates to rescue the ATT (the problem is 18 site-years and a
mismatched outcome, not omitted variables, and more covariates worsen the
positivity failure that already forced out `soil_water_table_depth`); loosening
the caliper or reverting to pixel-level clustering to recover significance (the
apparent precision of the pixel-clustered estimate is the artifact, not the
result); and reporting the ATT([[NUM: 2]]) = [[NUM: −0.096]] event-study point as
an effect, since at this cluster count one fire in one year drives it.

---

## References

> Every citation used in the drafted prose above. Verify and format to whatever
> style Cat wants; the modeling guide's reference list has full entries for the
> methods citations.

**Peatland fire and carbon**
- Richardson, C.J. et al. (2022). *Annual carbon sequestration and loss rates
  under altered hydrology and fire regimes in southeastern USA pocosin
  peatlands.* Global Change Biology 28:6370–6384.
- Reardon, J. et al. (2007). Peat consumption vs. water table position, Green
  Swamp, NC. [[TODO: full citation]]
- Flanagan, N.E. et al. (2020). Prescribed fire peat loss at Pocosin Lakes NWR.
  [[TODO: full citation]]
- Mickler, R.A. et al. (2017). Deep peat fire consumption, Pocosin Lakes.
  [[TODO: full citation]]
- Poulter, B. et al. (2006). Pocosin fire return intervals and emissions.
  [[TODO: full citation]]

**Design and methods**
- Callaway, B. & Sant'Anna, P.H.C. (2021). *Difference-in-differences with
  multiple time periods.* Journal of Econometrics 225(2):200–230.
- Castro, [[TODO: initials]] et al. (2026). [[TODO: title — the Kalimantan
  canal-block rewetting study this design transplants.]]
- Rosenbaum, P. & Rubin, D. (1983). *The central role of the propensity score in
  observational studies for causal effects.* Biometrika 70(1):41–55.
- Hansen, B.B. (2008). *The prognostic analogue of the propensity score.*
  Biometrika 95(2):481–488.

**Data**
- Lilleskov, E. et al. (2025). gSSURGO major-histosol percentage. [[TODO: full
  citation]]
- Humber, M.L. et al. (2019). *Spatial and temporal intercomparison of four
  global burned area products.* [[TODO: verify]]
- Vetrita, Y. et al. (2021). [[TODO: small-fire omission in tropical peatlands —
  verify]]

---

## Appendix A — Supplementary figures

[[CUT? Only if the main text is already long enough. Candidates: burn rate vs
covariate panels for the products/resolutions not shown in Figure 8
(FireCCIS311 at 300 m; MCD64A1 at 500 m — slides 20–34); per-site burn rate
plots (slides 35–40); propensity and prognostic score maps and overlap
diagnostics (slides 4–7, 52–54); the NC-wide (non-peat) validation results
(slides 108–109).]]

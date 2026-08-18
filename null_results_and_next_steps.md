# The null ATT: what it means, and where the project goes next

**Short version.** The finding is *"restored sites do not show significantly less
burned area than matched controls."* Before treating that as a result about
peatlands, it has to survive three checks, and it fails all three:

1. **Power.** Simulating this exact design — 6 restoration sites, 3 of them with
   both a pre- and a post-restoration year under FireCCIS311, 2019–2024 — gives
   **0.2–3% power against a treatment that prevents fire completely.** A design
   that would miss a 100%-effective treatment 97 times out of 100 cannot produce
   an informative null.
2. **Identification.** The event study reports ATT(−1) = −0.011 with the
   confidence band excluding zero. Parallel trends is violated *before* treatment
   begins, so the point estimate is not an ATT regardless of its p-value.
3. **Outcome–mechanism mismatch.** This is the important one. Burned area is not
   the variable rewetting is expected to change. Richardson et al. (2022) — the
   paper the CPRG carbon accounting is built on — says the opposite of what the
   analysis assumed. See §2.

So: this is not a finding about restoration. It is a finding about the design.
That is worth knowing, and the TNC roadmap already anticipated it — *"It may be
that treatment effects cannot be identified — if this is the case, document
limitations."* Documenting it rigorously is a **planned deliverable**, not a
consolation prize.

---

## 1. Why the null carries no information

### 1a. The denominator is site-years, not pixel-years

The DiD reports SEs around 0.0013 because it counts ~326,000 pixel-years. But
treatment is assigned to a **restoration site**, and NC peat fire arrives as a
handful of **large, landscape-scale events** — one fire paints thousands of
contiguous pixels burned in a single draw. The independent replicate is a
site-year, and there are at most 36 of them (6 sites × 6 years); under
FireCCIS311 only 3 sites have both a pre- and a post-period, so the usable count
is closer to 18.

`src/peatfire/modeling/power.py` (new, this branch) quantifies it. Simulating
the observed design under a *known* true effect:

| Scenario | P(site-year has fire) | Power vs. 100% prevention | Power vs. 50% reduction |
|---|---|---|---|
| FireCCIS311 as run (6 sites, 3 usable, 2019–2024) | 0.05 | **0.2%** | 0.3% |
| FireCCIS311 as run | 0.20 | **2.5%** | 0.7% |
| MCD64A1 (6 sites, 6 usable, 2001–2024) | 0.20 | 12.5% | 7.2% |
| MCD64A1 | 0.35 | 39.0% | 7.7% |
| Landsat-era (6 sites, 6 usable, 1986–2024) | 0.20 | 31.0% | 12.0% |
| Landsat-era | 0.35 | 57.8% | 17.2% |

(600 simulations per cell; site-level fire probability is a parameter — estimate
the real one from the panel with `design_from_panel` before quoting these.)

Read the top rows as: **the study as run would have detected complete fire
prevention roughly 1 time in 50.** Even switching to MODIS and going back to
2001 — which recovers the 2008 Evans Road and 2011 Pains Bay fires, the events
that actually matter — leaves power well under 50%.

And here is what *would* be required. Power to detect a **50% reduction**, by
design size (`sample_size_curve`, P(site-year fire) = 0.20, 400 sims per cell):

| sites \ years | 6 | 24 | 40 |
|---|---|---|---|
| **6** (this study) | 0.02 | 0.09 | 0.08 |
| 20 | 0.03 | 0.15 | 0.24 |
| 50 | 0.08 | 0.41 | 0.58 |
| 100 | 0.18 | 0.67 | **0.87** |
| 200 | 0.30 | **0.90** | 0.99 |

An 80%-powered test of a 50% reduction needs on the order of **4,000–5,000
site-years**. This study has 18–36. That is a factor of ~150, and no estimator
choice closes a gap of that size. Even the far easier target — detecting
*complete* fire prevention — needs ~50 sites × 24 years (power 0.98) or 100 sites
× 6 years (0.65); at 6 sites it never exceeds 0.22 no matter how long the record.

**The number of restoration sites, not the length of the fire record, is the
binding constraint.** That single conclusion should drive whatever comes next.

This is the single most defensible thing to report. It converts *"we found
nothing"* into *"we can rule out nothing; here is what would be required."*

### 1b. The estimator is failing its own diagnostics

Three separate red flags from the 8/11 deck and the notebook output, all pointing
the same way:

- **Parallel trends fails** — ATT(−1) = −0.011, band excludes zero. Worth
  investigating *why*: if sites were selected for restoration partly **because**
  they had recently burned or were visibly degraded, that is an Ashenfelter dip
  and naive DiD is not salvageable by adding covariates. Check the restoration
  records for post-fire rehabilitation projects. (The 2008 fire lowered the peat
  surface at the PD site by 0.3 m — restoration in this landscape genuinely does
  follow fire.)
- **Positivity fails** — `soil_water_table_depth` had to be dropped because
  control pixels get near-zero propensity for the later cohorts. The overlap
  needed for the doubly-robust estimator isn't there.
- **The bootstrap returns NaN** — some site-years contain zero fire in treated or
  control, so cluster-bootstrap replicates are degenerate.

The third has a clean fix: `randomization_inference()` in the new module. It
holds the fire history fixed and permutes *which sites were restored and when*,
which is the thing the null hypothesis is actually about. It is valid at 6
clusters, valid at 3, and has no NaN failure mode. Use it instead of the
bootstrap, and read `did_site_year()` (two-way FE on the site-year panel, t with
G−1 df) as the transparent cross-check on the Callaway–Sant'Anna fit.

### 1c. Only 2 cohorts survive, and they are 1–3 years post-restoration

`restrict_to_supported_cohorts` keeps 2021 and 2023 under FireCCIS311. The 2023
cohort has one post year. Peat rewetting takes time to raise water tables to the
15–30 cm that Richardson et al. identify as the threshold — a 1-year post window
may be measuring the pre-effect period.

---

## 2. The outcome variable is the wrong one — and the literature already said so

This is the finding that should reshape the project. From Richardson et al. 2022
(§3.7 and the discussion), all in the PDF already in hand:

> "C emissions from surface fires in undrained pocosin wetlands **are common and
> occur naturally** in shrub-scrub and pine woodlands at fire intervals of 20–80
> years … most surface fires are **low severity with respect to peat
> combustion**."

And the dose–response that matters, from the same discussion:

| Water table depth | Vertical peat loss in fire | Source |
|---|---|---|
| 66 cm (drained, dry) | **12.5–24 cm** | Reardon et al. 2007, Green Swamp |
| 30–38 cm (wet condition) | little or none | Reardon et al. 2007 |
| ~30 cm (PLNWR prescribed, 2015) | 1–2 cm | Flanagan et al. 2020 |
| **8 cm (restored block, PLNWR)** | **none observed** | Richardson et al. 2022 |
| drained, severe ground fire | ~40 cm average | Mickler et al. 2017 |

Restored pocosins **still burn at the surface**. What restoration changes is how
deep the fire goes — i.e. **burn depth and peat consumption**, which is exactly
the quantity a 300 m burned-area product cannot see. Slide 131 of the deck
already flags this ("Ideas for dealing with lack of products that measure
belowground smoldering?"); the null result is the empirical confirmation.

Two consequences:

- **The null is roughly the expected result**, not an anomaly. Reporting it that
  way — "restoration did not reduce burned *extent*, consistent with the
  literature; extent was never the mechanism" — is a substantive contribution.
- **The CPRG grant's high estimate rests on an assumption this analysis
  contradicts.** The grant assumed "restored sites would experience no wildfire
  emissions" (→ 22.6 Mg CO2e/ac/yr reduction). The data say restored sites *do*
  burn. What they plausibly don't do is lose peat. The CPRG memo's own
  alternative — shallow burn in restored (11.1 Mg CO2e/ac) vs. deep burn in
  drained (112.1 Mg CO2e/ac), → 3.4 Mg CO2e/ac/yr — is the assumption consistent
  with both the literature and this null. **The null result supports the revised,
  lower estimate.** That is a real answer to a real TNC decision.

---

## 3. Recommended pivot: the CPRG bounding analysis

**Decision (2026-08-18): close the project on option E below, paired with option A.
Do not open a new empirical analysis.**

### Why this one and not the others

The other four options are all *new analyses with uncertain outcomes*. VIIRS
detection persistence (C1) is the cheapest of them and might work — but if it also
returns null, three more weeks have bought nothing and the project is in exactly
the same place. The drainage gradient (B) has real power but is observational,
invites the confounding objections the matched design was built to avoid, and
would open more questions than it closes. Neither *wraps up*.

A bounding analysis cannot return null. It produces a number and an uncertainty
range by construction, which makes it the only option with a guaranteed landing
point — the property that matters when the goal is to finish.

It is also close to free on data: no new downloads, no new geospatial processing,
no pipeline changes. Every parameter is already in hand (table below).

**The key reframing: the null is an input to this analysis, not an obstacle to
it.** Restoration effectiveness enters as a parameter with a wide prior, and §1's
power result is what justifies the width. The analysis then asks whether that
width actually matters — which is a question the failed ATT can answer even though
it could not estimate the parameter itself.

### Parameters, and where each comes from

| Parameter | Range | Source |
|---|---|---|
| Fire return interval | 20–86 yr (most 30–60) | Poulter et al. 2006, via the CPRG memo |
| Burn depth, drained | 0.1–0.4 m | Reardon et al. 2007 (12.5–24 cm at WTD 66 cm); Mickler et al. 2017 (~40 cm, severe) |
| Burn depth, restored | 0–0.02 m | Richardson et al. 2022 (no loss at WTD 8 cm; 1–2 cm at WTD 30 cm) |
| Peat vs. aboveground C contribution | 30% at 0.01 m → 81% at 0.1 m | Poulter Table 5, via the CPRG memo |
| Emissions per fire event | 163 vs. 446 Mg CO2e/ac | the 2.7x discrepancy the CPRG memo already flags between Poulter's reported high-end and the Richardson supplement |
| Peat stock (bulk density 0.15 g/cm³, 52% C, 2.3 m depth) | 3,450 Mg peat/ha = 6,566 Mg CO2/ha | Richardson et al. 2022, PD site probes |
| **Restoration effectiveness** | **unconstrained by this study** | ← §1: the design had 0.2–3% power, so the data do not narrow this |

### Method

1. Monte Carlo over the seven parameters above → distribution of the **wildfire
   term** in Mg CO2e/ac/yr, and of the cumulative 2050 total.
2. Variance decomposition (Sobol, or one-at-a-time if that is enough) → which
   parameter dominates the spread.
3. Re-run with a shortened return-interval distribution → the phase-3 climate
   scenario, as a parameter change rather than a new model.

### The falsifiable expectation

**Burn depth and fire return interval will dominate the variance; restoration
effectiveness — the parameter this project spent its effort trying to estimate —
will contribute least.** Stated up front so it can be checked rather than
rationalized afterwards. If it holds, the conclusion is that the unestimable ATT
was never the parameter that mattered, and the cost of the null to TNC is much
smaller than it currently appears. If it does *not* hold, that is itself the
finding, and it argues for funding option F's monitoring design.

### Expected recommendation

The CPRG memo already contains the decision rule — *"If the fire return interval
exceeds 25 years, contributions from wildfire emission reductions should be
treated as a durability benefit rather than included directly in carbon
accounting"* — and Poulter's 20–86 yr (most 30–60) nearly settles it. The Monte
Carlo's job is to quantify how much of the interval distribution falls below 25
years under current and future climate, and to **size a buffer pool**, rather than
to credit 22.6 Mg CO2e/ac/yr as an annual flux.

### What it closes

- **Roadmap 3c** ("final model … and associated GHG reduction estimates") —
  currently blocked on the ATT; this unblocks it without one.
- **Roadmap 3b** ("document limitations") — §1 above plus `modeling/power.py`.
- **Roadmap phase 4** (future fire risk) — reduces to step 3 of the method.

Estimated effort: ~1 week, most of it writing. Deliverable: one memo with the
revised Mg CO2e/ac/yr distribution, a tornado plot of the variance decomposition,
and the buffer-pool recommendation.

### Status

Not yet built. Planned home: `src/peatfire/modeling/carbon.py` (Monte Carlo +
variance decomposition) and a memo alongside this one. The priors in the table
above are my reading of the sources and should be adjusted where the project
disagrees before anything is quoted externally.

---

## 4. Five ways forward, ranked by (value to TNC ÷ effort)

### A. Report the null as a bounded claim — 1 week, do this regardless

Deliverable: a limitations memo with the power table, the MDE, the randomization
p-value, and the design-summary table (sites × pre/post years × years with fire).
Everything needed is in `power.py`; §5 has the recipe. This closes roadmap
deliverable 3b ("document limitations") properly instead of leaving it as a
negative sentence.

### B. Switch the treatment variable from a 6-site binary to a landscape-wide hydrologic gradient — highest power gain

The binding constraint is 6 sites. But the *mechanism* — water table depth — varies
continuously across all ~83,000 peat pixels, and the covariate layers are already
built (`drainage` / HAND, `soil_water_table_depth`, `soil_drainage_class`).
Estimating `burned ~ f(drainage) + covariates` across the whole peat AOI and the
full MODIS record uses ~2 million pixel-years and every large fire since 2001,
rather than 18 site-years.

It is not a clean quasi-experiment — drainage is not randomly assigned — but it is
the same causal link the restoration analysis was a (badly underpowered) proxy
for, and it can be pushed hard: ditch-density gradients, distance-to-canal
regression discontinuity at drainage-district boundaries, matched drained/undrained
pairs. `fit_drainage_models` already exists for the levels version.

### C. Change the outcome to something that reflects burn depth — highest scientific value

Ranked by feasibility:

1. **VIIRS detection persistence.** Smoldering peat fires produce repeated active-fire
   detections over days to weeks at the same location; surface fires flash and
   stop. Detections-per-pixel-per-event, or fire duration, is a free proxy for
   smoldering that the project has already downloaded (375 m, 2012–present). This
   is the cheapest real move.
2. **dNBR / severity within fire perimeters.** The severity products (SE FireMap,
   MTBS, MOSEV) are already assessed in Appendix A. Conditional on burning, does
   severity differ by peat condition? This conditions on the fire happening, which
   removes the occurrence-rarity problem entirely — the sample becomes burned
   pixels, not all pixels.
3. **InSAR / lidar subsidence.** Peat surface elevation loss after a fire is the
   direct measurement of burn depth. Repeat lidar exists for coastal NC (NC
   Floodplain Mapping Program). Pre/post-fire DEM differencing over the 2008 and
   2011 fires would produce actual burn-depth numbers for NC — closing what the
   roadmap calls "likely a very large gap."

### D. Two-stage mechanism model — the way to get a defensible carbon number without a detectable ATT

Restoration → fire is unidentifiable here. But it factors:

```
restoration ──(link 1)──▶ water table depth ──(link 2)──▶ peat burn depth ──▶ CO2e
```

- **Link 1** is estimable with good power: WTD/soil moisture responds every year at
  every site (Richardson measured 60 cm drained vs. 35 cm restored at PLNWR
  directly; Sentinel-1/SMAP give a spatial version).
- **Link 2** is the Reardon/Flanagan/Mickler dose–response table in §2 — sparse,
  but real measurements, and improvable with C3.
- Compose them with Monte Carlo uncertainty propagation → a distribution for
  avoided emissions per restored acre.

This borrows strength from measurements that exist instead of demanding an
experiment that doesn't. It is also the honest structure for the phase-3
deliverable (fire risk under climate change), since climate scenarios enter at
link 1.

### E. Value-of-information / bounding analysis for the CPRG decision — cheapest thing with direct client impact

> **This is the recommended path — see §3 for the decision, the parameter table,
> and what it closes.** The rest of this entry is the original sketch.

TNC's actual decision, stated in the CPRG memo, is: *do avoided-fire emissions go
into carbon accounting, or into durability?* The memo's own rule — "if the fire
return interval exceeds 25 years, treat as a durability benefit" — plus Poulter's
20–86 year interval (most 30–60) nearly settles it.

Run a Monte Carlo over the four uncertain parameters (fire return interval
20–86 yr; burn depth 0.01–0.4 m; peat vs. aboveground contribution 30–81%;
restoration effectiveness, currently unconstrained) and report which one dominates
the variance in Mg CO2e/ac/yr. My expectation: **burn depth and return interval
dominate, and the restoration-effect parameter this project has been trying to
estimate contributes least.** If so, that is a defensible recommendation to
redirect effort — and it means the failed ATT costs TNC much less than it appears.

Recommended headline: **treat avoided fire as a durability/buffer-pool term, not a
credited annual flux**, and size the buffer from the distribution rather than
crediting 22.6 Mg CO2e/ac/yr.

### F. Prospective monitoring design — the deliverable that makes the next decade's answer possible

Use `sample_size_curve()` to state what would be needed for an 80%-powered test,
then recommend the instrumentation now: water-table wells at restored *and*
paired control sites, peat depth pins/rod-surface-elevation tables for burn-depth
measurement after the next fire, and pre-registration of the analysis. Pair it
with the negative control the roadmap already calls for (run the pipeline on
*planned but not-yet-restored* sites at their scheduled years — any "effect"
there is confounding).

---

## 5. What I would not do

- **Do not add more covariates to rescue the ATT.** The problem is 18 site-years
  and a mismatched outcome, not omitted variables. More covariates worsen the
  positivity problem that already forced `soil_water_table_depth` out.
- **Do not loosen the caliper or drop the site-level clustering to recover
  significance.** The pixel-clustered SEs are the reason the current estimate
  looks precise; that precision is an artifact.
- **Do not report the ATT(2) = −0.096 event-study point** as an effect. With this
  many clusters, one fire event in one year drives it.

---

## 6. Running the new code

```python
from peatfire.modeling import (
    site_year_panel, design_summary, did_site_year,
    randomization_inference, design_from_panel,
    minimum_detectable_effect, sample_size_curve,
)

# `panel` is the pixel-year frame from prepare_panel + attach_fire_response.
sy = site_year_panel(panel)              # collapse to the level treatment varies at

design_summary(sy)                       # the denominator: sites x pre/post x years with fire
did_site_year(sy)                        # transparent DiD, clustered by site, t(G-1)
randomization_inference(sy)              # p-value that survives 6 clusters and zero-fire years

spec = design_from_panel(sy)             # read the fire process off the real data
mde = minimum_detectable_effect(spec)    # -> {"mde": None} means: nothing was detectable
mde["curve"]                             # power at each true reduction, for the memo

sample_size_curve(spec, reduction=0.5)   # what a powered study would require
```

Start with `design_summary(sy)`. If `post_years_with_fire` in the `TOTAL` row is 0
or 1, that single number explains the null and everything else is confirmation.

## References

- Richardson, C.J. et al. (2022) *Annual carbon sequestration and loss rates under
  altered hydrology and fire regimes in southeastern USA pocosin peatlands.*
  Global Change Biology 28:6370–6384.
- Reardon, J. et al. (2007) — peat consumption vs. water table position, Green Swamp NC.
- Flanagan, N.E. et al. (2020) — prescribed fire peat loss at PLNWR.
- Mickler, R.A. et al. (2017) — deep peat fire consumption, Pocosin Lakes.
- Poulter, B. et al. (2006) — pocosin fire return intervals and emissions.
- Castro et al. (2026) — the Kalimantan design this pipeline transplants (11.3M
  pixel-years; see Part VI of `modeling_notebook_explained.md` for why the NC
  version cannot inherit its power).

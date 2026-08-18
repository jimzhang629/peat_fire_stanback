# Moving forward after a null burned-area result

## Executive recommendation

A non-significant difference in burned area is not evidence that rewetting has no
fire benefit. It means that this analysis did not resolve a burned-area effect at
its present spatial, temporal, and site-level sample size. In particular, the
current design has only about six restoration-site clusters, fire is rare, and a
binary burned-area product cannot measure the mechanism most likely to matter for
peat carbon: whether fire consumes the peat, and how deeply.

The defensible next step is therefore **not** to search through alternative model
specifications until one is significant. Report the null result, quantify the
effect sizes the study could and could not detect, and split the grant claim into
two independently supported components:

1. **Annual avoided soil emissions**, estimated from the water-table/soil
   respiration model; and
2. **Wildfire durability**, initially reported as a scenario or risk buffer, not
   booked as an expected annual credit unless the data identify both a change in
   fire probability and credible fire-loss parameters.

At the same time, redirect the fire analysis from extent alone to the more
plausible pathway

> rewetting -> wetter peat -> less peat combustion / shallower burn -> smaller
> carbon loss conditional on fire.

This preserves a useful and policy-relevant project even if restoration does not
change whether a satellite labels a pixel as burned.

## 1. What the null result does and does not say

“Not statistically significant” should be accompanied by the point estimate,
site-clustered confidence interval, number of independent restoration sites,
number of fires, and minimum detectable effect (MDE). Three interpretations then
become distinguishable:

* **Precise null:** the interval excludes reductions large enough to matter. This
  is evidence against a meaningful burned-area benefit.
* **Imprecise null:** the interval includes both a meaningful reduction and no
  effect (or harm). The study is inconclusive, usually because there are few site
  clusters or fire events.
* **Estimand mismatch:** burned extent is measured adequately, but restoration is
  expected to affect severity, peat burn depth, or suppression difficulty rather
  than ignition or perimeter. The burned-area result can be genuinely null while
  the carbon benefit is nonzero.

Avoid language such as “restoration had no effect” unless an equivalence test
rejects effects outside a pre-declared, policy-relevant margin. Prefer: “We did
not detect a change in burned area; the 95% interval was [x, y], which
does/does not exclude the effect assumed in the grant.”

## 2. Audit the burned-area result before changing the question

Run these checks as a fixed diagnostic sequence and show all results, including
unfavorable ones.

### Measurement and support

1. Tabulate actual pre- and post-restoration outcome years by cohort. Exclude
   cohorts without observations on both sides of treatment before matching.
2. Compare at least two fire products over their common years and repeat the
   headline estimate for each. Also compare products against incident perimeters
   or another independent reference source. Disagreement is a measurement result,
   not a reason to select the product with the smallest p-value.
3. Check whether small, low-severity, or under-canopy burns are detectable at the
   product resolution and whether mixed pixels near site boundaries dilute the
   treatment contrast. Repeat after an inward boundary buffer chosen before
   viewing treatment estimates.
4. List fires, not merely burned pixels. A single event can cover thousands of
   correlated pixels and must not be treated as thousands of independent events.

### Design and inference

5. Plot raw site-by-year outcomes and leave-one-site-out estimates. If one large
   fire or one restoration site determines the coefficient, say so.
6. Report site-clustered inference. With roughly six sites, conventional
   cluster-robust normal intervals remain fragile; supplement them with randomization
   inference based on the treatment-assignment design if defensible, or a
   wild-cluster bootstrap with small-cluster corrections. Neither method creates
   information that is absent, so emphasize the interval and sensitivity.
7. Show an event study with pre-treatment coefficients and site-specific treatment
   dates. A levels regression comparing restored and control pixels does not by
   itself establish a restoration effect.
8. Test sensitivity to matching choices, control-pool definitions, outcome window,
   and fire product as a small, pre-specified multiverse. Do not condition the
   reported result on significance.
9. Calculate an MDE using the site/event correlation structure. Compare it with
   the reduction needed to support the grant assumption. If the required effect
   is smaller than the MDE, the honest conclusion is “not identifiable with the
   available sites,” not “zero.”

The repository already supports the appropriate staggered difference-in-differences
design, pre/post support checks, event-study aggregation, matched controls, and
site-level clustering. Those should be the primary analysis; the clustered levels
models are useful descriptive sensitivity analyses.

## 3. Replace one binary question with a hierarchy of estimands

Pre-specify a short hierarchy so that the project is not a fishing exercise.

### Primary historical estimands

1. **Fire occurrence:** probability that a site (or sufficiently independent
   grid cell) intersects a fire in a year.
2. **Burned fraction:** fraction of the restored-site area burned per year. Use a
   fractional/two-part model or site-year aggregation rather than counting pixels
   as independent replicates.
3. **Severity conditional on fire:** high-severity fraction, continuous spectral
   severity, or field-observed organic-soil consumption among burned locations.
4. **Event scale:** patch size, reburn interval, and within-perimeter unburned
   refugia.

The first two measure the **extensive margin**; the latter two measure the
**intensive margin**. Conditioning severity on observed fire answers a useful
descriptive question but can introduce post-treatment selection if restoration
changes which fires occur. For a causal carbon estimand, combine occurrence and
loss in one unconditional expected-loss outcome or use a clearly labeled hurdle
model.

### Mechanism outcomes

Where data permit, add water-table depth or hydroperiod, soil moisture, drainage
distance/canal density, vegetation/fuel structure, and suppression/access proxies.
These should usually be mechanism or effect-modifier analyses, not controls added
indiscriminately after restoration. Post-treatment hydrology is part of the effect
rewetting is intended to cause; controlling for it would remove part of the total
treatment effect.

## 4. A carbon model that does not assume the result

Use an expected-loss identity instead of assuming that restored land never burns:

\[
E(L_s) = \lambda_s\,E(A_s \mid F)\,E(D_s\rho_s C_s\mid F)\,\frac{44}{12},
\]

where, for state \(s\) (drained or restored), \(\lambda\) is annual fire
probability, \(A\) is the fraction burned when a fire occurs, \(D\) is peat burn
depth, \(\rho\) is bulk density, and \(C\) is carbon fraction. Add aboveground
biomass, combustion completeness, CH4, and N2O terms explicitly when supported.
Do not multiply independent marginal means when these variables are correlated;
event-level Monte Carlo draws are preferable.

The wildfire benefit is

\[
\Delta E(L)=E(L_{drained})-E(L_{restored}).
\]

This decomposition is valuable even after a null extent result:

* set the occurrence/area ratio to the estimated value, including its uncertainty;
* estimate or bound the burn-depth/severity ratio separately; and
* propagate uncertainty in fire return interval, area, depth, density, carbon
  fraction, and combustion completeness to a distribution of avoided emissions.

Do not convert a one-time fire loss into an annual value solely by dividing by a
return interval without stating the stationarity and independence assumptions.
For climate projections, let the hazard vary through time and distinguish expected
emissions from a durability/reversal-risk metric.

## 5. Reframe the existing grant scenarios

The arithmetic in the proposed revised scenario is internally consistent as a
**conditional scenario**:

* restored peat contribution: \(2.51\times14.8\times0.30=11.1\) Mg CO2e/ac/fire;
* drained peat contribution: 112.1 Mg CO2e/ac/fire (using the stated deep-burn
  conversion and 81% peat share);
* difference: \((112.1-11.1)/30=3.37\), approximately 3.4 Mg CO2e/ac/yr; and
* combined with 7.8 Mg CO2e/ac/yr in avoided annual soil emissions: approximately
  11.2 Mg CO2e/ac/yr.

But this calculation should not yet be presented as an empirically estimated
restoration effect. It assumes that the 0.01 m and 0.1 m cases represent restored
and drained counterfactuals, respectively; that both states share the same
30-year return interval and burned area; and that carbon-pool percentages transfer
to the target sites. Each assumption needs a source, uncertainty distribution,
and sensitivity range. Also verify the units and carbon-to-CO2 conversion from the
original tables rather than applying the factor 14.8 as an undocumented constant.

Report at least these scenarios separately:

| Scenario | Annual soil benefit | Fire occurrence/area effect | Fire severity/depth effect | Accounting use |
|---|---:|---|---|---|
| Core | updated soil model | none credited | none credited | Quantified mitigation |
| Observed fire | updated soil model | fitted estimate + interval | none | Sensitivity until precise |
| Severity bounds | updated soil model | fitted estimate + interval | shallow/deep distributions | Probabilistic sensitivity |
| Grant upper case | grant soil estimate | no restored wildfire loss | maximum | Clearly labeled legacy upper bound |

This structure prevents an uncertain fire assumption from inflating the central
estimate while retaining wildfire risk reduction as a potentially important
benefit.

## 6. Highest-value new data

More pixels will not solve a six-site problem. Prioritize information that adds
independent treatment contrasts or observes the missing mechanism:

1. Add restored sites outside the current set, including restoration dates and
   comparable drained controls; extend the historical outcome series where product
   quality permits.
2. For every fire intersecting peat, assemble event perimeters, progression,
   suppression records, drought/weather, pre-fire water levels if available, and
   post-fire severity layers.
3. Establish a field protocol after future fires: paired restored/drained plots,
   residual peat surface or reference-pin burn-depth measurements, bulk density,
   carbon fraction, and vegetation consumption. Record zero consumption as well
   as deep burns.
4. Consider opportunistic comparisons within the same fire perimeter. Shared
   weather and ignition improve comparability, while pre-fire drainage, fuels,
   suppression, and position in the fire must still be addressed.
5. Treat remote-sensing severity as a proxy requiring calibration to organic-soil
   consumption. A spectral index alone should not be translated directly to peat
   carbon without local validation.

## 7. Revised roadmap and decision gates

### Near term: make the null result decision-useful

* Freeze the main design and produce the estimate, interval, MDE, event count,
  site count, product comparison, event study, and leave-one-site-out plot.
* Classify the result as precise null, imprecise null, or estimand mismatch.
* Deliver a short memo that explicitly retracts the “no wildfire emissions after
  restoration” assumption as a central estimate.

**Gate 1:** If the interval excludes the policy-relevant burned-area reduction,
stop treating avoided area as the main mechanism. If it includes that reduction,
describe the result as underpowered and seek more sites/years rather than claiming
either benefit or no benefit.

### Next: estimate loss conditional on fire and total expected loss

* Build an event-level database and compare severity metrics within fires.
* Calibrate burn depth and carbon density with field data or present explicit
  low/central/high distributions when calibration is unavailable.
* Fit the extensive and intensive margins, then propagate joint uncertainty by
  Monte Carlo simulation.

**Gate 2:** Include wildfire in quantified mitigation only if occurrence/area and
carbon-loss parameters are empirically defensible and the accounting rules permit
the implied return interval. Otherwise retain it as durability evidence and a
scenario analysis.

### Future climate risk

Project a time-varying hazard under multiple climate scenarios and report absolute
reversal risk for both drained and restored land. Avoid extrapolating the
historical restoration coefficient outside observed drought and water-table
conditions without a stated transportability assumption.

**Gate 3:** Before publishing 2050 totals, show results with and without wildfire,
with common and state-specific fire hazards, and across plausible burn-depth
distributions. The soil-only total should remain visible as the accounting core.

## Bottom line

The null burned-area estimate is a result, not a failed project. It directly
challenges the grant's strongest assumption—that restored sites incur no wildfire
emissions—and makes the revised accounting more credible. The most productive
scientific pivot is to determine whether rewetting changes **how peat burns**, not
only **whether a mapped pixel burns**, while treating wildfire benefits as uncertain
durability or scenario benefits until the relevant parameters can be identified.

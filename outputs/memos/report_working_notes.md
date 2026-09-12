# Project report — working notes

Scratch notes carried over from the draft of `project_report.docx` (the version
dated 2026-09-12, archived at `outputs/memos/archive/`). These were inline
to-dos, Word comments, and pasted email text sitting in the report body; they
are kept here so the report itself can stay clean.

---

## 1. The brief this report answers

From C. Chamberlain, by email:

> Could you add some more notes in your PPT or write up a quick Word document
> outlining: 1) what you've done to date; 2) detailing what decisions you made
> (e.g., which products you suggest and why); 3) explaining which factors best
> predict fire; and 4) your suggestions on next steps if you had more time?
>
> Also, one thing I was curious about, in your final model, do you have PDSI and
> GDD and not any other temperature or precipitation metrics? If so, that's
> perfect. If you have PDSI, GDD, Tmax, Tmin, Precip, or any combination of
> those, can you remove the other temperature and precipitation predictors and
> just keep PDSI and GDD? I'd be curious about those model outputs especially.
> If you don't have time to do that, could you provide your R code?

Where each is answered in the cleaned report:

| Ask | Section |
|---|---|
| 1. What's been done to date | §2 (what was done) and §3 (results) |
| 2. Decisions made and why | §4, with the log in Table 2 |
| 3. Which factors best predict fire | §5 |
| 4. Next steps | §6 |
| PDSI/GDD only? | §4.1 — yes, the exclusion set is in place |

The report is deliberately short (~11 pages including figures), since it is a
quick writeup for supervisors rather than a methods paper. Anything that needed
more depth was left in `methods.md` and pointed to from the report's header.

**Still outstanding on the PDSI/GDD question:** per-year GDD rasters had not been
built when the reported results were run, so `treated:gdd` is not yet estimated
(only `treated:pdsi` is). The code path exists. Re-run `modeling.ipynb` end to
end and the term appears. Worth doing before sending, since Cat asked about it
specifically.

---

## 2. Resolved in the cleaned report

Comments that were in the draft and have now been acted on:

- **Peatland mask + completed sites figures** — both in, as Figs. 1 and 2, plus
  a site table (Table 1) with completion years.
- **Data sources table** — replaced with a compact "datasets actually used"
  table (Table 2) that points to `data_inventory.csv` for the full inventory,
  rather than an empty skeleton table.
- **Reference fire datasets added to the data table** — NIFC IFPH, GeoMAC, TNC
  Coastal Plain are in Table 2.
- **"Preprocessing" section renamed** — now §2.3 "Common grid and
  preprocessing", which is what it actually describes.
- **Why these products were chosen** — §3.2 now gives the reasoning (small-fire
  vs large-fire recall trade-off) rather than asserting the conclusion.
- **Recall / precision / detection rate explained** — §2.4 defines tp, fn, fp,
  recall, precision, and all three roll-up statistics.
- **Recall tables in** — Fig. 4, panels labelled a = GeoMAC, b = NIFC IFPH,
  c = TNC Coastal Plain, with the ≥100 acre threshold stated.
- **Figure captions now explain how to read each figure** rather than naming it.
- **Modelling section names** — the wording that referred to "that section" is
  resolved; sections are numbered and cross-referenced.
- **Odds-ratio figure** — kept (Fig. 5), with the two numbers that matter
  (treatment and elevation) also stated in the text.

---

## 3. Still open — needs your call or your data

1. **Regenerate the raw burn-rate figure on the matched panel, then add it
   back.** It is currently left out of the report: the version that exists uses
   the full unmatched candidate pool (n = 515,684 pixel-years) as its control
   series, but the DiD identifies against matched controls, so the figure and
   the DiD results describe different comparisons. The notebook already produces
   the matched version (`plot_burn_rate_by_site(panel_m, ...)`). Worth adding
   back once regenerated — it shows the pre-treatment imbalance more clearly
   than any other figure. A placeholder note in §3.2 marks where it goes.

2. **Fix the x-axis label spacing** on that figure before sending.

3. **Run the power analysis and put it in §6.1.** `peatfire.modeling.power` has
   `design_summary`, `minimum_detectable_effect`, `simulate_power`, and
   `sample_size_curve`. This is the single highest-value addition: it turns the
   null into a quantified statement about what the design could have detected.

4. **Run `randomization_inference`** and report that p-value as the headline
   instead of the bootstrap interval — it is valid at six clusters.

5. **Jackknife by site and the negative control** (planned-but-unrestored sites
   at their scheduled years). Both are described in `methods.md` §9.

6. **Clustering question from the draft** ("maybe rerun this but cluster SEs by
   pixel?"): pixel-clustered SEs are the *deflated* ones — that is exactly the
   pseudo-replication `methods.md` §8 argues against, and it is what makes the
   unmatched-panel DiD look significant when it is not. Site clustering is the
   defensible choice and is what the report uses. If you want a cross-check,
   `att_collapsed()` gives the by-hand collapsed estimate with t(G−1)
   inference, which is more conservative, not less.

7. **Descriptive covariate views.** The draft note about plotting raw burn rate
   against PDSI and GDD by year is worth doing — `build_mask_frame` plus the
   burned-area-vs-covariate views cover the whole peat AOI, not just the matched
   sample, so they support a broader statement about what covaries with fire
   than the models can. Would slot into §5.

8. **Re-run on FireCCI S3.1.1** as a robustness check on the 2021 and 2023
   cohorts. One configuration switch; restart the kernel first, since the 300 m
   and 500 m intermediates must not mix.

9. **Site-splitting with Eric's hydrological data** — if each site can be split
   into 5–6 hydrologically distinct units, the cluster count rises from 6 toward
   30. Caveat noted in §6.2: sub-units of one site still share a water table and
   a fire, so verify independence rather than assuming the count.

---

## 4. Things deliberately dropped

- **Severity comparison against reference perimeters.** Perimeters carry no
  severity attribute and there is no field CBI, so there is nothing to score
  against. Stated in §2.4.
- **TNC Sandhills** as a reference source — does not overlap the peat frame.
- **Fire return interval as an outcome** — requires a pixel to burn twice inside
  the record, which would shrink the sample rather than grow it. Reasoning is
  kept in §6.2 so the question does not come back.

# Report TODO

Everything left to do, ordered so that you can start at the top and stop
anywhere. Section A is the critical path (the report is deliverable without
anything below it). Section B is what makes it good. Section C is optional.

Time estimates are for *you*, assuming the code already exists.

**Rule of thumb if you are stuck:** the report is already 60% written in
`report_draft.md`. What is missing is mostly (i) numbers you have to read off a
notebook, and (ii) figures you have already made once. Neither requires new
thinking. Do §A1 first — it is 20 minutes and it makes the rest feel finite.

---

## A. Critical path

### A1. Decide the report's spine (20 min, do this first)

- [ ] Confirm the section order in `report_draft.md` is what you want. It is:
      Exec summary → Intro → Methods → Results → Discussion → Next steps.
      Cat's four asks map to: Exec summary (all four), §4.4 Decisions (#2),
      §3.4 What predicts fire (#3), §5 Next steps (#4).
- [ ] Decide the figure budget. The draft has 8 main figures and 8 tables, which
      is too many for a short report. My cut list, in order:
      Fig 2 (burned area by product) → Tab 3 (recall table, redundant with Fig 4)
      → Fig 8 (burn rate vs covariates, or move to appendix) → Tab 5/6 (merge the
      two power tables into one) → Tab 2a/2b/2c (merge into one Table 2).
      A tight version is **5 figures, 4 tables**.
- [ ] Decide whether §4.3 (CPRG implications) is in scope for *this* report or
      belongs in a separate memo. If separate, §4.3 becomes two sentences and a
      pointer.

### A2. Run the two model specifications (Cat's explicit ask — highest priority)

Cat asked directly: *"in your final model, do you have PDSI and GDD and not any
other temperature or precipitation metrics? … if you have PDSI, GDD, Tmax, Tmin,
Precip, or any combination of those, can you remove the other temperature and
precipitation predictors and just keep PDSI and GDD?"* Your own comment 18 says
run both and report both.

- [ ] Run the matched logistic model on the **full** covariate set → odds ratios
      + 95% CIs, clustered by site.
- [ ] Run it on the **reduced** set: drop `tmax_normal`, `tmin_normal`,
      `precip_normal`; keep PDSI + GDD.
- [ ] Same two specs for the DiD, if it is cheap to do.
- [ ] Fill **Table 4** (side-by-side odds ratios) and **Figure 7**.
- [ ] Note anything that flips sign or magnitude between specs — that is a
      finding, and it is the direct answer to Cat's question.
- [ ] If you cannot get to this: Cat's fallback ask was *"could you provide your
      R code?"* — she means the modeling code. Point her at
      `src/peatfire/modeling/models.py` and `notebooks/modeling.ipynb`, and say
      it is Python, not R.

### A3. Write §3.4 "Which factors best predict fire" (1–2 hrs after A2)

- [ ] Write the paragraph. The skeleton with blanks is in the draft.
- [ ] Answer the specific question: **does drainage predict burn probability?**
      If it does, that is the most useful single result in the report — it is the
      restoration mechanism showing up where the six-site design could not see
      it, and it justifies next step 5.3.
- [ ] Say plainly that restoration status is *not* among the predictors, and
      cross-reference §3.5 so a reader knows why that is uninformative.

### A4. Fill in the numbers (2–3 hrs)

Grep the draft for `[[NUM` — every one is a number to read off a notebook.
The ones that carry real weight:

- [ ] Aggregated ATT + interval, and which inference method produced it
      (bootstrap or randomization). **Do not quote a bootstrap interval that
      returned NaN.**
- [ ] ATT(−1) and its band (the parallel-trends violation).
- [ ] Randomization-inference p-value from `randomization_inference()`.
- [ ] `treated` odds ratio + CI from the matched logistic.
- [ ] Number of treated pixels dropped for having no control inside the caliper.
- [ ] Recall numbers for FireCCIS311 vs MCD64A1 in peat (Fig 4 / §3.2).
- [ ] Sites usable per product (`design_summary`) → Table 1 and §3.5.
- [ ] Power table — **re-run with `design_from_panel(sy)` first.** The numbers in
      `null_results_and_next_steps.md` §1a use illustrative fire probabilities,
      not fitted ones. Do not paste them into the report as-is.

### A5. Place the figures you already have (2–3 hrs)

See `figure_table_manifest.md` for the source of each. In priority order:

- [ ] Fig 1 — peat extent map (you have it; it is already in the draft)
- [ ] Fig 3 — severity products over NC (you have it)
- [ ] Fig 4 — recall vs reference datasets, panels a–d (comment 13's panel
      order: a NIFC_IFPH, b GEOMAC, c TNC Coastal Plain, d TNC Sandhills)
- [ ] Fig 5 — matching + love plot (slides 17, 65, 66)
- [ ] Fig 6 — event study + aggregated ATT (**regenerate — slide 43's error bars
      are broken by the NaN bootstrap**)
- [ ] Fig 7 — odds ratio forest plot (needs A2)

### A6. Write the four `[[TODO]]` prose blocks (3–4 hrs)

These are the only places where you have to actually think. Each has a bulleted
skeleton in the draft; you are filling in sentences, not deciding what to say.

- [ ] §4.1 The null is a statement about the design
- [ ] §4.2 Burned area is the wrong outcome (**the scientific point — write this
      one properly, it is the part worth remembering**)
- [ ] §4.5 Limitations
- [ ] Executive summary (write last)

### A7. Resolve the Word comments (rolled up)

Every comment in `project_report.docx` is addressed somewhere in the draft.
Cross-reference so you can delete them as you go:

| # | Comment | Where it is handled |
|---|---|---|
| 0, 1 | peat mask figure + site completion years | Fig 1 + **Table 1** (table is the low-effort version you wanted) |
| 2 | describe the data collection notebook | §2.1 + `figure_table_manifest.md` — [ ] still needs a sentence naming `download_and_clip_data.ipynb` |
| 3 | add reference fire datasets to the data table | Table 2b |
| 4 | just point to `data_inventory.csv`, cut the table | §2.1 — kept a trimmed 3-part table; cut further if you disagree |
| 5 | rename "Preprocessing" | now §2.2 "Common grid and mask preprocessing" |
| 6 | "products chosen based on resolution, coverage, small fires…" | §3.1 opening paragraph, drafted |
| 7 | check why and add it in | §3.1 — [ ] verify the selection criteria against `fire_product_comparison.xlsx` |
| 8 | pooled/mean-event recall + detection rate → methods | §2.3, drafted with definitions |
| 9 | GEOMAC, TNC Coastal Plain, NIFC_IFPH; Sandhills has no peat | Table 2b caption + Fig 4 caption |
| 10 | figure caption should explain how to interpret | Fig 4 caption, drafted |
| 11 | say FireCCIS311 had best accuracy | §3.2, drafted (needs the numbers) |
| 12 | add a methods section for the product comparison | §2.3, drafted |
| 13 | recall tables, panel order a–d, note the <100 acre exclusion | Fig 4 + Tab 3 + §2.3 |
| 14 | explain recall and precision in methods | §2.3, drafted incl. why precision is conditional |
| 15 | best combo of small and large fire detection | §3.2, drafted |
| 16, 17 | reword to match final section names | done — sections are now named |
| 18 | run models on full covariates AND just GDD+PDSI, report both | **A2 above** |

- [ ] Also: comment 12 notes *"need to add the reference data to Box and to the
      data sources table."* The Box upload is a separate task — do not let it
      block the report.

---

## B. Makes it good

- [ ] **§4.4 decision log table (Table 8).** Drafted with ~14 rows in the draft;
      trim to the 8–10 a reader would actually question. This is Cat's ask #2 and
      it is nearly free — the content is already written.
- [ ] **Table 7** (water table depth → peat loss). Verify each row against the
      primary sources, not against `null_results_and_next_steps.md`. This table
      is the evidential backbone of §4.2 and will be scrutinized.
- [ ] **Check the Ashenfelter-dip question.** Were any of the six sites restored
      as post-fire rehabilitation? If yes, it explains the parallel-trends
      violation and belongs in §4.5. Check the TNC restoration records.
- [ ] **Fig 2** (annual burned area over NC peat, four products). Nice to have;
      cut first if time is short.
- [ ] **Fig 8** (burn rate vs covariate, 8 equal-count bins). You have these
      already for three product/resolution combinations — pick one for the main
      text, appendix the rest.
- [ ] References: fill the `[[TODO: full citation]]` entries.

---

## C. Optional / if time

- [ ] Appendix A supplementary figures (score maps, per-site burn rates, NC-wide
      validation) — these all exist in the deck already, so the cost is layout
      only.
- [ ] The negative-control placebo (run the pipeline on planned-but-unrestored
      sites at their scheduled years). Worth a sentence in §4.5 saying it was not
      run either way.
- [ ] Fire-product comparison figures for occurrence (VIIRS) — the report is
      currently burned-area-heavy.

---

## D. Next-steps items harvested from the meeting decks

Collected so you do not have to re-read the deck for §5. Struck-through items are
superseded by the null result.

**From 8/11 (most recent):**
- Other ways of producing a carbon reduction estimate besides burned area →
  **§5.2 and §5.4**
- How to increase the number of samples beyond 6 sites → **§5.3** (the answer is:
  you cannot add sites, so change the treatment variable to a gradient)
- ~~Add distance to road / light intensity / timber management extent as
  covariates~~ → superseded; more covariates worsen the positivity failure

**From 7/16:**
- Run on different fire products → **done** (§3.2, MCD64A1 vs FireCCIS311)
- Run on drainage with/instead of treated status → **§5.3**, still the best idea
  in the deck
- ~~Add land cover~~ → done
- ~~distance to road, light intensity, timber management~~ → superseded

**From 7/9:**
- ~~Add soil, land cover, temperature covariates~~ → done
- ~~Download land cover for years besides 2024~~ → check whether this matters for
  the exact-match key; probably not worth it now
- ~~Castro et al. DiD approach~~ → done
- ~~Try all burned-area products as the DV~~ → done for two

**From 7/2, 6/24, 6/10, 6/4:** all completed or superseded (covariate download,
clipping to peat extent, product correlation plots, fire data download). The one
live item is slide 131's *"ideas for dealing with lack of products that measure
belowground smoldering?"* — that question turned out to be the whole finding, and
its answer is **§5.4.1, VIIRS detection persistence.**

---

## E. Things to check before this goes to Cat

- [ ] Every `[[CHECK: ...]]` in the draft — these are claims I reconstructed from
      the repo docs and the deck rather than from primary sources.
- [ ] No number from `null_results_and_next_steps.md` is quoted without being
      re-derived from the actual panel (the power table especially).
- [ ] The CPRG numbers in §4.3 come from the CPRG memo itself, not from the
      summary in the repo.
- [ ] The Richardson/Reardon/Flanagan/Mickler rows in Table 7 come from the
      papers, not from the summary table.
- [ ] Say explicitly, once, that the modeling code is Python (not R) — Cat's
      email assumes R.

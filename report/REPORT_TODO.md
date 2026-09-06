# What's left on the report

`project_report.md` (and `project_report.docx`) is a complete draft. Every number
in it comes from the saved outputs in `notebooks/modeling.ipynb` — nothing is
invented. What remains is listed below.

`report_draft.md` is the earlier long scaffold. It is superseded, but it holds
extra material that did not fit the short version (a fuller decision log, the
next-steps items harvested from all seven meeting decks, an appendix figure list)
if you ever want to expand.

---

## 1. Drop in five figures

The report has five `[FIGURE n]` slots. Three of the files already exist:

- **Fig 1** — peat extent + restoration sites. Use the figure already in your
  Word draft.
- **Fig 2** — severity products over NC (a MTBS, b SE FireMap, c MOSEV). Also
  already in your draft.
- **Fig 3** — recall against reference perimeters.
  `outputs/figures/fire/validation_recall_nifc_ifph_<aoi>.png` and
  `outputs/figures/fire/nifc_ifph_burned_area_validation_summary.png`.
- **Fig 4** — covariate balance love plot.
  `outputs/figures/modeling/balance_love.png`.
- **Fig 5** — DiD event study.
  `outputs/figures/modeling/FireCCIS311_pixel_level_did_event_study_tidy.png`.

Optional sixth, if §3.3 needs more support: the per-site burn rate plot
(`FireCCIS311_burn_rate_per_site_event_time.png`) makes the "two sites, opposite
signs" point visually.

## 2. Two runs that would change the text

Both are configuration changes, not new code, and §5 promises them.

- [ ] **MCD64A1 over 2001–2024.** The report is currently entirely FireCCIS311,
      which has only three fire years. §3.4 and §4.4 both flag this. If you run
      it, update §3.3, §3.4, and drop the caveat in §4.4.
- [ ] **Reduced climate specification** — drop `precip_normal` and `tmax_normal`,
      keep PDSI and `gdd_normal`. This is the question raised in review; §3.4
      currently answers it by saying what the model contains and that the reduced
      run is outstanding. Given how wide the current climate intervals are, this
      will probably be better conditioned.

## 3. Verify before it goes out

- [ ] **Table 3** (water table depth → peat loss). Rows are compiled from the
      pocosin literature via `null_results_and_next_steps.md`; check each against
      the primary papers. This table carries §4.2, so it will be read closely.
- [ ] **§4.2's carbon paragraph.** It argues the result supports the revised lower
      CPRG estimate but deliberately quotes no figures. Take those from the CPRG
      memo directly if you want them in.
- [ ] **The Ashenfelter-dip question in §4.1.** Were any of the six sites restored
      as post-fire rehabilitation? If yes, it explains the parallel-trends
      violation. Check the TNC restoration records; either answer is worth a
      sentence.
- [ ] **References.** Seven entries are stubs — Castro, Flanagan, Lilleskov,
      Mickler, Poulter, Reardon, and the Castro title.

## 4. Worth knowing

- The report says the modelling is Python. The review question assumed R — worth
  saying so explicitly if you reply by email.
- **Power results are deliberately excluded.** The design's minimum detectable
  effect is in the notebook (cell 50: site SE 0.0148 vs pixel SE 0.0018, design
  effect 8.2, MDE 0.209 in P(burn) against a control base rate of 0.0125 — i.e.
  ~17× the baseline burn probability). §3.3 uses only the design-effect ratio, to
  explain why the printed confidence band is the wrong one. If you ever want the
  strongest single statement of why the null is uninformative, that MDE is it.
- The report does **not** claim a null in the sense of a non-significant estimate.
  The pooled ATT is negative with a band excluding zero. The argument is that it
  is not *identified* — parallel trends fails pre-treatment, two sites cancel, and
  the two estimation routes disagree in sign. Keep that distinction if you edit
  §3.3; it is the honest version.

## 5. Rebuilding the Word file

```bash
python report/build_docx.py                  # -> project_report.docx
python report/build_docx.py report_draft.md  # -> report_draft.docx
```

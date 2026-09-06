# Figure and table manifest

Where every figure and table in `report_draft.md` comes from, so you never have
to go hunting. "Deck slide N" refers to `cat_kemen_meetings.pptx` — several
figures already exist there and only need exporting.

Status key: **have** = exists, just place it · **regen** = exists but must be
re-run · **new** = does not exist yet.

## Figures

| # | Content | Status | Source |
|---|---|---|---|
| 1 | NC peat extent + restoration sites | have | already in your docx draft; deck slides 73, 80, 82 |
| 2 | Annual burned area over NC peat, 4 products | new | `fire_products_comparison/plotting.py`; common-grid annual totals. **First cut candidate.** |
| 3 | Severity products over NC (MTBS / SE FireMap / MOSEV) | have | already in your docx draft; deck slides 101, 123 |
| 4 | Recall vs reference perimeters, panels a–d | have | deck slides 78, 111; `validation.summarize_validation`. Panel order per comment 13: a NIFC_IFPH, b GEOMAC, c TNC Coastal Plain, d TNC Sandhills |
| 5 | Matching: geography · covariate space · love plot | have | deck slides 17, 65, 66; `modeling/plotting.py` (`plot_balance`) |
| 6 | Event study + aggregated ATT | **regen** | deck slide 43 — **error bars are broken there** (NaN bootstrap replicates). Re-run with `randomization_inference()` |
| 7 | Odds ratios, forest plot, both covariate specs | new | `modeling/models.py` odds-ratio table → forest plot. Needs TODO A2 |
| 8 | Burn rate vs covariate, 8 equal-count bins | have | deck slides 20–34 (FireCCIS311 300 m; MCD64A1 300 m and 500 m). Pick one combination for the main text |

## Tables

| # | Content | Status | Source |
|---|---|---|---|
| 1 | Completed restoration sites + years + pre/post availability | new | `load_treated_units` + `design_summary`; deck slide 82 has the site list |
| 2a | Fire products assessed | have | deck slides 75, 97, 100, 120, 122 — copy across |
| 2b | Reference perimeter datasets | have | deck slide 103; descriptions in `data_inventory.csv` |
| 2c | Covariate layers | part | deck slide 59 has the first five rows; soil/GDD layers from `get_climate&soil_data_updated.R` and `modeling/covariates.py` |
| 3 | Recall table underlying Fig 4 | have | `validation.summarize_validation` output. Redundant with Fig 4 — pick one |
| 4 | Odds ratios, full vs PDSI+GDD specs | new | TODO A2 |
| 5 | Power by scenario | **regen** | `power.minimum_detectable_effect`. **Re-run with `design_from_panel(sy)`** — the table in `null_results_and_next_steps.md` §1a uses illustrative fire probabilities |
| 6 | Sample-size curve (sites × years) | **regen** | `power.sample_size_curve`. Same caveat |
| 7 | Water table depth → vertical peat loss | new | `null_results_and_next_steps.md` §2 has the rows; **verify each against the primary papers** |
| 8 | Decision log | have (as prose) | Part V + Appendix A of `modeling_notebook_explained.md`; rows drafted in the report scaffold |

## Notebooks that produce each stage

- `notebooks/download_and_clip_data.ipynb` — data acquisition, clipping, peat
  mask construction (`PEAT_THRESHOLD = 80`). This is the notebook comment 2 asks
  you to describe in §2.1.
- `notebooks/run_fire_comparison.ipynb` — product intercomparison (Figs 2, 3).
- `notebooks/validate_against_reference.ipynb` — reference validation (Fig 4,
  Tab 3).
- `notebooks/peat_restoration_exploration.ipynb` — site exploration (Tab 1).
- `notebooks/modeling.ipynb` — matching, DiD, logistic (Figs 5, 6, 7, 8;
  Tabs 4, 5, 6).
- `notebooks/modeling_walkthrough.ipynb` — narrated version of the above.

## Output directories

Figures currently save under `outputs/figures/fire/` and
`outputs/figures/modeling/` (both empty in this checkout — the contents are
git-ignored). Save report-bound figures somewhere stable and note the filename
in the draft next to each `[[FIG n]]` marker as you place them.

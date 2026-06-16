log each methodological choice

## Fire-product comparison toolkit (src/peatfire/fire_products.py, fire_comparison.py, plotting.py)

- **Analysis CRS = EPSG:5070 (NAD83 / CONUS Albers Equal Area)**, fixed for all
  area math and grid comparisons rather than the input AOI's CRS. Area is only
  meaningful in an equal-area projection, and a fixed CRS keeps results
  comparable across AOIs (state vs peatland vs non-peatland). The AOI's own CRS
  is used only as a clip mask. Mirrors Humber et al. (2019), who had to
  cosine-weight their geographic products to recover equal-area totals.

- **Common grid with "max" aggregation** ("any sub-cell burn lights the cell")
  for all spatial comparisons, default 500 m (the coarsest burned-area product).
  This removes spatial resolution as a confound in the agreement matrices, as
  in Humber et al. (who aggregated to a shared ~6 km grid). The earlier sandbox
  matched GABAM onto MODIS's native grid, which privileged one product; the
  toolkit matches every product onto a single product-independent grid instead.

- **Resolution confound (supervisor's question): partially valid.** A coarse
  burned pixel attributes its whole large area as burned (mixed-pixel commission
  -> inflation per detection), but coarse products also omit small fires
  entirely. In small-fire landscapes (NC pocosins) omission usually dominates,
  so coarse products often report *less* area, not more (cf. Vetrita et al.
  2021). Handled by reporting the time series both at native resolution and on
  the common grid, and by doing all agreement on the common grid.

- **Annual burned area reported two ways**: native pixel area (reproduces Humber
  Figure 3) and common-grid (resolution-controlled). Also reported as % of AOI
  area for cross-AOI comparability.

- **Agreement metrics**: burned-area maps use binary metrics (Jaccard/IoU,
  Cohen's kappa, % agreement) -- the right choice for binary maps, vs Pearson
  which on binary data is only the phi coefficient. A separate temporal
  correlation matrix (Pearson/Spearman of annual totals) captures year-to-year
  co-variation. Severity products use Spearman correlation of continuous values
  on the common grid (their units -- CBI vs dNBR vs MTBS class -- are not
  directly comparable). VIIRS (occurrence) is included in both matrices: as
  per-cell presence for binary agreement, as per-cell detection count for
  correlation.

- **Monthly products** (MCD64A1, FireCCI51, MOSEV) are OR'd / max-aggregated
  into an annual layer. Within-year timing is preserved upstream and can feed a
  future seasonality plot (Humber Figures 5-6).

- **Plot style** (plotting.set_fire_style): products encoded by colour only
  (colour-blind-safe Okabe-Ito cycle), top/right spines removed, frameless
  legend.

- **Total least squares + RMSE for pairwise scatter comparison** (Humber et al.
  2019). Two-product comparison uses TLS (orthogonal regression, minimises
  perpendicular distance) rather than OLS, because both products carry error
  (errors-in-variables); OLS would bias the slope by assuming x is error-free.
  RMSE is reported against the y=x line. `agreement_matrix` gains
  `method="tls_slope"` (asymmetric: [a,b] uses a as x, b as y) and
  `method="rmse"`; `product_pair_scatter` + `plot_product_scatter` draw the
  per-cell scatter with the TLS and y=x lines. Each scatter point is a
  common-grid cell -- our per-spatial-unit analogue of Humber's per-TSA points.

- **Equal-area CRS vs TSA polygons are different roles.** Reprojecting to
  EPSG:5070 is the analogue of Humber's *latitude/area correction* (their
  geographic products are cosine-weighted; equal-area products are not), NOT of
  their TSA polygons. The TSA polygons are spatial *aggregation units*; our
  common-grid cells play that role.

- **Active fire (VIIRS) in the temporal comparison.** `period_totals_series`
  reports burned-area km^2 for BA products and detection *count* for occurrence
  products, so VIIRS joins the per-period (temporal) correlation/TLS against the
  BA products -- the quantitative analogue of Humber's temporal heat maps.
  Pearson (scale-invariant) includes VIIRS; TLS slope / RMSE on totals stay on
  the BA products (shared km^2 units).

## Ground-truth validation toolkit (reference_sources.py, validation.py)

- **References are a *separate* registry from the products under test**
  (`REFERENCE_SOURCES` vs `FIRE_PRODUCTS`). A reference (NIFC perimeters, TNC
  preserve burns, NCWRC Rx) is *truth within its own footprint* but **not a
  spatially-exhaustive census** of all landscape fire, so it cannot play the
  symmetric role a product does. Keeping it apart encodes that asymmetry.

- **Recall (1 - omission) is the headline metric; precision is conditional.**
  Of a reference perimeter's burned cells, recall = the fraction a product also
  maps -- clean, because we are confident a fire occurred inside the perimeter.
  Precision (product burns falling inside the perimeter) is reported **only
  per-event inside a buffered window** (default 5 km), and flagged conditional,
  because (1) a product burn outside an *incomplete* reference may be a real
  small fire, not a false alarm, so global precision is meaningless; and (2) a
  perimeter is an *outer boundary* with unburned islands, so it overestimates
  truth -- an upper bound that depresses precision and inflates omission.

- **Time-matched, per-event -- never a timeless union.** Each incident is scored
  against the product layer for its own year (annual matching by default, so a
  product is not penalised for sub-monthly timing), via `validate_event`. Events
  are kept distinct (not dissolved across years), so a detection in one year
  cannot "catch" a fire in another. `summarize_validation` then offers both a
  cell-weighted `recall_pooled` (dominated by big fires -- "what fraction of
  burned *area* was seen?") and `recall_mean_event` (each fire once -- surfaces
  small-fire omission).

- **Occurrence products judged at the event level.** Active fire (VIIRS) detects
  a burning front, not burned area, so per-cell recall is the wrong question;
  `detected` (any overlap with the perimeter) is the meaningful one. VIIRS is
  2012+, so it is simply absent (skipped) for the older incidents -- not 0.

- **Perimeters rasterised with `all_touched=True`** (`rasterize_polygons_to_grid`),
  matching the products' "any sub-cell burn lights the cell" (`how="max"`)
  convention, so small reference fires survive the 500 m grid and the reference
  stays the generous (upper-bound) truth boundary.

- **Severity is *not* validated against perimeters** (they carry no severity and
  we have no field CBI). Perimeters instead *restrict* the severity
  cross-comparison (SE FireMap vs MOSEV vs MTBS) to cells where a fire is known
  to have occurred, removing unburned-background noise. MTBS itself straddles
  product and reference (QA'd Landsat severity for >=500-ac eastern fires).

- **Two NIFC sources, not pooled.** `NIFC_IFPH` (Interagency Fire Perimeter
  History, long record, authoritative) is the default; `GEOMAC` (Historic
  Perimeters 2000-2018) overlaps it and is a *cross-check*, not a pool partner --
  the same fire is in both, so pooling double-counts. Different attribute schemas
  (upper- vs lower-case) are handled by per-role candidate-column lists per spec.

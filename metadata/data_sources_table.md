# Data Sources Used in Project

Fills out the "Data Sources Used in Project" table in the project report. Rows are
grouped by the role each dataset plays: (1) fire products that were compared,
(2) ground-truth references used to score them, (3) the treatment and sample frame,
(4) model covariates, and (5) boundaries/context layers. `Box path` is left blank
throughout — fill in per your Box layout.

Everything is clipped to NC, reprojected to **EPSG:5070**, and (for the modeling
stage) aggregated to a shared **300 m** analysis grid over the >80% histosol AOI.
"Native resolution" below is the resolution *as delivered*, before that regridding.

Compiled-but-unused datasets (UFMP undersized fires, SEUS TIMO forest-management
extent, NASA HAND drainage, NCWRC prescribed-fire perimeters) are listed at the
bottom rather than in the table; see `data_inventory.csv` for their full details.

---

## Fire products (occurrence, burned area, burn severity)

| Dataset | Role in analysis | Native resolution | Temporal coverage | Source/citation | Box path |
|---|---|---|---|---|---|
| **MCD64A1_061** | MODIS Terra+Aqua Burned Area Monthly 500 m (BurnDate band; burned = BurnDate > 0). Burned-area outcome for the **final DiD and logistic regression models** — chosen over FireCCIS311 because its long record supplies pre-treatment years for every restoration cohort. Also one of four burned-area products in the product comparison. | 500 m (sinusoidal, SR-ORG:6974) | Nov 2000 – present, monthly. Panel configured as 2001–2026 in `modeling.ipynb`. | NASA LP DAAC (https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/). Giglio, L., Justice, C., Boschetti, L., Roy, D. (2021). *MODIS/Terra+Aqua Burned Area Monthly L3 Global 500m SIN Grid V061*. NASA EOSDIS LP DAAC. https://doi.org/10.5067/MODIS/MCD64A1.061 | |
| **FireCCIS311** | ESA Fire_cci Sentinel-3 SYN Burned Area Pixel Product v1.1 (JD burn-date layer; CL confidence and LC land-cover layers also downloaded). **Best spatial accuracy over NC peatlands** in the reference validation, and the product used for the initial modeling run; record starts 2019, so it has no pre-treatment years for the 2019 restoration cohort. | 300 m | 2019-01-01 – 2024-12-31, monthly | ESA Climate Change Initiative, via CEDA Archive: https://data.ceda.ac.uk/neodc/esacci/fire/data/burned_area/Sentinel3_SYN/pixel/v1.1/uncompressed (CEDA account + access token required) | |
| **FireCCI51** | MODIS Fire_cci Burned Area Pixel Product v5.1 (band 1 = BurnDate). Burned-area **product comparison only**; not used in the models. | 250 m | 2001-01 – 2020-12, monthly | Google Earth Engine `ee.ImageCollection('ESA/CCI/FireCCI/5_1')`. Lizundia-Loiola, J., Otón, G., Ramo, R., Chuvieco, E. (2020). *Remote Sensing of Environment* 236:111493. | |
| **GABAM** | Global Annual Burned Area Map (Landsat). Finest-resolution burned-area product in the comparison; recommended for burned area on the basis of the inter-product comparison, but superseded by the reference-validation results. Comparison only. | 30 m | 1985–2021, annual | Google Earth Engine `ee.ImageCollection("projects/sat-io/open-datasets/GABAM")`. Long, T. et al. (2019). *Remote Sensing* 11(5):489. | |
| **VIIRS active fire** (S-NPP archive; NOAA-20 and NOAA-21 NRT also downloaded) | VIIRS 375 m Active Fire / Thermal Anomalies points. **Fire-occurrence product** recommended by the comparison; used as the occurrence layer scored against the reference perimeters. | 375 m (point detections) | S-NPP archive 2012-01-20 – present (downloaded through 2026-06-03). NOAA-20 NRT 2026-04-01 – 2026-06-03; NOAA-21 NRT 2026-01-17 – 2026-06-03. ~2 overpasses/day. | NASA FIRMS (https://firms.modaps.eosdis.nasa.gov/download/list.php); product readme: https://firms.modaps.eosdis.nasa.gov/download/Readme.txt | |
| **SE FireMap** (`cbi_mosaic`) | Composite Burn Index (CBI) burn severity, gradient-boosted model over Landsat ARD. Recommended severity product from the comparison, then **dropped**: severity coverage was too sparse over NC peatlands (report Fig. X). | 30 m (EPSG:5070) | 2000–2022, annual | USGS: https://burnseverity.cr.usgs.gov/products/southeastFiremap/data | |
| **MOSEV** | Global MODIS burn-severity database (band 1 = dNBR). Severity comparison only; **dropped** (sparse over NC peat). | 500 m (sinusoidal) | 2000–2020, monthly | Zenodo record 4265209. Alonso-González, E. & Fernández-García, V. (2021). *MOSEV: a global burn severity database from MODIS (2000–2020)*. *Earth System Science Data* 13:1925–1938. | |
| **MTBS** | Monitoring Trends in Burn Severity: Landsat thematic severity classes and perimeters for fires above the MTBS size thresholds. Severity comparison only; **dropped** (sparse over NC peat). | 30 m (EPSG:5070) | 1984–present, annual | USGS: https://burnseverity.cr.usgs.gov/direct-download | |

## Ground-truth reference datasets (product validation)

| Dataset | Role in analysis | Native resolution | Temporal coverage | Source/citation | Box path |
|---|---|---|---|---|---|
| **NIFC IFPH** (Interagency Fire Perimeter History, All Years) | **Primary ground-truth reference** for scoring product accuracy over NC peatlands (recall/omission; precision only conditionally, within a buffered per-event window). Source of the large pocosin incidents (Evans Road 2008, Pains Bay 2011, Juniper Road 2011, Whipping Creek 2016, Great Lakes 2023). | Vector polygons, per incident | ~1980 – present | National Interagency Fire Center. *InteragencyFirePerimeterHistory — All Years View*. NIFC Open Data. http://data-nifc.opendata.arcgis.com/datasets/e02b85c0ea784ce7bd8add7ae3d293d0_0 | |
| **GeoMAC** (Historic Perimeters Combined 2000–2018) | Independent **cross-check** on IFPH over 2000–2018. Overlaps IFPH, so scored separately rather than pooled (pooling double-counts the same fire). | Vector polygons, per incident | 2000-01-01 – 2018-12-31 | National Interagency Fire Center / GeoMAC. *Historic Perimeters Combined 2000–2018*. NIFC Open Data. https://data-nifc.opendata.arcgis.com/datasets/ef25d7e8c9f3499ba9e3d8e09606e488_0 | |
| **TNC_NC_Coastal_Plain_Fire_History_2025** | Reference for **small controlled burns** on TNC coastal-plain preserves — the peat-relevant program (pocosin / wet-pine). Tests small-fire omission, which the coarse products miss most. | Vector polygons, per burn unit | Burn history through 2025 (2025 export) | The Nature Conservancy, provided by Margaret Fields | |
| **TNC_NC_Sandhills_Fire_History_2025** | Same role for TNC sandhills preserves (longleaf uplands): a small-fire omission stress test outside peat. | Vector polygons, per burn unit | Burn history through 2025 (2025 export) | The Nature Conservancy, provided by Margaret Fields | |

## Treatment and sample frame

| Dataset | Role in analysis | Native resolution | Temporal coverage | Source/citation | Box path |
|---|---|---|---|---|---|
| **NC_Pocosin_Restoration_Sites_2026** | **The treatment.** Peatland restoration (rewetting) site polygons with restoration start/end years; `End_Yr` is used as the treatment year, and sites with `End_Yr = 0` (not completed) are dropped. Defines treated pixels and the DiD cohorts; 6 restored sites carry through to the analysis. | Vector polygons, per site | Restoration years through the 2026 export; supported cohorts under FireCCIS311 are 2021/2023, plus 2019 under MCD64A1 | Cat's Box database, *Peat Restoration* folder (TNC restoration records) | |
| **conus_histosol_major_percentage** | gSSURGO **percent of each pixel mapped as a major-component histosol** (H%, 0–100). Two roles: (a) the continuous matching covariate `histosol_pct`; (b) thresholded to define the peatland extent. | 30 m (EPSG:5070) | Static (gSSURGO snapshot) | Cat's Box database, *Peat Extent* folder. Lilleskov, E. et al. (2025). *Journal of Environmental Management*. | |
| **nc_peatlands_80_histosol_aoi** | **Sample frame for every model.** Dissolved polygon of pixels with H% > 80, derived from the histosol layer. All fire responses and covariates are clipped to it and put on a shared 300 m EPSG:5070 grid. | Derived from the 30 m mask; analysis grid 300 m | Static | Derived in `notebooks/download_and_clip_data.ipynb` from `conus_histosol_major_percentage` (Lilleskov et al. 2025) | |

## Model covariates

Static covariates are matching variables (nearest-neighbour in whitened covariate
space for continuous layers, exact match for categorical); per-year covariates enter
the outcome/DiD stage (e.g. a `treated × drought-year` interaction).

| Dataset | Role in analysis | Native resolution | Temporal coverage | Source/citation | Box path |
|---|---|---|---|---|---|
| **GLO30_DEM** (`elevation`) | Copernicus GLO-30 elevation (m a.s.l.). **Static continuous matching covariate.** | 30 m | Static (2011–2015 acquisitions) | Google Earth Engine `ee.ImageCollection('COPERNICUS/DEM/GLO30')`; ESA/Airbus Copernicus DEM. Also at https://portal.opentopography.org/raster?opentopoID=OTSDEM.032021.4326.3 | |
| **GHCN-Daily, NC stations** (PRCP, TMAX, TMIN) | Raw station records behind **every** climate covariate; pulled in R with `FedData::get_ghcn_daily`, unit-converted (precip → mm, temps → °C). Not used directly in the models. | Point (station) | Full station records, multi-decadal through ~2024; daily | NOAA NCEI GHCN-Daily (https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily). Menne, M.J. et al. (2012). *Global Historical Climatology Network-Daily*. NOAA NCEI. | |
| **Climate normals** — `precip_normal`, `tmax_normal`, `tmin_normal`, `gdd_normal` | **Static continuous matching covariates**: long-run annual precip total, mean daily TMAX/TMIN, and base-5 °C growing degree days, IDW-interpolated from the GHCN stations onto the analysis grid. `gdd_normal` is also in the headline logit adjustment set. | 300 m (interpolated from point stations) | 1991–2020 baseline (across-year mean) | Derived from GHCN-Daily (Menne et al. 2012) by `src/peatfire/modeling/climate.py`; GDD from `clim_monthly.gpkg` (`src/get_climate&soil_data_updated.R`) | |
| **Annual climate** — `precip`, `tmax`, `tmin` | **Per-year (temporal) covariates** for the outcome/DiD stage: year-specific weather (annual precip total, annual mean daily TMAX/TMIN). | 300 m (interpolated from point stations) | One grid per panel year (2019–2024 built; extends back with the MCD64A1 panel as far as station coverage allows), annual | Derived from GHCN-Daily (Menne et al. 2012) by `climate.build_annual_climate` | |
| **scPDSI** (`pdsi`) | **Per-year drought covariate** for the outcome/DiD stage (the `treated × drought-year` interaction). Self-calibrating Palmer Drought Severity Index computed per station in R from monthly precip + Hargreaves PET, then IDW-interpolated; annual value = mean of that year's monthly scPDSI. | 300 m (interpolated from point stations) | Station monthly panel 2000–2026; annual grids for the panel years | Derived from GHCN-Daily via `SPEI::hargreaves` + `scPDSI::pdsi`. Wells, N., Goddard, S., Hayes, M.J. (2004). *A self-calibrating Palmer Drought Severity Index*. *Journal of Climate*. | |
| **SSURGO soil (NC coastal plain)** | Raw relational SSURGO map-unit polygons + attribute tables (component, chorizon, muaggatt, mapunit) for the NC coastal-plain counties. Source behind the rasterised soil covariates; not used directly. | Soil map-unit polygons (vector) | Static (SSURGO snapshot, downloaded 2026-07) | USDA-NRCS SSURGO via `FedData::get_ssurgo` (https://websoilsurvey.nrcs.usda.gov). Soil Survey Staff, USDA-NRCS, *Soil Survey Geographic (SSURGO) Database*. | |
| **SSURGO soil covariates** — `soil_organic_matter`, `soil_awc`, `soil_drainage_class` | **Static matching covariates**: organic-matter content (`om_r`, % by weight; fuel-load proxy) and available water capacity (`awc_r`; moisture-retention proxy) as continuous axes, drainage class (`drainagecl`) as a categorical **exact-match key**. Aggregated per map unit, then rasterised. | Map-unit polygons rasterised at 30 m → 300 m grid | Static | Derived from SSURGO (Soil Survey Staff, USDA-NRCS) by `src/peatfire/modeling/soil.py` | |
| **SSURGO soil covariates (Cat's aggregated database)** — `soil_site_index`, `soil_water_table_depth` | **Static continuous matching covariates**, also in the headline logit adjustment set: forest-productivity site index (`industrial`/`coforprod.siteindex.r`) and April–June minimum water-table depth (`wtdepaprjunmin`, cm) — the latter the most direct drainage proxy for control sites. | Map-unit polygons rasterised → 300 m grid | Static | Cat's aggregated SSURGO database (`soil_database.gpkg`, built by `src/get_climate&soil_data_updated.R`); Soil Survey Staff, USDA-NRCS, SSURGO | |
| **LANDFIRE EVT 2024** (`land_cover`) | Existing Vegetation Type, registered as the categorical land-cover **exact-match key**, but **excluded from the reported model runs** (`excluded_covariates = {land_cover, drainage}`) — exact-matching on EVT class shrank the control pool too far. Retained for sensitivity checks. | 30 m (EPSG:5070) | 2024 snapshot (static) | Cat's Box database, *Land Cover* folder. LANDFIRE (2024). *Existing Vegetation Type Layer, LANDFIRE 2024*. U.S. Geological Survey / U.S. Dept. of Agriculture. https://doi.org/10.5066/P1XVKXRL | |

## Boundaries and context layers

| Dataset | Role in analysis | Native resolution | Temporal coverage | Source/citation | Box path |
|---|---|---|---|---|---|
| **nc_boundary** (Census TIGER state boundary) | NC clip mask for every download, and the AOI for the statewide product-comparison and validation figures. | Vector polygon | 2018 vintage (static) | US Census Bureau TIGER/Line cartographic boundary: https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_500k.zip | |
| **NC state & county boundaries (NAD83, 2011)** | County polygons used to scope the SSURGO and GHCN county-level pulls and for mapping. | Vector polygons | 2011 vintage (static) | NC OneMap / NCEM GIS: https://www.nconemap.gov/datasets/NCEM-GIS::north-carolina-state-and-county-boundary-polygons | |
| **Bailey's ecoregions** (all NC; coastal-plain subset) | Ecoregion context for the coastal-plain vs. sandhills framing of the TNC reference burns. Reference/mapping only — not a model covariate. Coastal-plain subset = `ECO_US_` 1895, 2926, 2946 (Mid-Atlantic, Southern, and East Gulf Coastal Plain). | Vector polygons | 1994 snapshot (static) | Bailey, R.G. (2016). *Bailey's ecoregions and subregions of the United States, Puerto Rico, and the U.S. Virgin Islands*. USDA Forest Service RDS-2016-0003. https://www.fs.usda.gov/rds/archive/catalog/RDS-2016-0003 | |

---

## Compiled but not used in the final analyses

Listed for completeness; full details in `data_inventory.csv`.

| Dataset | Why it isn't in the table above |
|---|---|
| **UFMP** (Undersized Fire Mapping Project) | MTBS-method severity/perimeters for 8 undersized NC fires. Dropped with the other severity products; coverage is opportunistic and federal-land-biased. |
| **TIMO / SEUS forest-management extent** | Downloaded as a candidate `management` covariate but never processed to the analysis grid, so it never entered matching. |
| **NASA HAND (drainage)** | Registered as the continuous `drainage` covariate but not downloaded; `soil_water_table_depth` covers the drainage signal instead. |
| **NCWRC prescribed-fire perimeters** | Registered as a reference source; no files on disk, so it contributed no validation events. |
| **VIIRS NOAA-20 / NOAA-21 NRT** | Near-real-time 2026 subsets only — too short for validation; the S-NPP archive is the VIIRS record actually used. |

## Two things worth double-checking before this goes in the report

1. **`modeling.ipynb` is currently set to `FIRE_PRODUCT = "FireCCIS311"`**, while the
   report says the headline results come from MCD64A1. Flip the switch (and restart
   the kernel) before regenerating the reported figures, or say in the methods which
   product each figure came from.
2. **`data_inventory.csv` lists MCD64A1's temporal extent as 2017-01-01 – 2018-04-30**,
   which was the first exploratory download. The modeling panel is configured for
   2001–2026, so confirm which years are actually in
   `data/processed/fire/MCD64A1_061/` and update both the inventory and the
   "Temporal coverage" cell above to the real processed span.

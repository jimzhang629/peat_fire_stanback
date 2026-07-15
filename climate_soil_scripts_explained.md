# Climate & Soil R scripts — line-by-line walkthrough

A plain-language reference for the two data-download scripts in `src/` — what every
block does, what each function takes and returns, and how the two scripts divide the
work. Written for someone who doesn't write R.

- `src/get_climate&soil_data.R` — the **pipeline** script (feeds the Python model)
- `src/get_climate&soil_data_updated.R` — **Cat's analysis** script (richer tables)

---

## The big picture first

These are **two different tools**, not two versions of one. The naming is genuinely
backwards from what you'd expect, so pin this down before reading either:

| | `get_climate&soil_data.R` (pipeline) | `get_climate&soil_data_updated.R` (analysis) |
|---|---|---|
| Purpose | Trimmed to feed the Python `peatfire` model | Cat's richer hand-analysis |
| Counties | all 40 | all 40 (recently expanded from a 3-county test) |
| Climate | daily frames → `.Rds` | daily frames **plus** GDD + combined `clim` table |
| Soil | raw 5-layer GeoPackage (model aggregates it) | pre-aggregated `soil_database` |
| Drought | scPDSI → `.Rds` (`YEAR`/`MONTH` cols) | same scPDSI (`month.year` string) |
| Extras | — | NLCD 2019 land cover |
| Saves? | yes | nothing originally — the save blocks were added recently |

Both run **independently and in either order** — neither reads the other's output.
The model now draws from *both* (see the last section). Below, the machinery the two
scripts **share** (the climate reshape, the SSURGO patches, the drought function) is
explained once and cross-referenced.

---

## R in 60 seconds

Five symbols carry almost every line:

| Symbol | Name | What it does |
|---|---|---|
| `<-` | assign | Store the right-hand thing into the name on the left. `x <- 5` |
| `%>%` | the pipe | Take the left result, feed it as the first input to the next function. Read it as *then*. |
| `f(a = 1)` | function call | Run an operation; `a = 1` are named inputs. |
| `$` | extract | Reach inside an object for a named part. `nc.climate$spatial` |
| `c(...)` | combine | Build a list/vector of values. |

Also: `#` starts a **comment** R ignores. A stack of piped steps assigned to a name is
a **pipeline** — an assembly line that starts with one object and passes it through
each step.

---

# Script 1 — `get_climate&soil_data.R` (pipeline)

Top-to-bottom, this downloads GHCN weather, computes a drought index, downloads SSURGO
soil, and downloads land cover — saving each as files the Python pipeline reads.

### Setup & study area — lines 1–30

- `rm(list=ls())` wipes memory clean; `options(stringsAsFactors = FALSE)` keeps text as text.
- `library(...)` loads packages: `sf` (spatial), `dplyr`/`tidyr` (wrangling), `FedData`
  (the downloader), `terra` (rasters), `tigris` (county shapes).
- `counties <- c("Beaufort", …)` — a list of all **40** NC Coastal Plain county names.
- `nc <- counties(state = "North Carolina") %>% filter(NAME %in% counties)` — downloads
  every NC county polygon, *then* keeps your 40. This `nc` spatial table is the
  **template** every download is clipped to.

### Download climate — lines 42–50

```r
get_ghcn_daily(
  template       = nc,
  label          = "nc_climate",
  elements       = c("tmax", "tmin", "prcp"),
  standardize    = TRUE,
  years          = c(1926:2026),
  raw.dir        = "data/raw/climate/ghcn",       # where raw downloads land
  extraction.dir = "data/interim/climate/ghcn")   # where the clipped cache lands
```

**Returns** `nc.climate`, a list with `$tabular` (per-station daily tables, wide — one
column per day `D1…D31`) and `$spatial` (station point locations).

### Reshape to tidy tables — lines 55–79

`nc.max`, `nc.min`, `nc.prcp` each turn one measurement from wide to long. This exact
pattern is in [the shared climate-reshape section](#shared-the-climate-reshape) — it's
identical in both scripts.

### 💾 Save the daily frames — lines 82–84

```r
saveRDS(nc.max,  "data/interim/climate/ghcn/nc_tmax_long.Rds")
saveRDS(nc.min,  "data/interim/climate/ghcn/nc_tmin_long.Rds")
saveRDS(nc.prcp, "data/interim/climate/ghcn/nc_prcp_long.Rds")
```

`saveRDS(object, "path")` writes one R object to a compact `.Rds` file (reload with
`readRDS`). It returns nothing useful — it runs for the side effect of creating the file.

### Drought index (scPDSI) — lines 86–165

- `totalprcp` — monthly precip totals per station (`group_by` then `summarize(sum)`).
- `pdsi_in` (first pass) — joins monthly precip + mean temps + station latitude and
  restricts to the 2000–2026 window. **No PET yet.**
- `complete(YEAR = 2000:2026, MONTH = 1:12)` — inserts missing rows so each station has a
  gap-free, ordered monthly series starting at January 2000 (the drought model requires it).
- **PET last, once per station.** Only *after* the series is gap-filled does the script
  compute **PET** (atmospheric "thirst") with `hargreaves(Tmin, Tmax, lat)`. This ordering
  matters: `hargreaves()` treats its input as one monthly series beginning in January, so it
  must see each station's full, contiguous 12-month cycle in order — not a month at a time.
  A single scalar `latitude` is passed because every row of a station shares one location.
- `run_scpdsi` — the per-station function, in [the shared drought section](#shared-the-drought-function).
- `pdsi_results` — runs it per station, stacks, re-attaches geometry.

```r
saveRDS(pdsi_results, "data/interim/climate/ghcn/nc_pdsi_long.Rds")
```

> **Shape:** here `pdsi_results` keeps clean `STATION / YEAR / MONTH / scPDSI` columns —
> the format the Python side prefers.

### Soil (SSURGO) — lines 160–324

Lines 160–282 are the [SSURGO patch functions](#shared-the-ssurgo-patches) (shared). Then:

```r
areanames  # "NC013", "NC015", … built from county codes
nc.soil <- get_ssurgo_fixed(template = areanames, label = "nc_soil",
                            raw.dir = "data/raw/soil/ssurgo",
                            extraction.dir = "data/interim/soil/ssurgo",
                            force.redo = FALSE)   # reuse cache if present
```

**Returns** `nc.soil` with `$spatial` (soil map-unit polygons, carrying `MUKEY`) and
`$tabular` (the relational attribute tables — where the soil *properties* actually live).

**💾 Write the raw 5-layer GeoPackage — lines 309–324.** The key design choice of the
pipeline script: it saves the soil **raw and relational**, so `soil.py` aggregates it itself.

```r
soil_gpkg <- "data/interim/soil/ssurgo/nc_soil_ssurgo.gpkg"
sf::st_write(nc.soil$spatial, soil_gpkg, layer = "mapunit_polys", delete_dsn = TRUE)
# then loop the attribute tables into the same file as extra layers:
for (tbl in c("component", "chorizon", "muaggatt", "mapunit")) { … }
```

A **GeoPackage** (`.gpkg`) is a single file holding many named "layers" (like tabs in a
spreadsheet, but spatial). A `for` loop writes each attribute table as its own layer.

### Land cover — lines 330–337

```r
nc.lc <- get_nlcd(template = nc, label = "nc_landcover",
                  year = 2020, dataset = "landcover",
                  extraction.dir = "data/interim/land_cover/nlcd")
```

Downloads the 2020 NLCD raster (each ~30 m cell classified as forest, wetland, water,
developed, …). *Note: the model actually uses a different land-cover product (LANDFIRE),
so this one is produced but not consumed.*

---

# Shared — the climate reshape

Both scripts build `nc.max` / `nc.min` / `nc.prcp` with this identical pipeline. Here it
is for max temperature:

```r
nc.max <- as.data.frame(do.call(bind_rows, nc.climate$tabular))$TMAX %>%
  pivot_longer(cols = c(D1:D31), names_to = "DAY", values_to = "TMAX") %>%
  mutate(TMAX = TMAX/10, DAY = as.numeric(gsub("D", "", DAY))) %>%
  filter(!is.na(TMAX)) %>%
  left_join(nc.climate$spatial %>% rename(STATION = ID))
```

1. **`do.call(bind_rows, …)`** stacks every station's table into one; `$TMAX` pulls out
   the wide max-temp sub-table (`STATION, YEAR, MONTH, D1…D31`).
2. **`pivot_longer`** — the key reshape. The 31 day-columns collapse into two: `DAY`
   (which day) and `TMAX` (the value). One row explodes into up to 31.
3. **`mutate`** — `TMAX/10` converts tenths-of-°C to °C; `gsub("D","",DAY)` strips the
   "D" and `as.numeric` makes it a number.
4. **`filter(!is.na(TMAX))`** — drops empty days (e.g. Feb 30/31 created by padding).
5. **`left_join(… rename(STATION = ID))`** — attaches each station's location to every
   reading, matched on `STATION`.

The wide → long transformation, concretely:

**Before (wide):**

| YEAR | MONTH | D1 | D2 | D3 |
|---|---|---|---|---|
| 2005 | 07 | 312 | 305 | 298 |

**After (long):**

| YEAR | MONTH | DAY | TMAX |
|---|---|---|---|
| 2005 | 07 | 1 | 31.2 |
| 2005 | 07 | 2 | 30.5 |
| 2005 | 07 | 3 | 29.8 |

`nc.min` and `nc.prcp` are the same pattern for min temp and precip (precip `/10` → mm).

---

# Shared — the drought function

Both scripts define `run_scpdsi` to compute one station's drought series (the pipeline
version is a little terser):

```r
run_scpdsi <- function(station_df) {
  start_year <- station_df$YEAR[1]                # first year in the series
  result <- scPDSI::pdsi(P = station_df$PRCP, PE = station_df$PET,
                         start = start_year, sc = TRUE)  # sc=TRUE → self-calibrating
  n <- length(result$X)                           # result$X = monthly PDSI series
  # rebuild calendar year/month from the running index:
  data.frame(STATION = station_df$STATION[1],
             YEAR  = start_year + (seq_len(n) - 1) %/% 12,   # %/% = integer divide
             MONTH = ((seq_len(n) - 1) %% 12) + 1,           # %%  = remainder
             scPDSI = as.numeric(result$X))
}
```

- `function(station_df) { … }` defines a reusable recipe; `station_df` is a placeholder
  for whatever table you hand it.
- `pdsi(P, PE, start, sc = TRUE)` computes the **self-calibrating Palmer Drought Severity
  Index** from precip `P` and PET `PE`. Scale: **~0 = normal, negative = drought, positive = wet.**
- `%/% 12` (integer division) turns a running month count into years; `%% 12` (remainder)
  gives the position within the year.

Applied to every station with:

```r
pdsi_results <- pdsi %>% group_by(STATION) %>% group_split() %>%
  purrr::map(run_scpdsi) %>% bind_rows()
```

`group_split()` breaks the table into one-per-station pieces; `purrr::map()` runs the
function on each; `bind_rows()` stacks the results.

---

# Shared — the SSURGO patches

SSURGO is the USDA's detailed soil survey. `FedData`'s built-in downloader uses some
**broken USDA URLs** and has a geometry-parsing bug, so both scripts define three
`fixed_*` "patch" functions plus a wrapper. You don't need to read these closely — think
of them as *plumbing repairs*.

| Function | What it repairs |
|---|---|
| `fixed_download_ssurgo_inventory` | Corrects the "which surveys exist" download URL. |
| `fixed_download_ssurgo_study_area` | Corrects each county's soil-zip download URL. |
| `fixed_get_ssurgo_inventory` | Fixes parsing of the map geometry returned by the USDA server. |
| `get_ssurgo_fixed` | Wrapper: swaps the fixes in, runs the download, restores the originals on exit. |

The wrapper uses `assignInNamespace(...)` to temporarily replace FedData's broken
internals, and `on.exit({...})` to schedule the originals to be put back when it
finishes — even if it errors — so FedData is left un-corrupted.

---

# Script 2 — `get_climate&soil_data_updated.R` (Cat's analysis)

Same download backbone, but it builds friendlier combined objects and (since the recent
edits) saves them. It shares the reshape, drought, and SSURGO machinery above — here's
what's *different*.

### Setup — lines 1–36

Loads `SPEI` and `scPDSI` up front (lines 20–21). Uses all **40** counties (recently
expanded from a 3-county test). No explicit `raw.dir` on the climate call, so it uses
FedData's default cache instead of `data/`.

### Climate reshape + growing degree days — lines 60–97

`nc.max` / `nc.min` are the usual [reshape](#shared-the-climate-reshape). Then this
script adds three things the pipeline never computes: `gdd` (see
[growing degree days](#growing-degree-days) below), plus `meanmax` / `meanmin` (average
monthly temperatures).

### Combined climate table `clim` — lines 123–136

```r
clim <- left_join(gdd, meanmax) %>% left_join(meanmin) %>% left_join(gsprcp) %>%
  left_join(nc.climate$spatial %>% rename(STATION = ID)) %>%
  mutate(month = month.name[as.numeric(MONTH)], year = as.numeric(YEAR)) %>%
  mutate(geometry = st_transform(geometry, crs = 4326)) %>%
  dplyr::select(year, month, totalGDD, TMAX, TMIN, PRCP, geometry)
```

- The chain of `left_join`s stitches GDD + mean temps + growing-season precip + station
  geometry into one table.
- `month.name[as.numeric(MONTH)]` — `month.name` is a built-in vector `c("January", …)`;
  indexing it turns `"07"` into `"July"`.
- `st_transform(…, crs = 4326)` reprojects the geometry to plain longitude/latitude.

> **💾 Added:** `clim` → `data/interim/climate/ghcn/clim_monthly.gpkg` — a GeoPackage so
> Python/geopandas can read it. This file supplies the model's **growing-degree-days** layer.

### Drought — lines 146–224

Same [drought function](#shared-the-drought-function) and per-station run, with two
differences from the pipeline script:

- **PET timing.** This script computes `hargreaves(...)` inside a
  `group_by(STATION, YEAR, MONTH)` — i.e. one month at a time — *before* `complete()`
  gap-fills the series. The pipeline script was later fixed to compute PET once over the
  full, ordered, gap-filled monthly series (see the pipeline drought section). If you rely
  on this file's PDSI, prefer the pipeline's; the model already reads the pipeline's
  `nc_pdsi_long.Rds`, so this discrepancy doesn't reach the covariates.
- **Output shape.** It keeps a `month.year` string (`"2005 July"`) instead of clean
  YEAR/MONTH columns.

> **💾 Added:** `pdsi_results` → `data/interim/climate/ghcn/pdsi_results.gpkg`. *(The model
> actually uses the pipeline's PDSI file, so this one is an extra.)*

### Example join code — lines 227–259

Two blocks wrapped in `if(FALSE){ … }` — R **never runs them**. They're a template showing
how to join `clim` and `pdsi_results` to site data by nearest weather station.
Documentation, not active code.

### Building `soil_database` — lines 401–443

This is what makes `_updated.R` distinctive. SSURGO is **relational** — properties are
scattered across linked tables — so this assembles them into one clean per-polygon table.
The key chain:

```
mukey ──(component)──> cokey ──(coforprod)──> siteindex.r   # forest productivity
                          └────(chorizon)───> awc.r         # available water
```

- `soil.spatial` — cleans the polygon layer, lowercases column names (`MUKEY` → `mukey`,
  geometry → `geom`).
- `soil_props` — starts from `component` (the `mukey↔cokey` bridge), joins `coforprod`
  for site index and `chorizon` for AWC, then `group_by(mukey) %>% summarize(mean(...))`
  collapses to **one value per map unit**.
- `soil_database` — joins those properties onto the polygons, computes acreage with
  `st_area(geom) * 0.000247105` (m² → acres), and keeps the final columns:

| Column | Meaning |
|---|---|
| `musym` / `mukey` | map-unit symbol / key (identifiers) |
| `muname` | map-unit name |
| `wtdepaprjunmin` | water-table depth, Apr–Jun minimum |
| `numacres` | acres in the polygon |
| `industrial` | site index / forest productivity |
| `awc` | available water content |

> **💾 Added:** `soil_database` → `data/interim/soil/ssurgo/soil_database.gpkg`. This
> supplies the model's **site index** and **water-table depth** layers. It is *not* the
> raw export `soil.py` reads — different file, different shape.

### Land cover — lines 474–488

`get_nlcd(… year = 2019 …)` downloads the 2019 NLCD raster.
**💾** `terra::writeRaster(nc.lc, "…/nc_landcover_2019.tif")` writes it as a GeoTIFF.
*(An extra output — the model uses LANDFIRE, not NLCD.)*

---

# Growing degree days

**GDD** measures accumulated warmth over the growing season. Each day contributes its
average temperature above a 5 °C base, floored at zero; those add up across the year.

```r
gdd <- left_join(nc.max, nc.min) %>%
  mutate(GDD = pmax(0, (TMAX + TMIN) / 2 - 5)) %>%       # daily degrees, floored at 0
  group_by(STATION, YEAR) %>% mutate(GDD = cumsum(GDD)) %>%   # running total
  group_by(STATION, YEAR, MONTH) %>% summarize(GDD = max(GDD)) %>%
  group_by(STATION, YEAR) %>% mutate(totalGDD = max(GDD, na.rm = TRUE))
```

- `pmax(0, x)` — element-wise "floor at zero": cold days contribute `0`, not a negative.
- `cumsum(GDD)` — a **running total** day by day within each station-year.
- `totalGDD` — the year's final accumulated GDD, carried onto every month's row.

| day | daily GDD | cumsum |
|---|---|---|
| Jan 1 | 0 | 0 |
| Apr 10 | 8 | 145 |
| Apr 11 | 9 | 154 |

This is the one genuinely new climate quantity in `_updated.R`, and it's why the model
now pulls `gdd_normal` from `clim_monthly.gpkg`.

---

# Outputs & how to run

Run both scripts once (either order — they're independent), then run the modeling
notebook. Every file below lands under `data/` but is git-ignored, so it stays on your
machine.

| Output file | Written by | Model uses it? | Supplies |
|---|---|---|---|
| `nc_tmax_long.Rds` · `nc_tmin_long` · `nc_prcp_long` | pipeline | ✅ | climate normals + per-year weather |
| `nc_pdsi_long.Rds` | pipeline | ✅ | drought (`pdsi`) |
| `nc_soil_ssurgo.gpkg` (raw, 5 layers) | pipeline | ✅ | organic matter, AWC, drainage |
| `clim_monthly.gpkg` | analysis | ✅ | `gdd_normal` |
| `soil_database.gpkg` | analysis | ✅ | `soil_site_index`, `soil_water_table_depth` |
| `pdsi_results.gpkg` | analysis | — | extra (model uses pipeline PDSI) |
| `nc_landcover_2019.tif` | analysis | — | extra (model uses LANDFIRE) |

1. In the R console, `source("src/get_climate&soil_data.R", encoding = "UTF-8")` and
   `source("src/get_climate&soil_data_updated.R", encoding = "UTF-8")`. Both need `SPEI` +
   `scPDSI` installed and a matching R version so the soil/graphics steps don't crash.
2. Confirm the `interim/` files above exist.
3. Run the modeling notebook (step 1b onward) — it reads these files and builds the
   gridded covariate rasters.

> **You don't re-run R every time.** Once the `interim/` files exist, you can re-run the
> notebook freely — the R step is only for refreshing the underlying data.

### Get historic climate data and soil data across North Carolina
## Precip data, GDD, max temperatures, PDSI and SSURGO data for soils
## Started 14 April 2026 by Cat

### Recent paper from Forest Service
# https://www.fs.usda.gov/rm/pubs_journals/2025/rmrs_2025_holden_z001.pdf

### housekeeping
rm(list=ls()) 
options(stringsAsFactors = FALSE)

### Load Libraries
library(sf)
library(dplyr)
library(tidyr)
library(FedData)
library(terra)
library(tigris)

## Counties in NC Coastal Plain
counties <- c("Beaufort", "Bertie", "Bladen", "Brunswick", "Camden", "Carteret", "Chowan",
              "Columbus", "Craven", "Cumberland", "Currituck", "Dare", "Duplin", "Edgecombe",
              "Gates", "Greene", "Halifax", "Harnett", "Hertford", "Hoke", "Hyde", "Johnston",
              "Jones", "Lenoir", "Martin", "Nash", "New Hanover", "Northampton", "Onslow",
              "Pamlico", "Pasquotank", "Pender", "Perquimans", "Pitt", "Robeson", "Sampson", 
              "Scotland", "Tyrrell", "Wayne", "Wilson")

### Load shapefile of North Carolina Coastal Plain
nc <- counties(state = "North Carolina") %>%
  filter(NAME %in% counties)


################################################################################
################################################################################
########################## Get historic climate data ###########################
################################################################################

## Can extract several climate metrics including precipitation, max temperature, min temperature,
## wind speed, wind direction, soil temperature, peak gust time, etc
## https://search.r-project.org/CRAN/refmans/FedData/html/get_ghcn_daily.html

nc.climate <- get_ghcn_daily(
  template = nc,
  label = "nc_climate",
  elements = c("tmax", "tmin", "prcp"),
  standardize = TRUE,
  years = c(1926:2026),
  raw.dir = "data/raw/climate/ghcn",
  extraction.dir = "data/interim/climate/ghcn"
)

############################## Get dataframes ##################################

### TMAX
nc.max <- as.data.frame(do.call(bind_rows, nc.climate$tabular))$TMAX %>%
  pivot_longer(cols = c(D1:D31), names_to = "DAY", values_to = "TMAX") %>%
  mutate(TMAX = TMAX/10, ### Convert to degrees C
         DAY = as.numeric(gsub("D", "", DAY))) %>%
  filter(!is.na(TMAX)) %>%
  ## Match station to geometry
  left_join(nc.climate$spatial %>% rename(STATION = ID))

### TMIN
nc.min <- as.data.frame(do.call(bind_rows, nc.climate$tabular))$TMIN %>%
  pivot_longer(cols = c(D1:D31), names_to = "DAY", values_to = "TMIN") %>%
  mutate(TMIN = TMIN/10, ### Convert to degrees C
         DAY = as.numeric(gsub("D", "", DAY))) %>%
  filter(!is.na(TMIN)) %>%
  ## Match station to geometry
  left_join(nc.climate$spatial %>% rename(STATION = ID))

### PRECIP
nc.prcp <- as.data.frame(do.call(bind_rows, nc.climate$tabular))$PRCP %>%
  pivot_longer(cols = c(D1:D31), names_to = "DAY", values_to = "PRCP") %>%
  mutate(PRCP = PRCP/10, ### Convert to mm 
         DAY = as.numeric(gsub("D", "", DAY))) %>%
  filter(!is.na(PRCP)) %>%
  ## Match station to geometry
  left_join(nc.climate$spatial %>% rename(STATION = ID))

# save above dataframes
saveRDS(nc.max,  "data/interim/climate/ghcn/nc_tmax_long.Rds")
saveRDS(nc.min,  "data/interim/climate/ghcn/nc_tmin_long.Rds")
saveRDS(nc.prcp, "data/interim/climate/ghcn/nc_prcp_long.Rds")

################################################################################
################################################################################
### Palmer Drought Severity Index (scPDSI) -- year-specific drought per station
################################################################################
## A TEMPORAL (per-year) climate covariate. scPDSI is a self-calibrating drought
## index (~0 = normal, negative = drought), computed monthly per station from
## precip + Hargreaves PET. We save the monthly per-station series as a long table
## (nc_pdsi_long.Rds); src/peatfire/modeling/climate.py reads it, takes each year's
## MEAN monthly scPDSI as that year's drought level, and IDW-interpolates it onto
## the grid like the annual precip/tmax layers (climate.DEFAULT_PDSI_ELEMENTS).
## scPDSI is self-calibrating, so a long-run *normal* is ~0 everywhere and carries
## no signal -- PDSI is therefore built only as a per-year layer, never a normal.
library(SPEI)     # hargreaves() reference-ET
library(scPDSI)   # pdsi() self-calibrating PDSI

## Monthly precip total per station-month.
totalprcp <- nc.prcp %>%
  group_by(STATION, YEAR, MONTH) %>%
  summarize(PRCP = sum(PRCP, na.rm = TRUE), .groups = "drop")

## Join monthly mean TMAX/TMIN + station latitude. PET is computed further down,
## AFTER the series is made gap-free -- hargreaves() treats its input as a monthly
## series starting in January, so it must see each station's full, ordered,
## contiguous 12-month cycle at once, not one month at a time. Restrict to the
## 2000-2026 window here so every station starts cleanly at January 2000.
pdsi_in <- totalprcp %>%
  left_join(nc.max %>% group_by(STATION, YEAR, MONTH) %>%
              summarize(TMAX = mean(TMAX, na.rm = TRUE), .groups = "drop")) %>%
  left_join(nc.min %>% group_by(STATION, YEAR, MONTH) %>%
              summarize(TMIN = mean(TMIN, na.rm = TRUE), .groups = "drop")) %>%
  left_join(nc.climate$spatial %>% rename(STATION = ID)) %>%
  mutate(latitude = st_coordinates(geometry)[, "Y"],
         MONTH    = as.numeric(MONTH),
         YEAR     = as.numeric(YEAR)) %>%
  st_drop_geometry() %>%
  filter(YEAR >= 2000, YEAR <= 2026)

## pdsi() needs a gap-free monthly series per station. Build the full
## Jan-2000..Dec-2026 grid, carry each station's latitude onto the filled-in
## months, then compute Hargreaves PET ONCE over the whole ordered series per
## station (verbose = FALSE silences the per-call message). A single scalar
## latitude is passed because every row of a station shares one location.
pdsi_in <- pdsi_in %>%
  group_by(STATION) %>%
  complete(YEAR = 2000:2026, MONTH = 1:12) %>%
  arrange(STATION, YEAR, MONTH) %>%
  mutate(latitude = latitude[!is.na(latitude)][1],
         PET = as.numeric(hargreaves(Tmin = TMIN, Tmax = TMAX,
                                     lat = latitude[1], na.rm = TRUE,
                                     verbose = FALSE))) %>%
  ungroup()

run_scpdsi <- function(station_df) {
  start_year <- station_df$YEAR[1]
  result <- scPDSI::pdsi(P = station_df$PRCP, PE = station_df$PET,
                         start = start_year, sc = TRUE)  # sc = TRUE -> scPDSI
  n   <- length(result$X)
  idx <- seq_len(n)
  data.frame(
    STATION = station_df$STATION[1],
    YEAR    = start_year + (idx - 1) %/% 12,
    MONTH   = ((idx - 1) %%  12) + 1,
    scPDSI  = as.numeric(result$X),
    stringsAsFactors = FALSE
  )
}

## Run per station, stack, and re-attach station point geometry (so the Python
## side has station locations to interpolate from). Kept as a long table with
## STATION/YEAR/MONTH/scPDSI -- the shape climate.load_ghcn_stations expects.
pdsi_results <- pdsi_in %>%
  group_by(STATION) %>%
  group_split() %>%
  purrr::map(run_scpdsi) %>%
  bind_rows() %>%
  left_join(nc.climate$spatial %>% rename(STATION = ID)) %>%
  st_as_sf()

saveRDS(pdsi_results, "data/interim/climate/ghcn/nc_pdsi_long.Rds")
message("wrote scPDSI long table -> data/interim/climate/ghcn/nc_pdsi_long.Rds")

################################################################################
################################################################################
####################### Get soil information from SSURGO #######################
################################################################################

#### NOTE 4/14/26: SSURGO data download links are broken. Below code does not work.

################################################################################
##### Issue with FedData using outdated links for SSURGO. 
## Below are some workarounds

# Patch 1: inventory shapefile download
fixed_download_ssurgo_inventory <- function(raw.dir, ...) {
  url <- "https://websoilsurvey.nrcs.usda.gov/DataAvailability/SoilDataAvailabilityShapefile.zip"
  destdir <- raw.dir
  FedData:::download_data(url = url, destdir = destdir, ...)
  return(normalizePath(paste(destdir, "/SoilDataAvailabilityShapefile.zip", sep = "")))
}

# Patch 2: individual study area zip download (nc=FALSE to avoid caching corrupt zips)
fixed_download_ssurgo_study_area <- function(area, date, raw.dir) {
  url <- paste("https://websoilsurvey.nrcs.usda.gov/DSD/Download/Cache/SSA/wss_SSA_",
               area, "_[", date, "].zip", sep = "")
  destdir <- raw.dir
  FedData:::download_data(url = url, destdir = destdir, nc = FALSE)
  return(normalizePath(paste(destdir, "/wss_SSA_", area, "_[", date, "].zip", sep = "")))
}

# Patch 3: fix MULTISURFACE geometry parsing from WFS
fixed_get_ssurgo_inventory <- function(template = NULL, raw.dir) {
  if (!is.null(template)) {
    template %<>%
      FedData:::template_to_sf() %>%
      sf::st_transform(4326)
  }
  
  if (
    !is.null(template) &&
    httr::status_code(
      httr::RETRY(
        verb = "GET",
        url = "https://sdmdataaccess.nrcs.usda.gov/Spatial/SDMWGS84Geographic.wfs"
      )
    ) == 200
  ) {
    bounds <- template %>% sf::st_bbox() %>% sf::st_as_sfc()
    
    if ((sf::st_bbox(template)[["xmax"]] - sf::st_bbox(template)[["xmin"]]) > 1 |
        (sf::st_bbox(template)[["ymax"]] - sf::st_bbox(template)[["ymin"]]) > 1) {
      bounds %<>% sf::st_intersection(FedData:::grid)
    }
    
    SSURGOAreas <- bounds %>%
      purrr::map_dfr(function(x) {
        bound <- x %>% sf::st_bbox()
        if (identical(bound["xmin"], bound["xmax"])) bound["xmax"] <- bound["xmax"] + 1e-04
        if (identical(bound["ymin"], bound["ymax"])) bound["ymax"] <- bound["ymax"] + 1e-04
        bbox.text <- paste(bound, collapse = ",")
        temp.file <- paste0(tempdir(), "/soils.gml")
        
        httr::RETRY(
          verb = "GET",
          url = "https://sdmdataaccess.nrcs.usda.gov/Spatial/SDMWGS84Geographic.wfs",
          query = list(
            Service = "WFS", Version = "1.1.0", Request = "GetFeature",
            Typename = "SurveyAreaPoly", BBOX = bbox.text,
            SRSNAME = "EPSG:4326", OUTPUTFORMAT = "GML3"
          ),
          httr::write_disk(temp.file, overwrite = TRUE)
        )
        
        tryCatch(
          suppressMessages(suppressWarnings(
            sf::read_sf(temp.file, drivers = "GML", type = 3) %>%  # type=3 forces MULTIPOLYGON
              dplyr::mutate(saverest = as.Date(
                lubridate::parse_date_time(saverest, orders = "b d Y HMOp", locale = "en_US")
              )) %>%
              sf::st_drop_geometry()
          )),
          error = function(e) return(NULL)
        )
      }) %>%
      dplyr::distinct() %>%
      dplyr::arrange(areasymbol)
  } else {
    tmpdir <- tempfile()
    if (!dir.create(tmpdir)) stop("failed to create my temporary directory")
    file <- FedData:::download_ssurgo_inventory(raw.dir = raw.dir)
    utils::unzip(file, exdir = tmpdir)
    SSURGOAreas <- sf::read_sf(normalizePath(tmpdir), layer = "soilsa_a_nrcs")
    if (!is.null(template)) {
      SSURGOAreas %<>%
        sf::st_make_valid() %>%
        sf::st_intersection(sf::st_transform(template, sf::st_crs(SSURGOAreas)))
    }
    unlink(tmpdir, recursive = TRUE)
  }
  
  if (0 %in% SSURGOAreas$iscomplete) {
    warning("Some of the soil surveys in your area are unavailable.\n",
            paste0(as.vector(SSURGOAreas[SSURGOAreas$iscomplete == 0, ]$areasymbol), collapse = "\n"))
  }
  
  return(SSURGOAreas)
}


# Wrapper that patches, runs get_ssurgo, then restores originals (even on error)
get_ssurgo_fixed <- function(template, label, ...) {
  # Save originals
  orig_download_ssurgo_inventory  <- FedData:::download_ssurgo_inventory
  orig_download_ssurgo_study_area <- FedData:::download_ssurgo_study_area
  orig_get_ssurgo_inventory       <- FedData:::get_ssurgo_inventory
  
  # Apply patches
  assignInNamespace("download_ssurgo_inventory",  fixed_download_ssurgo_inventory,  ns = "FedData")
  assignInNamespace("download_ssurgo_study_area", fixed_download_ssurgo_study_area, ns = "FedData")
  assignInNamespace("get_ssurgo_inventory",       fixed_get_ssurgo_inventory,       ns = "FedData")
  
  # Restore originals when function exits (even on error)
  on.exit({
    assignInNamespace("download_ssurgo_inventory",  orig_download_ssurgo_inventory,  ns = "FedData")
    assignInNamespace("download_ssurgo_study_area", orig_download_ssurgo_study_area, ns = "FedData")
    assignInNamespace("get_ssurgo_inventory",       orig_get_ssurgo_inventory,       ns = "FedData")
  })
  
  get_ssurgo(template = template, label = label, ...)
}

## Can extract several soil metrics 
## Here, I am focusing on soil moisture, organic matter content, and texture
## https://search.r-project.org/CRAN/refmans/FedData/html/get_ghcn_daily.html

areanames <- paste0("NC", nc$COUNTYFP)

nc.soil <- get_ssurgo_fixed(
  template = areanames,
  label = "nc_soil",
  raw.dir = "data/raw/soil/ssurgo",
  extraction.dir = "data/interim/soil/ssurgo",
  force.redo = FALSE
)

### Save the SSURGO download as ONE GeoPackage for the Python modeling pipeline.
## `get_ssurgo` returns a list: $spatial (the map-unit POLYGONS, carrying MUKEY)
## and $tabular (the relational attribute tables). SSURGO is relational, so the
## soil *properties* don't live on the polygons -- they're in the attribute
## tables: drainage class in `component`, organic matter / AWC in `chorizon`.
## src/peatfire/modeling/soil.py reads exactly this file, aggregates each
## property down to one value per MUKEY, joins it back to the polygons, and
## rasterises it onto the analysis grid. It looks for the polygon layer plus the
## `component`, `chorizon`, `muaggatt`, `mapunit` tables -- so we write each as a
## GeoPackage layer under its own name. Without this step the download stays as
## FedData's scattered per-table files and the Python side has nothing to read.
soil_gpkg <- "data/interim/soil/ssurgo/nc_soil_ssurgo.gpkg"
dir.create(dirname(soil_gpkg), recursive = TRUE, showWarnings = FALSE)

## Polygon layer first (delete_dsn = TRUE starts the GeoPackage fresh).
sf::st_write(nc.soil$spatial, soil_gpkg, layer = "mapunit_polys",
             delete_dsn = TRUE, quiet = TRUE)

## Relational attribute tables (non-spatial) as additional GeoPackage layers.
## soil.py matches layer names case-insensitively, so keep these exact names.
for (tbl in c("component", "chorizon", "muaggatt", "mapunit")) {
  if (!is.null(nc.soil$tabular[[tbl]])) {
    sf::st_write(as.data.frame(nc.soil$tabular[[tbl]]), soil_gpkg, layer = tbl,
                 delete_layer = TRUE, quiet = TRUE)
  }
}
message("wrote SSURGO GeoPackage -> ", soil_gpkg)

################################################################################
############## Get Land Cover data across years of interest ####################
################################################################################

## NOTE: NLCD Land Cover is released for specific epochs (2001, 2004, 2006,
## 2008, 2011, 2013, 2016, 2019, 2021) -- there is no standalone 2020 product,
## so 2019 is used as the nearest available epoch.
nc.lc <- get_nlcd(
  template = nc,
  label = "nc_landcover",
  year = 2019,
  dataset = "landcover",
  extraction.dir = "data/interim/land_cover/nlcd",
  force.redo = FALSE
)






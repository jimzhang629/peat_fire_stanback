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
### Get Palmer Drought Severity Index (PDSI) data for each site
library(SPEI)
library(scPDSI)

#### Jim - feel free to select either the relevant counties for your work or even just use a 
# shapefile from the final dataset

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
  years = c(1926:2026)
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


#### Added 10 July 2026 by Cat
### Get GDDs
gdd <- left_join(nc.max, nc.min) %>%
  mutate(GDD = pmax(0, (TMAX + TMIN) / 2 - 5)) %>%
  group_by(STATION, YEAR,) %>%
  mutate(GDD = cumsum(GDD)) %>%
  group_by(STATION, YEAR, MONTH) %>%
  summarize(GDD = max(GDD)) %>%
  ungroup() %>%
  group_by(STATION, YEAR) %>%
  mutate(totalGDD = max(GDD, na.rm = TRUE))

## Get Monthly Temps
meanmax <- nc.max %>%
  group_by(STATION, YEAR, MONTH) %>%
  summarize(TMAX = mean(TMAX, na.rm = TRUE))

meanmin <- nc.min %>%
  group_by(STATION, YEAR, MONTH) %>%
  summarize(TMIN = mean(TMIN, na.rm = TRUE))


### PRECIP
nc.prcp <- as.data.frame(do.call(bind_rows, nc.climate$tabular))$PRCP %>%
  pivot_longer(cols = c(D1:D31), names_to = "DAY", values_to = "PRCP") %>%
  mutate(PRCP = PRCP/10, ### Convert to mm 
         DAY = as.numeric(gsub("D", "", DAY))) %>%
  filter(!is.na(PRCP)) %>%
  ## Match station to geometry
  left_join(nc.climate$spatial %>% rename(STATION = ID))

## Monthly Precip
totalprcp <- nc.prcp %>%
  group_by(STATION, YEAR, MONTH) %>%
  summarize(PRCP = sum(PRCP, na.rm = TRUE))

### Growing Season Total Precip
gsprcp <- nc.prcp %>%
  filter(MONTH %in% c("04", "05", "06",
                      "07", "08", "09", "10")) %>%
  group_by(STATION, YEAR) %>%
  summarize(PRCP = sum(PRCP, na.rm = TRUE))


### Put all climate data together
clim <- left_join(gdd, meanmax) %>%
  left_join(meanmin) %>%
  left_join(gsprcp) %>%
  ## Match station to geometry
  left_join(nc.climate$spatial %>% rename(STATION = ID)) %>%
  ## Match formatting in swails dataset
  mutate(month = month.name[as.numeric(MONTH)],
         year = as.numeric(YEAR)) %>%
  ungroup() %>%
  mutate(geometry = st_transform(geometry, crs = 4326)) %>%
  dplyr::select(year, month, totalGDD, TMAX, TMIN, PRCP, geometry) %>%
  mutate(month.year = paste(year, month))

clim <- st_as_sf(clim, sf_column_name = "geometry")

### Save the combined monthly climate table (GDD, mean temps, growing-season precip).
## GeoPackage keeps the station point geometry so it can be read in Python/geopandas.
dir.create("data/interim/climate/ghcn", recursive = TRUE, showWarnings = FALSE)
sf::st_write(clim, "data/interim/climate/ghcn/clim_monthly.gpkg",
             delete_dsn = TRUE, quiet = TRUE)
message("wrote climate table -> data/interim/climate/ghcn/clim_monthly.gpkg")


################################################################################
################################################################################
### Calculate PDSI for each site and month

# Calculate PET using latitude and SPEI package
pdsi <- totalprcp %>%
  left_join(nc.max %>% group_by(STATION, YEAR, MONTH) %>% 
              summarize(TMAX = mean(TMAX, na.rm = TRUE))) %>%
  left_join(nc.min %>% group_by(STATION, YEAR, MONTH) %>% 
              summarize(TMIN = mean(TMIN, na.rm = TRUE))) %>%
  ## Match station to geometry
  left_join(nc.climate$spatial %>% rename(STATION = ID)) %>%
  mutate(latitude = st_coordinates(geometry)[,"Y"]) %>%
  group_by(STATION, YEAR, MONTH) %>%
  mutate(PET = hargreaves(TMIN, TMAX, lat = latitude, na.rm = TRUE),
         MONTH = as.numeric(MONTH),
         YEAR = as.numeric(YEAR)) %>%
  arrange(STATION, YEAR, MONTH)

### pdsi function needs a complete dataset to calculate for each station
# remove stations without some month's worth of data
pdsi <- pdsi %>%
  ungroup() %>%
  group_by(STATION) %>%
  complete(YEAR = 2000:2026, MONTH = 1:12)

run_scpdsi <- function(station_df) {
  
  # Get January start for each station
  start_year <- station_df$YEAR[1]
  
  # All months in dataset need to be complete
  P  <- station_df$PRCP   # precipitation (mm)
  PE <- station_df$PET    # pre-computed PET (mm)
  
  # Run self-calibrating PDSI
  result <- pdsi(P  = P,
                 PE = PE,
                 start = start_year,
                 sc = TRUE)   # sc = TRUE → scPDSI; FALSE → classic PDSI
  
  # pdsi() returns a list; $X is the PDSI time series
  n <- length(result$X)
  
  # Rebuild a year/month index that matches the output length
  all_months <- seq_len(n)
  years  <- start_year + (all_months - 1) %/% 12
  months <- ((all_months - 1) %% 12) + 1
  
  data.frame(
    STATION = station_df$STATION[1],
    YEAR    = years,
    MONTH   = months,
    scPDSI  = as.numeric(result$X),
    stringsAsFactors = FALSE
  )
}

pdsi_results <- pdsi %>%
  group_by(STATION) %>%
  # list of per-station data frames
  group_split() %>%  
  # run scPDSI for each
  purrr::map(run_scpdsi) %>%   
  # stack into one data frame
  bind_rows() %>%
  ## Match station to geometry
  left_join(nc.climate$spatial %>% rename(STATION = ID)) %>%
  mutate(MONTH = month.name[as.numeric(MONTH)],
         month.year = paste(YEAR, MONTH)) %>%
  dplyr::select(-MONTH, -YEAR)

pdsi_results <- st_as_sf(pdsi_results)

### Save the scPDSI drought results (one row per station-month, with geometry).
dir.create("data/interim/climate/ghcn", recursive = TRUE, showWarnings = FALSE)
sf::st_write(pdsi_results, "data/interim/climate/ghcn/pdsi_results.gpkg",
             delete_dsn = TRUE, quiet = TRUE)
message("wrote scPDSI drought table -> data/interim/climate/ghcn/pdsi_results.gpkg")


#### Jim - you will have to link up the pdsi_results on line 195 and the clim on lines 109-120
# to your overall dataset.
### Below is some example code of how to do that for the GDD & climate data:
if(FALSE) {
alldata <- alldata %>%
  mutate(month.year = paste(year, month)) %>% 
  st_as_sf(coords = c("longitude", "latitude"), crs = st_crs(clim$geometry)) %>%
  st_transform(4326)


### Join the site data to the nearest weather station for the month and year
all_climate <- do.call(rbind, lapply(unique(alldata$month.year), function(cat) {
  st_join(
    subset(alldata, month.year == cat), 
    subset(clim, month.year == cat), 
    join = st_nearest_feature
  )})) %>%
  select(-month.year.x, -month.y, -year.y, -month.year.y) %>%
  rename(month = month.x, year = year.x)
}

### And then some code for linking up the PDSI results:
if(FALSE){
all_climate$month.year <- paste(all_climate$year, all_climate$month)

all_climate <- do.call(rbind, lapply(unique(all_climate$month.year), function(cat) {
  st_join(
    subset(all_climate, month.year == cat), 
    subset(pdsi_results, month.year == cat), 
    join = st_nearest_feature
  )})) %>%
  select(-month.year.x, -month.year.y, -STATION, -NAME) 
}

################################################################################
################################################################################
####################### Get soil information from SSURGO #######################
################################################################################

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
  label = "nc_soil", force.redo = FALSE
)

#### Added 10 July 2026 by Cat
### Clean spatial data to join with other tables
soil.spatial <- nc.soil$spatial %>%
  mutate(MUKEY = as.numeric(MUKEY))
colnames(soil.spatial) <- tolower(colnames(soil.spatial))

#create and save functions - key with FedData
not_all_na <- function(x) any(!is.na(x)) #for data cleaning

#### Build major database 
# Pre-filter and aggregate lookup tables before joining
muaggatt_clean <- nc.soil$tabular$muaggatt %>%
  select(mukey, muname, musym, wtdepaprjunmin) %>%
  distinct()

# Aggregate soil properties by mukey to avoid cartesian explosions
soil_props <- nc.soil$tabular$component %>%
  select(mukey, cokey) %>%
  left_join(
    nc.soil$tabular$coforprod %>%
      select(cokey, siteindex.r) %>%
      mutate(siteindex.r = ifelse(is.na(siteindex.r), 0, siteindex.r)),
    by = "cokey"
  ) %>%
  left_join(
    nc.soil$tabular$chorizon %>%
      select(cokey, awc.r),
    by = "cokey"
  ) %>%
  group_by(mukey) %>%
  summarize(
    industrial = mean(siteindex.r, na.rm = TRUE),
    awc = mean(awc.r, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(!is.na(industrial) | !is.na(awc))  # Filter early

# Now do one clean join to spatial data
soil_database <- soil.spatial %>%
  left_join(muaggatt_clean, by = c("mukey", "musym")) %>%
  left_join(soil_props, by = c("mukey")) %>%
  filter(!is.na(muname)) %>%  # Remove unmatched rows
  mutate(numacres = as.vector(st_area(geom) * 0.000247105)) %>%
  distinct() %>%
  select(musym, mukey, geom, muname, wtdepaprjunmin, numacres, industrial, awc)

### Save the cleaned, aggregated soil database (one row per map-unit polygon).
## NOTE: this is Cat's *pre-aggregated* soil table -- it is NOT the raw relational
## SSURGO export that src/peatfire/modeling/soil.py reads (that one is written by
## get_climate&soil_data.R as nc_soil_ssurgo.gpkg). Different file, different shape.
dir.create("data/interim/soil/ssurgo", recursive = TRUE, showWarnings = FALSE)
sf::st_write(soil_database, "data/interim/soil/ssurgo/soil_database.gpkg",
             delete_dsn = TRUE, quiet = TRUE)
message("wrote soil database -> data/interim/soil/ssurgo/soil_database.gpkg")

### Use the SSURGO Metadata Table for more information: 
# https://www.nrcs.usda.gov/sites/default/files/2022-08/SSURGO-Metadata-Table-Column-Descriptions-Report.pdf

### Outputs include -
# musym: map unit symbol, unique identifier
# mukey: map unit key, another unique identifier
# geom: spatial geometry
# muname: name of the map unit
# wtdepaprjunmin: Water Table Depth - April - June - Minimum
  ## * this will potentially be useful for the drained control sites only!
  ## * for restored sites, we assume WTD to be 10cm below the soil surface
# numacres: number of acres within each map unit
# industrial: site index and annual productivity of forest aboveground
# awc: available water content


################################################################################
############## Get Land Cover data across years of interest ####################
################################################################################

nc.lc <- get_nlcd(
  template = nc,
  label = "nc_landcover",
  # Available years are: 2001, 2004, 2006, 2008, 2011, 2016, 2019. 
  ## User can only select one year at a time
  year = 2019, 
  dataset = "landcover",
  force.redo = FALSE
)

### Save the land cover raster as a GeoTIFF.
dir.create("data/interim/land_cover/nlcd", recursive = TRUE, showWarnings = FALSE)
terra::writeRaster(nc.lc, "data/interim/land_cover/nlcd/nc_landcover_2019.tif",
                   overwrite = TRUE)
message("wrote land cover raster -> data/interim/land_cover/nlcd/nc_landcover_2019.tif")




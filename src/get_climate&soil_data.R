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

################################################################################
############## Get Land Cover data across years of interest ####################
################################################################################

nc.lc <- get_nlcd(
  template = nc,
  label = "nc_landcover",
  year = 2020,
  dataset = "landcover",
  extraction.dir = "data/interim/land_cover/nlcd",
  force.redo = FALSE
)






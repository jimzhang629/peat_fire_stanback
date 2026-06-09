"""Helpers for preprocessing Google Earth Engine data

"""

import geopandas as gpd
from peatfire import data_path
from peatfire.preproc.data_loading import get_key
import ee
import geemap

def initialize_ee():
    '''
    This function grabs the EE_PROJECT_ID from your .env file and connects to the Google Earth Engine API
    '''
    ee_id = str(get_key('EE_PROJECT_ID'))
    ee.Authenticate()
    ee.Initialize(project=ee_id)
    
def mask_to_ee_geometry(mask):
    '''
    Converts a vector shapefile mask to ee crs (4326) and geometry
    
    Parameters
    ----------
    mask : GeoDataFrame
        Your mask GeoDataFrame (like nc)
        
    Returns
    -------
    aoi : area of interest (mask converted to ee geometry)
    
    Examples
    --------
    >>> nc = gpd.read_file(data_path('processed', 'boundaries', 'nc_boundary.gpkg'))
    >>> aoi = mask_to_ee_crs(nc)
    '''
    mask_ll = mask.to_crs(4326)
    aoi = geemap.gdf_to_ee(mask_ll).geometry()
    return aoi
    

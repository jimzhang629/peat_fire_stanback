"""Helpers for locating and loading data files.

Paths are resolved relative to the project root rather than the current
working directory, so these functions work the same whether they're called
from a notebook in ``notebooks/``, a script, or a test.
"""

import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
import rioxarray

# This file lives at <root>/src/peatfire/preproc/data_loading.py, so the
# project root is four levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"


def load_env() -> None:
    """Load secrets from the project's ``.env`` into ``os.environ``.

    Reads ``<repo>/.env`` regardless of the current working directory, so it
    works the same from a notebook, script, or test. Safe to call more than
    once. Does nothing (rather than erroring) if there is no ``.env`` file --
    keys exported in the shell environment are used as-is.

    Call this once near the top of a notebook, then read individual keys with
    :func:`get_key`. ``.env`` is gitignored; see ``.env.example`` for the
    expected keys.
    """
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")


def get_key(name: str) -> str:
    """Return the value of secret ``name`` from the environment.

    Calls :func:`load_env` first so a notebook only needs this one call.
    Raises a clear error if the key is missing or empty, rather than handing
    back ``None`` to fail confusingly downstream.

    Examples
    --------
    >>> get_key("FIRMS_MAP_KEY")  # doctest: +SKIP
    'abc123...'
    """
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {PROJECT_ROOT / '.env'} "
            f"(see .env.example) or export it in your shell."
        )
    return value

def data_path(*parts: str) -> Path:
    """Return an absolute path inside the ``data/`` directory.

    Examples
    --------
    >>> data_path("raw", "fires.csv")
    PosixPath('.../data/raw/fires.csv')
    """
    return DATA_DIR.joinpath(*parts)


def load_csv(*parts: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from the ``data/`` directory into a DataFrame.

    ``parts`` are joined onto ``data/`` and any extra keyword arguments are
    forwarded to :func:`pandas.read_csv`.

    Examples
    --------
    >>> df = load_csv("raw", "fires.csv")
    """
    return pd.read_csv(data_path(*parts), **kwargs)

def clip_vector_to_mask(vector_path, mask):
    '''
    Clips vector to a mask GeoDataFrame (like nc bounds)

    Parameters
    ----------
    vector_path : Path
        The path to your vector data (will read it in as a GeoDataFrame)
    mask : GeoDataFrame
        Your clipping mask GeoDataFrame
    Returns
    -------
    gdf_clipped : GeoDataFrame
        Your GeoDataFrame clipped to your mask
    '''
    gdf = gpd.read_file(vector_path)
    mask_in_gdf_crs = mask.to_crs(gdf.crs)
    gdf_clipped = gpd.sjoin(gdf, mask_in_gdf_crs[['geometry']], predicate='within').drop(columns='index_right')

    return gdf_clipped

def clip_vector_to_mask_and_save(vector_path, mask, out_path):
    '''
    Clips vector to a mask GeoDataFrame (like nc bounds) and saves it to out_path.

    Parameters
    ----------
    vector_path : Path
        The path to your vector data (will read it in as a GeoDataFrame)
    mask : GeoDataFrame
        Your clipping mask GeoDataFrame
    out_path : Path
        The path to where you want to save the clipped GeoDataFrame

    Returns
    -------
    gdf_clipped : GeoDataFrame
        Your GeoDataFrame clipped to your mask
    '''
    gdf_clipped = clip_vector_to_mask(vector_path, mask)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_clipped.to_file(out_path, driver='GPKG')

    return gdf_clipped

def clip_raster_to_mask(raster_path, mask):
    '''
    Clips a raster (e.g., .tif) to a mask GeoDataFrame (like nc bounds)

    Parameters
    ----------
    raster_path : Path
        The path to your raster data
    mask : GeoDataFrame
        Your clipping mask GeoDataFrame

    Returns
    -------
    raster_clipped : xarray Dataset
        Your raster data clipped to your mask
    '''
    raster = rioxarray.open_rasterio(raster_path, masked=True)
    mask_in_raster_crs = mask.to_crs(raster.rio.crs)
    raster_clipped = raster.rio.clip(mask_in_raster_crs.geometry, mask_in_raster_crs.crs)

    return raster_clipped

def clip_raster_to_mask_and_save(raster_path, mask, out_path):
    '''
    Clips a raster (e.g., .tif) to a mask GeoDataFrame (like nc bounds) and saves it to out_path.

    Parameters
    ----------
    raster_path : Path
        The path to your raster data
    mask : GeoDataFrame
        Your clipping mask GeoDataFrame
    out_path : Path
        The path to where you want to save the clipped raster

    Returns
    -------
    raster_clipped : xarray Dataset
        Your raster data clipped to your mask
    '''

    raster_clipped = clip_raster_to_mask(raster_path, mask)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raster_clipped.rio.to_raster(out_path)

    return raster_clipped

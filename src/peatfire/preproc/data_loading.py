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
import xarray as xr

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

def _as_gdf(vector):
    '''Return a GeoDataFrame from either an in-memory GeoDataFrame/GeoSeries or
    a path to a vector file (read with geopandas.read_file).'''
    if isinstance(vector, (gpd.GeoDataFrame, gpd.GeoSeries)):
        return vector
    return gpd.read_file(vector)

def _as_raster(raster):
    '''Return a rioxarray DataArray/Dataset from either an in-memory xarray
    object or a path to a raster file (read with masked=True).'''
    if isinstance(raster, (xr.DataArray, xr.Dataset)):
        return raster
    return rioxarray.open_rasterio(raster, masked=True)

def read_vector_resilient(path, **kwargs):
    """Read a vector file, working around GeoPackage "readonly database" errors.

    A GeoPackage is a SQLite database. GDAL/pyogrio sometimes fails to *read* one
    with ``attempt to write a readonly database`` when the file (or its directory)
    is not writable, or a stale ``-wal``/``-shm`` lock sidecar is present -- SQLite
    wants to touch the file even for a read. The fix is to read from a **writable
    copy**: this tries a direct read first and, on any failure, copies the file
    (plus any ``-wal``/``-shm`` sidecars) into a fresh temp dir and reads that.

    ``kwargs`` are forwarded to :func:`geopandas.read_file`.
    """
    import shutil
    import tempfile

    try:
        return gpd.read_file(path, **kwargs)
    except Exception:
        src = Path(path)
        tmp = Path(tempfile.mkdtemp(prefix="peatfire_gpkg_"))
        dst = tmp / src.name
        shutil.copy2(src, dst)
        for side in ("-wal", "-shm"):  # SQLite lock/journal sidecars, if any
            s = src.with_name(src.name + side)
            if s.exists():
                shutil.copy2(s, tmp / s.name)
        os.chmod(dst, 0o644)
        return gpd.read_file(dst, **kwargs)


def clip_vector_to_mask(vector, mask):
    '''
    Clips vector to a mask (like nc bounds).

    Parameters
    ----------
    vector : Path or GeoDataFrame
        Your vector data, either a path to a vector file (read in as a
        GeoDataFrame) or an already-loaded GeoDataFrame.
    mask : Path or GeoDataFrame
        Your clipping mask, either a path to a vector file or an already-loaded
        GeoDataFrame.
    Returns
    -------
    gdf_clipped : GeoDataFrame
        Your GeoDataFrame clipped to your mask
    '''
    gdf = _as_gdf(vector)
    mask_in_gdf_crs = _as_gdf(mask).to_crs(gdf.crs)
    gdf_clipped = gpd.clip(gdf, mask_in_gdf_crs)

    return gdf_clipped

def clip_vector_to_mask_and_save(vector, mask, out_path):
    '''
    Clips vector to a mask (like nc bounds) and saves it to out_path.

    Parameters
    ----------
    vector : Path or GeoDataFrame
        Your vector data, either a path to a vector file (read in as a
        GeoDataFrame) or an already-loaded GeoDataFrame.
    mask : Path or GeoDataFrame
        Your clipping mask, either a path to a vector file or an already-loaded
        GeoDataFrame.
    out_path : Path
        The path to where you want to save the clipped GeoDataFrame

    Returns
    -------
    gdf_clipped : GeoDataFrame
        Your GeoDataFrame clipped to your mask
    '''
    gdf_clipped = clip_vector_to_mask(vector, mask)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf_clipped.to_file(out_path, driver='GPKG')

    return gdf_clipped


def clip_raster_to_mask(raster, mask, from_disk=False):
    '''Clips a raster (e.g., .tif) to a mask (like nc bounds).

    Parameters
    ----------
    raster : Path or xarray DataArray/Dataset
        Your raster data, either a path to a raster file (read in with
        masked=True) or an already-loaded rioxarray object.
    mask : Path or GeoDataFrame
        Your clipping mask, either a path to a vector file or an already-loaded
        GeoDataFrame.
    from_disk : bool
        Set from_disk=True for very large rasters (e.g. CONUS-wide) so the
        clip reads only the needed windows from disk instead of loading the
        whole array into memory first -- avoids OOM / kernel crashes.

    Returns
    -------
    raster_clipped : xarray DataArray/Dataset
        Your raster data clipped to your mask
    '''
    raster = _as_raster(raster)
    mask_in_raster_crs = _as_gdf(mask).to_crs(raster.rio.crs)
    raster_clipped = raster.rio.clip(
        mask_in_raster_crs.geometry, mask_in_raster_crs.crs, from_disk=from_disk
    )
    return raster_clipped


def clip_raster_to_mask_and_save(raster, mask, out_path, from_disk=False):
    '''
    Clips a raster (e.g., .tif) to a mask (like nc bounds) and saves it to out_path.

    Parameters
    ----------
    raster : Path or xarray DataArray/Dataset
        Your raster data, either a path to a raster file (read in with
        masked=True) or an already-loaded rioxarray object.
    mask : Path or GeoDataFrame
        Your clipping mask, either a path to a vector file or an already-loaded
        GeoDataFrame.
    out_path : Path
        The path to where you want to save the clipped raster
    from_disk : bool
        Set from_disk=True for very large rasters (e.g. CONUS-wide) so the
        clip reads only the needed windows from disk instead of loading the
        whole array into memory first -- avoids OOM / kernel crashes.

    Returns
    -------
    raster_clipped : xarray DataArray/Dataset
        Your raster data clipped to your mask
    '''
    raster_clipped = clip_raster_to_mask(raster, mask, from_disk=from_disk)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raster_clipped.rio.to_raster(out_path)

    return raster_clipped
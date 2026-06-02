"""Importable helpers for the NC peatland fire project.

Re-exports the data-loading helpers so notebooks and scripts can do::

    from src import data_path, load_csv, DATA_DIR, PROJECT_ROOT
"""

from .data_loading import DATA_DIR, PROJECT_ROOT, data_path, load_csv

__all__ = ["DATA_DIR", "PROJECT_ROOT", "data_path", "load_csv"]

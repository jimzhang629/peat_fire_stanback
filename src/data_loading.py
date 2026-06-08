"""Helpers for locating and loading data files.

Paths are resolved relative to the project root rather than the current
working directory, so these functions work the same whether they're called
from a notebook in ``notebooks/``, a script, or a test.
"""

from pathlib import Path

import pandas as pd

# This file lives at <root>/src/data_loading.py, so the project root is two
# levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


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

# nc-peatland-fire
This project evaluates wildfire risk in North Carolina peatlands and quantifies how peatland restoration influences fire occurrence, severity, and associated greenhouse gas emissions.

## Layout

```
src/peatfire/    importable `peatfire` package (data_loading helpers live here)
data/            raw/, interim/, processed/ (contents git-ignored)
notebooks/       analysis notebooks; see example.ipynb
```

## Setup

Install the project once as an editable package. This puts `peatfire` on the
Python path so notebooks (and scripts/tests) can import it from anywhere,
without `sys.path` hacks or having to launch Jupyter from the repo root:

```bash
pip install -e .
```

If you have an old install of this project, uninstall it first so the import
name updates cleanly: `pip uninstall peat-fire-stanback nc-peatland-fire`.

## Importing from `peatfire` and loading data

After the editable install, import the package and its data helpers from any
notebook regardless of where Jupyter was started:

```python
from peatfire import data_path, load_csv

# Build an absolute path into the data/ folder (resolved relative to the repo
# root, never the current working directory):
data_path("raw", "fires.csv")          # -> <repo>/data/raw/fires.csv

# Or load a CSV straight into a DataFrame (kwargs forwarded to pandas.read_csv):
df = load_csv("raw", "fires.csv")
```

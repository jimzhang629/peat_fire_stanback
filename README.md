# nc-peatland-fire
This project evaluates wildfire risk in North Carolina peatlands and quantifies how peatland restoration influences fire occurrence and burned area.

## Layout

```
src/peatfire/    importable `peatfire` package (data_loading helpers live here)
data/            raw/, interim/, processed/ (contents git-ignored)
notebooks/       analysis notebooks; see example.ipynb
```

## Documentation

Please see `outputs/memos/project_report.docx` for project overview, methods, results, and next steps.

`notebooks/download_and_clip_data.ipynb` downloads raw data (e.g., fire products, covariates) and clips it to North Carolina. 

`notebooks/run_fire_comparison.ipynb` compares fire products against each other.  

`notebooks/validate_against_reference.ipynb` compares fire products against ground-truth reference data.  

`notebooks/modeling.ipynb` runs the logistic regression and DiD modeling pipelines.

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

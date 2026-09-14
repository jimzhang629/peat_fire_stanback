# nc-peatland-fire
This project evaluates wildfire risk in North Carolina peatlands and quantifies how peatland restoration influences fire occurrence and burned area, conditional on covariates.

## Layout

```
data/                                        raw/, interim/, processed/ (contents git-ignored)
metadata/data_inventory.csv                  contains metadata on all used data
                       
src/peatfire/                                importable `peatfire` package (data_loading helpers live here)
src/get_climate&soil_data_updated.R          script from Catherine Chamberlain to download climate/soil data

notebooks/                                   analysis notebooks - see documentation below.

outputs/memos/                               contains project outputs - see documentation below
```

## Notebooks
`notebooks/download_and_clip_data.ipynb` downloads raw data (e.g., fire products, covariates) and clips it to North Carolina. 

`notebooks/run_fire_comparison.ipynb` compares fire products against each other.  

`notebooks/validate_against_reference.ipynb` compares fire products against ground-truth reference data.  

`notebooks/modeling.ipynb` runs the logistic regression and DiD modeling pipelines.

## Memos

`outputs/memos/project_report.pdf` outlines the project overview, methods, results, and next steps.

`outputs/memos/cat_kemen_meetings.pptx` is the raw meeting notes from jim+cat+kemen summer 2026 meetings

`outputs/memos/fire_product_comparison.xlsx` is the theoretical comparison of several potential fire products for nc peat

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

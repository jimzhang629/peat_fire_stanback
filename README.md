# nc-peatland-fire
This project evaluates wildfire risk in North Carolina peatlands and quantifies how peatland restoration influences fire occurrence, severity, and associated greenhouse gas emissions.

## Setup

Install the project as an editable package once, from the repo root:

```bash
pip install -e .
```

This makes the `peatfire` package importable from anywhere — including
notebooks in `notebooks/` — without `sys.path` tweaks or relative `../` paths.

## Loading data

Data paths are resolved relative to the project root, so they work no matter
where the notebook or script runs from:

```python
from peatfire import data_path, load_csv

data_path("raw", "fires.csv")     # -> absolute Path into data/raw/fires.csv
df = load_csv("raw", "fires.csv")  # -> pandas DataFrame
```

## Layout

```
src/peatfire/   importable package (data_loading helpers live here)
data/           raw/, interim/, processed/ (contents git-ignored)
notebooks/      analysis notebooks; see example.ipynb
```

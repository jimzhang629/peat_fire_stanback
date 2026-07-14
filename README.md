# nc-peatland-fire
This project evaluates wildfire risk in North Carolina peatlands and quantifies how peatland restoration influences fire occurrence, severity, and associated greenhouse gas emissions.

## Layout

```
src/peatfire/    importable `peatfire` package (data_loading helpers live here)
data/            raw/, interim/, processed/ (contents git-ignored)
notebooks/       analysis notebooks; see example.ipynb
```

## Documentation

`modeling_notebook_explained.md` is the master guide: a layered walkthrough of
`notebooks/modeling.ipynb`, the math and statistics behind every step
(matching, propensity/prognostic scores, the staggered DiD / ATT, odds ratios),
the design-decision log, and the fire-product comparison & validation toolkit
decisions (Appendix A). It replaces the former `modeling_roadmap.md`,
`decisions.md`, `score_matching_and_did.md`, and `matching_assignment.md`.

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

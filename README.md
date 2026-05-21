# leapfrog-compare

Interactive tool for comparing Spectrum extract output with leapfrog-goals model output across a set of PJNZ files.

---

## Overview

The workflow has three steps:

```
1. run_extract.py       – runs Spectrum /ExtractBatch on your PJNZ folder → XLSX
2. run_goals_model.py   – runs leapfrog goals on each PJNZ → .npz per file
3. app.py               – launches an interactive Shiny dashboard to compare them
```

---

## Prerequisites

### 1. Python ≥ 3.10 and uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

### 2. Spectrum desktop application (for step 1 only)

Spectrum must be installed and its executable must be on your `PATH`.

### 3. leapfrog-goals Python package (for step 2 and the dashboard)

The `leapfrog-goals` package contains compiled C++ extensions and must be installed separately.

```bash
# From this project directory, point uv at your local clone of leapfrog/goals e.g.:
uv pip install ../leapfrog/goals

# Or using a full path:
uv pip install /path/to/leapfrog/goals
```

> **If you are on a different machine**, clone the leapfrog repository and adjust `leapfrog-goals` in `pyproject.toml` to match your local path.

---

## Setup

### 1. Install Python dependencies

```bash
uv sync
```

### 2. Edit `config.py`

Open `config.py` and adjust the paths for your machine:

| Variable | Description |
|---|---|
| `PJNZ_DIR` | Folder containing your `.PJNZ` files |
| `EXTRACT_CONFIG` | Path to the `.EX` extract configuration file |
| `EXTRACT_OUTPUT_DIR` | Where Spectrum extract output will be written |
| `GOALS_OUTPUT_DIR` | Where goals model output will be written |
| `SPECTRUM_EXE` | (Optional) Full path to Spectrum if not on `PATH` |

A minimal example:

```python
PJNZ_DIR           = Path("/home/user/data/pjnz_files")
EXTRACT_CONFIG     = Path("goals_extract_config.EX")
```

### 3. Create the output directories (optional — created automatically)

```bash
mkdir -p output/extract output/goals
```

---

## Usage

### Step 1 — Run Spectrum extract

```bash
uv run scripts/run_extract.py
```

This calls `spectrum /ExtractBatch` on every `.PJNZ` in `PJNZ_DIR` using the `.EX` config file and writes the output XLSX to `EXTRACT_OUTPUT_DIR`.

### Step 2 — Run leapfrog goals model

```bash
uv run scripts/run_goals_model.py
```

For each `.PJNZ` file this:
1. Reads the PJNZ using the Python PJNZ reader from `leapfrog/goals/src/_spectrum`
2. Converts parameters using `leapfrog_mapping.modvars_to_leapfrog`
3. Runs `leapfrog_goals.run_goals`
4. Saves the result to `output/goals/<pjnz_stem>.npz`

If a file fails, the error is printed and processing continues.

### Step 3 — Launch the dashboard

```bash
uv run shiny run app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

The dashboard will show:
- A **PJNZ dropdown** — select which projection to compare
- An **Indicators multi-select** — choose one or more indicators; the plot facets by indicator
- Each facet shows the **Spectrum (extract)** line and the **Leapfrog (goals)** line on the same axis

---

## Project structure

```
leapfrog-compare/
├── app.py                        # Shiny for Python dashboard
├── config.py                     # User configuration (paths, etc.)
├── pyproject.toml                # uv/pip project metadata
├── goals_extract_config.EX       # Spectrum extract config
├── scripts/
│   ├── run_extract.py            # Step 1: run Spectrum extract
│   └── run_goals_model.py        # Step 2: run leapfrog goals model
└── src/
    └── leapfrog_compare/
        ├── spectrum.py           # Spectrum CLI wrapper
        ├── pjnz_runner.py        # PJNZ loading + goals execution
        ├── extract_reader.py     # XLSX extract parser
        └── indicator_map.py      # Extract ↔ goals indicator mapping
```

---

## Adding new indicators

Edit `src/leapfrog_compare/indicator_map.py`.

1. Write a `compute` function `(output: dict[str, np.ndarray]) -> np.ndarray` that returns a 1-D array per output year.
2. Add an `IndicatorDef` to `INDICATOR_MAP` at the end of the `OrderedDict`.

Helper factories are available:
- `_sum(key)` — sum over all non-time dimensions of a goals array
- `_direct(key, scale)` — use a 1-D goals array directly (with optional scaling)
- `_age_slice_sum(key, age_start, age_end)` — sum over a slice of single-year age groups

Example — adding "Total non-AIDS deaths to HIV population":

```python
("Total non-AIDS deaths to HIV population", IndicatorDef(
    extract_name="Total non-AIDS deaths to HIV population",
    compute=_sum("p_deaths_background_hivpop"),
)),
```

---

## Troubleshooting

**`leapfrog_goals` not found**
Re-run `uv pip install /path/to/leapfrog/goals`. Check that a compiled `.pyd` (Windows) or `.so` (Linux/Mac) file is present in the installed package.

**`spectrum not found in PATH`**
Either add the Spectrum installation directory to `PATH`, or set `SPECTRUM_EXE = Path("C:/...")` in `config.py`.

**Dashboard shows "Extract: not loaded"**
Check that `EXTRACT_OUTPUT_DIR` contains an `.xlsx` file and that it was written without errors by `run_extract.py`.

**Goals output not appearing**
Check that `GOALS_OUTPUT_DIR` contains `.npz` files matching the PJNZ stems. Run `scripts/run_goals_model.py` again and look for errors.

# leapfrog-compare

Interactive dashboard for comparing Spectrum modvar output against leapfrog-goals model output across a set of PJNZ files.

---

## Overview

The app loads each PJNZ file directly, runs the leapfrog Goals model, and plots both the Goals output and the corresponding Spectrum modvar time series side by side.

```
1. Drop .PJNZ files into PJNZ_DIR
2. Edit config.py to point at that directory
3. uv run shiny run app.py
```

---

## Prerequisites

### 1. Python ≥ 3.10 and uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

### 2. leapfrog-goals Python package

The `leapfrog-goals` package contains compiled C++ extensions and must be installed separately.

```bash
# Point uv at your local clone of leapfrog/goals, e.g.:
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

### 2. Edit `src/leapfrog_compare/config.py`

Set `PJNZ_DIR` to the folder containing your `.PJNZ` files:

```python
PJNZ_DIR = Path("/home/user/data/pjnz_files")
```

### 3. Launch the dashboard

```bash
uv run shiny run app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## Dashboard controls

- **PJNZ** — select which projection to compare
- **Indicators** — choose one or more indicators; the plot facets by indicator
- **Year range** — slider set automatically from the PJNZ projection years
- **Disaggregation** — optionally break down by five-year age group and/or sex
- Each facet shows a **Leapfrog** (solid) line and a **Spectrum** (dashed) line in the same colour

---

## Project structure

```
leapfrog-compare/
├── app.py                        # Shiny for Python dashboard
├── pyproject.toml                # uv/pip project metadata
└── src/
    └── leapfrog_compare/
        ├── config.py             # User configuration (PJNZ_DIR)
        ├── pjnz_runner.py        # PJNZ loading + Goals model execution
        └── indicator_map.py      # Goals output ↔ Spectrum modvar mapping
```

---

## Indicator reference

For each indicator: the leapfrog Goals output array used, its dimensions, the Spectrum modvar tag, and the modvar dimensions.

> **Leapfrog array dimensions** — most population-like outputs share the shape `(81, 2, n_years)`:
> - axis 0 = single-year ages 0–80
> - axis 1 = sex (0 = male, 1 = female)
> - axis 2 = projection years

---

### Total population

| | Detail |
|---|---|
| **Leapfrog output** | `p_totpop` |
| **Leapfrog shape** | `(81, 2, n_years)` — ages × sex × years |
| **Leapfrog aggregation** | Sum over all ages and both sexes |
| **Spectrum modvar** | `DP_BigPop_V1` |
| **Spectrum PJNZ tag** | `<BigPop MV3>` |
| **Spectrum shape** | `(3, 81, 81)` — sex × age × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female) summed over all 81 age columns |

---

### Total Births - todo

| | Detail |
|---|---|
| **Leapfrog output** | `births` |
| **Leapfrog shape** | `(n_years,)` — total births per year |
| **Leapfrog aggregation** | Used directly (no aggregation) |
| **Spectrum modvar** | None - births is not read in yet |

---

### HIV population

| | Detail |
|---|---|
| **Leapfrog output** | `p_hivpop` |
| **Leapfrog shape** | `(81, 2, n_years)` — ages × sex × years |
| **Leapfrog aggregation** | Sum over all ages and both sexes |
| **Spectrum modvar** | `HV_TotalAdultsHIV_V1`|
| **Spectrum PJNZ tag** | `<TotalAdultsHIV MV>` |
| **Spectrum shape** | `(3, 81)` — sex × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female); no age aggregation (shape is sex × year, already age-aggregated) |

---

### New HIV infections

| | Detail |
|---|---|
| **Leapfrog output** | `p_infections` |
| **Leapfrog shape** | `(81, 2, n_years)` — ages × sex × years |
| **Leapfrog aggregation** | Sum over all ages and both sexes |
| **Spectrum modvar** | `HV_NewInfections_V1` |
| **Spectrum PJNZ tag** | `<NewInfections MV>` |
| **Spectrum shape** | `(3, 11, 5, 81)` — sex × risk group × vaccine state × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female) summed over all risk-group and vaccine-state dimensions |

Sex disaggregation sums the male (row 1) and female (row 2) sub-arrays over risk group and vaccine state. Age disaggregation is not available for this modvar.

---

### AIDS deaths

| | Detail |
|---|---|
| **Leapfrog output** | `p_hiv_deaths` |
| **Leapfrog shape** | `(81, 2, n_years)` — ages × sex × years |
| **Leapfrog aggregation** | Sum over all ages and both sexes |
| **Spectrum modvar** | `HV_AIDSDeaths_V1` |
| **Spectrum PJNZ tag** | `<AIDSDeaths MV>` |
| **Spectrum shape** | `(3, 11, 5, 81)` — sex × risk group × vaccine state × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female) summed over all risk-group and vaccine-state dimensions |

Same sex-disaggregation behaviour as New HIV infections. No per-age-group Spectrum line.

---

### Total number receiving ART (15-49)

| | Detail |
|---|---|
| **Leapfrog output** | `h_artpop` |
| **Leapfrog shape** | `(3, 7, 66, 2, n_years)` — ART duration stage × CD4 stage × single ages 15–49 × sex × years |
| **Leapfrog aggregation** | Sum over all ART duration stages, CD4 stages, ages, and both sexes |
| **Spectrum modvar** | `HV_TotalAdultsART_V1` |
| **Spectrum PJNZ tag** | `<TotalAdultsART MV>` |
| **Spectrum shape** | `(3, 81)` — sex × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female); no age aggregation (shape is sex × year, already age-aggregated) |

ART duration stages: 0 = <6 months, 1 = 6–12 months, 2 = >12 months.
Age disaggregation on the Goals side sums `h_artpop` over the appropriate single-year age slice; under-15 groups return zero (no adult ART data below age 15). Age disaggregation is not available for the Spectrum modvar.

---

### Prevalence (15–49)

| | Detail |
|---|---|
| **Leapfrog outputs** | `p_hivpop`, `p_totpop` |
| **Leapfrog shape** | Both `(81, 2, n_years)`; sliced to ages 15–49 before computing |
| **Leapfrog computation** | `100 × sum(p_hivpop[15:50]) / sum(p_totpop[15:50])` |
| **Spectrum modvar** | None — `HV_CalcPrevalence_V1` is all zeros in the current test files |
| **Spectrum PJNZ tag** | `<CalcPrevalence MV>` |
| **Spectrum shape** | `(3, 11, 81)` — sex × risk group × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female) summed over all 11 risk-group columns |

---

### Incidence (15–49) (Percent)

| | Detail |
|---|---|
| **Leapfrog outputs** | `p_infections`, `p_hivpop`, `p_totpop` |
| **Leapfrog shape** | All `(81, 2, n_years)`; sliced to ages 15–49 |
| **Leapfrog computation** | `100 × sum(p_infections[15:50]) / (sum(p_totpop[15:50]) − sum(p_hivpop[15:50]))` |
| **Spectrum modvar** | `HV_Incidence_V1` |
| **Spectrum PJNZ tag** | `<Incidence MV>` |
| **Spectrum shape** | `(81,)` — one rate per projection year |
| **Spectrum aggregation** | Multiplied by 100 to convert to percent; no sex or age disaggregation available |

---

## Adding new indicators

Edit [src/leapfrog_compare/indicator_map.py](src/leapfrog_compare/indicator_map.py).

1. Write a `compute_goals` function `(output: dict) -> np.ndarray` returning a 1-D array over years.
2. Write a `compute_goals_disagg` function for age/sex breakdown, or use `_disagg_std(key)` for standard `(81, 2, n_years)` arrays.
3. Optionally write `compute_spectrum` and `compute_spectrum_disagg` functions if a matching modvar exists.
4. Add an `IndicatorDef` entry to `INDICATOR_MAP`.

---

## Troubleshooting

**`leapfrog_goals` not found**
Re-run `uv pip install /path/to/leapfrog/goals`. Check that a compiled `.so` (Linux/Mac) or `.pyd` (Windows) file is present in the installed package.

**Dashboard shows "No PJNZ files found"**
Check that `PJNZ_DIR` in `config.py` points to a directory containing `.PJNZ` files (case-sensitive extension).

**A country's data looks truncated or goes NaN**
The Goals model may produce NaN values for some countries if a parameter causes a numerical instability. Check the terminal output for errors when the PJNZ is loaded.

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

### 1. Python ≥ 3.11 and uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you haven't already.

---

## Setup

### 1. Install Python dependencies

```bash
uv sync
```

If you want to use an in development leapfrog-goals you will need to check it out locally and then update the `tools.uv.sources` section in `pyproject.toml` to point to your local path.

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

Indicators are split into two groups:

- **All-ages** — compare Leapfrog DP/AIM vs Spectrum across all age groups; support age and sex disaggregation.
- **15–49** — compare all three sources restricted to the 15–49 population; **age disaggregation is disabled** for these (Goals outputs are pre-aggregated over 15–49).

Line styles:
- **Leapfrog DP/AIM** (solid) — from `run_goals` DP/AIM-derived arrays
- **Spectrum** (dashed) — read directly from PJNZ modvars
- **Leapfrog Goals** (dotted) — Goals-native outputs; 15–49 indicators only; hidden in age-faceted view

---

### Total population *(all ages)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM output** | `p_totpop` |
| **Leapfrog DP/AIM shape** | `(81, 2, n_years)` — ages × sex × years |
| **Leapfrog DP/AIM aggregation** | Sum over all ages and both sexes |
| **Spectrum modvar** | `DP_BigPop_V1` |
| **Spectrum PJNZ tag** | `<BigPop MV3>` |
| **Spectrum shape** | `(3, 81, 81)` — sex × age × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female) summed over all 81 age columns |

---

### Total Births *(all ages)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM output** | `births` |
| **Leapfrog DP/AIM shape** | `(n_years,)` — total births per year |
| **Leapfrog DP/AIM aggregation** | Used directly (no aggregation) |
| **Spectrum modvar** | None — births is not read in yet |

---

### HIV population *(all ages)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM output** | `p_hivpop` |
| **Leapfrog DP/AIM shape** | `(81, 2, n_years)` — ages × sex × years |
| **Leapfrog DP/AIM aggregation** | Sum over all ages and both sexes |
| **Spectrum modvar** | `HV_TotalAdultsHIV_V1` |
| **Spectrum PJNZ tag** | `<TotalAdultsHIV MV>` |
| **Spectrum shape** | `(3, 81)` — sex × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female); already age-aggregated |

---

### New HIV infections *(all ages)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM output** | `p_infections` |
| **Leapfrog DP/AIM shape** | `(81, 2, n_years)` — ages × sex × years |
| **Leapfrog DP/AIM aggregation** | Sum over all ages and both sexes |
| **Spectrum modvar** | `HV_NewInfections_V1` |
| **Spectrum PJNZ tag** | `<NewInfections MV>` |
| **Spectrum shape** | `(3, 11, 5, 81)` — sex × risk group × vaccine state × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female) summed over risk-group and vaccine-state dimensions |

Sex disaggregation sums rows 1 and 2 over inner dims. Age disaggregation not available for Spectrum.

---

### AIDS deaths *(all ages)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM output** | `p_hiv_deaths` |
| **Leapfrog DP/AIM shape** | `(81, 2, n_years)` — ages × sex × years |
| **Leapfrog DP/AIM aggregation** | Sum over all ages and both sexes |
| **Spectrum modvar** | `HV_AIDSDeaths_V1` |
| **Spectrum PJNZ tag** | `<AIDSDeaths MV>` |
| **Spectrum shape** | `(3, 11, 5, 81)` — sex × risk group × vaccine state × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female) summed over risk-group and vaccine-state dimensions |

Same sex-disaggregation behaviour as New HIV infections. Age disaggregation not available for Spectrum.

---

### Total number receiving ART (15-49) *(no Goals)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM output** | `h_artpop` |
| **Leapfrog DP/AIM shape** | `(3, 7, 66, 2, n_years)` — ART duration stage × CD4 stage × single ages 15–80 × sex × years |
| **Leapfrog DP/AIM aggregation** | Sum over all ART duration stages, CD4 stages, ages 15–49 (first 35 age indices), and both sexes |
| **Spectrum modvar** | `HV_TotalAdultsART_V1` |
| **Spectrum PJNZ tag** | `<TotalAdultsART MV>` |
| **Spectrum shape** | `(3, 81)` — sex × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female); already age-aggregated |

ART duration stages: 0 = <6 months, 1 = 6–12 months, 2 = >12 months. Under-15 age groups return zero on the DP/AIM side. Age disaggregation not available for Spectrum.

---

### Total population 15–49 *(15–49; age disagg disabled)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM** | `p_totpop` summed over ages 15–49 |
| **Spectrum** | `DP_BigPop_V1` rows 1+2 summed over age columns 15–49 |
| **Leapfrog Goals output** | `total_population` |
| **Leapfrog Goals shape** | `(n_years,)` — pre-computed 15–49 total; no sex disaggregation |

---

### New HIV infections 15–49 *(15–49; age disagg disabled)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM** | `p_infections` summed over ages 15–49 |
| **Spectrum** | `HV_NewInfections_V1` (all-ages modvar; no 15–49 slice available) |
| **Leapfrog Goals output** | `new_infections_goals` |
| **Leapfrog Goals shape** | `(3, n_years)` — sex × year; index 0 = male, 1 = female, 2 = both |
| **Leapfrog Goals aggregation** | Index 2 (both) for total; indices 0 and 1 for sex disaggregation |

---

### AIDS deaths 15–49 *(15–49; age disagg disabled)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM** | `p_hiv_deaths` summed over ages 15–49 |
| **Spectrum** | `HV_AIDSDeaths_V1` (all-ages modvar; no 15–49 slice available) |
| **Leapfrog Goals output** | `total_deaths_hiv` |
| **Leapfrog Goals shape** | `(n_years,)` — scalar total; no sex disaggregation (always shown as single total) |

---

### Prevalence (15–49) (%) *(15–49; age disagg disabled)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM** | `100 × sum(p_hivpop[15:50]) / sum(p_totpop[15:50])` |
| **Spectrum modvar** | `HV_CalcPrevalence_V1` |
| **Spectrum PJNZ tag** | `<CalcPrevalence MV>` |
| **Spectrum shape** | `(3, 11, 81)` — sex × risk group × year; index 0 = both, 1 = male, 2 = female |
| **Spectrum aggregation** | Rows 1 (male) + 2 (female) summed over all 11 risk-group columns |
| **Leapfrog Goals output** | `prevalence` × 100 |
| **Leapfrog Goals shape** | `(18, 3, n_years)` — risk group × sex × year |
| **Leapfrog Goals aggregation** | Last RG-aggregate row; sex index 0 = male, 1 = female, 2 = both |

---

### Incidence (15–49) (%) *(15–49; age disagg disabled)*

| | Detail |
|---|---|
| **Leapfrog DP/AIM** | `100 × sum(p_infections[15:50]) / (sum(p_totpop[15:50]) − sum(p_hivpop[15:50]))` |
| **Spectrum modvar** | `HV_Incidence_V1` × 100 |
| **Spectrum PJNZ tag** | `<Incidence MV>` |
| **Spectrum shape** | `(81,)` — one rate per projection year; no sex or age disaggregation |
| **Leapfrog Goals output** | `incidence_goals` × 100 |
| **Leapfrog Goals shape** | `(3, n_years)` — sex × year; index 0 = male, 1 = female, 2 = both |
| **Leapfrog Goals aggregation** | Index 2 (both) for total; indices 0 and 1 for sex disaggregation |

---

## Adding new indicators

Edit [src/leapfrog_compare/indicator_map.py](src/leapfrog_compare/indicator_map.py).

1. Write a `compute_goals` function `(output: dict) -> np.ndarray` returning a 1-D array over years.
2. Write a `compute_goals_disagg` function. Use `_disagg_std(key)` for all-ages arrays, `_disagg_std_1549(key)` for 15–49 restricted (returns `[]` when age disagg is on), or `_no_age_disagg(fn)` to disable age disagg on any existing function.
3. Optionally write `compute_spectrum` and `compute_spectrum_disagg` functions if a matching modvar exists.
4. Optionally write a `compute_leapfrog_goals_disagg` function `(goals_output, disagg_sex) -> list[(label, values)]` for Goals-native outputs.
5. Add an `IndicatorDef` entry to `INDICATOR_MAP`.

---

## Troubleshooting

**`leapfrog_goals` not found**
Re-run `uv pip install /path/to/leapfrog/goals`. Check that a compiled `.so` (Linux/Mac) or `.pyd` (Windows) file is present in the installed package.

**Dashboard shows "No PJNZ files found"**
Check that `PJNZ_DIR` in `config.py` points to a directory containing `.PJNZ` files (case-sensitive extension).

**A country's data looks truncated or goes NaN**
The Goals model may produce NaN values for some countries if a parameter causes a numerical instability. Check the terminal output for errors when the PJNZ is loaded.

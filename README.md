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

Global controls (sidebar):
- **PJNZ** — select which projection to compare
- **Year range** — slider set automatically from the PJNZ projection years

The main area has three tabs:

| Tab | Indicators | Disaggregation |
|---|---|---|
| **15-49** | 15–49 aggregate indicators; all three sources | By sex |
| **All ages** | All-ages indicators; Leapfrog DP/AIM vs Spectrum | By age group and/or sex |
| **Risk groups** | Fixed 5-panel risk-group fraction plot | By sex |

Line styles:
- **Leapfrog DP/AIM** (solid) — from `run_goals` DP/AIM-derived arrays
- **Spectrum** (dashed) — read directly from PJNZ modvars
- **Leapfrog Goals** (dotted) — Goals-native outputs; 15–49 tab only

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

> **Leapfrog array dimensions** — most population-like outputs share the shape `(81, 2, n_years)`:
> - axis 0 = single-year ages 0–80
> - axis 1 = sex (0 = male, 1 = female)
> - axis 2 = projection years

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
| **Spectrum modvar** | `HV_TotalAdultsHIVTag` and `DP_BigPopTag` |
| **Spectrum PJNZ tag** | `<TotalAdultsHIV MV>` and `<BigPop MV3>` |
| **Spectrum shape** | `(3, n_years)` and `(3, n_years)` 3 is n sexes both, male, female |
| **Spectrum aggregation** | Sum male and female total hiv / big pop |
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

### Total PLHIV (15-49)

| | Detail |
|---|---|
| **Leapfrog DP/AIM** | `p_hivpop` limited to 15-49 |
| **Spectrum modvar** | `HV_TotalAdultsHIVTag` |
| **Spectrum PJNZ tag** | `<TotalAdultsHIV MV>` |
| **Spectrum shape** | (3, n_years) |
| **Leapfrog Goals output** | `total_plhiv` |
| **Leapfrog Goals shape** | (n_years) |
| **Leapfrog Goals aggregation** | None needed, already aggregated |

---

### Total on ART (15-49)

| | Detail |
|---|---|
| **Leapfrog DP/AIM** | `total_on_art` summed over ages 15–49 |
| **Spectrum modvar** | `HV_TotalAdultsART_V1` |
| **Spectrum PJNZ tag** | `<TotalAdultsART MV>` |
| **Spectrum shape** | `(3, 81)` — sex × year; index 0 = both, 1 = male, 2 = female |
| **Leapfrog Goals output** | none available |

---

### Risk groups *(risk groups tab)*

Five fixed subplots showing the fraction of the 15–49 population in each risk group over time, multiplied by 100 to give percent. Computed independently for Goals and Spectrum.

**Goals** — `adults` array, shape `(nVAC+1=5, nRG+1=18, nCD4+1=17, nNS+1=3, n_years)`, indexed as `adults[VAC_ALL, rg_idx, CD4_ALL, sex_idx]`:

| Constant | Value | Meaning |
|---|---|---|
| `VAC_ALL` | 4 | All vaccination states |
| `CD4_ALL` | 16 | All CD4 stages |
| `RG_ALL` | 17 | Total across all risk groups (denominator) |
| sex_idx | 0 = male, 1 = female | |

| Risk group | Goals `rg_idx` |
|---|---|
| Low risk | 1 (RG_LRH) |
| Medium risk | 2 (RG_MRH) |
| High risk | 3 (RG_HRH) |
| PWID | 4 (RG_IDU) |
| MSM | 5 (RG_MSM) |

**Spectrum** — `HV_Adults_V1`, shape `(sex, rg, hiv, vac, n_years)`, indexed as `hv_adults[sex_idx, rg_idx, HV_AllHIV, RN_AllVacc, :]`:

| Constant | Value | Source |
|---|---|---|
| `HV_AllHIV` | 19 | `SpectrumCommon.Const.HV.HVConst` |
| `HV_AllRisk` | 0 | `SpectrumCommon.Const.HV.HVConst` (denominator) |
| `RN_AllVacc` | 0 | `SpectrumCommon.Const.RN.RNConst` |
| sex_idx | 1 = male, 2 = female | (index 0 = both, not used) |

| Risk group | Spectrum `rg_idx` | Constant |
|---|---|---|
| Low risk | 2 | `HV_LRH` |
| Medium risk | 3 | `HV_MRH` |
| High risk | 4 | `HV_HRH` |
| PWID | 5 | `HV_IDU` |
| MSM | 6 | `HV_MSM` |

When sex disaggregation is off, male (index 1) + female (index 2) are summed in both numerator and denominator before dividing.

## Adding new indicators

Edit [src/leapfrog_compare/indicator_map.py](src/leapfrog_compare/indicator_map.py). There are two indicator types depending on which tab the indicator belongs to:

**All-ages tab** — add an `AllAgesIndicatorDef` to `ALL_AGES_INDICATORS`:
```python
AllAgesIndicatorDef(
    compute_leapfrog=...,   # (output, disagg_age, disagg_sex) -> list[(label, ndarray)]
    compute_spectrum=...,   # (modvars, disagg_age, disagg_sex) -> list[(label, ndarray)] | None
)
```
Use `_disagg_std(key)` or `_disagg_art()` for the Leapfrog function.

**15–49 tab** — add an `Indicator1549Def` to `INDICATORS_1549`:
```python
Indicator1549Def(
    compute_leapfrog=...,   # (output, disagg_sex) -> list[(label, ndarray)]
    compute_spectrum=...,   # (modvars, disagg_sex) -> list[(label, ndarray)] | None
    compute_goals=...,      # (goals_output, disagg_sex) -> list[(label, ndarray)] | None
)
```
Use `_lf_1549(key)` for standard (81, 2, n_years) Goals arrays restricted to ages 15–49.

Label convention: use `"Total"` for all-ages unsplit series, `"15-49"` for 15–49 unsplit series, `"Male"` / `"Female"` for sex-disaggregated series.

---

## Troubleshooting

**`leapfrog_goals` not found**
Re-run `uv pip install /path/to/leapfrog/goals`. Check that a compiled `.so` (Linux/Mac) or `.pyd` (Windows) file is present in the installed package.

**Dashboard shows "No PJNZ files found"**
Check that `PJNZ_DIR` in `config.py` points to a directory containing `.PJNZ` files (case-sensitive extension).

**A country's data looks truncated or goes NaN**
The Goals model may produce NaN values for some countries if a parameter causes a numerical instability. Check the terminal output for errors when the PJNZ is loaded.

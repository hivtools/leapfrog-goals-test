"""
Indicator definitions: compute time series from Goals output and Spectrum modvars.

Each IndicatorDef provides:
  - compute_leapfrog_aim(output)            -> 1-D ndarray (total, all ages+sexes)
  - compute_leapfrog_aim_disagg(output, disagg_age, disagg_sex)
                                     -> list[(label, 1-D ndarray)]
  - compute_spectrum(modvars)        -> 1-D ndarray  |  None if unavailable
  - compute_spectrum_disagg(modvars, disagg_age, disagg_sex)
                                     -> list[(label, 1-D ndarray)]  |  None

Goals arrays for population-like indicators have shape (n_ages, 2, n_years):
  axis 0 = single-year ages 0-80
  axis 1 = sex (0=male, 1=female)
  axis 2 = years

h_artpop shape is (4, 7, 66, 2, n_years):
  axis 2 = adult single-year ages 15-80
  axis 3 = sex (0=male, 1=female)

Spectrum modvars shapes (confirmed):
  DP_BigPop_V1              (3, 81, 81)    sex × age × year;  [0]=both, [1]=male, [2]=female
  HV_NewInfections_V1       (3, K, M, 81)  sex × risk_grp × vaccine_state × year; [0]=both, [1]=male, [2]=female
  HV_AIDSDeaths_V1          same structure as HV_NewInfections_V1
  HV_TotalAdultsHIV_V1      (3, 81)        sex × year;  [0]=both, [1]=male, [2]=female
  HV_TotalAdultsART_V1      (3, 81)        sex × year;  [0]=both, [1]=male, [2]=female
  HV_CalcPrevalence_V1      (3, 11, 81)    sex × risk_grp × year; [0]=both, [1]=male, [2]=female
  HV_Incidence_V1           (81,)          rate per year (×100 → percent) — no disaggregation

For all modvars with a sex first-dimension, index 0 ("both") is a pre-computed total that we
do NOT use. Totals are always produced by manually summing indices 1 (male) + 2 (female).
"""

from __future__ import annotations

import leapfrog_compare.config  # noqa: F401 — ensures SpectrumCommon is on sys.path
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Callable

import numpy as np

from SpectrumCommon.Const.DP.DPTags import DP_BigPopTag  # type: ignore[import-untyped]
from SpectrumCommon.Const.HV.HVTags import (  # type: ignore[import-untyped]
    HV_AIDSDeathsTag,
    HV_CalcPrevalenceTag,
    HV_IncidenceTag,
    HV_NewInfectionsTag,
    HV_TotalAdultsARTTag,
    HV_TotalAdultsHIVTag,
)


# ---------------------------------------------------------------------------
# Age / sex group definitions for disaggregation
# ---------------------------------------------------------------------------

# 17 five-year age groups, 0-4 through 80+
AGE_GROUPS: list[tuple[int, int]] = [(i * 5, min(i * 5 + 4, 80)) for i in range(17)]
AGE_LABELS: list[str] = [
    f"{a}-{b}" if b < 80 else "80+" for a, b in AGE_GROUPS
]
SEX_LABELS = ["Male", "Female"]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _sum_std(arr: np.ndarray, age_slice: slice | None = None, sex: int | None = None) -> np.ndarray:
    """
    Sum a (n_ages, 2, n_years) array over age and/or sex.

    If age_slice is given, first restrict to that age range.
    If sex is given, take only that sex index.
    Returns a 1-D array of shape (n_years,).
    """
    if age_slice is not None:
        arr = arr[age_slice, :, :]
    if sex is not None:
        arr = arr[:, sex : sex + 1, :]
    return arr.reshape(-1, arr.shape[-1]).sum(axis=0)


def _disagg_std(key: str) -> Callable:
    """
    Return a disaggregation function for a Goals array with shape (81, 2, n_years).
    """
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = output[key]
        series: list[tuple[str, np.ndarray]] = []

        age_specs: list[tuple[str | None, slice | None]] = (
            [(label, slice(a, b + 1)) for (a, b), label in zip(AGE_GROUPS, AGE_LABELS)]
            if disagg_age else [(None, None)]
        )
        sex_specs: list[tuple[str | None, int | None]] = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )

        for age_label, age_sl in age_specs:
            for sex_label, sex_idx in sex_specs:
                data = _sum_std(arr, age_sl, sex_idx)
                parts = [p for p in [age_label, sex_label] if p]
                label = " / ".join(parts) if parts else "Total"
                series.append((label, data))
        return series
    return fn


def _disagg_art() -> Callable:
    """
    Disaggregation for h_artpop (4, 7, 66, 2, n_years): adult ages 15-80, sex axis=3.
    Age axis index 0 = age 15, index i = age 15+i.
    Under-15 age groups return zeros (no adult ART data below age 15).
    """
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = output["h_artpop"]  # (4, 7, 66, 2, n_years)
        n_years = arr.shape[-1]
        series: list[tuple[str, np.ndarray]] = []

        if disagg_age:
            age_items: list[tuple[str, slice | None]] = []
            for (a, b), lbl in zip(AGE_GROUPS, AGE_LABELS):
                if b < 15:
                    age_items.append((lbl, None))  # under-15: no data in h_artpop
                else:
                    art_start = max(0, a - 15)
                    art_end = min(65, b - 15) + 1
                    age_items.append((lbl, slice(art_start, art_end)))
        else:
            age_items = [(None, slice(None))]

        sex_items: list[tuple[str | None, int | None]] = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )

        for age_lbl, age_sl in age_items:
            for sex_lbl, sex_idx in sex_items:
                if age_sl is None:
                    data = np.zeros(n_years)
                elif sex_idx is not None:
                    data = arr[:, :, age_sl, sex_idx, :].reshape(-1, n_years).sum(axis=0)
                else:
                    data = arr[:, :, age_sl, :, :].reshape(-1, n_years).sum(axis=0)
                parts = [p for p in [age_lbl, sex_lbl] if p]
                label = " / ".join(parts) if parts else "Total"
                series.append((label, data))
        return series
    return fn


def _disagg_prevalence() -> Callable:
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        hiv = output["p_hivpop"]
        tot = output["p_totpop"]
        series: list[tuple[str, np.ndarray]] = []

        age_specs = (
            [(label, slice(a, b + 1)) for (a, b), label in zip(AGE_GROUPS, AGE_LABELS)]
            if disagg_age else [(None, None)]
        )
        sex_specs = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )

        for age_label, age_sl in age_specs:
            for sex_label, sex_idx in sex_specs:
                h = _sum_std(hiv, age_sl, sex_idx)
                t = _sum_std(tot, age_sl, sex_idx)
                data = 100.0 * h / np.where(t == 0, np.nan, t)
                parts = [p for p in [age_label, sex_label] if p]
                label = " / ".join(parts) if parts else "15-49"
                series.append((label, data))
        return series
    return fn


def _disagg_incidence() -> Callable:
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        inf = output["p_infections"]
        hiv = output["p_hivpop"]
        tot = output["p_totpop"]
        series: list[tuple[str, np.ndarray]] = []

        age_specs = (
            [(label, slice(a, b + 1)) for (a, b), label in zip(AGE_GROUPS, AGE_LABELS)]
            if disagg_age else [(None, None)]
        )
        sex_specs = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )

        for age_label, age_sl in age_specs:
            for sex_label, sex_idx in sex_specs:
                i_ = _sum_std(inf, age_sl, sex_idx)
                h = _sum_std(hiv, age_sl, sex_idx)
                t = _sum_std(tot, age_sl, sex_idx)
                hivneg = t - h
                data = 100.0 * i_ / np.where(hivneg == 0, np.nan, hivneg)
                parts = [p for p in [age_label, sex_label] if p]
                label = " / ".join(parts) if parts else "15-49"
                series.append((label, data))
        return series
    return fn


# ---------------------------------------------------------------------------
# Spectrum (modvars) extract functions — totals
# ---------------------------------------------------------------------------

def _spec_totpop(modvars: dict) -> np.ndarray:
    """DP_BigPop_V1: sum male (row 1) + female (row 2) over all 81 ages."""
    arr = np.array(modvars[DP_BigPopTag])  # (3, 81, 81)  [sex, age, year]
    return (arr[1] + arr[2]).sum(axis=0)


def _spec_newinf(modvars: dict) -> np.ndarray:
    """HV_NewInfections_V1: sum male (row 1) + female (row 2) over inner dims."""
    arr = np.array(modvars[HV_NewInfectionsTag])  # (3, risk_grp, vaccine_state, 81)
    n = arr.shape[-1]
    return (arr[1] + arr[2]).reshape(-1, n).sum(axis=0)


def _spec_aidsdeath(modvars: dict) -> np.ndarray:
    """HV_AIDSDeaths_V1: sum male (row 1) + female (row 2) over inner dims."""
    arr = np.array(modvars[HV_AIDSDeathsTag])
    n = arr.shape[-1]
    return (arr[1] + arr[2]).reshape(-1, n).sum(axis=0)


def _spec_hivpop(modvars: dict) -> np.ndarray:
    """HV_TotalAdultsHIV_V1: sum male (row 1) + female (row 2)."""
    arr = np.array(modvars[HV_TotalAdultsHIVTag])  # (3, 81)  sex × year
    return arr[1] + arr[2]


def _spec_art(modvars: dict) -> np.ndarray:
    """HV_TotalAdultsART_V1: sum male (row 1) + female (row 2)."""
    arr = np.array(modvars[HV_TotalAdultsARTTag])  # (3, 81)  sex × year
    return arr[1] + arr[2]


def _spec_prevalence(modvars: dict) -> np.ndarray:
    """HV_CalcPrevalence_V1: sum male (row 1) + female (row 2) over all risk groups."""
    # arr = np.array(modvars[HV_CalcPrevalenceTag])  # (3, 11, 81)  sex × risk_grp × year
    HIV = np.array(modvars[HV_TotalAdultsHIVTag]).sum(axis=0)  #
    POP = np.array(modvars[DP_BigPopTag]) # (81, 81)  age × year

        
    return HIV/(POP[1]+POP[2]).sum(axis=0)


def _spec_incidence(modvars: dict) -> np.ndarray:
    """HV_Incidence_V1 is a proportion; multiply by 100 for percent."""
    return np.array(modvars[HV_IncidenceTag]) 


def _spec_tot_plhiv(modvars: dict) -> np.ndarray:
    """Total PLHIV: same as HV_TotalAdultsHIV_V1."""
    return np.array(modvars[HV_TotalAdultsHIVTag])    


def _spec_tot_on_art(modvars: dict) -> np.ndarray:
    """Total on ART: same as HV_TotalAdultsART_V1."""
    return np.array(modvars[HV_TotalAdultsARTTag]) 


# ---------------------------------------------------------------------------
# Spectrum disaggregated (age + sex) extract functions
# ---------------------------------------------------------------------------

def _spec_totpop_disagg(modvars: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """DP_BigPop_V1 (3, 81, 81): [sex, age, year] — [0]=both, [1]=male, [2]=female.
    Totals are always male+female (index 1+2), never the pre-computed 'both' row."""
    arr = np.array(modvars[DP_BigPopTag])  # (3, 81_ages, 81_years)

    if disagg_age:
        age_items = [(lbl, slice(a, b + 1)) for (a, b), lbl in zip(AGE_GROUPS, AGE_LABELS)]
    else:
        age_items = [(None, slice(None))]

    series: list[tuple[str, np.ndarray]] = []
    for age_lbl, age_sl in age_items:
        if disagg_sex:
            for sex_lbl, sex_idx in [("Male", 1), ("Female", 2)]:
                data = arr[sex_idx, age_sl, :].sum(axis=0)
                parts = [p for p in [age_lbl, sex_lbl] if p]
                series.append((" / ".join(parts) if parts else sex_lbl, data))
        else:
            data = (arr[1] + arr[2])[age_sl, :].sum(axis=0)
            series.append((age_lbl if age_lbl else "Total", data))
    return series


def _spec_newinf_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_NewInfections_V1 (3, …, 81): sex [0]=both, [1]=male, [2]=female.
    Inner dims (risk group, vaccine state) are always summed; no age disagg available."""
    arr = np.array(modvars[HV_NewInfectionsTag])
    n = arr.shape[-1]
    if disagg_sex:
        return [
            ("Male", arr[1].reshape(-1, n).sum(axis=0)),
            ("Female", arr[2].reshape(-1, n).sum(axis=0)),
        ]
    return [("Total", (arr[1] + arr[2]).reshape(-1, n).sum(axis=0))]


def _spec_aidsdeath_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_AIDSDeaths_V1: same sex-first structure as HV_NewInfections_V1."""
    arr = np.array(modvars[HV_AIDSDeathsTag])
    n = arr.shape[-1]
    if disagg_sex:
        return [
            ("Male", arr[1].reshape(-1, n).sum(axis=0)),
            ("Female", arr[2].reshape(-1, n).sum(axis=0)),
        ]
    return [("Total", (arr[1] + arr[2]).reshape(-1, n).sum(axis=0))]


def _spec_newinf_1549_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_NewInfections_V1 for the 15-49 indicator: uses '15-49' demo key for color consistency."""
    arr = np.array(modvars[HV_NewInfectionsTag])
    n = arr.shape[-1]
    if disagg_sex:
        return [
            ("Male", arr[1].reshape(-1, n).sum(axis=0)),
            ("Female", arr[2].reshape(-1, n).sum(axis=0)),
        ]
    return [("15-49", (arr[1] + arr[2]).reshape(-1, n).sum(axis=0))]


def _spec_aidsdeath_1549_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_AIDSDeaths_V1 for the 15-49 indicator: uses '15-49' demo key for color consistency."""
    arr = np.array(modvars[HV_AIDSDeathsTag])
    n = arr.shape[-1]
    if disagg_sex:
        return [
            ("Male", arr[1].reshape(-1, n).sum(axis=0)),
            ("Female", arr[2].reshape(-1, n).sum(axis=0)),
        ]
    return [("15-49", (arr[1] + arr[2]).reshape(-1, n).sum(axis=0))]


def _spec_hivpop_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_TotalAdultsHIV_V1 (3, 81): sex [0]=both, [1]=male, [2]=female. No age disagg."""
    arr = np.array(modvars[HV_TotalAdultsHIVTag])
    if disagg_sex:
        return [("Male", arr[1]), ("Female", arr[2])]
    return [("Total", arr[1] + arr[2])]


def _spec_art_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_TotalAdultsART_V1 (3, 81): sex [0]=both, [1]=male, [2]=female. No age disagg."""
    arr = np.array(modvars[HV_TotalAdultsARTTag])
    if disagg_sex:
        return [("Male", arr[1]), ("Female", arr[2])]
    return [("Total", arr[1] + arr[2])]


# ---------------------------------------------------------------------------
# Disagg helpers for 15-49-restricted indicators
# ---------------------------------------------------------------------------

def _disagg_std_1549(key: str) -> Callable:
    """Disagg for a (81, 2, n_years) Goals array restricted to ages 15-49.
    Returns [] when disagg_age=True (no meaningful age faceting for a 15-49 aggregate)."""
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        if disagg_age:
            return []
        arr = output[key]
        sex_specs = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )
        series: list[tuple[str, np.ndarray]] = []
        for sex_label, sex_idx in sex_specs:
            data = _sum_std(arr, slice(15, 50), sex_idx)
            series.append((sex_label if sex_label else "15-49", data))
        return series
    return fn


def _no_age_disagg(disagg_fn: Callable) -> Callable:
    """Wraps any disagg function to return [] when disagg_age=True."""
    def wrapper(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        if disagg_age:
            return []
        return disagg_fn(output, False, disagg_sex)
    return wrapper


# ---------------------------------------------------------------------------
# Spectrum helpers for 15-49 sub-range
# ---------------------------------------------------------------------------

def _spec_totpop_1549(modvars: dict) -> np.ndarray:
    """DP_BigPop_V1: sum male + female over ages 15-49."""
    arr = np.array(modvars[DP_BigPopTag])  # (3, 81_ages, 81_years)
    return (arr[1, 15:50, :] + arr[2, 15:50, :]).sum(axis=0)


def _spec_totpop_1549_disagg(modvars: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """DP_BigPop_V1 ages 15-49 only; returns [] in age-faceted view."""
    if disagg_age:
        return []
    arr = np.array(modvars[DP_BigPopTag])
    if disagg_sex:
        return [
            ("Male", arr[1, 15:50, :].sum(axis=0)),
            ("Female", arr[2, 15:50, :].sum(axis=0)),
        ]
    return [("15-49", (arr[1, 15:50, :] + arr[2, 15:50, :]).sum(axis=0))]


# ---------------------------------------------------------------------------
# Leapfrog Goals compute functions (disagg_sex only; hidden in age-faceted view)
# ---------------------------------------------------------------------------

def _goals_total_pop_1549(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """total_population (n_years,) — scalar 15-49 aggregate, no sex disagg available.
    Returns empty when disagg_sex=True so a scalar doesn't appear alongside M/F lines."""
    if disagg_sex:
        return []
    return [("15-49", goals_output["total_population"])]


def _goals_total_deaths_hiv(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """total_deaths_hiv (n_years,) — scalar, no sex disagg available.
    Returns empty when disagg_sex=True."""
    if disagg_sex:
        return []
    return [("15-49", goals_output["total_deaths_hiv"])]


def _goals_newinf(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """new_infections_goals (3, n_years): [0]=Male, [1]=Female, [2]=Both."""
    arr = goals_output["new_infections_goals"]
    if disagg_sex:
        return [("Male", arr[0]), ("Female", arr[1])]
    return [("15-49", arr[2])]


def _goals_incidence(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """incidence_goals (3, n_years): [0]=Male, [1]=Female, [2]=Both. Multiplied by 100 → percent."""
    arr = goals_output["incidence_goals"] * 100.0
    if disagg_sex:
        return [("Male", arr[0]), ("Female", arr[1])]
    return [("15-49", arr[2])]


def _goals_prevalence(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """prevalence (18, 3, n_years): index [0] is RG aggregate; [0]=Male, [1]=Female, [2]=Both. Multiplied by 100 → percent."""
    arr = goals_output["prevalence"][len(goals_output["prevalence"])-1] * 100.0  # (3, n_years)
    if disagg_sex:
        return [("Male", arr[0]), ("Female", arr[1])]
    return [("15-49", arr[2])]


# ---------------------------------------------------------------------------
# Indicator definitions
# ---------------------------------------------------------------------------

@dataclass
class IndicatorDef:
    compute_leapfrog_aim: Callable[[dict], np.ndarray]
    compute_leapfrog_aim_disagg: Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]]
    compute_spectrum: Callable[[dict], np.ndarray] | None
    compute_spectrum_disagg: Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]] | None = field(default=None)
    # Goals-specific outputs (hidden in age-faceted view); signature: (goals_output, disagg_sex)
    compute_leapfrog_goals_disagg: Callable[[dict, bool], list[tuple[str, np.ndarray]]] | None = field(default=None)


INDICATOR_MAP: OrderedDict[str, IndicatorDef] = OrderedDict([
    # --- All-ages indicators: Leapfrog DP/AIM vs Spectrum only ---
    ("Total population", IndicatorDef(
        compute_leapfrog_aim=lambda o: _sum_std(o["p_totpop"]),
        compute_leapfrog_aim_disagg=_disagg_std("p_totpop"),
        compute_spectrum=_spec_totpop,
        compute_spectrum_disagg=_spec_totpop_disagg,
    )),
    ("Total Births", IndicatorDef(
        compute_leapfrog_aim=lambda o: o["births"],
        compute_leapfrog_aim_disagg=lambda o, _da, _ds: [("Total", o["births"])],
        compute_spectrum=None,  # DP_Births all-zeros in test files
    )),
    ("HIV population", IndicatorDef(
        compute_leapfrog_aim=lambda o: _sum_std(o["p_hivpop"]),
        compute_leapfrog_aim_disagg=_disagg_std("p_hivpop"),
        compute_spectrum=_spec_hivpop,
        compute_spectrum_disagg=_spec_hivpop_disagg,
    )),
    ("New HIV infections", IndicatorDef(
        compute_leapfrog_aim=lambda o: _sum_std(o["p_infections"]),
        compute_leapfrog_aim_disagg=_disagg_std("p_infections"),
        compute_spectrum=_spec_newinf,
        compute_spectrum_disagg=_spec_newinf_disagg,
    )),
    ("AIDS deaths", IndicatorDef(
        compute_leapfrog_aim=lambda o: _sum_std(o["p_hiv_deaths"]),
        compute_leapfrog_aim_disagg=_disagg_std("p_hiv_deaths"),
        compute_spectrum=_spec_aidsdeath,
        compute_spectrum_disagg=_spec_aidsdeath_disagg,
    )),
    ("Total number receiving ART (15-49)", IndicatorDef(
        compute_leapfrog_aim=lambda o: o["h_artpop"][:, :, :35, :, :].reshape(-1, o["h_artpop"].shape[-1]).sum(axis=0),
        compute_leapfrog_aim_disagg=_disagg_art(),
        compute_spectrum=_spec_art,
        compute_spectrum_disagg=_spec_art_disagg,
    )),

    # --- 15-49 indicators: all three sources; age disagg disabled ---
    ("Total population 15-49", IndicatorDef(
        compute_leapfrog_aim=lambda o: _sum_std(o["p_totpop"], slice(15, 50)),
        compute_leapfrog_aim_disagg=_disagg_std_1549("p_totpop"),
        compute_spectrum=_spec_totpop_1549,
        compute_spectrum_disagg=_spec_totpop_1549_disagg,
        compute_leapfrog_goals_disagg=_goals_total_pop_1549,
    )),
    ("New HIV infections 15-49", IndicatorDef(
        compute_leapfrog_aim=lambda o: _sum_std(o["p_infections"], slice(15, 50)),
        compute_leapfrog_aim_disagg=_disagg_std_1549("p_infections"),
        compute_spectrum=_spec_newinf,
        compute_spectrum_disagg=_spec_newinf_1549_disagg,
        compute_leapfrog_goals_disagg=_goals_newinf,
    )),
    ("AIDS deaths 15-49", IndicatorDef(
        compute_leapfrog_aim=lambda o: _sum_std(o["p_hiv_deaths"], slice(15, 50)),
        compute_leapfrog_aim_disagg=_disagg_std_1549("p_hiv_deaths"),
        compute_spectrum=_spec_aidsdeath,
        compute_spectrum_disagg=_spec_aidsdeath_1549_disagg,
        compute_leapfrog_goals_disagg=_goals_total_deaths_hiv,
    )),
    ("Prevalence (15-49) (%)", IndicatorDef(
        compute_leapfrog_aim=lambda o: 100.0 * _sum_std(o["p_hivpop"], slice(15, 50)) / np.where(
            _sum_std(o["p_totpop"], slice(15, 50)) == 0, np.nan,
            _sum_std(o["p_totpop"], slice(15, 50))
        ),
        compute_leapfrog_aim_disagg=_no_age_disagg(_disagg_prevalence()),
        compute_spectrum=_spec_prevalence,
        compute_leapfrog_goals_disagg=_goals_prevalence,
    )),
    ("Incidence (15-49) (%)", IndicatorDef(
        compute_leapfrog_aim=lambda o: 0.0 * _sum_std(o["p_infections"], slice(15, 50)) / np.where(
            (_sum_std(o["p_totpop"], slice(15, 50)) - _sum_std(o["p_hivpop"], slice(15, 50))) == 0,
            np.nan,
            _sum_std(o["p_totpop"], slice(15, 50)) - _sum_std(o["p_hivpop"], slice(15, 50)),
        ),
        compute_leapfrog_aim_disagg=_no_age_disagg(_disagg_incidence()),
        compute_spectrum=_spec_incidence,
        compute_leapfrog_goals_disagg=_goals_incidence,
        )),
        ("Total PLHIV", IndicatorDef(
            compute_leapfrog_aim=lambda o: _sum_std(o["total_plhiv"]),
            compute_leapfrog_aim_disagg=_disagg_std("total_plhiv"),
            compute_spectrum=_spec_tot_plhiv,
            # compute_spectrum_disagg=_spec_tot_plhiv_disagg,
        )),
        ("Total on ART", IndicatorDef(
            compute_leapfrog_aim=lambda o: _sum_std(o["total_on_art"]),
            compute_leapfrog_aim_disagg=_disagg_std("total_on_art"),
            compute_spectrum=_spec_tot_on_art,
            # compute_spectrum_disagg=_spec_tot_on_art_disagg,
        )),
])


def get_indicator_names() -> list[str]:
    return list(INDICATOR_MAP.keys())

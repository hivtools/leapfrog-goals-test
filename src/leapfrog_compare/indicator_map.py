"""
Indicator definitions: compute time series from Goals output and Spectrum modvars.

Two kinds of indicators:

AllAgesIndicatorDef — shown in the 'All ages' tab; supports age + sex disaggregation.
  compute_leapfrog(output, disagg_age, disagg_sex) -> list[(label, ndarray)]
  compute_spectrum(modvars, disagg_age, disagg_sex) -> list[(label, ndarray)] | None

Indicator1549Def — shown in the '15-49' tab; 15-49 aggregate, sex disaggregation only.
  compute_leapfrog(output, disagg_sex) -> list[(label, ndarray)]
  compute_spectrum(modvars, disagg_sex) -> list[(label, ndarray)] | None
  compute_goals(goals_output, disagg_sex) -> list[(label, ndarray)] | None

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
  HV_Incidence_V1           (81,)          rate per year (×100 → percent) — no sex disaggregation

For all modvars with a sex first-dimension, index 0 ("both") is a pre-computed total that we
do NOT use. Totals are always produced by manually summing indices 1 (male) + 2 (female).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Callable

import numpy as np

from SpectrumCommon.Const.DP.DPTags import DP_BigPopTag  # type: ignore[import-untyped]
from SpectrumCommon.Const.HV.HVTags import (  # type: ignore[import-untyped]
    HV_AdultsTag,
    HV_AIDSDeathsTag,
    HV_IncidenceTag,
    HV_NewInfectionsTag,
    HV_TotalAdultsARTTag,
    HV_TotalAdultsHIVTag,
    HV_PopulationsTag,
)
from SpectrumCommon.Const.HV.HVConst import (  # type: ignore[import-untyped]
    HV_AllHIV,
    HV_AllRisk,
    HV_HRH,
    HV_IDU,
    HV_LRH,
    HV_MRH,
    HV_MSM,
)
from SpectrumCommon.Const.RN.RNConst import RN_AllVacc, RN_UnV  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Age / sex group definitions
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
    """Sum a (n_ages, 2, n_years) Goals array over age and/or sex, returning (n_years,)."""
    if age_slice is not None:
        arr = arr[age_slice, :, :]
    if sex is not None:
        arr = arr[:, sex : sex + 1, :]
    return arr.reshape(-1, arr.shape[-1]).sum(axis=0)


# ---------------------------------------------------------------------------
# All-ages Leapfrog disagg functions — signature: (output, disagg_age, disagg_sex)
# ---------------------------------------------------------------------------

def _disagg_std(key: str) -> Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]]:
    """All-ages disaggregation for a Goals (81, 2, n_years) array."""
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = output[key]
        age_specs: list[tuple[str | None, slice | None]] = (
            [(label, slice(a, b + 1)) for (a, b), label in zip(AGE_GROUPS, AGE_LABELS)]
            if disagg_age else [(None, None)]
        )
        sex_specs: list[tuple[str | None, int | None]] = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )
        series: list[tuple[str, np.ndarray]] = []
        for age_label, age_sl in age_specs:
            for sex_label, sex_idx in sex_specs:
                data = _sum_std(arr, age_sl, sex_idx)
                parts = [p for p in [age_label, sex_label] if p]
                series.append((" / ".join(parts) if parts else "Total", data))
        return series
    return fn


def _disagg_art() -> Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]]:
    """All-ages disaggregation for h_artpop (4, 7, 66, 2, n_years): adult ages 15-80."""
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = output["h_artpop"]
        n_years = arr.shape[-1]
        if disagg_age:
            age_items: list[tuple[str, slice | None]] = []
            for (a, b), lbl in zip(AGE_GROUPS, AGE_LABELS):
                if b < 15:
                    age_items.append((lbl, None))  # under-15: no data in h_artpop
                else:
                    age_items.append((lbl, slice(max(0, a - 15), min(65, b - 15) + 1)))
        else:
            age_items = [(None, slice(None))]
        sex_items: list[tuple[str | None, int | None]] = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )
        series: list[tuple[str, np.ndarray]] = []
        for age_lbl, age_sl in age_items:
            for sex_lbl, sex_idx in sex_items:
                if age_sl is None:
                    data = np.zeros(n_years)
                elif sex_idx is not None:
                    data = arr[:, :, age_sl, sex_idx, :].reshape(-1, n_years).sum(axis=0)
                else:
                    data = arr[:, :, age_sl, :, :].reshape(-1, n_years).sum(axis=0)
                parts = [p for p in [age_lbl, sex_lbl] if p]
                series.append((" / ".join(parts) if parts else "Total", data))
        return series
    return fn


# ---------------------------------------------------------------------------
# 15-49 Leapfrog helpers — signature: (output, disagg_sex)
# ---------------------------------------------------------------------------

def _lf_1549(key: str) -> Callable[[dict, bool], list[tuple[str, np.ndarray]]]:
    """15-49 disaggregation for a Goals (81, 2, n_years) array (sex only)."""
    def fn(output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = output[key]
        if disagg_sex:
            return [(sl, _sum_std(arr, slice(15, 50), i)) for i, sl in enumerate(SEX_LABELS)]
        return [("15-49", _sum_std(arr, slice(15, 50)))]
    return fn


def _lf_prevalence_1549(output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Prevalence (%) ages 15-49 from Goals p_hivpop / p_totpop."""
    hiv = output["p_hivpop"]
    tot = output["p_totpop"]
    if disagg_sex:
        return [
            (sl, 100.0 * _sum_std(hiv, slice(15, 50), i) / np.where(
                _sum_std(tot, slice(15, 50), i) == 0, np.nan, _sum_std(tot, slice(15, 50), i)
            ))
            for i, sl in enumerate(SEX_LABELS)
        ]
    h = _sum_std(hiv, slice(15, 50))
    t = _sum_std(tot, slice(15, 50))
    return [("15-49", 100.0 * h / np.where(t == 0, np.nan, t))]


def _lf_incidence_1549(output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Incidence (%) ages 15-49 from Goals p_infections / (p_totpop - p_hivpop)."""
    inf = output["p_infections"]
    hiv = output["p_hivpop"]
    tot = output["p_totpop"]
    if disagg_sex:
        result = []
        for i, sl in enumerate(SEX_LABELS):
            i_ = _sum_std(inf, slice(15, 50), i)
            h = _sum_std(hiv, slice(15, 50), i)
            t = _sum_std(tot, slice(15, 50), i)
            result.append((sl, 100.0 * i_ / np.where(t - h == 0, np.nan, t - h)))
        return result
    i_ = _sum_std(inf, slice(15, 50))
    h = _sum_std(hiv, slice(15, 50))
    t = _sum_std(tot, slice(15, 50))
    return [("15-49", 100.0 * i_ / np.where(t - h == 0, np.nan, t - h))]


def _lf_artpop_1549(output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """h_artpop (hiv treatment stages, hiv disease stages, hiv age groups (15-80+), number sexes (2)"""
    art_pop = output["h_artpop"].sum(axis=(0, 1))
    if disagg_sex:
        return [("Male", art_pop[:35, 0, :].sum(axis=0)), ("Female", art_pop[:35, 1, :].sum(axis=0))]
    return [("15-49", art_pop[:35, :, :].sum(axis=(0, 1)))]


# ---------------------------------------------------------------------------
# Spectrum disagg functions — all-ages tab — signature: (modvars, disagg_age, disagg_sex)
# ---------------------------------------------------------------------------

def _spec_totpop_disagg(modvars: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """DP_BigPop_V1 (3, 81_ages, 81_years): [sex, age, year] — [0]=both, [1]=male, [2]=female."""
    arr = np.array(modvars[DP_BigPopTag])
    age_items = (
        [(lbl, slice(a, b + 1)) for (a, b), lbl in zip(AGE_GROUPS, AGE_LABELS)]
        if disagg_age else [(None, slice(None))]
    )
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


def _spec_hivpop_all_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_TotalAdultsHIV_V1 (3, 81): total adults HIV, no age disagg available."""
    arr = np.array(modvars[HV_TotalAdultsHIVTag])
    if disagg_sex:
        return [("Male", arr[1]), ("Female", arr[2])]
    return [("Total", arr[1] + arr[2])]


def _spec_newinf_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_NewInfections_V1 (3, …, 81): inner dims (risk group, vaccine state) always summed."""
    arr = np.array(modvars[HV_NewInfectionsTag])
    n = arr.shape[-1]
    if disagg_sex:
        return [("Male", arr[1].reshape(-1, n).sum(axis=0)), ("Female", arr[2].reshape(-1, n).sum(axis=0))]
    return [("Total", (arr[1] + arr[2]).reshape(-1, n).sum(axis=0))]


def _spec_aidsdeath_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_AIDSDeaths_V1: same sex-first structure as HV_NewInfections_V1."""
    arr = np.array(modvars[HV_AIDSDeathsTag])
    n = arr.shape[-1]
    if disagg_sex:
        return [("Male", arr[1].reshape(-1, n).sum(axis=0)), ("Female", arr[2].reshape(-1, n).sum(axis=0))]
    return [("Total", (arr[1] + arr[2]).reshape(-1, n).sum(axis=0))]


def _spec_art_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_TotalAdultsART_V1 (3, 81): sex [0]=both, [1]=male, [2]=female. No age disagg."""
    arr = np.array(modvars[HV_TotalAdultsARTTag])
    if disagg_sex:
        return [("Male", arr[1]), ("Female", arr[2])]
    return [("Total", arr[1] + arr[2])]


# ---------------------------------------------------------------------------
# Spectrum functions — 15-49 tab — signature: (modvars, disagg_sex)
# ---------------------------------------------------------------------------

def _spec_totpop_1549(modvars: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """DP_BigPop_V1 ages 15-49."""
    arr = np.array(modvars[DP_BigPopTag])
    if disagg_sex:
        return [("Male", arr[1, 15:50, :].sum(axis=0)), ("Female", arr[2, 15:50, :].sum(axis=0))]
    return [("15-49", (arr[1, 15:50, :] + arr[2, 15:50, :]).sum(axis=0))]


def _spec_hivpop_1549(modvars: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_TotalAdultsHIV_V1 — adults HIV (15+), labelled 15-49 for colour consistency."""
    arr = np.array(modvars[HV_TotalAdultsHIVTag])
    if disagg_sex:
        return [("Male", arr[1]), ("Female", arr[2])]
    return [("15-49", arr[1] + arr[2])]


def _spec_newinf_1549(modvars: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_NewInfections_V1 summed over risk groups / vaccine states."""
    arr = np.array(modvars[HV_NewInfectionsTag])
    n = arr.shape[-1]
    if disagg_sex:
        return [("Male", arr[1].reshape(-1, n).sum(axis=0)), ("Female", arr[2].reshape(-1, n).sum(axis=0))]
    return [("15-49", (arr[1] + arr[2]).reshape(-1, n).sum(axis=0))]


def _spec_aidsdeath_1549(modvars: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_AIDSDeaths_V1 summed over risk groups / vaccine states."""
    arr = np.array(modvars[HV_AIDSDeathsTag])
    n = arr.shape[-1]
    if disagg_sex:
        return [("Male", arr[1].reshape(-1, n).sum(axis=0)), ("Female", arr[2].reshape(-1, n).sum(axis=0))]
    return [("15-49", (arr[1] + arr[2]).reshape(-1, n).sum(axis=0))]


def _spec_prevalence_1549(modvars: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Prevalence (%) from HV_Populations and HV_TotalAdultsHIV."""
    pop = np.array(modvars[HV_PopulationsTag])
    hiv = np.array(modvars[HV_TotalAdultsHIVTag])
    if disagg_sex:
        return [("Male", 100 * hiv[1] / pop[1]), ("Female", 100 * hiv[2] / pop[2])]
    return [("15-49", 100 * (hiv[1] + hiv[2]) / (pop[1] + pop[2]))]


def _spec_incidence_1549(modvars: dict, _disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_Incidence_V1 (scalar per year, no sex disaggregation available)."""
    return [("15-49", 100 * np.array(modvars[HV_IncidenceTag]))]


def _spec_art_1549(modvars: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_TotalAdultsART_V1 (3, 81)."""
    arr = np.array(modvars[HV_TotalAdultsARTTag])
    if disagg_sex:
        return [("Male", arr[1]), ("Female", arr[2])]
    return [("15-49", arr[1] + arr[2])]


# ---------------------------------------------------------------------------
# Goals functions — 15-49 tab — signature: (goals_output, disagg_sex)
# ---------------------------------------------------------------------------

def _goals_total_pop_1549(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """total_population (n_years,) — scalar 15-49 aggregate, no sex disagg available."""
    if disagg_sex:
        return []
    return [("15-49", goals_output["total_population"])]


def _goals_total_deaths_hiv(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """total_deaths_hiv (n_years,) — scalar, no sex disagg available."""
    if disagg_sex:
        return []
    return [("15-49", goals_output["total_deaths_hiv"])]


def _goals_plhiv(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """total_plhiv (n_years,) — scalar, no sex disagg available."""
    if disagg_sex:
        return []
    return [("15-49", goals_output["total_plhiv"])]


def _goals_newinf(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """new_infections_goals (3, n_years): [0]=Male, [1]=Female, [2]=Both."""
    arr = goals_output["new_infections_goals"]
    if disagg_sex:
        return [("Male", arr[0]), ("Female", arr[1])]
    return [("15-49", arr[2])]


def _goals_incidence(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """incidence_goals (3, n_years): [0]=Male, [1]=Female, [2]=Both. ×100 → percent."""
    arr = goals_output["incidence_goals"] * 100.0
    if disagg_sex:
        return [("Male", arr[0]), ("Female", arr[1])]
    return [("15-49", arr[2])]


def _goals_total_on_art(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """total_on_art (n_years)"""
    arr = goals_output["total_on_art"]
    if disagg_sex:
        return []
    return [("15-49", arr)]


def _goals_prevalence(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """prevalence (18, 3, n_years): last row is 15-49 aggregate. ×100 → percent."""
    arr = goals_output["prevalence"][-1] * 100.0  # (3, n_years)
    if disagg_sex:
        return [("Male", arr[0]), ("Female", arr[1])]
    return [("15-49", arr[2])]


# ---------------------------------------------------------------------------
# Indicator definitions
# ---------------------------------------------------------------------------

@dataclass
class AllAgesIndicatorDef:
    """All-ages indicator shown in the 'All ages' tab. Supports age + sex disaggregation."""
    compute_leapfrog: Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]]
    compute_spectrum: Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]] | None = None


@dataclass
class Indicator1549Def:
    """15-49 aggregate indicator shown in the '15-49' tab. Sex disaggregation only."""
    compute_leapfrog: Callable[[dict, bool], list[tuple[str, np.ndarray]]]
    compute_spectrum: Callable[[dict, bool], list[tuple[str, np.ndarray]]] | None = None
    compute_goals: Callable[[dict, bool], list[tuple[str, np.ndarray]]] | None = None


ALL_AGES_INDICATORS: OrderedDict[str, AllAgesIndicatorDef] = OrderedDict([
    ("Total population", AllAgesIndicatorDef(
        compute_leapfrog=_disagg_std("p_totpop"),
        compute_spectrum=_spec_totpop_disagg,
    )),
    ("Total Births", AllAgesIndicatorDef(
        compute_leapfrog=lambda o, _da, _ds: [("Total", o["births"])],
    )),
    ("HIV population", AllAgesIndicatorDef(
        compute_leapfrog=_disagg_std("p_hivpop"),
        compute_spectrum=_spec_hivpop_all_disagg,
    )),
    ("New HIV infections", AllAgesIndicatorDef(
        compute_leapfrog=_disagg_std("p_infections"),
        compute_spectrum=_spec_newinf_disagg,
    )),
    ("AIDS deaths", AllAgesIndicatorDef(
        compute_leapfrog=_disagg_std("p_hiv_deaths"),
        compute_spectrum=_spec_aidsdeath_disagg,
    )),
    ("Total number receiving ART (15-49)", AllAgesIndicatorDef(
        compute_leapfrog=_disagg_art(),
        compute_spectrum=_spec_art_disagg,
    )),
])


INDICATORS_1549: OrderedDict[str, Indicator1549Def] = OrderedDict([
    ("Total population (15-49)", Indicator1549Def(
        compute_leapfrog=_lf_1549("p_totpop"),
        compute_spectrum=_spec_totpop_1549,
        compute_goals=_goals_total_pop_1549,
    )),
    ("Total PLHIV (15-49)", Indicator1549Def(
        compute_leapfrog=_lf_1549("p_hivpop"),
        compute_spectrum=_spec_hivpop_1549,
        compute_goals=_goals_plhiv,
    )),
    ("New HIV infections (15-49)", Indicator1549Def(
        compute_leapfrog=_lf_1549("p_infections"),
        compute_spectrum=_spec_newinf_1549,
        compute_goals=_goals_newinf,
    )),
    ("AIDS deaths (15-49)", Indicator1549Def(
        compute_leapfrog=_lf_1549("p_hiv_deaths"),
        compute_spectrum=_spec_aidsdeath_1549,
        compute_goals=_goals_total_deaths_hiv,
    )),
    ("Prevalence (15-49) (%)", Indicator1549Def(
        compute_leapfrog=_lf_prevalence_1549,
        compute_spectrum=_spec_prevalence_1549,
        compute_goals=_goals_prevalence,
    )),
    ("Incidence (15-49) (%)", Indicator1549Def(
        compute_leapfrog=_lf_incidence_1549,
        compute_spectrum=_spec_incidence_1549,
        compute_goals=_goals_incidence,
    )),
    ("Total on ART (15-49)", Indicator1549Def(
        compute_leapfrog=_lf_artpop_1549,
        compute_spectrum=_spec_art_1549,
        compute_goals=_goals_total_on_art
    )),
])


# ---------------------------------------------------------------------------
# Risk group definitions and compute functions
# ---------------------------------------------------------------------------

# Goals adults array: shape (nVAC+1=5, nRG+1=18, nCD4+1=17, nNS+1=3, n_years)
_VAC_ALL = 4
_VAC_UNV = 0
_CD4_ALL = 16
_RG_ALL  = 17

# (display_name, goals_rg_index) in display order
RISK_GROUPS: list[tuple[str, int]] = [
    ("Low risk",    1),  # RG_LRH
    ("Medium risk", 2),  # RG_MRH
    ("High risk",   3),  # RG_HRH
    ("PWID",        4),  # RG_IDU
    ("MSM",         5),  # RG_MSM
]

# Maps display name → Spectrum HV_Adults risk-group index
_SPEC_RG_INDICES: dict[str, int] = {
    "Low risk":    HV_LRH,
    "Medium risk": HV_MRH,
    "High risk":   HV_HRH,
    "PWID":        HV_IDU,
    "MSM":         HV_MSM,
}


def compute_rg_goals(
    goals_output: dict, disagg_sex: bool
) -> list[tuple[str, str, np.ndarray]]:
    """
    Risk-group fractions from Goals 'adults' (5, 18, 17, 3, n_years).
    Returns list of (rg_name, demo, ratio) where:
      ratio = adults[VAC_ALL, rg_idx, CD4_ALL, sex] / adults[VAC_ALL, RG_ALL, CD4_ALL, sex]
    When disagg_sex is False, male + female are summed before dividing.
    """
    adults = np.array(goals_output["adults"])
    result: list[tuple[str, str, np.ndarray]] = []
    for rg_name, rg_idx in RISK_GROUPS:
        if disagg_sex:
            for sex_idx, sex_label in enumerate(SEX_LABELS):
                num = adults[_VAC_ALL, rg_idx, _CD4_ALL, sex_idx]
                den = adults[_VAC_ALL, _RG_ALL, _CD4_ALL, sex_idx]
                result.append((rg_name, sex_label, 100 * num / np.where(den == 0, np.nan, den)))
        else:
            num = adults[_VAC_ALL, rg_idx, _CD4_ALL, 0] + adults[_VAC_ALL, rg_idx, _CD4_ALL, 1]
            if rg_name == "MSM":
                ## Only ever use men as denominator for MSM, women are 0 so numerator doesn't matter
                den = adults[_VAC_ALL, _RG_ALL, _CD4_ALL, 0]
            else:
                den = adults[_VAC_ALL, _RG_ALL, _CD4_ALL, 0] + adults[_VAC_ALL, _RG_ALL, _CD4_ALL, 1]
            result.append((rg_name, "Total", 100 * num / np.where(den == 0, np.nan, den)))
    return result


def compute_rg_spectrum(
    modvars: dict, disagg_sex: bool
) -> list[tuple[str, str, np.ndarray]]:
    """
    Risk-group fractions from Spectrum HV_Adults_V1 (sex, rg, hiv, vac, n_years).
    Indexed as hv_adults[sex, rg, HV_AllHIV, RN_AllVacc, :].
    Returns list of (rg_name, demo, ratio*100).
    When disagg_sex is False, male (1) + female (2) are summed before dividing.
    """
    hv_adults = np.array(modvars[HV_AdultsTag])
    result: list[tuple[str, str, np.ndarray]] = []
    for rg_name, _ in RISK_GROUPS:
        spec_rg_idx = _SPEC_RG_INDICES[rg_name]
        if disagg_sex:
            for sex_idx, sex_label in enumerate(SEX_LABELS, start=1):  # 1=male, 2=female
                num = hv_adults[sex_idx, spec_rg_idx, HV_AllHIV, RN_AllVacc]
                den = hv_adults[sex_idx, HV_AllRisk, HV_AllHIV, RN_AllVacc]
                result.append((rg_name, sex_label, 100 * num / np.where(den == 0, np.nan, den)))
        else:
            num = hv_adults[1, spec_rg_idx, HV_AllHIV, RN_AllVacc] + hv_adults[2, spec_rg_idx, HV_AllHIV, RN_AllVacc]
            if rg_name == "MSM":
                den = hv_adults[1, HV_AllRisk, HV_AllHIV, RN_AllVacc]
            else:
                den = hv_adults[1, HV_AllRisk, HV_AllHIV, RN_AllVacc] + hv_adults[2, HV_AllRisk, HV_AllHIV, RN_AllVacc]
            result.append((rg_name, "Total", 100 * num / np.where(den == 0, np.nan, den)))
    return result


def compute_new_infections_rg_goals(
    goals_output: dict, disagg_sex: bool
) -> list[tuple[str, str, np.ndarray]]:
    """
    New infections by risk group from Goals 'new_inf_vrs' (nVAC+1, nRG+1, nNS+1, n_years).
    Returns list of (rg_name, demo, count).
    """
    new_inf = np.array(goals_output["new_inf_vrs"])
    result: list[tuple[str, str, np.ndarray]] = []
    for rg_name, rg_idx in RISK_GROUPS:
        if disagg_sex:
            for sex_idx, sex_label in enumerate(SEX_LABELS):
                result.append((rg_name, sex_label, new_inf[_VAC_UNV, rg_idx, sex_idx]))
        else:
            values = new_inf[_VAC_UNV, rg_idx, 0] + new_inf[_VAC_UNV, rg_idx, 1]
            result.append((rg_name, "Total", values))
    return result


def compute_new_infections_rg_spectrum(
    modvars: dict, disagg_sex: bool
) -> list[tuple[str, str, np.ndarray]]:
    """
    New infections by risk group from Spectrum HV_NewInfections_V1 (sex, rg, vac, year).
    Uses RN_AllVacc for vaccine index. Returns list of (rg_name, demo, count).
    """
    arr = np.array(modvars[HV_NewInfectionsTag])
    result: list[tuple[str, str, np.ndarray]] = []
    for rg_name, _ in RISK_GROUPS:
        spec_rg_idx = _SPEC_RG_INDICES[rg_name]
        if disagg_sex:
            for sex_idx, sex_label in enumerate(SEX_LABELS, start=1):  # 1=male, 2=female
                result.append((rg_name, sex_label, arr[sex_idx, spec_rg_idx, RN_UnV]))
        else:
            values = arr[1, spec_rg_idx, RN_UnV] + arr[2, spec_rg_idx, RN_UnV]
            result.append((rg_name, "Total", values))
    return result

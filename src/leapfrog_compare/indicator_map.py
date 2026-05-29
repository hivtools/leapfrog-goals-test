"""
Indicator definitions: compute time series from Goals output and Spectrum modvars.

Each IndicatorDef provides:
  - compute_goals(output)            -> 1-D ndarray (total, all ages+sexes)
  - compute_goals_disagg(output, disagg_age, disagg_sex)
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
    arr = np.array(modvars[HV_CalcPrevalenceTag])  # (3, 11, 81)  sex × risk_grp × year
    return (arr[1] + arr[2]).sum(axis=0)


def _spec_incidence(modvars: dict) -> np.ndarray:
    """HV_Incidence_V1 is a proportion; multiply by 100 for percent."""
    return np.array(modvars[HV_IncidenceTag]) * 100.0


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
# Indicator definitions
# ---------------------------------------------------------------------------

@dataclass
class IndicatorDef:
    compute_goals: Callable[[dict], np.ndarray]
    compute_goals_disagg: Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]]
    compute_spectrum: Callable[[dict], np.ndarray] | None
    compute_spectrum_disagg: Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]] | None = field(default=None)


INDICATOR_MAP: OrderedDict[str, IndicatorDef] = OrderedDict([
    ("Total population", IndicatorDef(
        compute_goals=lambda o: _sum_std(o["p_totpop"]),
        compute_goals_disagg=_disagg_std("p_totpop"),
        compute_spectrum=_spec_totpop,
        compute_spectrum_disagg=_spec_totpop_disagg,
    )),
    ("Total Births", IndicatorDef(
        compute_goals=lambda o: o["births"],
        compute_goals_disagg=lambda o, _da, _ds: [("Total", o["births"])],
        compute_spectrum=None,  # DP_Births all-zeros in test files
    )),
    ("HIV population", IndicatorDef(
        compute_goals=lambda o: _sum_std(o["p_hivpop"]),
        compute_goals_disagg=_disagg_std("p_hivpop"),
        compute_spectrum=_spec_hivpop,
        compute_spectrum_disagg=_spec_hivpop_disagg,
    )),
    ("New HIV infections", IndicatorDef(
        compute_goals=lambda o: _sum_std(o["p_infections"]),
        compute_goals_disagg=_disagg_std("p_infections"),
        compute_spectrum=_spec_newinf,
        compute_spectrum_disagg=_spec_newinf_disagg,
    )),
    ("AIDS deaths", IndicatorDef(
        compute_goals=lambda o: _sum_std(o["p_hiv_deaths"]),
        compute_goals_disagg=_disagg_std("p_hiv_deaths"),
        compute_spectrum=_spec_aidsdeath,
        compute_spectrum_disagg=_spec_aidsdeath_disagg,
    )),
    ("Total number receiving ART (15-49)", IndicatorDef(
        compute_goals=lambda o: o["h_artpop"][:, :, :35, :, :].reshape(-1, o["h_artpop"].shape[-1]).sum(axis=0),
        compute_goals_disagg=_disagg_art(),
        compute_spectrum=_spec_art,
        compute_spectrum_disagg=_spec_art_disagg,
    )),
    ("Prevalence (15-49)", IndicatorDef(
        compute_goals=lambda o: 100.0 * _sum_std(o["p_hivpop"], slice(15, 50)) / np.where(
            _sum_std(o["p_totpop"], slice(15, 50)) == 0, np.nan,
            _sum_std(o["p_totpop"], slice(15, 50))
        ),
        compute_goals_disagg=_disagg_prevalence(),
        compute_spectrum=_spec_prevalence,
    )),
    ("Incidence (15-49) (Percent)", IndicatorDef(
        compute_goals=lambda o: 100.0 * _sum_std(o["p_infections"], slice(15, 50)) / np.where(
            (_sum_std(o["p_totpop"], slice(15, 50)) - _sum_std(o["p_hivpop"], slice(15, 50))) == 0,
            np.nan,
            _sum_std(o["p_totpop"], slice(15, 50)) - _sum_std(o["p_hivpop"], slice(15, 50)),
        ),
        compute_goals_disagg=_disagg_incidence(),
        compute_spectrum=_spec_incidence,
    )),
])


def get_indicator_names() -> list[str]:
    return list(INDICATOR_MAP.keys())

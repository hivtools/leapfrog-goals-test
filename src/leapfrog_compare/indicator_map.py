"""
Indicator definitions: compute time series from Goals output and Spectrum modvars.

Each IndicatorDef holds a `disagg` dict keyed by source id (e.g. "dp_aim", "spectrum",
"goals", "eppasm", "eppasm_lf"). Each value has signature:
  (data, disagg_age, disagg_sex) -> list[(label, 1-D ndarray)]
A source is omitted from the dict entirely when it has no data for that indicator.
Callers should use `.get(source_key)` and treat a missing key the same as an empty
result.

Goals arrays for population-like indicators have shape (n_ages, 2, n_years):
  axis 0 = single-year ages 0-80
  axis 1 = sex (0=male, 1=female)
  axis 2 = years

h_artpop shape is (4, 7, 66, 2, n_years):
  axis 2 = adult single-year ages 15-80
  axis 3 = sex (0=male, 1=female)

Spectrum modvars shapes (confirmed):
  DP_BigPop_V1              (3, 81, 81)    sex × age × year;  [0]=both, [1]=male, [2]=female
  DP_Births_V1              (3, 18, n_years) sex × 5y-age-band × year; [0]=Both Sexes,
                            [1]=male, [2]=female; age axis [0]="All Ages" (a genuine
                            pre-aggregated total, distinct from the 17 5-year bands at
                            [1..17]) — unlike DP_BigPop_V1, index 0 on *both* axes here
                            is the correct value to read directly (see GetDP_Births in
                            SpectrumEngine's DPUtil.py), not a value to recompute
  HV_NewInfections_V1       (3, K, M, 81)  sex × risk_grp × vaccine_state × year; [0]=both, [1]=male, [2]=female
  HV_AIDSDeaths_V1          same structure as HV_NewInfections_V1
  HV_TotalAdultsHIV_V1      (3, 81)        sex × year;  [0]=both, [1]=male, [2]=female
  HV_TotalAdultsART_V1      (3, 81)        sex × year;  [0]=both, [1]=male, [2]=female
  HV_CalcPrevalence_V1      (3, 11, 81)    sex × risk_grp × year; [0]=both, [1]=male, [2]=female
  HV_Incidence_V1           (81,)          rate per year (×100 → percent) — no sex disaggregation

For all modvars with a sex first-dimension EXCEPT DP_Births_V1, index 0 ("both") is a
pre-computed total that we do NOT use. Totals are always produced by manually summing
indices 1 (male) + 2 (female).
"""

from __future__ import annotations

import leapfrog_compare.config  # noqa: F401 — ensures SpectrumCommon is on sys.path
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Callable

import numpy as np

from SpectrumCommon.Const.AM.AMTags import (  # type: ignore[import-untyped]
    AM_AIDSDeathsARTSingleAgeTag,
    AM_AIDSDeathsNoARTSingleAgeTag,
    AM_CD4DistributionChildTag,
    AM_HIVBySingleAgeTag,
    AM_NewInfectionsBySingleAgeTag,
    AM_ChildNeedPMTCTTag,
    AM_NonAIDSExcessDeathsSingleAgeTag,
    AM_OnARTBySingleAgeTag,
)
from SpectrumCommon.Const.DP.DPConst import (  # type: ignore[import-untyped]
    DP_AllAges,
    DP_CD4_0t4,
    DP_CD4_5t14,
    DP_CD4_Ped_GT1000,
    DP_CD4_Per_GT30,
    DP_D_ARTlt6m,
    DP_NoTreat,
    DP_OnART,
    DP_P_Perinatal,
)
from SpectrumCommon.Const.DP.DPTags import (  # type: ignore[import-untyped]
    DP_BigPopTag,
    DP_BirthsTag,
    DP_DeathsByAgeTag,
    DP_DeathsTag,
)
from SpectrumCommon.Const.GB.GBConst import (  # type: ignore[import-untyped]
    GB_BothSexes,
    GB_Female,
    GB_Male,
)
from SpectrumCommon.Const.HV.HVTags import (  # type: ignore[import-untyped]
    HV_AdultsTag,
    HV_AIDSDeathsTag,
    HV_IncidenceTag,
    HV_NewInfectionsTag,
    HV_PopulationsTag,
    HV_TotalAdultsARTTag,
    HV_TotalAdultsHIVTag,
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
from SpectrumCommon.Const.RN.RNConst import (  # type: ignore[import-untyped]
    RN_AllVacc,
    RN_UnV,
    RN_CompSexEd,
    RN_FSW,
    RN_EconomicStrengthening,
    RN_IDUHarmRed,
    RN_IDUNSEP,
    RN_IDUDrugSub,
    RN_MSMOutreach,
    RN_Condom,
    RN_CondomSupply,
    RN_ANCTesting,
    RN_MC15_49,
    RN_PrEPOralDaily,
    RN_PrEPOralMonthly,
    RN_PrEPOralPlusCon,
    RN_PrEPInject1Mo,
    RN_PrEPInject2Mo,
    RN_PrEPInject6Mo,
    RN_PrEPRing,
    RN_PrEPbNABs,
    RN_PrEPImplant,
    RN_PrEP_PEP,
    RN_HIVTestANCVisit,
    RN_HIVWomanCounceled,
    RN_ExposedInfantTest,
    RN_AdultOnART,
    RN_ChildOnART,
    RN_Vaccines,
    RN_CureAdultsChildren,
    RN_CureNeonates,
    RN_CURE_CHILDREN,
    RN_AHDTreatment,
    RN_POC_CD4_Int,
    RN_POC_VL_Int,
    RN_VMM,
    RN_THERAPEUTIC_VAC,
    RN_FUNC_CURE,
    RN_LONG_ACT_TREAT,
)
from SpectrumCommon.Const.RN.RNTags import (  # type: ignore[import-untyped]
    RN_NumberPeopleReachedTag,
    RN_ResourcesRequiredTag,
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

# Single-year ages 0-80 — the actual resolution of the underlying arrays
# (dp_aim's p_* keys and the AM_*BySingleAge* modvars are indexed by single
# year of age already; AGE_GROUPS above is only a 5-year *display* bucketing
# used by the age-facet grid). Used by the "By age" sub-tab, which plots the
# full single-year age profile rather than re-bucketing into 5-year groups.
AGE_GROUPS_SINGLE: list[tuple[int, int]] = [(a, a) for a in range(81)]
AGE_LABELS_SINGLE: list[str] = [str(a) for a in range(81)]

# CD4 stages for the pediatric age bands (hc1 = 0-4, hc2 = 5-14), fixed order
# matching leapfrog's hc1DS/hc2DS array index 0..N-1.
CD4_LABELS_HC1: list[str] = [
    "CD4 >=30", "CD4 [26, 30)", "CD4 [21, 25)", "CD4 [16, 20)",
    "CD4 [11, 15)", "CD4 [5, 10)", "CD4 < 5",
]
CD4_LABELS_HC2: list[str] = [
    "CD4 >=1000", "CD4 [750, 999)", "CD4 [500, 749)",
    "CD4 [350, 499)", "CD4 [200, 349)", "CD4 <200",
]


def cd4_facet_desc(cd4_labels: list[str]) -> str:
    """Row-heading description for a ChildCD4IndicatorDef's `cd4_labels`, shared by
    the single- and multi-PJNZ "0-14" sub-tabs. The death indicators use a single
    ["Total"] row (Spectrum has no CD4-stratified child-deaths output), so the
    heading shouldn't claim a CD4 breakdown for those. Of the CD4-faceted population
    indicators, 0-4 (hc1) stages are CD4 *percentage* bands ("CD4 distribution"),
    while 5-14 (hc2) stages are CD4 *count* bands ("CD4 count") — the standard
    child HIV-staging convention switches at age 5."""
    if cd4_labels == CD4_LABELS_HC1:
        return "CD4 distribution"
    elif cd4_labels == CD4_LABELS_HC2:
        return "CD4 count"
    else:
        return "total"


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


def _disagg_std(
    key: str,
    *,
    age_groups: list[tuple[int, int]] = AGE_GROUPS,
    age_labels: list[str] = AGE_LABELS,
) -> Callable:
    """
    Return a disaggregation function for a Goals array with shape (81, 2, n_years).
    `age_groups`/`age_labels` default to the 5-year display buckets; pass
    AGE_GROUPS_SINGLE/AGE_LABELS_SINGLE for single-year-of-age resolution
    (the "By age" sub-tab's age_profile disagg functions).
    """
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = output[key]
        series: list[tuple[str, np.ndarray]] = []

        age_specs: list[tuple[str | None, slice | None]] = (
            [(label, slice(a, b + 1)) for (a, b), label in zip(age_groups, age_labels)]
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


def _child_art_by_single_age(output: dict) -> np.ndarray:
    """hc1_artpop (hTS, CD4, AGE=5, NS=2, T; ages 0-4) and hc2_artpop (hTS, CD4,
    AGE=10, NS=2, T; ages 5-14) summed over their CD4/ART-duration axes and
    stitched together into a single (age=15, NS=2, T) array covering ages 0-14 —
    the child-side complement to h_artpop's adult (15+) coverage, for _disagg_art
    to draw on instead of reporting zeros under age 15."""
    hc1 = output["hc1_artpop"]
    hc2 = output["hc2_artpop"]
    hc1_sum = hc1.sum(axis=tuple(range(hc1.ndim - 3)))  # -> (age=5, NS=2, T)
    hc2_sum = hc2.sum(axis=tuple(range(hc2.ndim - 3)))  # -> (age=10, NS=2, T)
    return np.concatenate([hc1_sum, hc2_sum], axis=0)  # -> (age=15, NS=2, T)


def _disagg_art(
    *,
    age_groups: list[tuple[int, int]] = AGE_GROUPS,
    age_labels: list[str] = AGE_LABELS,
) -> Callable:
    """
    Disaggregation for h_artpop (4, 7, 66, 2, n_years): adult ages 15-80, sex axis=3.
    Age axis index 0 = age 15, index i = age 15+i.
    Under-15 age groups are drawn from hc1_artpop/hc2_artpop instead (see
    _child_art_by_single_age) — h_artpop alone has no data below age 15, and every
    AGE_GROUPS/AGE_GROUPS_SINGLE band either falls entirely below 15 or entirely
    at/above it (the 5-year buckets split exactly on the 15 boundary), so no band
    ever needs to mix the two sources.
    `age_groups`/`age_labels` default to the 5-year display buckets; pass
    AGE_GROUPS_SINGLE/AGE_LABELS_SINGLE for single-year-of-age resolution.
    """
    def fn(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = output["h_artpop"]  # (4, 7, 66, 2, n_years)
        child_arr = _child_art_by_single_age(output)  # (age=15, NS=2, n_years)
        n_years = arr.shape[-1]
        series: list[tuple[str, np.ndarray]] = []

        if disagg_age:
            age_items: list[tuple[str, str, slice]] = []
            for (a, b), lbl in zip(age_groups, age_labels):
                if b < 15:
                    age_items.append((lbl, "child", slice(a, b + 1)))
                else:
                    art_start = max(0, a - 15)
                    art_end = min(65, b - 15) + 1
                    age_items.append((lbl, "adult", slice(art_start, art_end)))
        else:
            age_items = [(None, "total", slice(None))]

        sex_items: list[tuple[str | None, int | None]] = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )

        for age_lbl, source, age_sl in age_items:
            for sex_lbl, sex_idx in sex_items:
                if source == "child":
                    data = (
                        child_arr[age_sl, sex_idx, :] if sex_idx is not None
                        else child_arr[age_sl, :, :].sum(axis=(0, 1))
                    )
                elif source == "adult":
                    data = (
                        arr[:, :, age_sl, sex_idx, :].reshape(-1, n_years).sum(axis=0) if sex_idx is not None
                        else arr[:, :, age_sl, :, :].reshape(-1, n_years).sum(axis=0)
                    )
                else:  # "total": adult (15+) + child (0-14)
                    if sex_idx is not None:
                        data = (
                            arr[:, :, :, sex_idx, :].reshape(-1, n_years).sum(axis=0)
                            + child_arr[:, sex_idx, :].sum(axis=0)
                        )
                    else:
                        data = (
                            arr.reshape(-1, n_years).sum(axis=0)
                            + child_arr.sum(axis=(0, 1))
                        )
                parts = [p for p in [age_lbl, sex_lbl] if p]
                label = " / ".join(parts) if parts else "Total"
                series.append((label, data))
        return series
    return fn


def _disagg_prevalence() -> Callable:
    """Prevalence (%) restricted to ages 15-49. Always called via _no_age_disagg,
    which forces disagg_age=False, so the age range is hardcoded rather than
    branching on disagg_age (matching _disagg_std_1549's pattern)."""
    def fn(output: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        hiv = output["p_hivpop"]
        tot = output["p_totpop"]
        series: list[tuple[str, np.ndarray]] = []

        sex_specs = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )

        for sex_label, sex_idx in sex_specs:
            h = _sum_std(hiv, slice(15, 50), sex_idx)
            t = _sum_std(tot, slice(15, 50), sex_idx)
            data = 100.0 * h / np.where(t == 0, np.nan, t)
            label = sex_label if sex_label else "15-49"
            series.append((label, data))
        return series
    return fn


def _lf_pregnant_prevalence_disagg(output: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Prevalence (%) in pregnant women = pmtct_need (HIV+ pregnant women needing
    PMTCT) / births — both already flat per-year totals (n_years,), no age/sex
    axis to sum over (see the "Total Births" indicator's o["births"] usage).
    Inherently female-only, so disagg_sex only changes the line label, never
    adds a Male line."""
    need = output["pmtct_need"]
    births = output["births"]
    data = 100.0 * need / np.where(births == 0, np.nan, births)
    return [("Female", data)] if disagg_sex else [("Total", data)]


def _spec_pregnant_prevalence_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Prevalence (%) in pregnant women = AM_ChildNeedPMTCT_V1 (children needing
    PMTCT) / DP_Births_V1, mirroring the dp_aim column's pmtct_need / births
    computation. Both are year-only (no age/sex axis — see _spec_births_disagg
    for DP_Births_V1's 'Both Sexes'/'All Ages' pre-aggregated total). Inherently
    female-only, so disagg_sex only changes the line label."""
    need = np.array(modvars[AM_ChildNeedPMTCTTag])
    births = np.array(modvars[DP_BirthsTag])[GB_BothSexes, DP_AllAges, :]
    data = 100.0 * need / np.where(births == 0, np.nan, births)
    return [("Female", data)] if disagg_sex else [("Total", data)]


def _disagg_incidence() -> Callable:
    """Incidence (%) restricted to ages 15-49. Always called via _no_age_disagg,
    which forces disagg_age=False, so the age range is hardcoded rather than
    branching on disagg_age (matching _disagg_std_1549's pattern)."""
    def fn(output: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        inf = output["p_infections"]
        hiv = output["p_hivpop"]
        tot = output["p_totpop"]
        series: list[tuple[str, np.ndarray]] = []

        sex_specs = (
            [(sl, i) for i, sl in enumerate(SEX_LABELS)]
            if disagg_sex else [(None, None)]
        )

        for sex_label, sex_idx in sex_specs:
            i_ = _sum_std(inf, slice(15, 50), sex_idx)
            h = _sum_std(hiv, slice(15, 50), sex_idx)
            t = _sum_std(tot, slice(15, 50), sex_idx)
            hivneg = t - h
            data = 100.0 * i_ / np.where(hivneg == 0, np.nan, hivneg)
            label = sex_label if sex_label else "15-49"
            series.append((label, data))
        return series
    return fn


# ---------------------------------------------------------------------------
# Spectrum (modvars) extract functions — totals
# ---------------------------------------------------------------------------

def _spec_births_disagg(modvars: dict, _disagg_age: bool, _disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """DP_Births_V1 (3, 18, n_years): sex × 5-year age band × year. Unlike
    DP_BigPop_V1, index 0 on both axes ('Both Sexes' / 'All Ages') is a genuine
    pre-aggregated total meant to be read directly — mirrors SpectrumEngine's
    own GetDP_Births(modvars, t) accessor. No sex or age disaggregation
    available (births has no per-sex breakdown to split by)."""
    arr = np.array(modvars[DP_BirthsTag])
    return [("Total", arr[GB_BothSexes, DP_AllAges, :])]


def _spec_prevalence_1549_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Prevalence (%) computed as HV_TotalAdultsHIV_V1 / HV_Populations_V1 (both (3, 81)
    sex x year), matching how the dp_aim/goals columns derive prevalence as a ratio
    rather than reading a separate pre-computed prevalence modvar."""
    pop = np.array(modvars[HV_PopulationsTag])
    hiv = np.array(modvars[HV_TotalAdultsHIVTag])
    if disagg_sex:
        return [("Male", 100 * hiv[1] / pop[1]), ("Female", 100 * hiv[2] / pop[2])]
    return [("15-49", 100 * (hiv[1] + hiv[2]) / (pop[1] + pop[2]))]


def _spec_incidence(modvars: dict) -> np.ndarray:
    """HV_Incidence_V1 is a proportion; multiply by 100 for percent."""
    return np.array(modvars[HV_IncidenceTag]) * 100.0


def _am_prevalence_1549_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Prevalence (%) for the AIM tab: no prevalence-result modvar is written by a plain
    AIM run (update_modvars_from_state never populates one), so compute it the same way
    as the dp_aim column — AM_HIVBySingleAge_V1 / DP_BigPop_V1 ages 15-49, as a ratio."""
    hiv = np.array(modvars[AM_HIVBySingleAgeTag])
    tot = np.array(modvars[DP_BigPopTag])
    if disagg_sex:
        result = []
        for sex_idx, sex_lbl in [(1, "Male"), (2, "Female")]:
            h = hiv[sex_idx, 15:50, :].sum(axis=0)
            t = tot[sex_idx, 15:50, :].sum(axis=0)
            result.append((sex_lbl, 100.0 * h / np.where(t == 0, np.nan, t)))
        return result
    h = (hiv[1, 15:50, :] + hiv[2, 15:50, :]).sum(axis=0)
    t = (tot[1, 15:50, :] + tot[2, 15:50, :]).sum(axis=0)
    return [("15-49", 100.0 * h / np.where(t == 0, np.nan, t))]


def _am_incidence_1549_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Incidence (%) for the AIM tab, computed the same way as the dp_aim column:
    AM_NewInfectionsBySingleAge_V1 / (DP_BigPop_V1 - AM_HIVBySingleAge_V1) ages 15-49."""
    inf = np.array(modvars[AM_NewInfectionsBySingleAgeTag])
    hiv = np.array(modvars[AM_HIVBySingleAgeTag])
    tot = np.array(modvars[DP_BigPopTag])
    if disagg_sex:
        result = []
        for sex_idx, sex_lbl in [(1, "Male"), (2, "Female")]:
            i_ = inf[sex_idx, 15:50, :].sum(axis=0)
            h = hiv[sex_idx, 15:50, :].sum(axis=0)
            t = tot[sex_idx, 15:50, :].sum(axis=0)
            hivneg = t - h
            result.append((sex_lbl, 100.0 * i_ / np.where(hivneg == 0, np.nan, hivneg)))
        return result
    i_ = (inf[1, 15:50, :] + inf[2, 15:50, :]).sum(axis=0)
    h = (hiv[1, 15:50, :] + hiv[2, 15:50, :]).sum(axis=0)
    t = (tot[1, 15:50, :] + tot[2, 15:50, :]).sum(axis=0)
    hivneg = t - h
    return [("15-49", 100.0 * i_ / np.where(hivneg == 0, np.nan, hivneg))]


# ---------------------------------------------------------------------------
# Spectrum disaggregated (age + sex) extract functions
# ---------------------------------------------------------------------------

def _am_disagg_from_array(
    get_arr: Callable[[dict], np.ndarray],
    *,
    age_groups: list[tuple[int, int]] = AGE_GROUPS,
    age_labels: list[str] = AGE_LABELS,
) -> Callable:
    """Disagg factory for any modvar (or derived array) sharing DP_BigPop_V1's
    (3, 81, 81) [sex, age, year] shape — [0]=both, [1]=male, [2]=female. Totals
    are always male+female (index 1+2), never the pre-computed 'both' row.
    Used both for DP_BigPop_V1 itself (Total population, same for both the
    Goals and AIM tabs) and for the AM_* modvars that back the AIM tab's
    DP/AIM-derived comparisons (as opposed to the Goals tab's HV_*-based
    "spectrum" functions — HV_* tags are Goals-specific and aren't populated
    by a plain AIM-only Spectrum run). `age_groups`/`age_labels` default to the
    5-year display buckets; pass AGE_GROUPS_SINGLE/AGE_LABELS_SINGLE for
    single-year-of-age resolution (the "By age" sub-tab)."""
    def fn(modvars: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = get_arr(modvars)  # (3, 81_ages, 81_years)

        if disagg_age:
            age_items = [(lbl, slice(a, b + 1)) for (a, b), lbl in zip(age_groups, age_labels)]
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
    return fn


def _am_disagg(
    tag,
    *,
    age_groups: list[tuple[int, int]] = AGE_GROUPS,
    age_labels: list[str] = AGE_LABELS,
) -> Callable:
    """_am_disagg_from_array, reading a single modvar tag directly."""
    return _am_disagg_from_array(lambda modvars: np.array(modvars[tag]), age_groups=age_groups, age_labels=age_labels)


_spec_totpop_disagg = _am_disagg(DP_BigPopTag)
_am_hivpop_disagg = _am_disagg(AM_HIVBySingleAgeTag)
_am_newinf_disagg = _am_disagg(AM_NewInfectionsBySingleAgeTag)
_am_art_disagg = _am_disagg(AM_OnARTBySingleAgeTag)

# AIDS deaths (all ages) for the AIM tab, derived as AM_AIDSDeathsARTSingleAgeTag +
# AM_AIDSDeathsNoARTSingleAgeTag — both share DP_BigPop_V1's (sex=3, single_age=81, T)
# shape, so the same age/sex disaggregation logic applies directly.
_am_aidsdeath_disagg = _am_disagg_from_array(
    lambda modvars: np.array(modvars[AM_AIDSDeathsARTSingleAgeTag]) + np.array(modvars[AM_AIDSDeathsNoARTSingleAgeTag])
)


def _lf_total_deaths_disagg(output: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Total deaths = background non-HIV deaths + HIV deaths + excess non-AIDS
    deaths, each (81, 2, n_years). Never branches on disagg_age (same idiom as
    "Total Births"'s dp_aim fn) — no age breakdown is offered for either source,
    since Spectrum's DP_Deaths_V1 is only available in 5-year age bands, not
    single-year-of-age."""
    arr = (
        output["p_deaths_background_totpop"]
        + output["p_hiv_deaths"]
        + output["p_deaths_excess_nonaids"]
    )
    if disagg_sex:
        return [("Male", _sum_std(arr, sex=0)), ("Female", _sum_std(arr, sex=1))]
    return [("Total", _sum_std(arr))]


def _spec_deaths_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """DP_Deaths_V1 (3, 18, n_years): sex x 5-year age band x year, same shape as
    DP_Births_V1 — but unlike DP_Births_V1's GB_BothSexes row (a genuine
    pre-aggregated total), DP_Deaths_V1's GB_BothSexes row isn't reliable, so the
    total is always Male + Female summed manually, matching this module's usual
    convention for every other non-Births modvar. No age disaggregation offered
    (deaths are only available in 5-year age bands here).

    DP_Deaths_V1 alone is *non-AIDS* deaths only — it excludes AIDS mortality,
    which Spectrum reports separately via AM_AIDSDeathsARTSingleAge_V1 /
    AM_AIDSDeathsNoARTSingleAge_V1 (confirmed empirically: leapfrog's dp_aim total
    deaths, background + HIV + excess non-AIDS, minus DP_Deaths_V1 reproduces
    Spectrum's own AIDS death count to within rounding, across the full
    projection). Both AIDS-death tags share DP_BigPop_V1's (sex=3, single_age=81,
    T) shape, so they're summed over age and added in here to make this an
    apples-to-apples total against the dp_aim side's _lf_total_deaths_disagg."""
    arr = np.array(modvars[DP_DeathsTag])
    male = arr[GB_Male, DP_AllAges, :]
    female = arr[GB_Female, DP_AllAges, :]

    aids_art = np.array(modvars[AM_AIDSDeathsARTSingleAgeTag])
    aids_noart = np.array(modvars[AM_AIDSDeathsNoARTSingleAgeTag])
    aids_male = (aids_art[1] + aids_noart[1]).sum(axis=0)
    aids_female = (aids_art[2] + aids_noart[2]).sum(axis=0)

    male = male + aids_male
    female = female + aids_female
    if disagg_sex:
        return [("Male", male), ("Female", female)]
    return [("Total", male + female)]


# ---------------------------------------------------------------------------
# "Deaths" sub-tab: drill-down into the components of "Total deaths", each
# aggregated fully over age (year on the x-axis), with an optional sex split.
# Every dp_aim array here ends in (..., NS=2, T) — p_deaths_* end in (age, NS,
# T), h_hiv_deaths_*/h_deaths_excess_* end in (..., hAG, NS, T), hc1/hc2_*
# end in (..., hc{1,2}AG, NS, T) — so summing every axis but the last two
# always collapses to a (NS, T) total, regardless of which array it is.
# ---------------------------------------------------------------------------

def _lf_deaths_disagg(*keys: str) -> Callable:
    """Sum one or more dp_aim/goals output arrays (each ending in (..., NS=2, T))
    fully over every axis except sex and year, then add them together. Used for
    every "Deaths" sub-tab indicator's dp_aim/goals side — most map to a single
    output key, but e.g. "AIDS deaths (on ART)" is the sum of the adult
    (h_hiv_deaths_art) and child (hc1_art_aids_deaths, hc2_art_aids_deaths)
    arrays, matching how LeapfrogDataMapping.py assembles Spectrum's single-age
    AM_AIDSDeathsARTSingleAge_V1 from the same three leapfrog outputs. Never
    branches on disagg_age (same idiom as _lf_total_deaths_disagg) — the
    "Deaths" sub-tab offers no age facet."""
    def fn(output: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        total = None
        for key in keys:
            arr = output[key]
            summed = arr.sum(axis=tuple(range(arr.ndim - 2)))  # -> (NS=2, T)
            total = summed if total is None else total + summed
        if disagg_sex:
            return [("Male", total[0]), ("Female", total[1])]
        return [("Total", total[0] + total[1])]
    return fn


# AM_AIDSDeathsARTSingleAgeTag/AM_AIDSDeathsNoARTSingleAgeTag: see _am_aidsdeath_disagg
# above — same (sex=3, single_age=81, T) shape as DP_BigPop_V1, split by ART status
# rather than summed together.
_spec_aidsdeath_art_disagg = _am_disagg(AM_AIDSDeathsARTSingleAgeTag)
_spec_aidsdeath_noart_disagg = _am_disagg(AM_AIDSDeathsNoARTSingleAgeTag)

# DP_DeathsByAgeTag: background (non-HIV) deaths, single year of age — same
# (sex=3, single_age=81, T) shape as DP_BigPop_V1, unlike the 5-year-banded
# DP_DeathsTag used by "Total deaths"/_spec_deaths_disagg above.
_spec_bg_deaths_totpop_disagg = _am_disagg(DP_DeathsByAgeTag)

# AM_NonAIDSExcessDeathsSingleAgeTag (ART_status=3, sex=3, single_age=81, T):
# index 0 is a precomputed "NoTreat + OnART" sum, but per this module's usual
# convention (see the file docstring) precomputed totals aren't trusted — the
# "total" variant below sums DP_NoTreat + DP_OnART manually instead of reading
# index 0.
_spec_excess_nonaids_art_disagg = _am_disagg_from_array(
    lambda modvars: np.array(modvars[AM_NonAIDSExcessDeathsSingleAgeTag])[DP_OnART]
)
_spec_excess_nonaids_noart_disagg = _am_disagg_from_array(
    lambda modvars: np.array(modvars[AM_NonAIDSExcessDeathsSingleAgeTag])[DP_NoTreat]
)
_spec_excess_nonaids_total_disagg = _am_disagg_from_array(
    lambda modvars: (
        np.array(modvars[AM_NonAIDSExcessDeathsSingleAgeTag])[DP_NoTreat]
        + np.array(modvars[AM_NonAIDSExcessDeathsSingleAgeTag])[DP_OnART]
    )
)

# Non-AIDS deaths in the HIV+ population, by ART status — background and
# excess deaths bundled together, since Spectrum has no isolated single-cause
# modvar to compare against leapfrog's separate p_deaths_nonaids_artpop/hivpop
# (background only). The dp_aim side sums the matching leapfrog background +
# excess pair so both lines represent the same combined total (see
# _lf_deaths_disagg calls below).
#
# Rather than reading the precomputed AM_ResNonAIDSDeathsHIVPopART_V1/NoART_V1
# report modvars directly (Spectrum only populates those once a report screen
# actually requests them — they read back as all-zero on a plain PJNZ import),
# this is a direct port of Spec5's DP/MAIN/DPUTIL.PAS
# Get_NonAIDSDeathsHIVPopART/NoART, which derive the same figure from four
# modvars that *are* always populated: DP_BigPop_V1, AM_OnARTBySingleAge_V1,
# AM_HIVBySingleAge_V1, DP_DeathsByAge_V1, and AM_NonAIDSExcessDeathsSingleAge_V2.
# DP_DeathsByAge_V1 ("deaths_all") is *every* non-AIDS death — background and
# excess combined, HIV+ and HIV- people alike; subtracting the two excess
# terms isolates the background-only share, which Get_NonAIDSDeathsHIVPopART
# reallocates by ART population share (Get_NonAIDSDeathsHIVPopNoART is the
# mirror image for the not-on-ART share of the HIV+ population):
#   ART:    deaths_art + (deaths_all - deaths_off - deaths_art) * pop_art / pop_all
#   NoART:  deaths_off + (deaths_all - deaths_off - deaths_art) * (pop_hiv - pop_art) / pop_all
def _spec_nonaids_hivpop_disagg(art: bool) -> Callable:
    def fn(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        pop_all = np.array(modvars[DP_BigPopTag])              # (sex=3, age=81, T)
        pop_art = np.array(modvars[AM_OnARTBySingleAgeTag])     # (sex=3, age=81, T)
        pop_hiv = np.array(modvars[AM_HIVBySingleAgeTag])       # (sex=3, age=81, T)
        deaths_all = np.array(modvars[DP_DeathsByAgeTag])       # (sex=3, age=81, T)
        excess = np.array(modvars[AM_NonAIDSExcessDeathsSingleAgeTag])  # (ART_status=3, sex=3, age=81, T)
        deaths_off = excess[DP_NoTreat]
        deaths_art = excess[DP_OnART]

        bg_share = deaths_all - deaths_off - deaths_art
        safe_pop_all = np.where(pop_all > 0, pop_all, 1.0)
        if art:
            per_age = deaths_art + np.where(pop_all > 0, bg_share * pop_art / safe_pop_all, 0.0)
        else:
            per_age = deaths_off + np.where(pop_all > 0, bg_share * (pop_hiv - pop_art) / safe_pop_all, 0.0)

        if disagg_sex:
            return [
                ("Male", per_age[GB_Male].sum(axis=0)),
                ("Female", per_age[GB_Female].sum(axis=0)),
            ]
        return [("Total", (per_age[GB_Male] + per_age[GB_Female]).sum(axis=0))]
    return fn


_spec_nonaids_hivpop_art_disagg = _spec_nonaids_hivpop_disagg(art=True)
_spec_nonaids_hivpop_noart_disagg = _spec_nonaids_hivpop_disagg(art=False)


# ---------------------------------------------------------------------------
# Single-year-of-age variants for the "By age" sub-tab (see IndicatorDef.age_profile
# below). Same underlying arrays/logic as the 5-year-bucketed versions above,
# just called with AGE_GROUPS_SINGLE/AGE_LABELS_SINGLE instead of the display
# bucketing, since the raw data is single-year resolution already.
# ---------------------------------------------------------------------------
_disagg_std_single_age = lambda key: _disagg_std(key, age_groups=AGE_GROUPS_SINGLE, age_labels=AGE_LABELS_SINGLE)
_disagg_art_single_age = _disagg_art(age_groups=AGE_GROUPS_SINGLE, age_labels=AGE_LABELS_SINGLE)

_spec_totpop_disagg_single = _am_disagg(DP_BigPopTag, age_groups=AGE_GROUPS_SINGLE, age_labels=AGE_LABELS_SINGLE)
_am_hivpop_disagg_single = _am_disagg(AM_HIVBySingleAgeTag, age_groups=AGE_GROUPS_SINGLE, age_labels=AGE_LABELS_SINGLE)
_am_newinf_disagg_single = _am_disagg(AM_NewInfectionsBySingleAgeTag, age_groups=AGE_GROUPS_SINGLE, age_labels=AGE_LABELS_SINGLE)
_am_art_disagg_single = _am_disagg(AM_OnARTBySingleAgeTag, age_groups=AGE_GROUPS_SINGLE, age_labels=AGE_LABELS_SINGLE)
_am_aidsdeath_disagg_single = _am_disagg_from_array(
    lambda modvars: np.array(modvars[AM_AIDSDeathsARTSingleAgeTag]) + np.array(modvars[AM_AIDSDeathsNoARTSingleAgeTag]),
    age_groups=AGE_GROUPS_SINGLE, age_labels=AGE_LABELS_SINGLE,
)


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


def _spec_hivpop_1549_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_TotalAdultsHIV_V1 — Spectrum's Goals/HV-module adult PLHIV total. Like
    leapfrog's own Goals output, it covers only the 15-49 risk-group cohort
    (verified against a PJNZ import: matches AM_HIVBySingleAgeTag summed over ages
    15-49, ~0.35x the 15-80 sum), so the '15-49' label is accurate — not a
    stand-in for a missing slice."""
    arr = np.array(modvars[HV_TotalAdultsHIVTag])
    if disagg_sex:
        return [("Male", arr[1]), ("Female", arr[2])]
    return [("15-49", arr[1] + arr[2])]


def _am_art_15plus_disagg(modvars: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Genuine 15+ on-ART total, from AM_OnARTBySingleAgeTag (sex=3, single_age=81,
    T) restricted to ages 15-80 — this lines up with leapfrog's h_artpop-derived
    "dp_aim" line, which spans ages 15-80 and has no under-15 data.

    Used for BOTH the "spectrum" and "spectrum_aim" sources of the "Total number
    receiving ART (15+)" indicator. HV_TotalAdultsART_V1 is deliberately NOT used
    here: it is Spectrum's Goals/HV-module total and — like leapfrog's own
    hv.total_on_art — only covers the 15-49 risk-group cohort (verified against a
    PJNZ import: it matches AM_OnARTBySingleAgeTag summed over ages 15-49 to
    within ~1%, and is only ~0.35x the 15-80 sum). Using it here compared a
    Spectrum 15-49 figure against a leapfrog 15+ figure.

    In the age-faceted view, falls back to the standard 5-year-band
    disaggregation (same as the other AM_* single-age sources)."""
    if disagg_age:
        return _am_art_disagg(modvars, True, disagg_sex)
    arr = np.array(modvars[AM_OnARTBySingleAgeTag])
    if disagg_sex:
        return [
            ("Male", arr[1, 15:81, :].sum(axis=0)),
            ("Female", arr[2, 15:81, :].sum(axis=0)),
        ]
    return [("Total", (arr[1, 15:81, :] + arr[2, 15:81, :]).sum(axis=0))]


def _spec_art_1549_disagg(modvars: dict, _disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """HV_TotalAdultsART_V1 — Spectrum's Goals/HV-module on-ART total. Covers only
    the 15-49 risk-group cohort (verified against a PJNZ import: matches
    AM_OnARTBySingleAgeTag summed over ages 15-49 to within ~1%, vs ~0.35x the
    15-80 sum), so the '15-49' label is accurate. The genuine 15+ figure comes
    from AM_OnARTBySingleAgeTag instead — see _am_art_15plus_disagg."""
    arr = np.array(modvars[HV_TotalAdultsARTTag])
    if disagg_sex:
        return [("Male", arr[1]), ("Female", arr[2])]
    return [("15-49", arr[1] + arr[2])]


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


def _lf_total_deaths_1549_disagg(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """Total deaths restricted to ages 15-49 (see _lf_total_deaths_disagg for the
    all-ages version). Returns [] when disagg_age=True, matching every other
    15-49 indicator's _disagg_std_1549-style idiom."""
    if disagg_age:
        return []
    arr = (
        output["p_deaths_background_totpop"]
        + output["p_hiv_deaths"]
        + output["p_deaths_excess_nonaids"]
    )
    if disagg_sex:
        return [("Male", _sum_std(arr, slice(15, 50), 0)), ("Female", _sum_std(arr, slice(15, 50), 1))]
    return [("15-49", _sum_std(arr, slice(15, 50)))]


def _spec_deaths_1549_disagg(modvars: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """DP_Deaths_V1 restricted to the 15-49 age bands: the 5-year bands 15-19..45-49
    are age-axis indices 4-10 (AGE_GROUPS index 3-9, offset +1 for the 'All Ages'
    slot at index 0). Unlike the all-ages version, GB_BothSexes can't be used
    directly here — that pre-aggregated total is only valid for the full
    DP_AllAges slice — so Male+Female bands are summed manually, matching
    _am_disagg_1549's convention.

    Like _spec_deaths_disagg, DP_Deaths_V1 is non-AIDS deaths only, so
    AM_AIDSDeathsARTSingleAge_V1/NoARTSingleAge_V1 (single-year-of-age, ages
    15-49 = indices 15:50) are added in to match the dp_aim side's
    _lf_total_deaths_1549_disagg, which includes HIV deaths."""
    if disagg_age:
        return []
    arr = np.array(modvars[DP_DeathsTag])

    aids_art = np.array(modvars[AM_AIDSDeathsARTSingleAgeTag])
    aids_noart = np.array(modvars[AM_AIDSDeathsNoARTSingleAgeTag])
    aids_male = (aids_art[1] + aids_noart[1])[15:50, :].sum(axis=0)
    aids_female = (aids_art[2] + aids_noart[2])[15:50, :].sum(axis=0)

    male = arr[GB_Male, 4:11, :].sum(axis=0) + aids_male
    female = arr[GB_Female, 4:11, :].sum(axis=0) + aids_female
    if disagg_sex:
        return [("Male", male), ("Female", female)]
    return [("15-49", male + female)]


def _no_age_disagg(disagg_fn: Callable) -> Callable:
    """Wraps any disagg function to return [] when disagg_age=True."""
    def wrapper(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        if disagg_age:
            return []
        return disagg_fn(output, False, disagg_sex)
    return wrapper


def _lf_artpop_1549(output: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """h_artpop (4, 7, 66, 2, n_years) summed over ART-duration/CD4, restricted to ages
    15-49 (the first 35 single-year entries of the 66-length adult age axis, which
    starts at age 15). Returns [] when disagg_age=True."""
    if disagg_age:
        return []
    art_pop = output["h_artpop"].sum(axis=(0, 1))  # (66, 2, n_years)
    if disagg_sex:
        return [("Male", art_pop[:35, 0, :].sum(axis=0)), ("Female", art_pop[:35, 1, :].sum(axis=0))]
    return [("15-49", art_pop[:35, :, :].sum(axis=(0, 1)))]


# ---------------------------------------------------------------------------
# Spectrum helpers for 15-49 sub-range
# ---------------------------------------------------------------------------

def _am_disagg_1549(tag) -> Callable:
    """Disagg factory for a DP_BigPop_V1-shaped modvar restricted to ages 15-49;
    returns [] in the age-faceted view. Used for both DP_BigPop_V1 itself and the
    AM_* modvars (see _am_disagg above)."""
    def fn(modvars: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        if disagg_age:
            return []
        arr = np.array(modvars[tag])
        if disagg_sex:
            return [
                ("Male", arr[1, 15:50, :].sum(axis=0)),
                ("Female", arr[2, 15:50, :].sum(axis=0)),
            ]
        return [("15-49", (arr[1, 15:50, :] + arr[2, 15:50, :]).sum(axis=0))]
    return fn


_spec_totpop_1549_disagg = _am_disagg_1549(DP_BigPopTag)
_am_hivpop_1549_disagg = _am_disagg_1549(AM_HIVBySingleAgeTag)
_am_newinf_1549_disagg = _am_disagg_1549(AM_NewInfectionsBySingleAgeTag)
_am_art_1549_disagg = _am_disagg_1549(AM_OnARTBySingleAgeTag)


def _am_aidsdeath_1549_disagg(modvars: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """AIDS deaths (15-49) for the AIM tab, derived as AM_AIDSDeathsARTSingleAgeTag +
    AM_AIDSDeathsNoARTSingleAgeTag summed over ages 15-49 (each shape (sex=3,
    single_age=81, T); index 0='both' unused, always sum male(1)+female(2)
    manually — same convention as the rest of this module's AM_* handling)."""
    if disagg_age:
        return []
    total = np.array(modvars[AM_AIDSDeathsARTSingleAgeTag]) + np.array(modvars[AM_AIDSDeathsNoARTSingleAgeTag])
    if disagg_sex:
        return [
            ("Male", total[1, 15:50, :].sum(axis=0)),
            ("Female", total[2, 15:50, :].sum(axis=0)),
        ]
    return [("15-49", (total[1, 15:50, :] + total[2, 15:50, :]).sum(axis=0))]


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


def _goals_plhiv(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """total_plhiv (n_years,) — scalar, no sex disagg available."""
    if disagg_sex:
        return []
    return [("15-49", goals_output["total_plhiv"])]


def _goals_total_on_art(goals_output: dict, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
    """total_on_art (n_years,) — scalar, no sex disagg available."""
    if disagg_sex:
        return []
    return [("15-49", goals_output["total_on_art"])]


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
    """prevalence (18, 3, n_years): last risk-group index (17, "RG_ALL") is the aggregate row;
    sex axis [0]=Male, [1]=Female, [2]=Both. Multiplied by 100 → percent."""
    arr = goals_output["prevalence"][len(goals_output["prevalence"])-1] * 100.0  # (3, n_years)
    if disagg_sex:
        return [("Male", arr[0]), ("Female", arr[1])]
    return [("15-49", arr[2])]


# ---------------------------------------------------------------------------
# Risk group definitions and compute functions ("Risk groups" / "New infections"
# sub-tabs). Ported verbatim from the reference implementation (see git history)
# rather than derived independently — these plots use a dedicated one-row-per-
# risk-group layout (plotting.render_risk_group_comparison), not the standard
# IndicatorDef/render_comparison path, so they are wired directly in app.py.
# ---------------------------------------------------------------------------

# Goals adults array: shape (nVAC+1=5, nRG+1=18, nCD4+1=17, nNS+1=3, n_years)
_VAC_ALL = 4
_VAC_UNV = 0
_CD4_ALL = 16
_RG_ALL = 17

# (display_name, goals_rg_index) in display order
RISK_GROUPS: list[tuple[str, int]] = [
    ("Low risk", 1),     # RG_LRH
    ("Medium risk", 2),  # RG_MRH
    ("High risk", 3),    # RG_HRH
    ("PWID", 4),         # RG_IDU
    ("MSM", 5),          # RG_MSM
]

# Maps display name -> Spectrum HV_Adults risk-group index
_SPEC_RG_INDICES: dict[str, int] = {
    "Low risk": HV_LRH,
    "Medium risk": HV_MRH,
    "High risk": HV_HRH,
    "PWID": HV_IDU,
    "MSM": HV_MSM,
}


def compute_rg_goals(goals_output: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
    """
    Risk-group fractions (% of 15-49 population in each risk group) from Goals
    'adults' (5, 18, 17, 3, n_years). Returns list of (rg_name, demo, ratio) where:
      ratio = 100 * adults[VAC_ALL, rg_idx, CD4_ALL, sex] / adults[VAC_ALL, RG_ALL, CD4_ALL, sex]
    When disagg_sex is False, male + female are summed before dividing (except MSM,
    which always uses the male-only denominator since female MSM values are 0).
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
                # Only ever use men as denominator for MSM; women are 0 so numerator doesn't matter
                den = adults[_VAC_ALL, _RG_ALL, _CD4_ALL, 0]
            else:
                den = adults[_VAC_ALL, _RG_ALL, _CD4_ALL, 0] + adults[_VAC_ALL, _RG_ALL, _CD4_ALL, 1]
            result.append((rg_name, "Total", 100 * num / np.where(den == 0, np.nan, den)))
    return result


def compute_rg_spectrum(modvars: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
    """
    Risk-group fractions from Spectrum HV_Adults_V1 (sex, rg, hiv, vac, n_years).
    Indexed as hv_adults[sex, rg, HV_AllHIV, RN_AllVacc, :].
    Returns list of (rg_name, demo, ratio*100). When disagg_sex is False, male (1) +
    female (2) are summed before dividing (except MSM: male-only denominator).
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


def compute_new_infections_rg_goals(goals_output: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
    """New infections by risk group from Goals 'new_inf_vrs' (nVAC+1, nRG+1, nNS+1, n_years).
    Returns list of (rg_name, demo, count)."""
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


def compute_new_infections_rg_spectrum(modvars: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
    """New infections by risk group from Spectrum HV_NewInfections_V1 (sex, rg, vac, year).
    Uses RN_UnV for vaccine index. Returns list of (rg_name, demo, count)."""
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


@dataclass
class RiskGroupIndicatorDef:
    """One selectable indicator for the risk-group sub-tabs' Indicator dropdown
    (comparison_module.risk_group_panel_ui/server, multi_risk_group_panel_server).
    `compute_fns` has the same (data, disagg_sex) -> list[(rg_name, demo, ndarray)]
    shape as a bare RiskGroupSubTab.compute_fns dict; `title_prefix`/`y_title` are
    carried alongside since the two indicators here (population share vs. new
    infections) have different units and the row layout (risk-group names) gives no
    clue which one is showing."""
    compute_fns: dict[str, Callable[[dict, bool], list[tuple[str, str, np.ndarray]]]]
    title_prefix: str
    y_title: str | None = None


RISK_GROUP_INDICATOR_NAMES: list[str] = ["Risk group population share", "New infections"]

RISK_GROUP_INDICATOR_MAP: OrderedDict[str, RiskGroupIndicatorDef] = OrderedDict([
    ("Risk group population share", RiskGroupIndicatorDef(
        compute_fns={"goals": compute_rg_goals, "spectrum": compute_rg_spectrum},
        title_prefix="Risk group population share",
        y_title="% of population aged 15-49",
    )),
    ("New infections", RiskGroupIndicatorDef(
        compute_fns={"goals": compute_new_infections_rg_goals, "spectrum": compute_new_infections_rg_spectrum},
        title_prefix="New infections by risk group",
        y_title="New infections",
    )),
])


# ---------------------------------------------------------------------------
# Child (0-14) CD4-faceted indicators ("0-14" sub-tab). Each compute fn has
# the same (data, disagg_sex) -> list[(facet_label, demo, ndarray)] shape as
# the risk-group compute fns above, so they plug directly into
# plotting.render_risk_group_comparison unmodified — the "facet_label" is a
# CD4 stage (population indicators) or "Total" (death indicators, since no
# Spectrum modvar is CD4-stratified for child deaths — see below).
# ---------------------------------------------------------------------------

def _lf_child_cd4_noart(key: str, cd4_labels: list[str]) -> Callable:
    """Factory for hc{1,2}_hivpop, shape (CD4, hcTT=4, AGE, NS=2, T). Sums
    hcTT (axis 1) and AGE (axis 2); CD4 (axis 0) is the facet axis; NS
    (axis 3, 0=Male/1=Female) is the optional sex split."""
    def fn(output: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
        summed = output[key].sum(axis=(1, 2))  # -> (CD4, NS=2, T)
        result: list[tuple[str, str, np.ndarray]] = []
        for c_idx, cd4_label in enumerate(cd4_labels):
            if disagg_sex:
                for sex_idx, sex_label in enumerate(SEX_LABELS):
                    result.append((cd4_label, sex_label, summed[c_idx, sex_idx, :]))
            else:
                result.append((cd4_label, "Total", summed[c_idx, 0, :] + summed[c_idx, 1, :]))
        return result
    return fn


def _lf_child_cd4_art(key: str, cd4_labels: list[str]) -> Callable:
    """Factory for hc{1,2}_artpop, shape (hTS=3, CD4, AGE, NS=2, T). Sums
    hTS (axis 0) and AGE (axis 2); CD4 is axis 1 here."""
    def fn(output: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
        summed = output[key].sum(axis=(0, 2))  # -> (CD4, NS=2, T)
        result: list[tuple[str, str, np.ndarray]] = []
        for c_idx, cd4_label in enumerate(cd4_labels):
            if disagg_sex:
                for sex_idx, sex_label in enumerate(SEX_LABELS):
                    result.append((cd4_label, sex_label, summed[c_idx, sex_idx, :]))
            else:
                result.append((cd4_label, "Total", summed[c_idx, 0, :] + summed[c_idx, 1, :]))
        return result
    return fn


def _spec_child_cd4_noart(age_grp: int, cd4_offset: int, cd4_labels: list[str]) -> Callable:
    """AM_CD4DistributionChildTag[sex, age_grp, CD4, TT, DP_NoTreat, T]. Sums
    the TT axis (DP_P_Perinatal..+4, 4 values). Sex axis: 1=Male, 2=Female
    (index 0='both' is unused, per this module's docstring — always sum
    1+2 manually)."""
    def fn(modvars: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
        arr = np.array(modvars[AM_CD4DistributionChildTag])
        result: list[tuple[str, str, np.ndarray]] = []
        for c_idx, cd4_label in enumerate(cd4_labels):
            c = c_idx + cd4_offset
            sub = arr[:, age_grp, c, DP_P_Perinatal:DP_P_Perinatal + 4, DP_NoTreat, :].sum(axis=1)
            if disagg_sex:
                result.append((cd4_label, "Male", sub[1]))
                result.append((cd4_label, "Female", sub[2]))
            else:
                result.append((cd4_label, "Total", sub[1] + sub[2]))
        return result
    return fn


def _spec_child_cd4_art(age_grp: int, cd4_offset: int, cd4_labels: list[str]) -> Callable:
    """AM_CD4DistributionChildTag[sex, age_grp, CD4, ARTdur, DP_OnART, T].
    Sums the ART-duration axis (DP_D_ARTlt6m..+3, 3 values)."""
    def fn(modvars: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
        arr = np.array(modvars[AM_CD4DistributionChildTag])
        result: list[tuple[str, str, np.ndarray]] = []
        for c_idx, cd4_label in enumerate(cd4_labels):
            c = c_idx + cd4_offset
            sub = arr[:, age_grp, c, DP_D_ARTlt6m:DP_D_ARTlt6m + 3, DP_OnART, :].sum(axis=1)
            if disagg_sex:
                result.append((cd4_label, "Male", sub[1]))
                result.append((cd4_label, "Female", sub[2]))
            else:
                result.append((cd4_label, "Total", sub[1] + sub[2]))
        return result
    return fn


_spec_child_cd4_hc1_noart = _spec_child_cd4_noart(DP_CD4_0t4, DP_CD4_Per_GT30, CD4_LABELS_HC1)
_spec_child_cd4_hc2_noart = _spec_child_cd4_noart(DP_CD4_5t14, DP_CD4_Ped_GT1000, CD4_LABELS_HC2)
_spec_child_cd4_hc1_art = _spec_child_cd4_art(DP_CD4_0t4, DP_CD4_Per_GT30, CD4_LABELS_HC1)
_spec_child_cd4_hc2_art = _spec_child_cd4_art(DP_CD4_5t14, DP_CD4_Ped_GT1000, CD4_LABELS_HC2)


def _lf_child_total(key: str) -> Callable:
    """Sum an hc{1,2}_*_aids_deaths array fully over every axis except the
    last two (NS, T) — both death-array families reduce the same way (3
    axes to collapse, whichever CD4/TT-or-ARTdur/age axes those are for
    that particular array). Produces a single 'Total' facet row, since
    Spectrum has no CD4 breakdown to compare against for child deaths."""
    def fn(output: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
        arr = output[key]
        summed = arr.sum(axis=tuple(range(arr.ndim - 2)))  # -> (NS=2, T)
        if disagg_sex:
            return [("Total", SEX_LABELS[0], summed[0]), ("Total", SEX_LABELS[1], summed[1])]
        return [("Total", "Total", summed[0] + summed[1])]
    return fn


def _spec_child_death_total(tag, age_slice: slice) -> Callable:
    """AM_AIDSDeathsARTSingleAgeTag / AM_AIDSDeathsNoARTSingleAgeTag, shape
    (sex=3, single_age=81, T) — no CD4 axis exists in this modvar. Restrict
    to the hc1 (0:5) or hc2 (5:15) age slice and sum over age."""
    def fn(modvars: dict, disagg_sex: bool) -> list[tuple[str, str, np.ndarray]]:
        arr = np.array(modvars[tag])
        sub = arr[:, age_slice, :].sum(axis=1)  # -> (sex=3, T)
        if disagg_sex:
            return [("Total", "Male", sub[1]), ("Total", "Female", sub[2])]
        return [("Total", "Total", sub[1] + sub[2])]
    return fn


_spec_child_death_hc1_art = _spec_child_death_total(AM_AIDSDeathsARTSingleAgeTag, slice(0, 5))
_spec_child_death_hc2_art = _spec_child_death_total(AM_AIDSDeathsARTSingleAgeTag, slice(5, 15))
_spec_child_death_hc1_noart = _spec_child_death_total(AM_AIDSDeathsNoARTSingleAgeTag, slice(0, 5))
_spec_child_death_hc2_noart = _spec_child_death_total(AM_AIDSDeathsNoARTSingleAgeTag, slice(5, 15))


@dataclass
class ChildCD4IndicatorDef:
    """Like IndicatorDef, but for the CD4-faceted child sub-tab: `compute_fns`
    has the (data, disagg_sex) -> list[(facet_label, demo, ndarray)] shape
    used by plotting.render_risk_group_comparison, and `cd4_labels` supplies
    that indicator's row order/count (7 for hc1/0-4, 6 for hc2/5-14, or a
    single "Total" row for the death indicators)."""
    cd4_labels: list[str]
    compute_fns: dict[str, Callable[[dict, bool], list[tuple[str, str, np.ndarray]]]]


CHILD_CD4_INDICATOR_NAMES: list[str] = [
    "0-4 HIV population (not on ART)", "5-14 HIV population (not on ART)",
    "0-4 On ART population (by CD4 at ART initiation)", "5-14 On ART population (by CD4 at ART initiation)",
    "0-4 AIDS deaths (on ART)", "5-14 AIDS deaths (on ART)",
    "0-4 AIDS deaths (not on ART)", "5-14 AIDS deaths (not on ART)",
]

CHILD_CD4_INDICATOR_MAP: OrderedDict[str, ChildCD4IndicatorDef] = OrderedDict([
    # "HIV population" here is specifically the not-on-ART stock: both dp_aim's
    # hc1_hivpop/hc2_hivpop (see leapfrog's variables.Rmd: "PLHIV ... not on
    # ART ...") and Spectrum's AM_CD4DistributionChildTag DP_NoTreat branch are
    # defined that way — neither model has a single combined "PLHIV regardless
    # of ART status" stock, so the two sides are a correct match, just not an
    # all-PLHIV total. Labelled accordingly rather than changing what's plotted.
    ("0-4 HIV population (not on ART)", ChildCD4IndicatorDef(CD4_LABELS_HC1, {
        "dp_aim": _lf_child_cd4_noart("hc1_hivpop", CD4_LABELS_HC1),
        "spectrum_aim": _spec_child_cd4_hc1_noart, "spectrum": _spec_child_cd4_hc1_noart,
    })),
    ("5-14 HIV population (not on ART)", ChildCD4IndicatorDef(CD4_LABELS_HC2, {
        "dp_aim": _lf_child_cd4_noart("hc2_hivpop", CD4_LABELS_HC2),
        "spectrum_aim": _spec_child_cd4_hc2_noart, "spectrum": _spec_child_cd4_hc2_noart,
    })),
    # "On ART population" is broken down by the CD4 category a child was in
    # *at ART initiation* — both dp_aim's hc1_artpop/hc2_artpop (the CD4 index
    # is fixed at initiation and never reclassified as ART duration advances,
    # see child_model_simulation.hpp's progress_time_on_art) and Spectrum's
    # AM_CD4DistributionChildTag DP_OnART branch (same fixed-CD4 + ART-duration
    # axis structure) track it this way, not by current/recovered CD4. Summed
    # across CD4 rows the total is still the true current on-ART population;
    # only the per-row breakdown reflects CD4 at initiation.
    ("0-4 On ART population (by CD4 at ART initiation)", ChildCD4IndicatorDef(CD4_LABELS_HC1, {
        "dp_aim": _lf_child_cd4_art("hc1_artpop", CD4_LABELS_HC1),
        "spectrum_aim": _spec_child_cd4_hc1_art, "spectrum": _spec_child_cd4_hc1_art,
    })),
    ("5-14 On ART population (by CD4 at ART initiation)", ChildCD4IndicatorDef(CD4_LABELS_HC2, {
        "dp_aim": _lf_child_cd4_art("hc2_artpop", CD4_LABELS_HC2),
        "spectrum_aim": _spec_child_cd4_hc2_art, "spectrum": _spec_child_cd4_hc2_art,
    })),
    # Death indicators: no Spectrum modvar is CD4-stratified for deaths, so
    # these use cd4_labels=["Total"] — render_risk_group_comparison then
    # renders a single row instead of 6/7.
    ("0-4 AIDS deaths (on ART)", ChildCD4IndicatorDef(["Total"], {
        "dp_aim": _lf_child_total("hc1_art_aids_deaths"),
        "spectrum_aim": _spec_child_death_hc1_art, "spectrum": _spec_child_death_hc1_art,
    })),
    ("5-14 AIDS deaths (on ART)", ChildCD4IndicatorDef(["Total"], {
        "dp_aim": _lf_child_total("hc2_art_aids_deaths"),
        "spectrum_aim": _spec_child_death_hc2_art, "spectrum": _spec_child_death_hc2_art,
    })),
    ("0-4 AIDS deaths (not on ART)", ChildCD4IndicatorDef(["Total"], {
        "dp_aim": _lf_child_total("hc1_noart_aids_deaths"),
        "spectrum_aim": _spec_child_death_hc1_noart, "spectrum": _spec_child_death_hc1_noart,
    })),
    ("5-14 AIDS deaths (not on ART)", ChildCD4IndicatorDef(["Total"], {
        "dp_aim": _lf_child_total("hc2_noart_aids_deaths"),
        "spectrum_aim": _spec_child_death_hc2_noart, "spectrum": _spec_child_death_hc2_noart,
    })),
])


# ---------------------------------------------------------------------------
# Indicator definitions
# ---------------------------------------------------------------------------

@dataclass
class IndicatorDef:
    disagg: dict[str, Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]]]
    # Single-year-of-age disagg functions for the "By age" sub-tab, same 3-arg
    # (data, disagg_age, disagg_sex) shape as `disagg` — only populated for
    # indicators whose underlying arrays actually carry single-year age
    # resolution (see the "_*_single" functions above). A missing source key
    # (or a missing/empty dict entirely) means that source has no age-resolved
    # data for this indicator and is skipped in the "By age" view, same
    # missing-key convention as `disagg`.
    age_profile: dict[str, Callable[[dict, bool, bool], list[tuple[str, np.ndarray]]]] = field(default_factory=dict)


def _as_full_disagg(fn: Callable[[dict, bool], list[tuple[str, np.ndarray]]]) -> Callable:
    """Adapt a 2-arg (data, disagg_sex) Goals-native function to the 3-arg
    (data, disagg_age, disagg_sex) shape, hiding it in the age-faceted view —
    matches the pre-refactor behavior where this source was never rendered there."""
    def wrapper(data: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        if disagg_age:
            return []
        return fn(data, disagg_sex)
    return wrapper


def _totals_only(fn: Callable[[dict], np.ndarray], label: str = "Total") -> Callable:
    """Adapt a plain single-array function to the 3-arg disagg shape. Used for the
    Spectrum indicators that have no age/sex breakdown available at all — the single
    combined value is always shown, regardless of the requested disaggregation
    (there's nothing to split it by); it is naturally excluded from the age-faceted
    view since its label carries no age-group prefix."""
    def wrapper(data: dict, disagg_age: bool, disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        return [(label, fn(data))]
    return wrapper


INDICATOR_MAP: OrderedDict[str, IndicatorDef] = OrderedDict([
    # --- All-ages indicators: Leapfrog DP/AIM vs Spectrum only ---
    # "spectrum" (HV_*-based) is used by the Goals tab; "spectrum_aim" (AM_*/DP_*-
    # based) is used by the AIM tab, since HV_* modvars are Goals-specific and
    # aren't populated by a plain AIM-only Spectrum run. DP_BigPop_V1 (Total
    # population) is identical for both, so both keys point at the same function.
    ("Total population", IndicatorDef(
        disagg={
            "dp_aim": _disagg_std("p_totpop"),
            "spectrum": _spec_totpop_disagg,
            "spectrum_aim": _spec_totpop_disagg,
        },
        # DP_BigPopTag (unlike the HV_* modvars below) isn't Goals-specific, so
        # "spectrum" has real single-age data here too, same as "spectrum_aim".
        age_profile={
            "dp_aim": _disagg_std_single_age("p_totpop"),
            "spectrum": _spec_totpop_disagg_single,
            "spectrum_aim": _spec_totpop_disagg_single,
        },
    )),
    ("Total Births", IndicatorDef(disagg={
        "dp_aim": lambda o, _da, _ds: [("Total", o["births"])],
        "spectrum": _spec_births_disagg,
        "spectrum_aim": _spec_births_disagg,
    })),
    ("HIV population", IndicatorDef(
        disagg={
            "dp_aim": _disagg_std("p_hivpop"),
            "spectrum": _am_hivpop_disagg,
            "spectrum_aim": _am_hivpop_disagg,
        },
        # "spectrum" (Goals tab) here means the same raw Spectrum `modvars` dict
        # as "spectrum_aim" (see _goals_run_fn in app.py). HV_TotalAdultsHIV_V1
        # is restricted to Spectrum's Goals/HV adult cohort (ages 15-49) and so
        # is NOT a valid stand-in for an all-ages total; AM_HIVBySingleAge_V1 is
        # populated for Goals-classified PJNZ runs too, with values matching
        # dp_aim's p_hivpop closely, so it's used for "spectrum" here as well
        # as for "spectrum_aim" (same as age_profile below).
        age_profile={
            "dp_aim": _disagg_std_single_age("p_hivpop"),
            "spectrum": _am_hivpop_disagg_single,
            "spectrum_aim": _am_hivpop_disagg_single,
        },
    )),
    ("New HIV infections", IndicatorDef(
        disagg={
            "dp_aim": _disagg_std("p_infections"),
            # "spectrum" uses the same AM_NewInfectionsBySingleAge_V1 modvar as
            # "spectrum_aim" rather than HV_NewInfections_V1, which is restricted
            # to Spectrum's Goals/HV adult cohort (ages 15-49) and so excludes
            # pediatric (MTCT) infections — see "HIV population" above.
            "spectrum": _am_newinf_disagg,
            "spectrum_aim": _am_newinf_disagg,
        },
        age_profile={
            "dp_aim": _disagg_std_single_age("p_infections"),
            "spectrum": _am_newinf_disagg_single,
            "spectrum_aim": _am_newinf_disagg_single,
        },
    )),
    ("AIDS deaths", IndicatorDef(
        disagg={
            "dp_aim": _disagg_std("p_hiv_deaths"),
            # "spectrum" uses the same AM_AIDSDeathsARTSingleAge_V1 +
            # AM_AIDSDeathsNoARTSingleAge_V1 combination as "spectrum_aim" rather
            # than HV_AIDSDeaths_V1, which is restricted to Spectrum's Goals/HV
            # adult cohort (ages 15-49) and so excludes child AIDS deaths — see
            # "HIV population" above.
            "spectrum": _am_aidsdeath_disagg,
            "spectrum_aim": _am_aidsdeath_disagg,
        },
        age_profile={
            "dp_aim": _disagg_std_single_age("p_hiv_deaths"),
            "spectrum": _am_aidsdeath_disagg_single,
            "spectrum_aim": _am_aidsdeath_disagg_single,
        },
    )),
    ("Total number receiving ART (15+)", IndicatorDef(
        disagg={
            "dp_aim": _disagg_art(),
            # Both use AM_OnARTBySingleAgeTag summed over ages 15-80 — a genuine
            # 15+ figure. HV_TotalAdultsART_V1 is NOT used here: it is Spectrum's
            # Goals/HV 15-49 total (see _am_art_15plus_disagg / _spec_art_1549_disagg).
            "spectrum": _am_art_15plus_disagg,
            "spectrum_aim": _am_art_15plus_disagg,
        },
        age_profile={
            "dp_aim": _disagg_art_single_age,
            "spectrum": _am_art_disagg_single,
            "spectrum_aim": _am_art_disagg_single,
        },
    )),
    # Total deaths / prevalence in pregnant women: no "goals" key — the Leapfrog
    # Goals tab shows only the dp_aim/spectrum lines for these, since there's no
    # separate Goals-native output to compare (see _goals_run_fn in app.py).
    ("Total deaths", IndicatorDef(disagg={
        "dp_aim": _lf_total_deaths_disagg,
        "spectrum": _spec_deaths_disagg,
        "spectrum_aim": _spec_deaths_disagg,
    })),

    # --- "Deaths" sub-tab: drill-down into "Total deaths"'s components, all
    # aggregated over age (year on the x-axis). No "goals" key on any of these
    # — same reasoning as "Total deaths" above. "AIDS deaths (on ART)" +
    # "AIDS deaths (not on ART)" sum to the existing "AIDS deaths" indicator;
    # "Excess non-AIDS deaths (on ART)" + "(not on ART)" sum to "Excess
    # non-AIDS deaths (total)"; together with "Background deaths (total
    # population)", all five sum back to "Total deaths". "Non-AIDS deaths in
    # HIV+ population (on/not on ART)" is a separate drill-down (of all
    # background deaths, how many fall among PLHIV) that Spectrum only
    # reports as a combined background+excess total — see
    # _spec_nonaids_hivpop_art_disagg's docstring above.
    ("Background deaths (total population)", IndicatorDef(disagg={
        "dp_aim": _lf_deaths_disagg("p_deaths_background_totpop"),
        "spectrum": _spec_bg_deaths_totpop_disagg,
        "spectrum_aim": _spec_bg_deaths_totpop_disagg,
    })),
    ("AIDS deaths (on ART)", IndicatorDef(disagg={
        "dp_aim": _lf_deaths_disagg("h_hiv_deaths_art", "hc1_art_aids_deaths", "hc2_art_aids_deaths"),
        "spectrum": _spec_aidsdeath_art_disagg,
        "spectrum_aim": _spec_aidsdeath_art_disagg,
    })),
    ("AIDS deaths (not on ART)", IndicatorDef(disagg={
        "dp_aim": _lf_deaths_disagg("h_hiv_deaths_no_art", "hc1_noart_aids_deaths", "hc2_noart_aids_deaths"),
        "spectrum": _spec_aidsdeath_noart_disagg,
        "spectrum_aim": _spec_aidsdeath_noart_disagg,
    })),
    ("Excess non-AIDS deaths (total)", IndicatorDef(disagg={
        "dp_aim": _lf_deaths_disagg("p_deaths_excess_nonaids"),
        "spectrum": _spec_excess_nonaids_total_disagg,
        "spectrum_aim": _spec_excess_nonaids_total_disagg,
    })),
    ("Excess non-AIDS deaths (on ART)", IndicatorDef(disagg={
        "dp_aim": _lf_deaths_disagg("p_excess_deaths_nonaids_on_art"),
        "spectrum": _spec_excess_nonaids_art_disagg,
        "spectrum_aim": _spec_excess_nonaids_art_disagg,
    })),
    ("Excess non-AIDS deaths (not on ART)", IndicatorDef(disagg={
        "dp_aim": _lf_deaths_disagg("p_excess_deaths_nonaids_no_art"),
        "spectrum": _spec_excess_nonaids_noart_disagg,
        "spectrum_aim": _spec_excess_nonaids_noart_disagg,
    })),
    ("Non-AIDS deaths in HIV+ population (on ART)", IndicatorDef(disagg={
        "dp_aim": _lf_deaths_disagg("p_deaths_nonaids_artpop", "p_excess_deaths_nonaids_on_art"),
        "spectrum": _spec_nonaids_hivpop_art_disagg,
        "spectrum_aim": _spec_nonaids_hivpop_art_disagg,
    })),
    ("Non-AIDS deaths in HIV+ population (not on ART)", IndicatorDef(disagg={
        "dp_aim": _lf_deaths_disagg("p_deaths_nonaids_hivpop", "p_excess_deaths_nonaids_no_art"),
        "spectrum": _spec_nonaids_hivpop_noart_disagg,
        "spectrum_aim": _spec_nonaids_hivpop_noart_disagg,
    })),

    ("Prevalence in pregnant women (%)", IndicatorDef(disagg={
        "dp_aim": _lf_pregnant_prevalence_disagg,
        "spectrum": _spec_pregnant_prevalence_disagg,
        "spectrum_aim": _spec_pregnant_prevalence_disagg,
    })),

    # --- 15-49 indicators: all three sources; age disagg disabled ---
    ("Total population (15-49)", IndicatorDef(disagg={
        "dp_aim": _disagg_std_1549("p_totpop"),
        "spectrum": _spec_totpop_1549_disagg,
        "spectrum_aim": _spec_totpop_1549_disagg,
        "goals": _as_full_disagg(_goals_total_pop_1549),
    })),
    ("Total PLHIV (15-49)", IndicatorDef(disagg={
        "dp_aim": _disagg_std_1549("p_hivpop"),
        "spectrum": _spec_hivpop_1549_disagg,
        "spectrum_aim": _am_hivpop_1549_disagg,
        "goals": _as_full_disagg(_goals_plhiv),
    })),
    ("New HIV infections (15-49)", IndicatorDef(disagg={
        "dp_aim": _disagg_std_1549("p_infections"),
        "spectrum": _spec_newinf_1549_disagg,
        "spectrum_aim": _am_newinf_1549_disagg,
        "goals": _as_full_disagg(_goals_newinf),
    })),
    ("AIDS deaths (15-49)", IndicatorDef(disagg={
        "dp_aim": _disagg_std_1549("p_hiv_deaths"),
        "spectrum": _spec_aidsdeath_1549_disagg,
        "spectrum_aim": _am_aidsdeath_1549_disagg,
        "goals": _as_full_disagg(_goals_total_deaths_hiv),
    })),
    # No "goals" key: no separate Goals-native total-deaths output to compare.
    ("Total deaths (15-49)", IndicatorDef(disagg={
        "dp_aim": _lf_total_deaths_1549_disagg,
        "spectrum": _spec_deaths_1549_disagg,
        "spectrum_aim": _spec_deaths_1549_disagg,
    })),
    ("Prevalence (15-49) (%)", IndicatorDef(disagg={
        "dp_aim": _no_age_disagg(_disagg_prevalence()),
        "spectrum": _spec_prevalence_1549_disagg,
        "spectrum_aim": _am_prevalence_1549_disagg,
        "goals": _as_full_disagg(_goals_prevalence),
    })),
    ("Incidence (15-49) (%)", IndicatorDef(disagg={
        "dp_aim": _no_age_disagg(_disagg_incidence()),
        "spectrum": _totals_only(_spec_incidence, label="15-49"),
        "spectrum_aim": _am_incidence_1549_disagg,
        "goals": _as_full_disagg(_goals_incidence),
    })),
    ("Total on ART (15-49)", IndicatorDef(disagg={
        "dp_aim": _lf_artpop_1549,
        "spectrum": _spec_art_1549_disagg,
        "spectrum_aim": _am_art_1549_disagg,
        "goals": _as_full_disagg(_goals_total_on_art),
    })),
])

# Named indicator groupings, used to populate each top-level tab's inner plot
# sub-tabs. Spectrum reuses ALL_AGES/FIFTEEN_49 too, but its sources list simply
# omits the "goals" source so goals-native lines never render there.
ALL_AGES_INDICATOR_NAMES: list[str] = [
    "Total population", "Total Births", "HIV population",
    "New HIV infections", "AIDS deaths", "Total number receiving ART (15+)",
    "Total deaths", "Prevalence in pregnant women (%)",
]
FIFTEEN_49_INDICATOR_NAMES: list[str] = [
    "Total population (15-49)", "Total PLHIV (15-49)", "New HIV infections (15-49)",
    "AIDS deaths (15-49)", "Total deaths (15-49)", "Prevalence (15-49) (%)",
    "Incidence (15-49) (%)", "Total on ART (15-49)",
]
# "Deaths" sub-tab: drill-down into the components of "Total deaths"/"AIDS
# deaths" above, each aggregated over age with year on the x-axis.
DEATHS_INDICATOR_NAMES: list[str] = [
    "Total deaths", "Background deaths (total population)",
    "AIDS deaths", "AIDS deaths (on ART)", "AIDS deaths (not on ART)",
    "Excess non-AIDS deaths (total)", "Excess non-AIDS deaths (on ART)",
    "Excess non-AIDS deaths (not on ART)",
    "Non-AIDS deaths in HIV+ population (on ART)",
    "Non-AIDS deaths in HIV+ population (not on ART)",
]
# Indicators with a real single-year-of-age breakdown available (i.e. a
# non-empty `age_profile` dict) — populates the "By age" sub-tab's indicator
# multiselect. "Total Births" is excluded: DP_Births_V1 has no age or sex axis.
AGE_PROFILE_INDICATOR_NAMES: list[str] = [
    name for name in ALL_AGES_INDICATOR_NAMES if INDICATOR_MAP[name].age_profile
]


def get_indicator_names() -> list[str]:
    return list(INDICATOR_MAP.keys())


# ---------------------------------------------------------------------------
# "Resource needs" tab: leapfrog Goals' num_people_reached / resources_required
# vs Spectrum's RN_NumberPeopleReached_V1 / RN_ResourcesRequired_V1. Both the
# Goals output arrays and the Spectrum modvars are (nIntervnRN + 4, n_years),
# indexed by the RN_* intervention constant (row = the constant's integer
# value); the year axis is already aligned to output_years on the Goals side and
# positionally offset from first_year on the Spectrum side (needs_offset_align).
# One plot per intervention row, each a single flat total line per source —
# these outputs carry no age or sex axis.
#
# Spectrum only populates RN_NumberPeopleReached_V1 / RN_ResourcesRequired_V1
# once its Resource Needs module has actually run and written results back; on a
# plain PJNZ import they read back empty (0-d), same as the AM_ResNonAIDSDeaths*
# report modvars noted in the "Deaths" section above. _spec_rn_row detects that
# and returns [] so the Spectrum line is simply omitted rather than erroring.
# ---------------------------------------------------------------------------

# (row index, plot title) for every intervention, in display order. The row
# index is the RN_* intervention constant — the same integer indexes both the
# Goals output arrays and the Spectrum modvars.
_REACHED_INTERVENTIONS: tuple[tuple[int, str], ...] = (
    (RN_CompSexEd, "Comprehensive sexuality education"),
    (RN_FSW, "Female sex workers and clients"),
    (RN_EconomicStrengthening, "Economic strengthening"),
    (RN_IDUHarmRed, "PWID harm reduction"),
    (RN_IDUNSEP, "PWID needle and syringe exchange"),
    (RN_IDUDrugSub, "Opioid agonist maintenance therapy"),
    (RN_MSMOutreach, "Men who have sex with men"),
    (RN_Condom, "Condom promotion"),
    (RN_CondomSupply, "Condom supply"),
    (RN_ANCTesting, "ANC testing"),
    (RN_MC15_49, "Male circumcision"),
    (RN_PrEPOralDaily, "Oral PrEP (daily)"),
    (RN_PrEPOralMonthly, "Oral PrEP (monthly)"),
    (RN_PrEPOralPlusCon, "Oral PrEP plus contraceptive"),
    (RN_PrEPInject1Mo, "Injectable PrEP (1 month)"),
    (RN_PrEPInject2Mo, "Injectable PrEP (2 month)"),
    (RN_PrEPInject6Mo, "Injectable PrEP (6 month)"),
    (RN_PrEPRing, "PrEP ring"),
    (RN_PrEPbNABs, "bNABs"),
    (RN_PrEPImplant, "Implantable PrEP"),
    (RN_PrEP_PEP, "PEP"),
    (RN_HIVTestANCVisit, "HIV test at ANC visit"),
    (RN_HIVWomanCounceled, "HIV-positive women counselled"),
    (RN_ExposedInfantTest, "Infant males circumcised"),
    (RN_AdultOnART, "Adults on ART"),
    (RN_ChildOnART, "Children on ART"),
    (RN_Vaccines, "Prophylactic vaccine"),
    (RN_CureAdultsChildren, "Cure (adults and children)"),
    (RN_CureNeonates, "Cure (neonates)"),
    (RN_CURE_CHILDREN, "Cure (children)"),
    (RN_AHDTreatment, "AHD treatment"),
    (RN_POC_CD4_Int, "POC CD4 test"),
    (RN_POC_VL_Int, "POC VL test"),
    (RN_VMM, "Vaginal microbiome modification"),
    (RN_THERAPEUTIC_VAC, "Therapeutic vaccine"),
    (RN_FUNC_CURE, "Functional cure"),
    (RN_LONG_ACT_TREAT, "Long acting treatment"),
)


def _goals_rn_row(key: str, idx: int) -> Callable:
    """One intervention row of a Goals (nIntervnRN + 4, n_years) resource-needs
    array (`num_people_reached` or `resources_required`), returned as a single
    flat total line — no age/sex axis to split on."""
    def fn(output: dict, _disagg_age: bool, _disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        return [("Total", output[key][idx])]
    return fn


def _spec_rn_row(tag: str, idx: int) -> Callable:
    """Matching intervention row of a Spectrum resource-needs modvar
    (RN_NumberPeopleReached_V1 / RN_ResourcesRequired_V1). Only populated once
    Spectrum's Resource Needs module has run; on a plain import the modvar comes
    back empty (0-d) or absent — return [] so the Spectrum line is dropped
    rather than raising."""
    def fn(modvars: dict, _disagg_age: bool, _disagg_sex: bool) -> list[tuple[str, np.ndarray]]:
        arr = np.asarray(modvars.get(tag))
        if arr.ndim < 2 or idx >= arr.shape[0]:
            return []
        return [("Total", arr[idx])]
    return fn


def _resource_needs_map(goals_key: str, spec_tag: str) -> OrderedDict[str, IndicatorDef]:
    """One IndicatorDef per intervention, reading `goals_key` from the Goals
    output dict and `spec_tag` from the Spectrum modvars — keyed by the plot
    title so each "Resource needs" sub-tab gets its own map (the two sub-tabs
    share intervention names, so they can't live in one dict)."""
    return OrderedDict(
        (name, IndicatorDef(disagg={
            "goals": _goals_rn_row(goals_key, idx),
            "spectrum": _spec_rn_row(spec_tag, idx),
        }))
        for idx, name in _REACHED_INTERVENTIONS
    )


PEOPLE_REACHED_INDICATOR_MAP: OrderedDict[str, IndicatorDef] = _resource_needs_map(
    "num_people_reached", RN_NumberPeopleReachedTag,
)
RESOURCES_REQUIRED_INDICATOR_MAP: OrderedDict[str, IndicatorDef] = _resource_needs_map(
    "resources_required", RN_ResourcesRequiredTag,
)

# Intervention plot titles, shared by both "Resource needs" sub-tabs.
RESOURCE_NEEDS_INDICATOR_NAMES: list[str] = [name for _idx, name in _REACHED_INTERVENTIONS]

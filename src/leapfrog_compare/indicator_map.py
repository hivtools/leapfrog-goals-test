"""
Mapping from Spectrum extract indicator names to leapfrog-goals output arrays.

To add a new indicator:
1. Write a compute function  f(output: dict[str, ndarray]) -> ndarray  that
   returns a 1-D array with one value per output year.
2. Add an IndicatorDef entry to INDICATOR_MAP.

Helper factories
----------------
_sum(key)
    Sum a goals array over every dimension except the last (time).
_direct(key, scale)
    Use a goals array directly, assuming it is already 1-D per year.
    Multiply by `scale` (e.g. 100 to convert 0-1 fraction to percent).
_age_slice_sum(key, age_start, age_end)
    Sum over a specific age slice (0-indexed single ages) and all sexes,
    keeping the time axis.  Useful for e.g. ages 15-49 (indices 15..49).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Callable

import numpy as np


@dataclass
class IndicatorDef:
    # Name as it appears in the Spectrum extract XLSX "Indicator" column.
    extract_name: str
    # Function (goals_output_dict) -> 1-D ndarray  (one value per year).
    compute: Callable[[dict[str, np.ndarray]], np.ndarray]
    # Which "Configuration" row to use from the extract (default: Male+Female total).
    extract_configuration: str = "Male+Female"


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _sum(key: str) -> Callable:
    """Sum all non-time dimensions of a goals output array."""
    def compute(output: dict) -> np.ndarray:
        arr = output[key]
        # Flatten everything except the last (time) dimension, then sum.
        return arr.reshape(-1, arr.shape[-1]).sum(axis=0)
    return compute


def _direct(key: str, scale: float = 1.0) -> Callable:
    """Use a 1-D goals output directly, with optional scaling."""
    def compute(output: dict) -> np.ndarray:
        arr = output[key].flatten()
        return arr * scale
    return compute


def _age_slice_sum(key: str, age_start: int, age_end: int) -> Callable:
    """
    Sum a goals array over age indices [age_start, age_end] (inclusive) and
    all sex indices, keeping the time axis.

    Assumes array shape is (n_ages, n_sexes, n_years) — i.e. the leapfrog
    standard for single-age population outputs.
    """
    def compute(output: dict) -> np.ndarray:
        arr = output[key]          # (ages, sexes, years)
        sliced = arr[age_start:age_end + 1, :, :]   # (slice, sexes, years)
        return sliced.reshape(-1, sliced.shape[-1]).sum(axis=0)
    return compute


def _prevalence_15to49(output: dict) -> np.ndarray:
    """HIV prevalence ages 15-49 as a percentage."""
    hiv = _age_slice_sum("p_hivpop", 15, 49)(output)
    tot = _age_slice_sum("p_totpop", 15, 49)(output)
    return 100.0 * hiv / np.where(tot == 0, np.nan, tot)


def _incidence_15to49(output: dict) -> np.ndarray:
    """HIV incidence rate ages 15-49 as a percentage of the HIV-negative population."""
    infections = _age_slice_sum("p_infections", 15, 49)(output)
    tot = _age_slice_sum("p_totpop", 15, 49)(output)
    hiv = _age_slice_sum("p_hivpop", 15, 49)(output)
    hivneg = tot - hiv
    return 100.0 * infections / np.where(hivneg == 0, np.nan, hivneg)


# ---------------------------------------------------------------------------
# Indicator map
# ---------------------------------------------------------------------------
# Keys are the display names used in the dashboard dropdown AND the exact
# "Indicator" string from the Spectrum extract XLSX.
# ---------------------------------------------------------------------------

INDICATOR_MAP: OrderedDict[str, IndicatorDef] = OrderedDict([
    ("Total population", IndicatorDef(
        extract_name="Total population",
        compute=_sum("p_totpop"),
    )),
    ("Total Births", IndicatorDef(
        extract_name="Total Births",
        compute=_sum("births"),
    )),
    ("HIV population", IndicatorDef(
        extract_name="HIV population",
        compute=_sum("p_hivpop"),
    )),
    ("New HIV infections", IndicatorDef(
        extract_name="New HIV infections",
        compute=_sum("p_infections"),
    )),
    ("AIDS deaths", IndicatorDef(
        extract_name="AIDS deaths",
        compute=_sum("p_hiv_deaths"),
    )),
    ("Total number receiving ART (15+)", IndicatorDef(
        extract_name="Total number receiving ART (15+)",
        compute=_sum("h_artpop"),
    )),
    ("Prevalence (15-49)", IndicatorDef(
        extract_name="Prevalence (15-49)",
        compute=_prevalence_15to49,
    )),
    ("Incidence (15-49) (Percent)", IndicatorDef(
        extract_name="Incidence (15-49) (Percent)",
        compute=_incidence_15to49,
    )),
])


def get_indicator_names() -> list[str]:
    """Return ordered list of mapped indicator display names."""
    return list(INDICATOR_MAP.keys())


def compute_goals_series(
    indicator_name: str,
    goals_output: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Apply the indicator's compute function to the goals output dict.

    Returns a 1-D numpy array (one value per year in goals_output's time axis).
    """
    defn = INDICATOR_MAP[indicator_name]
    return defn.compute(goals_output)

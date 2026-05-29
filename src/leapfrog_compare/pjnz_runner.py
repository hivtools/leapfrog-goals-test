"""
Load a PJNZ file and run the leapfrog goals model.
"""
from pathlib import Path

import numpy as np

from leapfrog_goals import get_goals_ss, run_goals
from SpectrumCommon.Util.LeapfrogDataMapping import modvars_to_leapfrog
from Tools.ImportPJNZ.Importer import GB_ImportProjectionFromFile
from SpectrumCommon.Const.PJ.PJNTags import PJN_FirstYearTag, PJN_FinalYearTag


def run_pjnz(pjnz_path: Path) -> tuple[dict, dict[str, np.ndarray], range]:
    """
    Load a PJNZ file and run the leapfrog Goals model.

    Returns
    -------
    (modvars, goals_output, output_years)
        modvars      : raw Spectrum modvars dict (list values converted to numpy arrays)
        goals_output : dict of numpy arrays from run_goals()
        output_years : range(first_year, final_year + 1)
    """
    raw_modvars, _, _, _ = GB_ImportProjectionFromFile(str(pjnz_path))
    modvars = _modvars_to_numpy(raw_modvars)

    ss = get_goals_ss()
    lf_data = modvars_to_leapfrog(modvars, ss, "Spectrum")
    lf_data["ex_input"] = np.full((81, 2), 1.0)

    first_year = int(modvars[PJN_FirstYearTag])
    final_year = int(modvars[PJN_FinalYearTag])
    output_years = range(first_year, final_year + 1)

    goals_output = run_goals(lf_data, output_years=output_years)

    return modvars, goals_output, output_years


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _modvars_to_numpy(modvars: dict) -> dict:
    """Convert list values in modvars to numpy arrays."""
    result = {}
    for tag, value in modvars.items():
        if isinstance(value, list):
            try:
                if len(value) > 0 and isinstance(value[0], (dict, str, bool)):
                    value = np.array(value, order="C")
                else:
                    value = np.array(value, order="C", dtype=np.float64)
            except Exception as exc:
                raise ValueError(f"Failed to convert modvar '{tag}' to numpy array") from exc
        result[tag] = value
    return result

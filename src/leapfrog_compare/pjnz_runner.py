"""
Load a PJNZ file and run the leapfrog goals model.
"""

from pathlib import Path

import numpy as np

from leapfrog_goals import get_goals_ss, run_goals
from leapfrog_goals import modvars_to_leapfrog
from Tools.ImportPJNZ.Importer import GB_ImportProjectionFromFile
from SpectrumCommon.Const.PJ.PJNTags import PJN_FirstYearTag, PJN_FinalYearTag


def load_pjnz_parameters(pjnz_path: Path) -> tuple[dict, range]:
    """
    Read a PJNZ file and return (leapfrog_parameters, output_years).

    Parameters
    ----------
    pjnz_path:
        Path to the .PJNZ file.

    Returns
    -------
    (parameters, output_years)
        parameters   : dict ready to pass to run_goals()
        output_years : range covering projection first to final year
    """
    modvars, _params, _epp, _shiny90 = GB_ImportProjectionFromFile(str(pjnz_path))

    modvars = _modvars_to_numpy(modvars)

    ss = get_goals_ss()
    lf_data = modvars_to_leapfrog(modvars, ss)

    lf_data["ex_input"] = np.full((81, 2), 1.0)

    first_year: int = modvars[PJN_FirstYearTag]
    final_year: int = modvars[PJN_FinalYearTag]

    return lf_data, range(first_year, final_year + 1)


def run_goals_for_pjnz(pjnz_path: Path) -> tuple[dict[str, np.ndarray], range]:
    """Load a PJNZ and return (goals_output, output_years)."""
    parameters, output_years = load_pjnz_parameters(pjnz_path)
    output = run_goals(parameters, output_years=output_years)
    return output, output_years


def save_goals_output(
    output: dict[str, np.ndarray],
    output_years: range,
    dest: Path,
) -> None:
    """Save goals output as a compressed numpy archive (.npz)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(dest),
        _years=np.array(list(output_years)),
        **output,
    )


def load_goals_output(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Load a saved goals output .npz file.

    Returns
    -------
    (output_dict, years_array)
    """
    data = np.load(str(path), allow_pickle=False)
    years = data["_years"]
    output = {k: data[k] for k in data.files if k != "_years"}
    return output, years


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _modvars_to_numpy(modvars: dict) -> dict:
    """Convert any list values in modvars to numpy arrays (mirrors run_goals.py)."""
    result = {}
    for tag, value in modvars.items():
        if isinstance(value, list):
            try:
                if len(value) > 0 and isinstance(value[0], (dict, str, bool)):
                    value = np.array(value, order="C")
                else:
                    value = np.array(value, order="C", dtype=np.float64)
            except Exception as exc:
                raise ValueError(
                    f"Failed to convert modvar '{tag}' to numpy array"
                ) from exc
        result[tag] = value
    return result

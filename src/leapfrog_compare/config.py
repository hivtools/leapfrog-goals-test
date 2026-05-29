"""
User configuration for leapfrog-compare.
Edit this file to match your local setup before running any scripts.
"""

import sys
from pathlib import Path

# Root of the leapfrog-goals repository clone (contains src/leapfrog_goals/).
LEAPFROG_GOALS_REPO: Path = Path("../leapfrog/goals")

# Directory that contains the .PJNZ files to process.
PJNZ_DIR: Path = Path("C:\\Users\\Test\\Downloads\\pjnz")

# Put the leapfrog-goals src on sys.path so SpectrumCommon etc. are importable
# from anywhere in the package without each module repeating this setup.
_leapfrog_src = str((LEAPFROG_GOALS_REPO / "src").resolve())
if _leapfrog_src not in sys.path:
    sys.path.insert(0, _leapfrog_src)

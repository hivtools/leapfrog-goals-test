"""
User configuration for leapfrog-compare.
Edit this file to match your local setup before running any scripts.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust for your machine
# ---------------------------------------------------------------------------

# Directory that contains the .PJNZ files to process.
PJNZ_DIR: Path = Path("C:\\Users\\Test\\Downloads\\pjnz")

# Spectrum extract configuration file (.EX).
EXTRACT_CONFIG: Path = Path("goals_extract_config.EX")

# ---------------------------------------------------------------------------
# Output directories (created automatically on first run)
# ---------------------------------------------------------------------------

# Path to the XLSX file that Spectrum will write (parent directory is created automatically).
EXTRACT_OUTPUT_FILE: Path = Path("output/extract/extract.xlsx")
GOALS_OUTPUT_DIR: Path = Path("output/goals")

# ---------------------------------------------------------------------------
# Spectrum executable
# On Windows, Spectrum must be on your PATH or set the full path here, e.g.:
#   SPECTRUM_EXE = Path("C:/Program Files/Spectrum/spectrum.exe")
# Leave as None to use shutil.which("spectrum").
# ---------------------------------------------------------------------------

SPECTRUM_EXE: Path | None = None

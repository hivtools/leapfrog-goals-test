"""
User configuration for leapfrog-compare.
Edit this file to match your local setup before running the app.
"""
from pathlib import Path

# Directory that contains the .PJNZ files to process.
PJNZ_DIR: Path = Path("~/Downloads/LFG_Spectrum_Compare_Files")

# Seconds between checks of PJNZ_DIR for added/removed/modified .PJNZ files.
# Files are picked up automatically within this interval — no app restart needed.
PJNZ_POLL_INTERVAL_SECS: float = 5.0

# --- EPPASM tab settings ---------------------------------------------------
# R executable used to shell out to the eppasm / eppasm-leapfrog wrapper script.
R_EXECUTABLE: str = "Rscript"

# Root directories of the two R package checkouts (only read when the
# corresponding USE_LOCAL_CHECKOUT flag below is True).
EPPASM_DIR: Path = Path("C:/Users/Test/projects/eppasm")
EPPASM_LEAPFROG_DIR: Path = Path("C:/Users/Test/projects/eppasm-leapfrog")

# Explicit, independent choice per package: True loads the package from its
# checkout directory above via pkgload::load_all() (development / not-yet-
# installed); False uses the regular installed copy via library(). No
# auto-detection — this is the single source of truth for which copy ran.
# eppasm.lf is not currently R-installed on this machine (and the installed
# `leapfrog` R package is older than the >=0.1.8 it requires), so its flag
# defaults to True; flip to False once both are properly installed.
EPPASM_USE_LOCAL_CHECKOUT: bool = False
EPPASM_LF_USE_LOCAL_CHECKOUT: bool = False

# Where cached EPPASM CSV results are stored, keyed by (pjnz stem, package).
EPPASM_CACHE_DIR: Path = Path("output/eppasm")

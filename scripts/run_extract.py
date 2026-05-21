#!/usr/bin/env python
"""
Run Spectrum /ExtractBatch on the configured PJNZ directory.

Usage:
    uv run scripts/run_extract.py

Reads configuration from config.py in the project root.
Output is written to EXTRACT_OUTPUT_DIR (default: output/extract/).
"""

import sys

import leapfrog_compare.config as config
from leapfrog_compare.spectrum import run_extract, spectrum_version


def main() -> None:
    print("=== Spectrum Extract ===")

    version = spectrum_version(spectrum_exe=config.SPECTRUM_EXE)
    if version:
        print(f"Spectrum version: {version}")
    else:
        print(
            "WARNING: Could not determine Spectrum version. "
            "Make sure Spectrum is on PATH or set SPECTRUM_EXE in config.py."
        )

    pjnz_dir = config.PJNZ_DIR
    extract_config = config.EXTRACT_CONFIG
    output_file = config.EXTRACT_OUTPUT_FILE

    if not pjnz_dir.exists():
        print(f"ERROR: PJNZ directory does not exist: {pjnz_dir.resolve()}")
        print("Edit PJNZ_DIR in config.py to point at your PJNZ files.")
        sys.exit(1)

    pjnz_files = list(pjnz_dir.glob("*.PJNZ"))

    if not pjnz_files:
        print(f"ERROR: No .PJNZ files found in {pjnz_dir.resolve()}")
        sys.exit(1)

    print(f"PJNZ directory : {pjnz_dir.resolve()}")
    print(f"Extract config : {extract_config.resolve()}")
    print(f"Output dir     : {output_file.resolve()}")
    print(f"PJNZ files     : {len(pjnz_files)} found")
    print()

    try:
        result = run_extract(
            pjnz_dir=pjnz_dir,
            config=extract_config,
            output_file=output_file,
            spectrum_exe=config.SPECTRUM_EXE,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Extract failed — {exc}")
        sys.exit(1)

    if result.stdout:
        print(result.stdout)
    if result.returncode == 0:
        print("Extract completed successfully")
    else:
        print("Error: extract returned non-zero exit code")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Run the leapfrog goals model on every PJNZ in PJNZ_DIR.

Usage:
    uv run scripts/run_goals_model.py

Results are saved as compressed numpy archives (.npz) in GOALS_OUTPUT_DIR,
one file per PJNZ named after the PJNZ file stem, e.g.:
    output/goals/Mozambique_2025.npz

If a PJNZ fails, the error is printed and execution continues with the next file.
"""

import sys
import traceback
from pathlib import Path

import leapfrog_compare.config as config
from leapfrog_compare.pjnz_runner import run_goals_for_pjnz, save_goals_output


def main() -> None:
    print("=== Run leapfrog goals model ===")

    pjnz_dir = config.PJNZ_DIR
    output_dir = config.GOALS_OUTPUT_DIR

    if not pjnz_dir.exists():
        print(f"ERROR: PJNZ directory does not exist: {pjnz_dir.resolve()}")
        print("Edit PJNZ_DIR in config.py to point at your PJNZ files.")
        sys.exit(1)

    pjnz_files = sorted(
        list(pjnz_dir.glob("*.PJNZ"))
    )

    if not pjnz_files:
        print(f"ERROR: No .PJNZ files found in {pjnz_dir.resolve()}")
        sys.exit(1)

    print(f"PJNZ directory : {pjnz_dir.resolve()}")
    print(f"Output dir     : {output_dir.resolve()}")
    print(f"PJNZ files     : {len(pjnz_files)} found")
    print()

    succeeded = 0
    failed: list[tuple[Path, str]] = []

    for pjnz_path in pjnz_files:
        dest = output_dir / f"{pjnz_path.stem}.npz"
        print(f"  [{pjnz_files.index(pjnz_path) + 1}/{len(pjnz_files)}] {pjnz_path.name}", end=" ... ", flush=True)

        try:
            goals_output, output_years = run_goals_for_pjnz(pjnz_path)
            save_goals_output(goals_output, output_years, dest)
            print(f"OK  →  {dest}")
            succeeded += 1
        except Exception:  # noqa: BLE001
            msg = traceback.format_exc()
            print("FAILED")
            print(f"    {msg.splitlines()[-1]}")
            failed.append((pjnz_path, msg))

    print()
    print(f"Completed: {succeeded} succeeded, {len(failed)} failed.")

    if failed:
        print("\nFailed files:")
        for path, err in failed:
            print(f"  {path.name}")
            for line in err.splitlines():
                print(f"    {line}")
        sys.exit(1)


if __name__ == "__main__":
    main()

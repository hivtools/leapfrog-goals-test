"""
Wrapper around the Spectrum desktop CLI for running /ExtractBatch.

Spectrum must be on PATH (or SPECTRUM_EXE set in config.py).
On Windows the installation directory is typically added to PATH by the installer.
"""

import shutil
import subprocess
from pathlib import Path


def spectrum_version(spectrum_exe: Path | None = None, timeout: int = 30) -> str | None:
    """Return the Spectrum version string, or None if Spectrum is not found."""
    exe = _resolve_exe(spectrum_exe)
    if exe is None:
        return None
    try:
        result = subprocess.run(
            [str(exe), "/Version"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return (result.stdout or result.stderr).strip() or None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def run_extract(
    pjnz_dir: Path,
    config: Path,
    output_file: Path,
    spectrum_exe: Path | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess:
    """
    Run Spectrum /ExtractBatch on a folder of PJNZ files.

    Parameters
    ----------
    pjnz_dir:
        Directory containing .PJNZ files.
    config:
        Path to the .EX extract configuration file.
    output_file:
        Full path (including filename) for the XLSX output file Spectrum will write.
        The parent directory is created automatically.
    spectrum_exe:
        Full path to the spectrum executable. Falls back to PATH lookup.
    timeout:
        Seconds before the subprocess is killed.

    Returns
    -------
    subprocess.CompletedProcess
    """
    exe = _resolve_exe(spectrum_exe)
    if exe is None:
        raise RuntimeError(
            "Spectrum executable not found. Add it to PATH or set SPECTRUM_EXE in config.py."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(exe),
        "/ExtractBatch",
        str(pjnz_dir.resolve()),
        str(config.resolve()),
        str(output_file.resolve()),
    ]

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )


def _resolve_exe(spectrum_exe: Path | None) -> Path | None:
    if spectrum_exe is not None:
        return spectrum_exe
    found = shutil.which("spectrum")
    return Path(found) if found else None

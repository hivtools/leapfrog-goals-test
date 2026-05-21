"""
Parse the XLSX output from Spectrum /ExtractBatch.

Sheet layout (one sheet per indicator)
---------------------------------------
- Row 1  : "Spectrum Version: X.YZ"  — skipped
- Row 2  : column headers
           File name | Country | ISO 3166-1 alpha-3 | Subnational region
           | Module | Indicator | Configuration | Year | Value
- Row 3+ : data rows in long format — one row per PJNZ × year

All PJNZs appear in every indicator sheet.
Sheet names are truncated to 31 chars by Excel; the full indicator name is read
from the "Indicator" data column instead.
"""

from pathlib import Path

import pandas as pd


def read_extract_xlsx(path: Path) -> pd.DataFrame:
    """
    Parse an extract XLSX and return a long-form DataFrame.

    Columns: pjnz_stem, indicator, configuration, year, value

    Parameters
    ----------
    path:
        Path to the XLSX file produced by Spectrum /ExtractBatch.

    Returns
    -------
    pandas.DataFrame with columns:
        pjnz_stem     — file stem of the source PJNZ (no directory, no extension)
        indicator     — full indicator name string (from data, not truncated sheet name)
        configuration — e.g. "Male+Female", "Male", "Female"
        year          — integer year
        value         — numeric value
    """
    xl = pd.ExcelFile(path, engine="openpyxl")
    frames = []

    for sheet_name in xl.sheet_names:
        df = pd.read_excel(
            xl,
            sheet_name=sheet_name,
            skiprows=1,   # skip version row
            header=0,
        )

        df = _normalise_headers(df)
        if df is None:
            continue

        df["pjnz_stem"] = df["file_name"].apply(
            lambda p: Path(str(p)).stem if pd.notna(p) else None
        )
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        frames.append(df[["pjnz_stem", "indicator", "configuration", "year", "value"]])

    if not frames:
        raise ValueError(f"No usable sheets found in {path}")

    return pd.concat(frames, ignore_index=True)


def get_pjnz_names(extract_df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique PJNZ stem names present in the extract data."""
    return sorted(extract_df["pjnz_stem"].dropna().unique().tolist())


def filter_extract(
    extract_df: pd.DataFrame,
    pjnz_stem: str,
    indicator: str,
    configuration: str = "Male+Female",
) -> pd.Series:
    """
    Return a year-indexed Series of extract values for one PJNZ + indicator.

    Falls back to the first configuration found if the requested one is absent.
    """
    subset = extract_df[
        (extract_df["pjnz_stem"] == pjnz_stem)
        & (extract_df["indicator"] == indicator)
    ]

    if subset.empty:
        return pd.Series(dtype=float)

    if configuration in subset["configuration"].values:
        subset = subset[subset["configuration"] == configuration]
    else:
        subset = subset[subset["configuration"] == subset["configuration"].iloc[0]]

    return subset.set_index("year")["value"].sort_index()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HEADER_ALIASES = {
    "file name": "file_name",
    "country": "country",
    "iso 3166-1 alpha-3": "iso",
    "subnational region": "subnational",
    "module": "module",
    "indicator": "indicator",
    "configuration": "configuration",
    "year": "year",
    "value": "value",
}


def _normalise_headers(df: pd.DataFrame) -> pd.DataFrame | None:
    """Rename known text headers to lower-snake-case; return None if required cols missing."""
    rename = {}
    for col in df.columns:
        key = str(col).strip().lower()
        if key in _HEADER_ALIASES:
            rename[col] = _HEADER_ALIASES[key]

    df = df.rename(columns=rename)

    required = {"file_name", "indicator", "configuration", "year", "value"}
    if not required.issubset(df.columns):
        return None

    return df

"""
Excel loader for raw PSA experiment files.

Responsibilities
----------------
* Open the workbook and find every sheet whose name starts with ``||PSA``.
* Concatenate them in sheet-name order into a single tidy DataFrame.
* Resolve sensor columns by keyword search (per ``COLUMN_KEYWORDS``)
  so the analyser is robust to column reordering between files.

Nothing in this module imports Qt — it can be used from a CLI or unit test.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import (
    COLUMN_EXACT, COLUMN_KEYWORDS, OPTIONAL_COLUMN_KEYWORDS, SHEET_PREFIX,
)


class ColumnResolutionError(KeyError):
    """Raised when a required sensor column cannot be located."""


def discover_psa_sheets(xlsx_path: Path) -> list[str]:
    """
    Return sheet names beginning with ``||PSA`` in natural-sort order.

    Natural sort means ``||PSA00009`` comes before ``||PSA00010``.
    """
    xl = pd.ExcelFile(xlsx_path)
    sheets = [s for s in xl.sheet_names if s.startswith(SHEET_PREFIX)]
    if not sheets:
        raise ValueError(
            f"No sheets starting with '{SHEET_PREFIX}' found in {xlsx_path.name}."
        )

    def natural_key(name: str) -> list:
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", name)]

    return sorted(sheets, key=natural_key)


def resolve_column(df_columns: Iterable[str], keywords: list[str]) -> str:
    """
    Return the first column name in ``df_columns`` matching any keyword.

    Matching is case-insensitive and substring-based.
    Raises ColumnResolutionError if nothing matches.
    """
    cols = list(df_columns)
    lower = [str(c).lower() for c in cols]
    for kw in keywords:
        kw_l = kw.lower()
        for i, c in enumerate(lower):
            if kw_l in c:
                return cols[i]
    raise ColumnResolutionError(
        f"None of the keywords {keywords} matched any column. "
        f"Available columns: {cols[:10]}..."
    )


def build_column_map(df_columns: Iterable[str]) -> dict[str, str]:
    """
    Return a dict ``{logical_name: actual_column_name}`` for every entry
    in ``COLUMN_KEYWORDS``. Tries the exact name from ``COLUMN_EXACT`` first,
    then falls back to keyword search. Raises if anything is missing.
    """
    cols = list(df_columns)
    cols_set = set(cols)
    mapping: dict[str, str] = {}
    errors: list[str] = []
    for logical, kws in COLUMN_KEYWORDS.items():
        exact = COLUMN_EXACT.get(logical)
        if exact and exact in cols_set:
            mapping[logical] = exact
            continue
        try:
            mapping[logical] = resolve_column(cols, kws)
        except ColumnResolutionError as e:
            errors.append(f"  - {logical}: {e}")
    if errors:
        raise ColumnResolutionError(
            "Could not auto-detect required columns:\n" + "\n".join(errors)
        )
    # Optional sensors — included in the map only if present
    for logical, kws in OPTIONAL_COLUMN_KEYWORDS.items():
        try:
            mapping[logical] = resolve_column(cols, kws)
        except ColumnResolutionError:
            pass
    return mapping


def load_psa_workbook(xlsx_path: str | Path,
                      progress_cb=None) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load every ``||PSA*`` sheet and concatenate into a single DataFrame.

    Parameters
    ----------
    xlsx_path : path to the raw experiment .xlsx file
    progress_cb : optional callable ``(percent: int, msg: str) -> None``
                  invoked between sheets so the UI can update its progress bar.

    Returns
    -------
    df : concatenated DataFrame, original column names preserved
    colmap : dict mapping logical names ('time', 'co2_in', ...) to
             the actual column names found in this workbook.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(xlsx_path)

    sheets = discover_psa_sheets(xlsx_path)
    frames: list[pd.DataFrame] = []
    for i, name in enumerate(sheets, start=1):
        if progress_cb:
            pct = int(100 * (i - 1) / len(sheets))
            progress_cb(pct, f"Reading sheet {name} ({i}/{len(sheets)})")
        frames.append(pd.read_excel(xlsx_path, sheet_name=name))

    df = pd.concat(frames, ignore_index=True)
    colmap = build_column_map(df.columns)

    # Stable-sort by parsed timestamp so out-of-order sheets don't create
    # date jumps downstream. Uses the same parser as the analyzer.
    from .analyzer import _build_timestamp  # local import to avoid cycle
    ts = _build_timestamp(df, colmap)
    if ts.notna().any():
        order = ts.argsort(kind="mergesort").to_numpy()
        df = df.iloc[order].reset_index(drop=True)

    if progress_cb:
        progress_cb(100, f"Loaded {len(df):,} rows from {len(sheets)} sheet(s).")

    return df, colmap

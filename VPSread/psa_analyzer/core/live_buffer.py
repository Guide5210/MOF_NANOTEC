"""
Accumulate live sample rows into a DataFrame that the *existing* PSA pipeline
can analyse unchanged.

The whole point of the live mode is to reuse :func:`run_analysis` verbatim.
That function only needs a DataFrame whose columns carry the same names as the
raw Excel sheets (``DATE / TIME``, ``MFC-01 ...``, ``BPR-01 ...``, ``CO2 ...``)
plus a ``colmap``. eServer's CSV already uses those names, so the buffer simply
collects the dict rows, coerces the numeric columns, and hands them off.

No Qt here — the worker thread owns the buffer; the UI reads snapshots from it.
"""

from __future__ import annotations

import warnings

import pandas as pd

from .analyzer import AnalysisResult, run_analysis, window_metrics
from .constants import AnalysisParams
from .data_loader import ColumnResolutionError, build_column_map


# eServer writes timestamps as  "M/D/YYYY  h:mm:ss AM/PM"  (US, 12-hour, with
# a double space before the time). Parse that explicitly — it's faster than
# dateutil and silences pandas' "could not infer format" warning.
_ESERVER_TS_FMT = "%m/%d/%Y %I:%M:%S %p"


def parse_timestamps(series: pd.Series) -> pd.Series:
    """
    Parse an eServer DATE/TIME column to datetimes.

    Tries the known 12-hour AM/PM format first (after collapsing repeated
    whitespace); if every value fails that format, falls back to pandas'
    flexible parser so the reader still works if eServer is reconfigured.
    """
    s = series.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out = pd.to_datetime(s, format=_ESERVER_TS_FMT, errors="coerce")
    if out.isna().all():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = pd.to_datetime(s, errors="coerce", dayfirst=False)
    return out


class LiveBuffer:
    """Growing in-memory table of samples plus on-demand analysis."""

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._colmap: dict[str, str] | None = None

    # -- ingest -------------------------------------------------------------
    def add_rows(self, rows: list[dict]) -> int:
        """Append raw sample dicts; returns the new total row count."""
        if rows:
            self._rows.extend(rows)
        return len(self._rows)

    def clear(self) -> None:
        self._rows.clear()
        self._colmap = None

    def rows_snapshot(self) -> list[dict]:
        """A shallow copy of the raw rows (for the data-table editor)."""
        return list(self._rows)

    def drop_indices(self, indices) -> int:
        """Delete rows by position; returns the number removed."""
        drop = set(int(i) for i in indices)
        if not drop:
            return 0
        self._rows = [r for i, r in enumerate(self._rows) if i not in drop]
        self._colmap = None
        return len(drop)

    def __len__(self) -> int:
        return len(self._rows)

    # -- views --------------------------------------------------------------
    @staticmethod
    def _coerce(df: pd.DataFrame) -> pd.DataFrame:
        """
        CSV values arrive as strings (unlike ``read_excel`` which infers types).
        Convert the timestamp column to real datetimes (US month-first, matching
        eServer's ``M/D/YYYY`` format) and every other column to numeric so the
        analyzer's arithmetic works unchanged.
        """
        for col in df.columns:
            name = str(col).strip().lower()
            if "date" in name or "time" in name:
                df[col] = parse_timestamps(df[col])
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def to_dataframe(self) -> pd.DataFrame:
        """Full buffer as a typed DataFrame with eServer column names intact."""
        return self._coerce(pd.DataFrame(self._rows))

    def latest(self) -> dict | None:
        """The most recent raw row (for the big live numbers), or None."""
        return self._rows[-1] if self._rows else None

    def colmap(self) -> dict[str, str] | None:
        """Resolve (and cache) the logical->actual column map, or None if the
        required sensor columns aren't present yet."""
        if self._colmap is not None:
            return self._colmap
        if not self._rows:
            return None
        try:
            self._colmap = build_column_map(self.to_dataframe().columns)
        except ColumnResolutionError:
            return None
        return self._colmap

    # -- analysis -----------------------------------------------------------
    def can_analyze(self) -> bool:
        """True once the required columns are present and there is data."""
        return self.colmap() is not None and len(self._rows) > 0

    def analyze(self, params: AnalysisParams) -> AnalysisResult:
        """Run the unchanged PSA pipeline over everything collected so far."""
        df = self.to_dataframe()
        colmap = build_column_map(df.columns)   # raises if columns missing
        return run_analysis(df, colmap, params)

    def window_metrics(self, minutes: float = 5.0) -> dict:
        """Purity / Recovery over the trailing ``minutes``.

        Unlike :meth:`analyze` this needs no BPR edges, so it keeps producing
        numbers while the cycle detector still has nothing to segment.
        """
        cm = self.colmap()
        if cm is None or not self._rows:
            return {"purity": float("nan"), "recovery": float("nan"),
                    "n": 0, "minutes": float(minutes),
                    "ad_gas_sum": 0.0, "flow_negative": False}
        return window_metrics(self.to_dataframe(), cm, minutes)

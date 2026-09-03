"""
PSA analysis pipeline — pure pandas/numpy, no UI dependencies.

Implements the four-step calculation defined in the project spec:
    1. Row-level derived columns
    2. Per-step aggregation (Purity, Recovery)
    3. Per-cycle aggregation (4 steps -> 1 cycle, Productivity)
    4. Final-stage averaging over stable cycles

The single public entry point is :func:`run_analysis`, which returns
a :class:`AnalysisResult` bundle that the UI can plot / export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .constants import AnalysisParams


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """Everything a downstream consumer (UI, exporter) needs."""
    rows: pd.DataFrame          # row-level df with derived columns
    steps: pd.DataFrame         # one row per bed step
    cycles: pd.DataFrame        # one row per cycle — primary KPI table
    final_purity: float
    final_recovery: float
    final_productivity: float
    total_elapsed_hours: float
    params: AnalysisParams
    # Per-metric descriptive statistics over the final averaging window.
    # Shape: {"purity": {"mean","std","min","max","cv"}, "recovery": {...},
    #         "productivity": {...}}.  cv = coefficient of variation [%].
    final_stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------
def _find_date_time_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """
    Find the separate DATE and TIME columns by name, ignoring case,
    whitespace, and the combined 'DATE / TIME' column.
    """
    date_col: str | None = None
    time_col: str | None = None
    for c in df.columns:
        norm = str(c).strip().lower()
        if "/" in norm:                # skip the combined 'DATE / TIME'
            continue
        if date_col is None and norm == "date":
            date_col = c
        elif time_col is None and norm == "time":
            time_col = c
    return date_col, time_col


def _build_timestamp(df: pd.DataFrame, colmap: dict[str, str]) -> pd.Series:
    """
    Build a continuous datetime Series following the original Excel logic:
        full_ts = df["DATE"] + df["TIME"]   (Excel serial days)
    """
    date_col, time_col = _find_date_time_columns(df)
    if date_col is not None and time_col is not None:
        d = pd.to_numeric(df[date_col], errors="coerce")
        t = pd.to_numeric(df[time_col], errors="coerce")
        if d.notna().any() and t.notna().any():
            return pd.to_datetime(d + t, unit="D", origin="1899-12-30",
                                  errors="coerce")

    # Last-resort fallbacks for workbooks that store the timestamp differently.
    raw_t = df[colmap["time"]]

    # Already a datetime dtype (the common case — openpyxl converts Excel
    # datetime cells to pandas Timestamps). Use it directly; running it
    # through to_numeric would yield ns-since-epoch and then mis-parse.
    if pd.api.types.is_datetime64_any_dtype(raw_t):
        return raw_t

    # Numeric Excel serial (days since 1899-12-30).
    serial = pd.to_numeric(raw_t, errors="coerce")
    if serial.notna().sum() > 0.5 * len(raw_t):
        return pd.to_datetime(serial, unit="D", origin="1899-12-30",
                              errors="coerce")

    # String form (e.g. '21/04/2026 06:00:00') — dayfirst handles DD/MM/YYYY.
    t = pd.to_datetime(raw_t, errors="coerce", dayfirst=True)
    if t.notna().any():
        return t
    return pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")


# ---------------------------------------------------------------------------
# Forward-looking rolling max (Excel MAX(AY_n : AY_{n+window-1}))
# ---------------------------------------------------------------------------
def _forward_rolling_max(arr: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Max of the current row and the next ``window-1`` rows, matching the
    researcher's Excel ``=MAX(AY2:AY8)`` (7-row forward window). Near the
    end of the data the window shrinks to whatever rows remain (Excel reads
    the trailing empty cells as nothing). NaNs are ignored.
    """
    rev = arr[::-1]
    m = pd.Series(rev).rolling(window, min_periods=1).max().to_numpy()
    return m[::-1]


# ---------------------------------------------------------------------------
# Step 1 — row-level calculations
# ---------------------------------------------------------------------------
def _compute_row_level(df: pd.DataFrame,
                       colmap: dict[str, str],
                       params: AnalysisParams) -> pd.DataFrame:
    """
    Add elapsed time, outlet flows, edge detection, and a cumulative
    step counter to the raw row-level DataFrame.

    The input is *not* mutated; a new DataFrame is returned.
    """
    out = df.copy()

    # Time: build a continuous serial timestamp. If separate DATE and TIME
    # columns exist in the sheet, combine them; otherwise parse the single
    # 'DATE / TIME' column. Either way we then patch midnight crossovers
    # so a multi-day run produces a monotonically increasing elapsed_s.
    t = _build_timestamp(out, colmap)
    out["timestamp"] = t
    t_valid = t.dropna()
    t0 = t_valid.iloc[0]
    out["elapsed_s"] = (t - t0).dt.total_seconds()

    # NOTE: do NOT convert MFC-07 (AD-GAS, SMLM) to SLPM. The original
    # Excel sheet uses the raw SMLM/SLPM ratio for Recovery, which yields
    # ~100% — matching the researcher's reference workbook.
    ad_gas = pd.to_numeric(out[colmap["ad_gas"]], errors="coerce")
    co2_pct = pd.to_numeric(out[colmap["co2_pct"]], errors="coerce")
    # Recovery / Productivity use the instantaneous CO2 vol% (matches the
    # researcher's reference exactly).
    out["co2_outlet_Lmin"] = ad_gas * (co2_pct / 100.0)
    out["n2_outlet_Lmin"]  = ad_gas * ((100.0 - co2_pct) / 100.0)

    # Purity uses a *forward-looking* 7-row rolling MAX of the CO2 vol%
    # (Excel BB = K × MAX(AY_n:AY_{n+6})/100). Same flow source (ad_gas),
    # different vol% — this is the only difference from Recovery.
    co2_pct_max7 = _forward_rolling_max(co2_pct.to_numpy(dtype=float), 7)
    out["co2_pct_max7"]        = co2_pct_max7
    out["co2_outlet_pur_Lmin"] = ad_gas * (co2_pct_max7 / 100.0)
    out["n2_outlet_pur_Lmin"]  = ad_gas * ((100.0 - co2_pct_max7) / 100.0)

    bpr = pd.to_numeric(out[colmap["bpr"]], errors="coerce")
    out["edge"] = (bpr <= params.edge_threshold_bar).astype(int)

    # Step ID = cumulative count of 0->1 rising edges, +1 so the rows
    # BEFORE the first BPR edge become step 1 (matches the researcher's
    # Excel convention: step 1 = 0 → first BPR drop).
    rising = (out["edge"].diff().fillna(0) > 0).astype(int)
    out["step_id"] = rising.cumsum() + 1

    return out


# ---------------------------------------------------------------------------
# Step 2 — per-step aggregation
# ---------------------------------------------------------------------------
def _aggregate_steps(rows: pd.DataFrame,
                     colmap: dict[str, str]) -> pd.DataFrame:
    """
    Group by ``step_id`` and compute the per-step KPIs.

    Note: rows where ``step_id == 0`` (before the first rising edge)
    are discarded as pre-experiment data.
    """
    valid = rows[rows["step_id"] > 0].copy()

    grouped = valid.groupby("step_id").agg(
        sum_co2_out=("co2_outlet_Lmin", "sum"),
        sum_n2_out =("n2_outlet_Lmin", "sum"),
        sum_co2_out_pur=("co2_outlet_pur_Lmin", "sum"),
        sum_n2_out_pur =("n2_outlet_pur_Lmin", "sum"),
        sum_co2_in =(colmap["co2_in"], "sum"),
        elapsed_s  =("elapsed_s", "last"),
    ).reset_index()

    # Purity uses the MAX7-vol% outlet sums (Excel BQ/BR).
    denom_purity = grouped["sum_co2_out_pur"] + grouped["sum_n2_out_pur"]
    grouped["purity_pct"]   = np.where(denom_purity > 0,
                                       grouped["sum_co2_out_pur"] / denom_purity * 100.0,
                                       np.nan)
    grouped["recovery_pct"] = np.where(grouped["sum_co2_in"] > 0,
                                       grouped["sum_co2_out"] / grouped["sum_co2_in"] * 100.0,
                                       np.nan)
    return grouped


# ---------------------------------------------------------------------------
# Adaptive smoothing helper
# ---------------------------------------------------------------------------
def _adaptive_centered_mean(series: pd.Series) -> np.ndarray:
    """
    Centred moving average with an adaptive window, matching the lab's
    revised Excel smoothing (columns BW / BX):

        cycle 1–4 -> window = 3   (centre: n-1, n, n+1)
        cycle 5+  -> window = 5   (centre: n-2 … n+2)

    Edges shrink the window to whatever neighbours are available, and NaNs
    are skipped, so the result has no gaps.
    """
    vals = series.to_numpy(dtype=float)
    n = len(vals)
    out = np.full(n, np.nan)
    for i in range(n):
        half = 1 if (i + 1) <= 4 else 2   # cycle number is 1-based (i+1)
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window = vals[lo:hi]
        window = window[~np.isnan(window)]
        if window.size:
            out[i] = window.mean()
    return out


# ---------------------------------------------------------------------------
# Step 3 — per-cycle aggregation
# ---------------------------------------------------------------------------
def _aggregate_cycles(steps: pd.DataFrame,
                      rows: pd.DataFrame,
                      params: AnalysisParams) -> pd.DataFrame:
    """
    Roll up every ``steps_per_cycle`` consecutive bed-steps into one cycle.

    Productivity is computed in tCO2 per m³ of adsorbent per day, using
    the bed volume and CO2 density supplied via ``params``.
    """
    n = params.steps_per_cycle
    # Trim any trailing partial cycle so the reshape is clean
    usable = (len(steps) // n) * n
    s = steps.iloc[:usable].copy()
    s["cycle_id"] = (s["step_id"] - 1) // n + 1   # 1-indexed

    cyc = s.groupby("cycle_id").agg(
        sum_co2_out=("sum_co2_out", "sum"),
        sum_n2_out =("sum_n2_out",  "sum"),
        sum_co2_out_pur=("sum_co2_out_pur", "sum"),
        sum_n2_out_pur =("sum_n2_out_pur",  "sum"),
        sum_co2_in =("sum_co2_in",  "sum"),
        elapsed_s  =("elapsed_s",   "last"),
    ).reset_index()

    # dt per cycle = end-of-this minus end-of-previous
    cyc["dt_s"] = cyc["elapsed_s"].diff().fillna(cyc["elapsed_s"])
    cyc["dt_min"] = cyc["dt_s"] / 60.0

    # Purity uses the MAX7-vol% outlet sums (Excel BQ/BR).
    denom_purity = cyc["sum_co2_out_pur"] + cyc["sum_n2_out_pur"]
    cyc["purity_pct"]   = np.where(denom_purity > 0,
                                   cyc["sum_co2_out_pur"] / denom_purity * 100.0,
                                   np.nan)
    cyc["recovery_pct"] = np.where(cyc["sum_co2_in"] > 0,
                                   cyc["sum_co2_out"] / cyc["sum_co2_in"] * 100.0,
                                   np.nan)

    # Productivity [tCO2 / m³ / day] — matches researcher's reference value.
    # Their Excel formula scales bed_volume with a 1e-9 factor (equivalent
    # to treating the per-bed value in µL, then ×N_beds), which yields the
    # 0.5457 reference for the 4-bed × 60 mL standard rig.
    total_bed_m3 = params.steps_per_cycle * params.bed_volume_mL * 1e-9
    sum_co2_out_L = cyc["sum_co2_out"] / 1000.0
    numerator   = (sum_co2_out_L * params.co2_density_t_per_L) / cyc["dt_min"]
    denominator = total_bed_m3 * (cyc["dt_s"] / 86400.0)
    cyc["productivity_tCO2_m3_day"] = np.where(denominator > 0,
                                               numerator / denominator,
                                               np.nan)

    # Smoothed lines for plotting — adaptive centred window (3 for the first
    # 4 cycles, 5 thereafter), matching the lab's revised Excel columns BW/BX.
    cyc["purity_avg_pct"]        = _adaptive_centered_mean(cyc["purity_pct"])
    cyc["recovery_avg_pct"]      = _adaptive_centered_mean(cyc["recovery_pct"])
    cyc["productivity_avg"]      = _adaptive_centered_mean(
        cyc["productivity_tCO2_m3_day"])
    # Corrected = averaged then capped at 100% (Excel column BY). Display-only;
    # the raw recovery_pct column above is left untouched for the data table.
    cyc["recovery_corrected_pct"] = np.minimum(cyc["recovery_avg_pct"], 100.0)
    cyc["productivity_corrected"] = cyc["productivity_avg"]

    # Drop the trailing cycle if it's clearly incomplete: row count below
    # 50% of the median row-count per cycle.
    active = rows[rows["step_id"] > 0].copy()
    if len(active):
        active["cycle_id"] = (active["step_id"] - 1) // n + 1
        counts = active.groupby("cycle_id").size()
        counts = counts[counts.index.isin(cyc["cycle_id"])]
        if len(counts) >= 2:
            median = counts.median()
            last_id = counts.index.max()
            if counts.loc[last_id] < 0.5 * median:
                cyc = cyc[cyc["cycle_id"] != last_id].reset_index(drop=True)

    return cyc


# ---------------------------------------------------------------------------
# Step 4 — final stable-cycle averaging
# ---------------------------------------------------------------------------
def _describe(series: pd.Series) -> dict:
    """Descriptive stats for one metric over the final window."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "max": float("nan"), "cv": float("nan")}
    mean = float(s.mean())
    std = float(s.std(ddof=0))           # population std (whole window)
    cv = float(std / mean * 100.0) if mean else float("nan")
    return {"mean": mean, "std": std,
            "min": float(s.min()), "max": float(s.max()), "cv": cv}


def _final_metrics(cycles: pd.DataFrame,
                   params: AnalysisParams) -> tuple[float, float, float, dict]:
    """
    Average Purity / Recovery / Productivity over the stable-cycle window
    from ``params.final_cycle_start`` (default cycle 3) to the last cycle.

    Matches the lab's revised Excel convention:
        final_purity       = mean(purity[start : end])
        final_recovery     = mean(recovery[start : end])      # NOT capped
        final_productivity = mean(productivity[start : end])

    The 100% cap is a *display-only* correction (plot "Corrected" mode /
    recovery_corrected_pct column); the researcher's Excel reports the raw
    uncapped mean for the Final value, so neither Final nor the statistics
    apply the cap.

    If there aren't enough cycles to reach the cutoff, falls back to
    averaging all available cycles rather than returning NaN.

    Also returns a ``stats`` dict with mean/std/min/max/cv for each metric
    over the same window.
    """
    if cycles.empty:
        return float("nan"), float("nan"), float("nan"), {}

    start = max(1, int(params.final_cycle_start))
    sub = cycles[cycles["cycle_id"] >= start]
    if sub.empty:                       # cutoff past the last cycle — use all
        sub = cycles

    stats = {
        "purity":       _describe(sub["purity_pct"]),
        "recovery":     _describe(sub["recovery_pct"]),   # uncapped
        "productivity": _describe(sub["productivity_tCO2_m3_day"]),
    }
    return (stats["purity"]["mean"],
            stats["recovery"]["mean"],
            stats["productivity"]["mean"],
            stats)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def run_analysis(df: pd.DataFrame,
                 colmap: dict[str, str],
                 params: AnalysisParams,
                 progress_cb: Optional[Callable[[int, str], None]] = None
                 ) -> AnalysisResult:
    """
    Execute the full four-step PSA pipeline.

    Parameters
    ----------
    df : concatenated raw-row DataFrame from :func:`load_psa_workbook`
    colmap : logical-to-actual column map from the same loader
    params : :class:`AnalysisParams` with experimental constants
    progress_cb : optional ``(percent, message)`` callback for the UI

    Returns
    -------
    :class:`AnalysisResult`
    """
    def _tick(p: int, m: str) -> None:
        if progress_cb:
            progress_cb(p, m)

    _tick(10, "Computing row-level derivatives...")
    rows = _compute_row_level(df, colmap, params)

    _tick(40, "Aggregating bed-steps...")
    steps = _aggregate_steps(rows, colmap)

    _tick(65, "Aggregating cycles...")
    cycles = _aggregate_cycles(steps, rows, params)

    _tick(85, "Computing final stable-cycle averages...")
    fp, fr, fpr, stats = _final_metrics(cycles, params)
    # Total elapsed time = end-of-last-cycle elapsed_s (matches Excel's
    # "Elapsed time [h]" at the last cycle row, ignoring any trailing
    # rows that run past the experiment).
    total_hours = 0.0
    if not cycles.empty and "elapsed_s" in cycles.columns:
        last_elapsed = cycles["elapsed_s"].dropna()
        if len(last_elapsed):
            total_hours = float(last_elapsed.iloc[-1]) / 3600.0
    if total_hours <= 0.0:
        ts = rows["timestamp"].dropna()
        if len(ts):
            total_hours = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 3600.0

    _tick(100, "Analysis complete.")
    return AnalysisResult(
        rows=rows,
        steps=steps,
        cycles=cycles,
        final_purity=fp,
        final_recovery=fr,
        final_productivity=fpr,
        total_elapsed_hours=total_hours,
        params=params,
        final_stats=stats,
    )


# ---------------------------------------------------------------------------
# Trailing-time-window metrics (live monitor)
# ---------------------------------------------------------------------------
def window_metrics(df: pd.DataFrame,
                   colmap: dict[str, str],
                   minutes: float = 5.0) -> dict:
    """Purity / Recovery over the last ``minutes`` of data.

    Identical arithmetic to :func:`_aggregate_steps` -- the *only* difference
    is what defines a group. There, a group is one BPR-delimited bed step;
    here it is simply "every row inside the trailing time window". That makes
    the numbers available continuously, without waiting for the cycle detector
    to find a BPR edge, which is what the live monitor needs while an
    experiment is still running.

    Returns ``{"purity", "recovery", "n", "minutes"}`` with NaN metrics when
    there is not enough usable data yet.
    """
    blank = {"purity": float("nan"), "recovery": float("nan"),
             "n": 0, "minutes": float(minutes)}
    if df is None or df.empty:
        return blank

    tcol = colmap.get("time")
    if not tcol or tcol not in df.columns:
        return blank
    t = pd.to_datetime(df[tcol], errors="coerce")
    if t.notna().sum() == 0:
        return blank

    ad_gas = pd.to_numeric(df[colmap["ad_gas"]], errors="coerce")
    co2_pct = pd.to_numeric(df[colmap["co2_pct"]], errors="coerce")
    co2_in = pd.to_numeric(df[colmap["co2_in"]], errors="coerce")

    # The 7-row forward MAX is computed over the WHOLE series before slicing,
    # so rows near the end of the window still see their real future samples
    # instead of being truncated by the window edge.
    co2_max7 = pd.Series(
        _forward_rolling_max(co2_pct.to_numpy(dtype=float), 7), index=df.index)

    cutoff = t.max() - pd.Timedelta(minutes=float(minutes))
    sel = t >= cutoff
    if not sel.any():
        return blank

    ad_w = ad_gas[sel]
    co2_out_pur = ad_w * (co2_max7[sel] / 100.0)
    n2_out_pur = ad_w * ((100.0 - co2_max7[sel]) / 100.0)
    co2_out = ad_w * (co2_pct[sel] / 100.0)

    denom_purity = co2_out_pur.sum() + n2_out_pur.sum()
    sum_co2_in = co2_in[sel].sum()

    # A mass-flow meter cannot read negative. When it does, the wiring or the
    # register is wrong, and every metric derived from it is meaningless -- so
    # say so instead of letting a confident-looking number reach a report.
    sum_ad = float(ad_w.sum())

    return {
        "purity": (co2_out_pur.sum() / denom_purity * 100.0
                   if denom_purity > 0 else float("nan")),
        "recovery": (co2_out.sum() / sum_co2_in * 100.0
                     if sum_co2_in > 0 else float("nan")),
        "n": int(sel.sum()),
        "minutes": float(minutes),
        "ad_gas_sum": sum_ad,
        "flow_negative": sum_ad < 0,
    }

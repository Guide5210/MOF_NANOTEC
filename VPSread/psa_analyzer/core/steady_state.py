"""
Steady-state window detection and KPI extraction.

A row is "steady" when several conditions hold simultaneously (CO2 high,
O2 low, vacuum active, bed temperatures near setpoint). The longest run
of consecutive steady rows in a DataFrame is the steady-state window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SS_THRESHOLDS = {
    "co2_pct_min":  75.0,    # CO2 analyzer > 75 % vol
    "o2_pct_max":    5.0,    # O2 < 5 %
    "pt10_max":   -500.0,    # PT-10 vacuum < -500 mbar
    "temp_tol":      3.0,    # bed TICs within ±3 °C of setpoint
    "window_rows":  200,     # need ≥ 200 consecutive steady rows
}

# Keyword fragments used to locate the auxiliary sensor columns.
_SS_COLUMN_KEYWORDS = {
    "o2_pct":  ["O2 ( vol%)", "O2 (vol%)", "O2_vol"],
    "pt10":    ["PT-10",      "PT10"],
    "tic":     ["TI1-1",      "TI2-1", "TI3-1", "TI4-1"],   # bed TICs
}


@dataclass
class SteadyStateWindow:
    start_idx: int           # positional row index in the source df
    end_idx: int             # inclusive
    n_rows: int
    start_time: pd.Timestamp | None
    end_time: pd.Timestamp | None


@dataclass
class SteadyStateResult:
    window: SteadyStateWindow
    purity_pct: float
    recovery_pct: float
    productivity_tCO2_m3_day: float
    co2_feed_slpm: float
    co2_product_slpm: float


# ---------------------------------------------------------------------------
def _resolve(df_columns, keywords) -> str | None:
    cols = list(df_columns)
    lower = [str(c).lower() for c in cols]
    for kw in keywords:
        kw_l = kw.lower()
        for i, c in enumerate(lower):
            if kw_l in c:
                return cols[i]
    return None


def _resolve_tic_columns(df_columns) -> list[str]:
    """Return up to 4 bed-temperature columns (TI1-1, TI2-1, TI3-1, TI4-1)."""
    out: list[str] = []
    for kw in _SS_COLUMN_KEYWORDS["tic"]:
        c = _resolve(df_columns, [kw])
        if c is not None and c not in out:
            out.append(c)
    return out


# ---------------------------------------------------------------------------
def find_steady_state(df: pd.DataFrame,
                      colmap: dict[str, str],
                      temp_setpoint_C: float,
                      thresholds: dict | None = None,
                      ) -> SteadyStateWindow | None:
    """
    Locate the longest run of rows satisfying every steady-state condition.

    Returns ``None`` if no qualifying window is found. Missing optional
    sensors (O2, PT-10, TICs) are silently skipped — only the conditions
    we have data for are enforced.
    """
    th = {**SS_THRESHOLDS, **(thresholds or {})}
    n = len(df)
    if n == 0:
        return None

    # CO2 product purity is mandatory
    if colmap.get("co2_pct") not in df.columns:
        return None
    mask = pd.to_numeric(df[colmap["co2_pct"]], errors="coerce") >= th["co2_pct_min"]

    o2_col = _resolve(df.columns, _SS_COLUMN_KEYWORDS["o2_pct"])
    if o2_col:
        mask &= pd.to_numeric(df[o2_col], errors="coerce") <= th["o2_pct_max"]

    pt10_col = _resolve(df.columns, _SS_COLUMN_KEYWORDS["pt10"])
    if pt10_col:
        mask &= pd.to_numeric(df[pt10_col], errors="coerce") <= th["pt10_max"]

    tic_cols = _resolve_tic_columns(df.columns)
    if tic_cols:
        # Each bed within ±tol of setpoint; ignore values <= -1 (closed valve).
        for c in tic_cols:
            t = pd.to_numeric(df[c], errors="coerce")
            t_ok = ((t - temp_setpoint_C).abs() <= th["temp_tol"]) | (t <= -1)
            mask &= t_ok

    mask = mask.fillna(False).to_numpy()

    # Longest run of True
    best_start, best_end, best_len = -1, -1, 0
    cur_start, cur_len = -1, 0
    for i, ok in enumerate(mask):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = i
        else:
            cur_len = 0

    if best_len < th["window_rows"]:
        return None

    # Best-effort timestamps
    t_col = colmap.get("time")
    if t_col and t_col in df.columns:
        ts = pd.to_datetime(df[t_col], errors="coerce")
        t_start = ts.iloc[best_start] if best_start < len(ts) else None
        t_end = ts.iloc[best_end] if best_end < len(ts) else None
    else:
        t_start = t_end = None

    return SteadyStateWindow(
        start_idx=best_start,
        end_idx=best_end,
        n_rows=best_len,
        start_time=t_start,
        end_time=t_end,
    )


# ---------------------------------------------------------------------------
CO2_DENSITY_STP_g_per_L: float = 1.964   # g CO2 / L at STP


def extract_results(df: pd.DataFrame,
                    colmap: dict[str, str],
                    window: SteadyStateWindow,
                    bed_volume_mL: float = 60.0,
                    n_beds: int = 4,
                    waste_min_slpm: float = 1.0,
                    ) -> SteadyStateResult:
    """Compute purity / recovery / productivity over the SS window.

    To guard against a window that still includes warmup rows (where the
    waste line isn't open yet), rows with MFC-08 < ``waste_min_slpm`` are
    excluded. If fewer than 50 rows remain, recovery is reported as NaN.
    """
    sl = df.iloc[window.start_idx:window.end_idx + 1]

    # Apply waste-active mask: desorption blowdown should be flowing
    waste_col = colmap.get("waste_gas")
    if waste_col in sl.columns:
        waste = pd.to_numeric(sl[waste_col], errors="coerce").fillna(0.0)
        keep = waste > waste_min_slpm
        if int(keep.sum()) >= 50:
            sl = sl.loc[keep]

    co2_pct = pd.to_numeric(sl[colmap["co2_pct"]], errors="coerce")
    purity = float(co2_pct.mean())

    co2_feed = pd.to_numeric(sl[colmap["co2_in"]], errors="coerce")
    ad_gas = pd.to_numeric(sl[colmap["ad_gas"]], errors="coerce")  # SMLM
    # AD-GAS is in SMLM (mL/min) -> convert to SLPM for a like-for-like ratio
    co2_product = (ad_gas / 1000.0) * (co2_pct / 100.0)
    feed_mean = float(co2_feed.mean())
    prod_mean = float(co2_product.mean())
    if feed_mean > 0 and len(sl) >= 50:
        recovery = prod_mean / feed_mean * 100.0
    else:
        recovery = float("nan")

    # Productivity in tCO2 / m^3 / day
    total_bed_L = (n_beds * bed_volume_mL) / 1000.0
    total_bed_m3 = total_bed_L / 1000.0
    productivity = (
        prod_mean * CO2_DENSITY_STP_g_per_L * 60.0 * 24.0 / 1e6 / total_bed_m3
    ) if total_bed_m3 > 0 else float("nan")

    return SteadyStateResult(
        window=window,
        purity_pct=purity,
        recovery_pct=recovery,
        productivity_tCO2_m3_day=productivity,
        co2_feed_slpm=feed_mean,
        co2_product_slpm=prod_mean,
    )

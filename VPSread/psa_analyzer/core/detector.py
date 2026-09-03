"""
Auto-detect contiguous experiment blocks in a multi-experiment workbook.

Primary signal: MFC-01 (CO2 feed). When CO2 feed is on, an experiment is
running; when it drops to baseline, we're between experiments. A short
rolling mean is used to ignore spikes / noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .analyzer import _build_timestamp


@dataclass
class ExperimentSegment:
    """One detected experiment window."""
    index: int                  # 1-based position within the detected list
    start_idx: int              # row index (positional) into the source df
    end_idx: int                # inclusive
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_h: float
    mean_co2_slpm: float
    bpr_pulses: int = 0         # how many PSA cycles happened in this block
    flag: str = ""              # "warmup", "long", "short", or "" if normal
    # 4-signal pattern (snapped to standard setpoints when within tolerance)
    pressure_bar: float | None = None       # BPR-01 max during cycling (1/2/3)
    flow_mlmin: float | None = None         # (MFC-01 + MFC-02) × 1000 (500/1000/1500/2000)
    temperature_C: float | None = None      # median of TIC-B1/B2/B3 (30/40/50)
    ads_time_s: float | None = None         # BPR pulse interval − 50 s overhead

    def label(self) -> str:
        d = self.start_time.strftime("%d %b %Y")
        s = self.start_time.strftime("%H:%M")
        e = self.end_time.strftime("%H:%M")
        return f"{d}  {s} → {e}  ({self.duration_h:.2f} h)"

    def signature(self) -> str:
        """Compact 'P=3 F=1000 T=40 t=480' identifier for matching tables."""
        parts = []
        if self.pressure_bar is not None:
            parts.append(f"P={self.pressure_bar:g}")
        if self.flow_mlmin is not None:
            parts.append(f"F={self.flow_mlmin:.0f}")
        if self.temperature_C is not None:
            parts.append(f"T={self.temperature_C:.0f}")
        if self.ads_time_s is not None:
            parts.append(f"t={self.ads_time_s:.0f}")
        return " ".join(parts) if parts else "—"


# Standard setpoints used by the lab — extracted values are snapped to the
# nearest one if within tolerance (avoids labels like "2.97 bar" → "3 bar").
_PRESSURE_LEVELS = [1.0, 2.0, 3.0]
_FLOW_LEVELS     = [500.0, 1000.0, 1500.0, 2000.0]
_TEMP_LEVELS     = [30.0, 40.0, 50.0]
_ADS_TIME_LEVELS = [180.0, 300.0, 480.0, 600.0, 900.0, 1200.0]


def _snap(value: float, levels: list[float], tol_frac: float = 0.10) -> float:
    """Snap *value* to the nearest entry in *levels* if within tol_frac."""
    if value is None or not np.isfinite(value):
        return value
    nearest = min(levels, key=lambda L: abs(L - value))
    if abs(nearest - value) / nearest <= tol_frac:
        return nearest
    return value


def _extract_signature(df: pd.DataFrame, colmap: dict[str, str],
                       ts: pd.Series, s: int, e: int,
                       bpr_threshold_bar: float = 0.001,
                       overhead_s: float = 50.0,
                       ) -> dict:
    """Return {pressure_bar, flow_mlmin, temperature_C, ads_time_s} for one block."""
    out: dict = {"pressure_bar": None, "flow_mlmin": None,
                 "temperature_C": None, "ads_time_s": None}
    sl = slice(s, e + 1)

    # 1. Pressure = BPR-01 max during cycling (use 95th percentile to ignore stray spikes)
    bpr_col = colmap.get("bpr")
    if bpr_col in df.columns:
        bpr = pd.to_numeric(df[bpr_col].iloc[sl], errors="coerce")
        active = bpr[bpr > bpr_threshold_bar]
        if len(active):
            out["pressure_bar"] = _snap(float(active.quantile(0.95)),
                                        _PRESSURE_LEVELS)

    # 2. Flow = (MFC-01 + MFC-02) × 1000 mL/min during BPR > 0
    co2_col = colmap.get("co2_in")
    n2_col = colmap.get("n2_in")
    if co2_col in df.columns:
        co2 = pd.to_numeric(df[co2_col].iloc[sl], errors="coerce").fillna(0.0)
        n2 = (pd.to_numeric(df[n2_col].iloc[sl], errors="coerce").fillna(0.0)
              if n2_col in df.columns else pd.Series(0.0, index=co2.index))
        if bpr_col in df.columns:
            bpr = pd.to_numeric(df[bpr_col].iloc[sl], errors="coerce").fillna(0.0)
            mask = bpr > bpr_threshold_bar
            co2 = co2[mask]
            n2 = n2[mask]
        if len(co2):
            total_slpm = float(co2.mean()) + float(n2.mean())
            out["flow_mlmin"] = _snap(total_slpm * 1000.0, _FLOW_LEVELS)

    # 3. Temperature = median of TIC-B1/B2/B3 (skip values ≤ -1 = closed)
    tic_means: list[float] = []
    for key in ("tic_b1", "tic_b2", "tic_b3"):
        col = colmap.get(key)
        if col in df.columns:
            t = pd.to_numeric(df[col].iloc[sl], errors="coerce")
            t = t[t > -1.0]
            if len(t):
                tic_means.append(float(t.median()))
    if tic_means:
        out["temperature_C"] = _snap(float(np.median(tic_means)), _TEMP_LEVELS)

    # 4. Adsorption time = median BPR rising-edge interval − overhead
    if bpr_col in df.columns:
        bpr = pd.to_numeric(df[bpr_col].iloc[sl], errors="coerce")
        # rising edge = BPR > threshold after being ≤ threshold
        above = (bpr > bpr_threshold_bar).astype(np.int8)
        rising = np.flatnonzero(above.diff().fillna(0).to_numpy() == 1)
        if len(rising) >= 2:
            ts_block = ts.iloc[sl].reset_index(drop=True)
            edge_times = ts_block.iloc[rising].dropna()
            if len(edge_times) >= 2:
                gaps_s = np.diff(edge_times.astype("int64").to_numpy()) / 1e9
                if len(gaps_s):
                    period_s = float(np.median(gaps_s))
                    ads_s = max(0.0, period_s - overhead_s)
                    out["ads_time_s"] = _snap(ads_s, _ADS_TIME_LEVELS)
    return out


def _classify_segment(dur_h: float, mean_co2_pct: float, n_pulses: int,
                      min_pulses: int) -> str:
    """Tag a segment so the UI can highlight it.

    A block is flagged 'warmup' when BOTH conditions hold:
      - too few BPR pulses (no real PSA cycling), AND
      - low product CO2% (< 30 % vol).
    A real low-flow experiment that still cycles is therefore not lost.
    """
    if n_pulses < min_pulses and mean_co2_pct < 30.0:
        return "warmup"
    if n_pulses < max(2, min_pulses // 2):
        # Almost no cycling at all — definitely not a PSA run
        return "warmup"
    if dur_h * 60.0 < 30.0:
        return "short"
    if dur_h > 50.0:
        return "long"
    return ""


def detect_experiments(df: pd.DataFrame,
                       colmap: dict[str, str],
                       threshold_slpm: float = 0.05,
                       waste_threshold_slpm: float = 5.0,
                       pt10_threshold_mbar: float = -500.0,
                       rolling_n: int = 10,
                       min_gap_min: float = 10.0,
                       min_duration_min: float = 5.0,    # filter zero-duration
                       level_change_frac: float = 0.15,
                       bpr_threshold_bar: float = 0.001,
                       bpr_pulse_gap_min: float = 18.0,
                       min_bpr_pulses: int = 5,
                       max_block_hours: float = 6.0,
                       resplit_std_window: int = 50,
                       resplit_std_threshold: float = 0.005,
                       ) -> list[ExperimentSegment]:
    """
    Return a list of detected experiment windows, in chronological order.

    Parameters
    ----------
    threshold_slpm
        CO2 inlet flow above this counts as "active".
    rolling_n
        Smoothing window for the inlet flow signal (in rows).
    min_gap_min
        Two active blocks separated by a gap shorter than this are merged
        unless the mean CO2 level changes by more than ``level_change_frac``
        (then they're kept as separate experiments).
    min_duration_min
        Discard any block shorter than this.
    level_change_frac
        Relative change in mean MFC-01 level that prevents short-gap merging.
    """
    if colmap.get("co2_in") not in df.columns:
        return []

    ts = _build_timestamp(df, colmap)
    co2 = pd.to_numeric(df[colmap["co2_in"]], errors="coerce")

    # 1. Smooth MFC-01 and apply 3-signal voting:
    #    a row is "active" if at least 2 of {CO2 feed on, waste-gas flowing,
    #    vacuum active} hold. Any missing optional sensor doesn't count for
    #    or against the row — voting threshold scales with available signals.
    smooth = co2.rolling(rolling_n, min_periods=1).mean().fillna(0.0)
    votes = (smooth > threshold_slpm).astype(np.int8)
    n_signals = 1

    waste_col = colmap.get("waste_gas")
    if waste_col in df.columns:
        waste = pd.to_numeric(df[waste_col], errors="coerce").fillna(0.0)
        waste_smooth = waste.rolling(rolling_n, min_periods=1).mean().fillna(0.0)
        votes = votes + (waste_smooth > waste_threshold_slpm).astype(np.int8)
        n_signals += 1

    pt10_col = colmap.get("pt10")
    if pt10_col in df.columns:
        pt10 = pd.to_numeric(df[pt10_col], errors="coerce").fillna(0.0)
        pt10_smooth = pt10.rolling(rolling_n, min_periods=1).mean().fillna(0.0)
        votes = votes + (pt10_smooth < pt10_threshold_mbar).astype(np.int8)
        n_signals += 1

    # Need ≥2 signals when ≥2 are available; otherwise fall back to 1
    required = 2 if n_signals >= 2 else 1
    active = (votes >= required).astype(np.int8)

    # 2. Find rising / falling edges
    diff = active.diff().fillna(0)
    rise = np.flatnonzero(diff.to_numpy() == 1)
    fall = np.flatnonzero(diff.to_numpy() == -1)

    # Handle file that starts or ends mid-experiment
    if active.iloc[0] == 1:
        rise = np.insert(rise, 0, 0)
    if active.iloc[-1] == 1:
        fall = np.append(fall, len(active) - 1)

    # Pair them up
    blocks: list[tuple[int, int]] = []
    for s, e in zip(rise, fall):
        if e > s:
            blocks.append((int(s), int(e)))

    if not blocks:
        return []

    # 3. Merge short gaps unless the flow level differs noticeably
    def mean_level(a: int, b: int) -> float:
        seg = co2.iloc[a:b + 1].dropna()
        return float(seg.mean()) if len(seg) else 0.0

    merged: list[tuple[int, int]] = [blocks[0]]
    for s, e in blocks[1:]:
        prev_s, prev_e = merged[-1]
        gap_min = (ts.iloc[s] - ts.iloc[prev_e]).total_seconds() / 60.0
        if gap_min < min_gap_min:
            lvl_prev = mean_level(prev_s, prev_e)
            lvl_curr = mean_level(s, e)
            ref = max(lvl_prev, lvl_curr, 1e-9)
            if abs(lvl_curr - lvl_prev) / ref <= level_change_frac:
                merged[-1] = (prev_s, e)
                continue
        merged.append((s, e))

    # 3b. Within each block, split further when BPR-01 stops pulsing for
    # long stretches — MFC-01 may stay on between consecutive experiments,
    # but BPR cycling pauses, so the absence of pulses marks a real gap.
    bpr_col = colmap.get("bpr")
    rising_bpr = np.array([], dtype=int)
    if bpr_col in df.columns:
        bpr = pd.to_numeric(df[bpr_col], errors="coerce")
        below = (bpr <= bpr_threshold_bar).astype(np.int8)
        rising_bpr = np.flatnonzero(below.diff().fillna(0).to_numpy() == 1)
        if len(rising_bpr):
            split_blocks: list[tuple[int, int]] = []
            for s, e in merged:
                in_block = rising_bpr[(rising_bpr >= s) & (rising_bpr <= e)]
                if len(in_block) < 2:
                    split_blocks.append((s, e))
                    continue
                # Walk the pulses; if the time between two consecutive pulses
                # exceeds bpr_pulse_gap_min, end the current sub-block.
                sub_start = s
                last_pulse = in_block[0]
                for idx in in_block[1:]:
                    gap_min = (ts.iloc[idx] - ts.iloc[last_pulse]).total_seconds() / 60.0
                    if gap_min > bpr_pulse_gap_min:
                        split_blocks.append((sub_start, last_pulse))
                        sub_start = idx
                    last_pulse = idx
                split_blocks.append((sub_start, e))
            merged = split_blocks

    def pulses_in(a: int, b: int) -> int:
        if not len(rising_bpr):
            return 0
        return int(((rising_bpr >= a) & (rising_bpr <= b)).sum())

    # 3c. For any block still longer than max_block_hours, resplit using the
    # MFC-01 rolling-std valley method: low std for ≥30 min = "machine idle"
    # (no PSA cycling), so use those valleys as additional split points.
    def _resplit_long(s: int, e: int) -> list[tuple[int, int]]:
        if not len(ts):
            return [(s, e)]
        dur_h = (ts.iloc[e] - ts.iloc[s]).total_seconds() / 3600.0
        if dur_h <= max_block_hours:
            return [(s, e)]
        sub = co2.iloc[s:e + 1]
        if len(sub) < resplit_std_window * 2:
            return [(s, e)]
        std = sub.rolling(resplit_std_window, min_periods=1).std().fillna(0.0)
        low = (std < resplit_std_threshold).to_numpy()
        # Find runs of 'low std' lasting > 30 min and use the middle as a split
        splits: list[int] = []
        run_start = None
        for i, v in enumerate(low):
            if v:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_end = i - 1
                    abs_lo = s + run_start
                    abs_hi = s + run_end
                    gap_min = (ts.iloc[abs_hi] - ts.iloc[abs_lo]).total_seconds() / 60.0
                    if gap_min >= 30.0:
                        splits.append((abs_lo + abs_hi) // 2)
                    run_start = None
        if not splits:
            return [(s, e)]
        out_blocks: list[tuple[int, int]] = []
        cur = s
        for sp in splits:
            if sp > cur:
                out_blocks.append((cur, sp))
                cur = sp + 1
        out_blocks.append((cur, e))
        # Recurse — sometimes one valley isn't enough
        final: list[tuple[int, int]] = []
        for a, b in out_blocks:
            sub_dur = (ts.iloc[b] - ts.iloc[a]).total_seconds() / 3600.0
            if sub_dur > max_block_hours and (b - a) > resplit_std_window * 2:
                final.extend(_resplit_long(a, b))
            else:
                final.append((a, b))
        return final

    expanded: list[tuple[int, int]] = []
    for s, e in merged:
        expanded.extend(_resplit_long(s, e))
    merged = expanded

    # 4. Build the result list. Drop zero/near-zero-duration blocks
    # (often the snap-to-valid-timestamp lands on a single row).
    co2_pct_col = colmap.get("co2_pct")
    co2_pct_series = (
        pd.to_numeric(df[co2_pct_col], errors="coerce")
        if co2_pct_col in df.columns else None
    )

    out: list[ExperimentSegment] = []
    for s, e in merged:
        ts_block = ts.iloc[s:e + 1]
        valid_mask = ts_block.notna()
        if not valid_mask.any():
            continue
        valid_positions = np.flatnonzero(valid_mask.to_numpy())
        s_eff = s + int(valid_positions[0])
        e_eff = s + int(valid_positions[-1])
        t_start = ts.iloc[s_eff]
        t_end = ts.iloc[e_eff]
        if pd.isna(t_start) or pd.isna(t_end):
            continue

        dur_h = (t_end - t_start).total_seconds() / 3600.0
        # Hard noise filters — drop these completely (don't even show):
        #   * end ≤ start (timestamps still crossed despite sort)
        #   * shorter than min_duration_min minutes
        #   * longer than 2× max_block_hours (resplit failed; almost certainly bogus)
        #   * effectively zero signal: no pulses AND mean CO2 feed < 0.01 SLPM
        if dur_h <= 0:
            continue
        if dur_h * 60.0 < min_duration_min:
            continue
        if dur_h > max_block_hours * 2:
            continue

        m_co2 = mean_level(s, e)
        n_pulses = pulses_in(s, e)
        if n_pulses == 0 and m_co2 < 0.01:
            continue

        if co2_pct_series is not None:
            m_co2_pct = float(co2_pct_series.iloc[s:e + 1].dropna().mean() or 0.0)
        else:
            m_co2_pct = 0.0
        flag = _classify_segment(dur_h, m_co2_pct, n_pulses, min_bpr_pulses)

        # Hide short warmup blocks too — they're noise, not useful context.
        if flag == "warmup" and dur_h < 0.5:
            continue

        # Pattern Detection — extract pressure/flow/temp/ads_time
        sig = _extract_signature(df, colmap, ts, s_eff, e_eff,
                                 bpr_threshold_bar=bpr_threshold_bar)

        out.append(ExperimentSegment(
            index=len(out) + 1,
            start_idx=s_eff,
            end_idx=e_eff,
            start_time=t_start,
            end_time=t_end,
            duration_h=dur_h,
            mean_co2_slpm=m_co2,
            bpr_pulses=n_pulses,
            flag=flag,
            pressure_bar=sig["pressure_bar"],
            flow_mlmin=sig["flow_mlmin"],
            temperature_C=sig["temperature_C"],
            ads_time_s=sig["ads_time_s"],
        ))

    return out

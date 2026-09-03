"""
Default parameters and physical constants for PSA analysis.

All values defined here are *defaults*. The UI exposes them as editable
fields so the same script works for CALF-20, Zeolite 13X, or any other
adsorbent without code changes.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Physical / experimental constants
# ---------------------------------------------------------------------------
CO2_DENSITY_T_PER_L: float = 1.8e-6       # tonne CO2 per litre — researcher's reference value
SHEET_PREFIX: str = "||PSA"           # auto-detect sheets starting with this
EDGE_THRESHOLD_BAR: float = 0.001     # BPR-01 cycle boundary detector
FINAL_CYCLE_START: int = 3            # cycle # to begin averaging "final" KPIs (lab's revised Excel starts at cycle 3)


# ---------------------------------------------------------------------------
# Column-keyword map — used by the loader to auto-detect columns
# regardless of column letter order. Each list is searched in order;
# the first match wins.
# ---------------------------------------------------------------------------
# Exact column names as they appear in the raw workbook. The loader tries
# these first; if not found it falls back to keyword search.
COLUMN_EXACT: dict[str, str] = {
    "time":    "DATE / TIME",
    "co2_in":  "MFC-01 (CO2 ) SLPM",        # SLPM (L/min)
    "ad_gas":  "MFC-07 (AD-GAS ) SMLM",     # SMLM (mL/min) — converted to L/min internally
    "bpr":     "BPR -01 SLPM",
    "co2_pct": "CO2 ( vol%)",
}

COLUMN_KEYWORDS: dict[str, list[str]] = {
    "time":     ["DATE / TIME", "DATE/TIME", "DATE", "TIME", "Timestamp"],
    "co2_in":   ["MFC-01", "MFC01", "CO2 inlet"],
    "ad_gas":   ["MFC-07", "MFC07", "AD-GAS", "AD GAS"],
    "bpr":      ["BPR -01", "BPR-01", "BPR01", "BPR"],
    "co2_pct":  ["CO2 ( vol%)", "CO2 (vol%)", "CO2_vol"],
}

# Optional columns — used by the experiment detector and steady-state
# extractor for 3-signal voting / sanity checks. If missing, the relevant
# check is skipped instead of failing the load.
OPTIONAL_COLUMN_KEYWORDS: dict[str, list[str]] = {
    "waste_gas": ["MFC-08", "MFC08", "WASTE-GAS", "WASTE GAS"],
    "pt10":      ["PT-10", "PT10"],
    "n2_in":     ["MFC-02", "MFC02", "( N2 )", "N2 inlet"],
    "tic_b1":    ["TI1-1"],
    "tic_b2":    ["TI2-1"],
    "tic_b3":    ["TI3-1"],
}

# Outlet flow column (ad_gas) is reported in SMLM. Convert to SLPM (L/min)
# to match co2_in units before any ratio/recovery math.
SMLM_TO_SLPM: float = 1.0 / 1000.0


# ---------------------------------------------------------------------------
# User-tunable parameters (exposed in the UI sidebar)
# ---------------------------------------------------------------------------
@dataclass
class AnalysisParams:
    """
    Tunable parameters for a single analysis run.

    Passed by value (not by reference to globals) so each run is reproducible
    and parameters can be serialised alongside results.
    """
    bed_volume_mL: float = 60.0           # per single bed (×steps_per_cycle = total)
    steps_per_cycle: int = 4              # 4 bed-steps make one full cycle
    # Kept for backwards compatibility with saved parameter sets only.
    # Nothing reads it: plot smoothing uses a fixed adaptive window
    # (3 cycles for the first 4, 5 thereafter) to match the lab's Excel.
    rolling_window: int = 4
    final_cycle_start: int = FINAL_CYCLE_START
    co2_density_t_per_L: float = CO2_DENSITY_T_PER_L
    edge_threshold_bar: float = EDGE_THRESHOLD_BAR
    temperature_C: float = 40.0           # informational; logged with results
    adsorbent_name: str = "CALF-20"       # informational; for plot titles

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict for JSON/CSV metadata export."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Plot colour palette — fixed, accessible, print-safe
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlotColors:
    """Distinct, colour-blind-friendly palette. No neon."""
    purity:       str = "#1f77b4"   # blue
    recovery:     str = "#ff7f0e"   # orange
    productivity: str = "#2ca02c"   # green
    corrected:    str = "#d62728"   # red (for "corrected" overlay)
    grid:         str = "#cccccc"
    background_light: str = "#ffffff"
    background_dark:  str = "#1e1e1e"


PALETTE = PlotColors()

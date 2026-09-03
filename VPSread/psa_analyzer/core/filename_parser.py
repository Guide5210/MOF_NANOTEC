"""
Parse experimental parameters from the workbook filename.

Naming convention used by the lab:
    YYYYMMDD_MATERIAL_TempC_FlowML_PressureBAR_AdsTimeMIN.xlsx

Example:
    20260331_CALF-20_40C_1000ml_3bar_8min.xlsx
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


# OFAT baseline (matches Exp 11 / 11* per lab convention)
BASELINE: dict = {
    "temp_C":         40,
    "flow_mlmin":     1000,
    "pressure_barg":  3,
    "ads_time_s":     480,   # 8 min
}


@dataclass
class FilenameParams:
    date: date
    material: str
    temp_C: int
    flow_mlmin: int
    pressure_barg: int
    ads_time_s: int

    @property
    def ads_time_min(self) -> int:
        return self.ads_time_s // 60

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "material": self.material,
            "temp_C": self.temp_C,
            "flow_mlmin": self.flow_mlmin,
            "pressure_barg": self.pressure_barg,
            "ads_time_s": self.ads_time_s,
        }


_PATTERN = re.compile(
    r"""
    ^(?P<date>\d{8})_
    (?P<material>[^_]+)_
    (?P<temp>\d+)C_
    (?P<flow>\d+)ml_
    (?P<pressure>\d+)bar_
    (?P<ads_min>\d+)min$
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_filename(filename: str) -> FilenameParams:
    """Extract experiment parameters from the workbook filename."""
    name = os.path.basename(filename)
    name = re.sub(r"\.xlsx?$", "", name, flags=re.IGNORECASE)
    m = _PATTERN.match(name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    return FilenameParams(
        date=datetime.strptime(m.group("date"), "%Y%m%d").date(),
        material=m.group("material"),
        temp_C=int(m.group("temp")),
        flow_mlmin=int(m.group("flow")),
        pressure_barg=int(m.group("pressure")),
        ads_time_s=int(m.group("ads_min")) * 60,
    )


def try_parse_filename(filename: str) -> Optional[FilenameParams]:
    """Like ``parse_filename`` but returns None on mismatch instead of raising."""
    try:
        return parse_filename(filename)
    except ValueError:
        return None


def classify_group(params: FilenameParams | dict) -> str:
    """Tag which OFAT variable this run varies vs the baseline."""
    if isinstance(params, FilenameParams):
        params = params.to_dict()
    diffs = {k for k in BASELINE if params.get(k) != BASELINE[k]}
    if not diffs:
        return "Repeat (baseline)"
    if diffs == {"ads_time_s"}:
        return "Effect of adsorption/desorption time"
    if diffs == {"flow_mlmin"}:
        return "Effect of inlet flowrate"
    if diffs == {"pressure_barg"}:
        return "Effect of pressure"
    if diffs == {"temp_C"}:
        return "Effect of temperature"
    return f"Multi-variable (changed: {sorted(diffs)})"

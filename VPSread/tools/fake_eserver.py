"""
Fake eServer CSV writer — for testing the Live Monitor without the real PLC.

It writes a growing CSV in eServer's exact format (M/D/YYYY  h:mm:ss AM/PM
timestamps, the same column layout incl. the swapped CO/CH4 and unused O2)
and appends one new row every `--every` real seconds, advancing the timestamp
by `--step` simulated seconds. BPR-01 drops every 10 rows so 4 steps = 1 cycle,
exactly like the rig.

Usage (run in its own terminal, leave it running):

    python tools/fake_eserver.py
    python tools/fake_eserver.py --path D:\PSA_live\log.csv --every 1 --step 5

Then launch the app, click "Live Monitor (PLC)", and select the same file.
Press Ctrl+C here to stop.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

HEADER = (
    "DATE / TIME,MFC-01 (CO2) SLPM,MFC-02 (N2) SLPM,MFC-03 (CH4) SLPM,"
    "MFC-06 (MIX) NLPM,MFC-07 (AD-GAS) SMLM,MFC-08 (WASTE-GAS) SMLM,"
    "BPR-01 SLPM,PT-1 (bar),TI3-1 (C),TIC-B1 (C),PIC-PPE (bar),"
    "CO (vol%),CO2 (vol%),CH4 (vol%),O2 (vol%)"
)


def make_row(i: int, t: datetime, rows_per_step: int = 10) -> str:
    """One eServer-style data line for sample index i at sim-time t."""
    ts = t.strftime("%m/%d/%Y  %I:%M:%S %p")          # double space, AM/PM
    bpr = 0.0 if (i % rows_per_step) == 0 else 1.0     # drop marks a new step
    co2 = 85.0 + 2.0 * math.sin(i / 25.0) + random.uniform(-0.5, 0.5)
    mfc01 = 1.0 + random.uniform(-0.02, 0.02)
    mfc07 = 1.2 + random.uniform(-0.03, 0.03)
    temp = 40.0 + random.uniform(-0.3, 0.3)
    return (
        f"{ts},{mfc01:.3f},0.500,0.000,0.000,{mfc07:.3f},0.000,"
        f"{bpr:.3f},-0.10,{temp:.1f},{temp:.1f},2.50,"
        f"0.00,{co2:.2f},0.00,0.00"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(Path.home() / "Desktop" / "psa_live_test.csv"),
                    help="CSV file to write/append to")
    ap.add_argument("--every", type=float, default=1.0,
                    help="real seconds between appended rows")
    ap.add_argument("--step", type=float, default=5.0,
                    help="simulated seconds added to the timestamp per row")
    ap.add_argument("--seed", type=int, default=200,
                    help="number of rows to pre-fill before live appending")
    args = ap.parse_args()

    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t = datetime.now().replace(microsecond=0)
    i = 0
    # Fresh file with header + a seed block so there's history to analyse.
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(HEADER + "\n")
        for _ in range(args.seed):
            f.write(make_row(i, t) + "\n")
            i += 1
            t += timedelta(seconds=args.step)
    print(f"Wrote {args.seed} seed rows to {path}")
    print(f"Appending 1 row every {args.every}s (sim step {args.step}s). Ctrl+C to stop.")

    try:
        while True:
            time.sleep(args.every)
            with path.open("a", encoding="utf-8", newline="") as f:
                f.write(make_row(i, t) + "\n")
            i += 1
            t += timedelta(seconds=args.step)
            if i % 10 == 0:
                print(f"  appended row {i} ({t:%I:%M:%S %p})")
    except KeyboardInterrupt:
        print(f"\nStopped at {i} rows. File: {path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

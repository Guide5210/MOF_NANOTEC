"""
Read the HORIBA VA-5000 gas analyzer over Modbus/TCP — per its own manual.

Settings come straight from chapter 8 "External Input/Output" of the VA-5000
instruction manual, so nothing here is guessed:

    Physical interface : TCP/IP
    Protocol           : Modbus/TCP
    TCP port           : 502
    Slave address      : **fixed to 255**   <- not 1, and not ignored
    Function code      : 03 (Read Holding Registers)
    Data length 2      : stored in consecutive addresses, **high word first**
                         (Fig. 94: 0x12345678 -> [N]=0x1234, [N+1]=0x5678)
                         i.e. float32 word order = BIG (the PLC uses little)

Concentrations live at (manual "Address map", data length 2, float):

    10506  Instantaneous value - Component 1 concentration value
    10508  Instantaneous value - Component 2 concentration value
    10510  Instantaneous value - Component 3 concentration value
    10512  Instantaneous value - Component 4 concentration value

IMPORTANT — the manual states:
    "Both of read and write address not specified in the address map are
     prohibited. Do not read and write the prohibited address. That may
     affect the performance of the analyzer."
So this script never sweeps blindly; it touches documented addresses only.

USAGE:

    python tools/read_horiba.py                       # 192.168.1.100, one shot
    python tools/read_horiba.py 192.168.1.100         # explicit host
    python tools/read_horiba.py 192.168.1.100 --watch # refresh every 2 s
    python tools/read_horiba.py 192.168.1.100 502 255 # host port unit

Requires pymodbus:   pip install pymodbus
"""

from __future__ import annotations

import struct
import sys
import time

DEFAULT_HOST = "192.168.1.100"
DEFAULT_PORT = 502
DEFAULT_UNIT = 255          # manual: "Slave address — Fixed to 255"

# Component N concentration float sits at CONC_BASE + 2*(N-1).
CONC_BASE = 10506
CORR_BASE = 10522           # "correction concentration value" (O2-corrected)

UNIT_CODES = {0: "vol%", 1: "ppm", 2: "(reserved)", 3: "g/m3", 4: "mg/m3",
              5: "(reserved)"}


# ---------------------------------------------------------------------------
# Modbus plumbing (tolerant of pymodbus 2.x / 3.x keyword differences)
# ---------------------------------------------------------------------------
def read_holding(client, addr: int, count: int, unit: int):
    """Read holding registers; returns the word list, or None on error."""
    for kw in ("device_id", "slave", "unit"):
        try:
            rr = client.read_holding_registers(addr, count=count, **{kw: unit})
            break
        except TypeError:
            continue
        except Exception:
            return None
    else:
        try:
            rr = client.read_holding_registers(addr, count=count)
        except Exception:
            return None
    if rr is None or not hasattr(rr, "registers"):
        return None
    try:
        if rr.isError():
            return None
    except Exception:
        return None
    return list(rr.registers)


def f32(words: list[int], i: int) -> float:
    """Decode words[i], words[i+1] as float32, high word first (manual Fig. 94)."""
    raw = ((words[i] & 0xFFFF) << 16) | (words[i + 1] & 0xFFFF)
    return struct.unpack(">f", struct.pack(">I", raw))[0]


def u32(words: list[int], i: int) -> int:
    """Decode words[i], words[i+1] as ulong, high word first."""
    return ((words[i] & 0xFFFF) << 16) | (words[i + 1] & 0xFFFF)


# ---------------------------------------------------------------------------
def snapshot(client, unit: int) -> dict:
    """Read one full picture of the analyzer from documented addresses only."""
    out: dict = {}

    # -- instantaneous value block ------------------------------------------
    # 10495 update mode; 10496..10501 timestamp; 10502 request
    w = read_holding(client, 10495, 8, unit)
    if w:
        out["update_mode"] = w[0]
        out["stamp"] = tuple(w[1:7])          # Y M D H M S (year 0 == 2000)

    # 10506..10513 = component 1..4 concentration (float, 2 words each)
    w = read_holding(client, CONC_BASE, 8, unit)
    if w:
        out["conc"] = [f32(w, 2 * i) for i in range(4)]

    # 10522..10529 = component 1..4 correction concentration
    w = read_holding(client, CORR_BASE, 8, unit)
    if w:
        out["corr"] = [f32(w, 2 * i) for i in range(4)]

    # -- analyzer status ----------------------------------------------------
    w = read_holding(client, 10840, 6, unit)   # status 1, 2, 3 (ulong each)
    if w:
        st1 = u32(w, 0)
        out["measuring"] = bool(st1 >> 9 & 1)
        out["maintenance"] = bool(st1 >> 16 & 1)
        out["auto_cal"] = bool(st1 >> 18 & 1)
        out["auto_zero"] = bool(st1 >> 19 & 1)

    # -- per-component metadata --------------------------------------------
    w = read_holding(client, 10858, 4, unit)   # present range enable flag
    if w:
        out["enabled"] = [bool(x) for x in w]
    w = read_holding(client, 10866, 4, unit)   # digits after decimal point
    if w:
        out["digits"] = list(w)
    w = read_holding(client, 10874, 4, unit)   # unit code
    if w:
        out["units"] = [UNIT_CODES.get(x, f"?{x}") for x in w]
    w = read_holding(client, 10882, 8, unit)   # range value (x100), ulong each
    if w:
        out["range"] = [u32(w, 2 * i) / 100.0 for i in range(4)]

    # -- gas line -----------------------------------------------------------
    w = read_holding(client, 11520, 2, unit)
    if w:
        out["gas_flow"] = w[0]     # 0 sample gas / 1 calibration gas
        out["gas_line"] = w[1]

    return out


def print_snapshot(snap: dict, host: str, port: int, unit: int) -> None:
    """Pretty-print one snapshot."""
    print("=" * 72)
    print(f" HORIBA VA-5000  {host}:{port}  unit {unit}   "
          f"(FC3, float32 word order = big)")
    print("=" * 72)

    if "stamp" in snap:
        y, mo, d, h, mi, s = snap["stamp"]
        print(f"  analyzer time   : {2000 + y if y < 100 else y:04d}-"
              f"{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d}")
    if "update_mode" in snap:
        mode = ("auto-update (default)" if snap["update_mode"]
                else "manual — value refreshes only when 10502 is read/written")
        print(f"  update mode     : {snap['update_mode']}  ({mode})")
    if "measuring" in snap:
        flags = []
        if snap.get("measuring"):    flags.append("MEASURING")
        if snap.get("maintenance"):  flags.append("MAINTENANCE")
        if snap.get("auto_cal"):     flags.append("AUTO-CAL")
        if snap.get("auto_zero"):    flags.append("AUTO-ZERO")
        print(f"  analyzer state  : {', '.join(flags) or 'idle'}")
    if "gas_flow" in snap:
        gas = "sample gas" if snap["gas_flow"] == 0 else "CALIBRATION gas"
        print(f"  gas flowing     : {gas}  (line code {snap.get('gas_line')})")

    if "conc" not in snap:
        print("\n  [ERROR] could not read the concentration block (10506).")
        return

    print()
    print(f"  {'Comp':<6}{'Address':<9}{'Concentration':>16}"
          f"  {'Unit':<7}{'Range':>9}  {'Digits':>6}  {'Enabled':>7}")
    print("  " + "-" * 68)
    for i in range(4):
        addr = CONC_BASE + 2 * i
        val = snap["conc"][i]
        u = snap.get("units", ["?"] * 4)[i]
        rng = snap.get("range", [float("nan")] * 4)[i]
        dg = snap.get("digits", ["?"] * 4)[i]
        en = snap.get("enabled", [None] * 4)[i]
        en_s = "-" if en is None else ("yes" if en else "no")
        print(f"  {i + 1:<6}{addr:<9}{val:>16.4f}  {u:<7}{rng:>9.2f}  "
              f"{dg:>6}  {en_s:>7}")

    if "corr" in snap:
        print("\n  correction (O2-corrected) values, addresses 10522+:")
        print("    " + "  ".join(f"C{i + 1}={snap['corr'][i]:.4f}"
                                 for i in range(4)))


# ---------------------------------------------------------------------------
def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    watch = "--watch" in sys.argv

    host = args[0] if args else DEFAULT_HOST
    port = int(args[1]) if len(args) > 1 else DEFAULT_PORT
    unit = int(args[2]) if len(args) > 2 else DEFAULT_UNIT

    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError:
        print("[ERROR] pymodbus is not installed.  Run:  pip install pymodbus")
        return 1

    client = ModbusTcpClient(host, port=port, timeout=2.0)
    if not client.connect():
        print(f"[ERROR] cannot connect to {host}:{port}")
        print("        - is this PC on the same 192.168.1.x network?")
        print("        - check the analyzer screen: COMMUNICATION 1/2")
        return 1

    try:
        while True:
            snap = snapshot(client, unit)
            if not snap:
                print(f"[ERROR] connected to {host}:{port}, but unit {unit} "
                      f"answered nothing.")
                print("        The manual says the slave address is fixed to "
                      "255 — try:  python tools/read_horiba.py "
                      f"{host} {port} 255")
                return 1
            print_snapshot(snap, host, port, unit)
            if not watch:
                break
            print("\n(press Ctrl+C to stop)\n")
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        client.close()

    print("\nNext: compare the four numbers above with the analyzer's own")
    print("screen to learn which Component is CO / CO2 / CH4 / O2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

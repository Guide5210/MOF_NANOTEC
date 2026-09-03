"""
Watch the PLC while the rig runs and report which registers actually MOVE.

WHY THIS EXISTS
---------------
A static scan cannot identify a tag when everything reads zero at standstill.
But a process value has one property a constant does not: **it changes when
the plant runs.** So: sample the whole readable range repeatedly across a few
PSA cycles, then list every address that moved, biggest swing first.

Whatever moves in step with the plant is a real measurement. In particular
BPR-01 - the signal the cycle detector keys on - must swing every cycle, so
it will stand out. Anything that never budges is a constant, a setpoint, or
simply not connected.

If NOTHING moves while the plant is clearly running, that is just as decisive:
it means the process values do not live in the PLC's Modbus window at all, and
they have to be read from whatever device does own them (the way CO2 turned
out to belong to the HORIBA analyzer, not the PLC).

USAGE (on the eServer PC):

    python tools/watch_plc.py                       # 192.168.1.5, 0..4000
    python tools/watch_plc.py 192.168.1.5
    python tools/watch_plc.py 192.168.1.5 502 1    # host port unit
    python tools/watch_plc.py 192.168.1.5 502 1 0 4000
    python tools/watch_plc.py --seconds 600         # stop after 10 minutes
    python tools/watch_plc.py --out D:\run1.txt     # where to save the findings

Start it, run the rig for two or three full cycles, then press Ctrl+C.

The findings are written by this program itself (default watch_result.txt in
the current folder) -- do NOT pipe it through Tee-Object, because Ctrl+C tears
down a PowerShell pipeline before the summary is ever written to the file.

Requires pymodbus:   pip install pymodbus
"""

from __future__ import annotations

import struct
import sys
import time

WORD_ORDER = "little"     # same as the PLC's other registers
POLL_INTERVAL = 1.0       # seconds between sweeps
CHUNK = 125               # Modbus max per request

# A float this close to its own min/max counts as "did not move" - filters out
# the last-bit jitter of an analog input sitting still.
NOISE_FLOOR = 1e-4


def read_holding(client, addr: int, count: int, unit: int):
    """Read holding registers, tolerant of pymodbus API differences."""
    for kw in ("device_id", "slave", "unit"):
        try:
            return client.read_holding_registers(addr, count=count, **{kw: unit})
        except TypeError:
            continue
        except Exception:
            return None
    try:
        return client.read_holding_registers(addr, count=count)
    except Exception:
        return None


def words_of(rr) -> list[int] | None:
    if rr is None or not hasattr(rr, "registers"):
        return None
    try:
        if rr.isError():
            return None
    except Exception:
        return None
    return list(rr.registers)


def decode_float(w0: int, w1: int) -> float:
    if WORD_ORDER == "little":
        combined = ((w1 & 0xFFFF) << 16) | (w0 & 0xFFFF)
    else:
        combined = ((w0 & 0xFFFF) << 16) | (w1 & 0xFFFF)
    return struct.unpack(">f", struct.pack(">I", combined))[0]


# ---------------------------------------------------------------------------
def discover(client, start: int, end: int, unit: int) -> list[tuple[int, int]]:
    """Find what the PLC will actually serve, as contiguous (addr, count) runs.

    Done once up front: a block containing an unreadable address is refused as
    a whole, so failures are subdivided rather than thrown away. The resulting
    runs are then re-read cheaply on every later sweep.
    """
    readable: list[int] = []

    def walk(addr: int, count: int) -> None:
        if words_of(read_holding(client, addr, count, unit)) is not None:
            readable.extend(range(addr, addr + count))
            return
        if count == 1:
            return
        half = count // 2
        walk(addr, half)
        walk(addr + half, count - half)

    addr = start
    while addr <= end:
        count = min(CHUNK, end - addr + 1)
        walk(addr, count)
        addr += count

    runs: list[tuple[int, int]] = []
    for a in readable:
        if runs and a == runs[-1][0] + runs[-1][1] and runs[-1][1] < CHUNK:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((a, 1))
    return runs


def sweep(client, runs: list[tuple[int, int]], unit: int) -> dict[int, int]:
    """Read every discovered run once; returns {address: word}."""
    out: dict[int, int] = {}
    for addr, count in runs:
        words = words_of(read_holding(client, addr, count, unit))
        if words:
            for j, w in enumerate(words):
                out[addr + j] = w
    return out


def build_report(lo, hi, last, hits, sweeps, elapsed) -> str:
    """The full findings, as text (printed AND written to the output file)."""
    out: list[str] = []
    w = out.append

    changed = sorted(((hi[a] - lo[a], a) for a in lo
                      if hi[a] - lo[a] > NOISE_FLOOR), reverse=True)

    w("=" * 68)
    w(f" RESULTS after {sweeps} sweeps over {elapsed:.0f}s")
    w("=" * 68)
    w("")
    w(f" ADDRESSES THAT MOVED  ({len(changed)}) - biggest swing first")
    w("-" * 68)
    if not changed:
        w("  NONE.")
        w("")
        w("  Every readable register held still for the whole run.")
        w("  If the rig really was running, the process values are not in")
        w("  the PLC's Modbus window at all - they belong to another")
        w("  device, exactly like CO2 belongs to the HORIBA analyzer.")
        w("  Next step: open eServer's tag/address configuration and see")
        w("  which device each tag is bound to.")
    else:
        w(f"  {'Addr':>6}  {'min':>13}  {'max':>13}  {'swing':>13}"
          f"  {'now':>13}  {'steps':>6}")
        for swing, a in changed[:40]:
            w(f"  {a:>6}  {lo[a]:>13.5f}  {hi[a]:>13.5f}  {swing:>13.5f}"
              f"  {last[a]:>13.5f}  {hits.get(a, 0):>6}")
        if len(changed) > 40:
            w(f"  ... and {len(changed) - 40} more")
        w("")
        w("  'steps' = how many times the value actually changed. A signal")
        w("  with many steps and a large swing is a live process value; two")
        w("  or three steps over a small range is just analog jitter.")

    w("")
    w(" NON-ZERO BUT UNCHANGING - constants / setpoints / idle sensors")
    w("-" * 68)
    still = sorted(a for a in lo
                   if hi[a] - lo[a] <= NOISE_FLOOR and abs(last[a]) > 1e-9)
    if not still:
        w("  (none)")
    for a in still[:40]:
        w(f"  {a:>6} = {last[a]:>13.5f}")
    if len(still) > 40:
        w(f"  ... and {len(still) - 40} more")
    return "\n".join(out)


# ---------------------------------------------------------------------------
def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seconds = 0.0
    if "--seconds" in sys.argv:
        try:
            seconds = float(sys.argv[sys.argv.index("--seconds") + 1])
            args = [a for a in args if a != str(seconds).rstrip("0").rstrip(".")]
        except (IndexError, ValueError):
            seconds = 0.0

    out_path = "watch_result.txt"
    if "--out" in sys.argv:
        try:
            out_path = sys.argv[sys.argv.index("--out") + 1]
            args = [a for a in args if a != out_path]
        except IndexError:
            pass

    host = args[0] if args else "192.168.1.5"
    port = int(args[1]) if len(args) > 1 else 502
    unit = int(args[2]) if len(args) > 2 else 1
    start = int(args[3]) if len(args) > 3 else 0
    end = int(args[4]) if len(args) > 4 else 4000

    print("=" * 68)
    print(f" PLC change watcher  -  {host}:{port}  unit {unit}  "
          f"addresses {start}..{end}")
    print("=" * 68)

    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError:
        print("\n[ERROR] pymodbus is not installed.  Run:  pip install pymodbus")
        return 1

    client = ModbusTcpClient(host, port=port, timeout=2)
    if not client.connect():
        print(f"\n[ERROR] Cannot connect to {host}:{port}")
        return 1
    print("[OK] Connected.\n")

    print("Discovering which addresses the PLC will serve (once) ...")
    runs = discover(client, start, end, unit)
    total = sum(c for _, c in runs)
    print(f"  {total} readable words in {len(runs)} block(s)\n")
    if not total:
        print("[ERROR] nothing readable - check host/port/unit.")
        return 1

    print("-" * 68)
    print(" WATCHING.  Run the rig now - let it complete 2-3 full cycles.")
    print(" Press Ctrl+C when done to see what moved.")
    print(f" Findings are written to {out_path} either way.")
    print("-" * 68)

    lo: dict[int, float] = {}
    hi: dict[int, float] = {}
    last: dict[int, float] = {}
    hits: dict[int, int] = {}      # how many times each address actually changed
    sweeps = 0
    t0 = time.time()

    try:
        while True:
            words = sweep(client, runs, unit)
            for a in range(start, end):
                if a % 2 or a not in words or (a + 1) not in words:
                    continue
                v = decode_float(words[a], words[a + 1])
                if v != v or abs(v) > 1e12:      # NaN / nonsense
                    continue
                if a not in lo:
                    lo[a] = hi[a] = v
                else:
                    if abs(v - last[a]) > NOISE_FLOOR:
                        hits[a] = hits.get(a, 0) + 1
                    lo[a] = min(lo[a], v)
                    hi[a] = max(hi[a], v)
                last[a] = v
            sweeps += 1

            moved = sum(1 for a in lo if hi[a] - lo[a] > NOISE_FLOOR)
            elapsed = time.time() - t0
            print(f"  sweep {sweeps:>5}   {elapsed:>6.0f}s   "
                  f"{moved} address(es) have moved so far", flush=True)

            # An interim list every 30 sweeps, so the answer is on screen even
            # if the run is cut short and never reaches the summary.
            if moved and sweeps % 30 == 0:
                addrs = sorted((hi[a] - lo[a], a) for a in lo
                               if hi[a] - lo[a] > NOISE_FLOOR)
                print("     so far: " + ", ".join(
                    f"{a}({s:.4g})" for s, a in reversed(addrs[-10:])),
                    flush=True)

            if seconds and elapsed >= seconds:
                break
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n  (stopped by Ctrl+C)")

    client.close()
    print("")

    # -- results ------------------------------------------------------------
    # Written by this program rather than piped, so Ctrl+C killing a shell
    # pipeline can never take the findings with it.
    report = build_report(lo, hi, last, hits, sweeps, time.time() - t0)
    print(report)
    try:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"PLC change watcher - {host}:{port} unit {unit} "
                     f"addresses {start}..{end}\n")
            fh.write(f"{total} readable words in {len(runs)} block(s)\n\n")
            fh.write(report + "\n")
        print(f"\n  Findings saved to: {out_path}")
    except Exception as exc:        # noqa: BLE001
        print(f"\n  [WARN] could not write {out_path}: {exc}")

    print("\n" + "=" * 68)
    print(" DONE.  Copy everything above and send it back.")
    print(" Please also note what the rig was doing during the watch.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

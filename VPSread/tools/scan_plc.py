"""
One-shot Modbus scan for the Delta AS PLC (run on the eServer PC).

Run this on the machine that can already reach the PLC/HMI (the eServer PC).
It reads a wide block of registers as 32-bit floats, lists every non-zero
value with its address, and flags the ones that look like our sensors
(CO2 vol%, the MFC flows, BPR). It also probes the high address where CO2
lives in eServer (D28091) to confirm whether it is reachable at all.

USAGE (on the eServer PC, in PowerShell / cmd):

    python tools/scan_plc.py
    python tools/scan_plc.py 192.168.1.5
    python tools/scan_plc.py 192.168.1.5 502 1          # host port unit
    python tools/scan_plc.py 192.168.1.5 502 1 0 4000   # ... start end

    # locate a tag: read its value off eServer, then find which register holds it
    python tools/scan_plc.py --find 0.068
    python tools/scan_plc.py 192.168.1.5 502 1 0 4000 --find 0.068 -0.8

    # sweep EVERY device file (D, X, Y, SR, T, C, E) with the right function
    # code. X holds the analog inputs and needs FC04 -- no earlier scan of ours
    # ever issued that, so X has never once been read.
    python tools/scan_plc.py 192.168.1.5 502 1 0 4000 --devices

Requires pymodbus:   pip install pymodbus

Then copy EVERYTHING the script prints and send it back.
"""

from __future__ import annotations

import struct
import sys

# ---- args ----------------------------------------------------------------
# --find VALUE [VALUE ...] : report every address holding (approximately) each
# value. This is how a tag is located: read the number off eServer's screen,
# search for it here, and the address that answers is that tag's register.
_argv = sys.argv[1:]
FIND: list[float] = []
if "--find" in _argv:
    i = _argv.index("--find")
    for tok in _argv[i + 1:]:
        try:
            FIND.append(float(tok))
        except ValueError:
            break
    _argv = _argv[:i]

# --devices : sweep every AS-series device file (D, X, Y, SR, T, C, E) with the
# right function code, instead of only the D range given on the command line.
SWEEP_DEVICES = "--devices" in _argv
if SWEEP_DEVICES:
    _argv = [a for a in _argv if a != "--devices"]

HOST  = _argv[0] if len(_argv) > 0 else "192.168.1.5"
PORT  = int(_argv[1]) if len(_argv) > 1 else 502
UNIT  = int(_argv[2]) if len(_argv) > 2 else 1
START = int(_argv[3]) if len(_argv) > 3 else 0
END   = int(_argv[4]) if len(_argv) > 4 else 4000
WORD_ORDER = "little"   # confirmed correct earlier (floats at even addresses)

# A match is "close enough" within this relative tolerance, so a screen value
# rounded to 3 decimals (0.068) still finds a register holding 0.0683.
FIND_RTOL = 0.02

# Delta's "Modbus TCP Specifications" sheet: FC03/FC04 take 1-100 words per
# request (plain Modbus allows 125), and the server accepts only 8 simultaneous
# connections in total -- shared with eServer and the HMI. Run one tool at a
# time.
MAX_READ_WORDS = 100

# Each AS-series device file at its raw protocol address (the published
# 4xxxx / 3xxxx number minus its base), with the function code that reaches it.
# Our sweeps have only ever covered 0..4000 of the D file, so everything below
# marked FC04 -- or sitting above D4000 -- has never been looked at.
#   name: (first, last, function code)
AS_DEVICES: dict[str, tuple[int, int, int]] = {
    "D  data registers":    (0,     29999, 3),
    "X  inputs (word)":     (32768, 32831, 4),   # FC04 ONLY
    "Y  outputs (word)":    (40960, 41023, 3),
    "SR special registers": (49152, 51199, 3),
    "T  timers (word)":     (57344, 57855, 3),
    "C  counters (word)":   (61440, 61951, 3),
    "E  index registers":   (65024, 65038, 3),
}


def decode_float(w0: int, w1: int, word_order: str = "little") -> float:
    """Decode two 16-bit registers into a float32."""
    if word_order == "little":
        combined = ((w1 & 0xFFFF) << 16) | (w0 & 0xFFFF)
    else:
        combined = ((w0 & 0xFFFF) << 16) | (w1 & 0xFFFF)
    return struct.unpack(">f", struct.pack(">I", combined))[0]


def read_holding(client, addr: int, count: int, unit: int, fc: int = 3):
    """Read registers, tolerant of pymodbus API differences.

    ``fc`` 3 reads holding registers (D, Y, SR, T, C, E); ``fc`` 4 reads input
    registers, which is the ONLY way to reach the X device file. An FC03 sweep
    reports X as absent rather than as an error, so anything the PLC program
    leaves in X stays invisible until the function code is right.
    """
    fn = (client.read_input_registers if int(fc) == 4
          else client.read_holding_registers)
    for kw in ("device_id", "slave", "unit"):
        try:
            return fn(addr, count=count, **{kw: unit})
        except TypeError:
            continue
    return fn(addr, count=count)


def _try_block(client, addr: int, count: int, unit: int,
               fc: int = 3) -> list[int] | None:
    """Read one block; None if the device refuses it."""
    try:
        rr = read_holding(client, addr, count, unit, fc)
    except Exception:
        return None
    if rr is None or rr.isError() or not hasattr(rr, "registers"):
        return None
    return list(rr.registers)


def read_range(client, start: int, end: int, unit: int,
               fc: int = 3) -> dict[int, int]:
    """Read start..end inclusive, keeping every word the device will give.

    Reads in MAX_READ_WORDS blocks for speed, but a block containing even one
    unreadable address is refused *as a whole* — so on failure the block is
    subdivided down to single registers instead of being thrown away. Without
    this, readable registers sitting next to a gap stay invisible.
    """
    out: dict[int, int] = {}

    def walk(addr: int, count: int) -> None:
        words = _try_block(client, addr, count, unit, fc)
        if words is not None:
            for j, w in enumerate(words):
                out[addr + j] = w
            return
        if count == 1:
            return                      # genuinely not exposed
        half = count // 2
        walk(addr, half)
        walk(addr + half, count - half)

    addr = start
    while addr <= end:
        count = min(MAX_READ_WORDS, end - addr + 1)
        walk(addr, count)
        addr += count
    return out


def main() -> int:
    print("=" * 60)
    print(f" PLC Modbus Scan  -  {HOST}:{PORT}  unit {UNIT}")
    print("=" * 60)

    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError:
        print("\n[ERROR] pymodbus is not installed on this PC.")
        print("        Run:  pip install pymodbus")
        print("        Then run this script again.")
        return 1

    client = ModbusTcpClient(HOST, port=PORT, timeout=2)
    if not client.connect():
        print(f"\n[ERROR] Cannot connect to {HOST}:{PORT}")
        print("        - Is this PC on the same network as the PLC?")
        print("        - Is Modbus TCP enabled on the HMI?")
        return 1
    print("[OK] Connected.\n")

    # ---- wide scan -------------------------------------------------------
    print(f"Scanning addresses {START}..{END} "
          f"(float32, word order = {WORD_ORDER}) ...")
    words = read_range(client, START, END, UNIT)
    total = END - START + 1
    print(f"  readable words: {len(words)} / {total}\n")

    # Which parts of the address space the PLC actually publishes. The shape of
    # these blocks is the clue to how its Modbus map relates to D-numbers: a
    # tag can only live inside one of them.
    if words:
        addrs = sorted(words)
        blocks: list[list[int]] = [[addrs[0], addrs[0]]]
        for a in addrs[1:]:
            if a == blocks[-1][1] + 1:
                blocks[-1][1] = a
            else:
                blocks.append([a, a])
        print("-" * 60)
        print(f" READABLE BLOCKS ({len(blocks)})")
        print("-" * 60)
        for lo, hi in blocks:
            nz = sum(1 for a in range(lo, hi + 1) if words.get(a))
            print(f"  {lo:>6} .. {hi:<6}  ({hi - lo + 1:>5} words, "
                  f"{nz} non-zero)")
        print()

    # decode every even-aligned float pair we have both words for
    floats: list[tuple[int, float]] = []
    for addr in range(START, END + 1, 2):
        if addr in words and (addr + 1) in words:
            v = decode_float(words[addr], words[addr + 1], WORD_ORDER)
            floats.append((addr, v))

    def ok(v: float) -> bool:
        return (v == v) and abs(v) > 1e-6 and abs(v) < 1e7  # finite, non-zero

    print("-" * 60)
    print(" NON-ZERO FLOAT VALUES  (addr = value)")
    print("-" * 60)
    shown = 0
    for addr, v in floats:
        if ok(v):
            print(f"  {addr:>6} = {v:>14.5f}")
            shown += 1
    if shown == 0:
        print("  (none - the readable range held only zeros)")

    # ---- raw 16-bit words -------------------------------------------------
    # Decoding only float32 at even addresses hides anything stored as a plain
    # 16-bit integer -- and eServer's own address table shows the TI channels
    # are exactly that ("Unsigned / Word / Read Count 1"), so a temperature of
    # 29.5 C is sitting in memory as 295. Those words are invisible in the
    # float view above, which is why the block reports far more non-zero words
    # than the float list can account for.
    # Words already accounted for by a non-zero float above are not news; what
    # matters is the remainder, because those are the tags we still cannot see.
    claimed: set[int] = set()
    for addr, v in floats:
        if ok(v):
            claimed.add(addr)
            claimed.add(addr + 1)

    nonzero_words = [(a, w) for a, w in sorted(words.items()) if w]
    unexplained = [(a, w) for a, w in nonzero_words if a not in claimed]

    print()
    print("-" * 60)
    print(f" UNEXPLAINED NON-ZERO WORDS ({len(unexplained)} of "
          f"{len(nonzero_words)})  -- 16-bit view")
    print("-" * 60)
    print("  Holding data, but NOT part of any float above. eServer's own")
    print("  address table stores the TI channels as 'Unsigned / Word /")
    print("  Read Count 1', so 29.5 C sits in memory as the integer 295.")
    print()
    if not unexplained:
        print("  (none - every non-zero word belongs to a float)")
    for a, w in unexplained[:150]:
        signed = w - 65536 if w > 32767 else w
        print(f"  {a:>6} = {w:>6}   signed {signed:>7}"
              f"   /10 = {signed/10:>9.2f}   /100 = {signed/100:>9.3f}")
    if len(unexplained) > 150:
        print(f"  ... and {len(unexplained) - 150} more")
    print()
    print("  Read a value off eServer, find it in the /10 or /100 column, and")
    print("  that address is that tag. Send this whole list back.")
    print()

    # ---- every device file ------------------------------------------------
    # The X file holds the analog/digital inputs and answers ONLY to FC04, so
    # every scan we have run so far reported it as absent rather than as an
    # error. If the PLC program never MOVs a channel into a D register, this is
    # the only place its value exists.
    device_words: dict[str, dict[int, int]] = {}
    if SWEEP_DEVICES:
        print()
        print("-" * 60)
        print(" DEVICE FILE SWEEP  (D / X / Y / SR / T / C / E)")
        print("-" * 60)
        for name, (lo, hi, fc) in AS_DEVICES.items():
            top = min(hi, lo + 600)
            w = read_range(client, lo, top, UNIT, fc)
            device_words[name] = w
            nz = [(a, v) for a, v in sorted(w.items()) if v]
            print(f"  {name:22} FC{fc:02d}  {lo}..{top}  "
                  f"readable {len(w):>4}  non-zero {len(nz)}")
            for a, v in nz[:20]:
                sv = v - 65536 if v > 32767 else v
                print(f"        {a:>6} ({name.split()[0]}{a - lo}) = {v:>6}"
                      f"   signed {sv:>7}   /10 = {sv / 10:>9.2f}")
            if len(nz) > 20:
                print(f"        ... and {len(nz) - 20} more non-zero")
        print()
        print("  Anything listed under X was invisible to every earlier scan.")
        print()

    # ---- explicit value search -------------------------------------------
    if FIND:
        print("\n" + "-" * 60)
        print(" VALUE SEARCH  (addresses holding the values you asked for)")
        print("-" * 60)
        for want in FIND:
            tol = max(abs(want) * FIND_RTOL, 5e-4)
            hits = [(a, v, "float") for a, v in floats if abs(v - want) <= tol]
            # The same physical value may be stored as a scaled integer, so
            # look for it that way too rather than reporting "not found".
            for a, w in words.items():
                s = w - 65536 if w > 32767 else w
                for scale, tag in ((1.0, "int"), (10.0, "int/10"),
                                   (100.0, "int/100")):
                    v = s / scale
                    if s and abs(v - want) <= max(abs(want) * FIND_RTOL,
                                                  0.5 / scale):
                        hits.append((a, v, tag))
                        break
            # The device files are a separate address space, so a tag that
            # lives in X or SR would read as "not found" without this.
            for dname, dwords in device_words.items():
                base = AS_DEVICES[dname][0]
                for a, w in dwords.items():
                    sv = w - 65536 if w > 32767 else w
                    for scale, tag in ((1.0, "int"), (10.0, "int/10"),
                                       (100.0, "int/100")):
                        v = sv / scale
                        if sv and abs(v - want) <= max(abs(want) * FIND_RTOL,
                                                       0.5 / scale):
                            label = f"{dname.split()[0]}{a - base}"
                            hits.append((a, v, f"{tag}, {label}"))
                            break
            if hits:
                print(f"  {want:g}  ->  " + ", ".join(
                    f"addr {a} ({v:.5g} as {k})" for a, v, k in hits[:15]))
                if len(hits) > 15:
                    print(f"        ... and {len(hits) - 15} more")
            else:
                extra = " or any swept device file" if SWEEP_DEVICES else                     " (add --devices to search X / Y / SR too)"
                print(f"  {want:g}  ->  (not found in {START}..{END}"
                      f"{extra}, as float or scaled integer)")
        print("\n  A value that matches at exactly one address identifies that")
        print("  tag. If several addresses match, note a second value from")
        print("  eServer at the same moment and search for both.")

    # ---- target matches --------------------------------------------------
    print("\n" + "-" * 60)
    print(" TARGET MATCHES (best guesses)")
    print("-" * 60)
    # ok() screens out zeros. Without it the BPR line printed a page of
    # "addr N=0.0000" candidates and buried the real ones.
    flow = [(a, v) for a, v in floats if ok(v) and 0.001 <= v <= 1.0]
    bpr  = [(a, v) for a, v in floats
            if ok(v) and -3.5 <= v <= 5 and not (0.001 <= v <= 1.0)]

    def fmt(items):
        return ", ".join(f"addr {a}={v:.4f}" for a, v in items[:12]) or "(none)"

    print(f"  MFC flows (0.001-1) : {fmt(flow)}")
    print(f"  BPR       (-3.5..5) : {fmt(bpr)}")
    print()
    print("  No CO2 guess is offered any more: CO2 is known to live at D28091")
    print("  (see the probe below), and the 40-105 band that used to be")
    print("  suggested here only ever caught flows held in mL/min -- e.g. a")
    print("  0.070 L/min trickle also stored as the integer 70.")
    print()
    print("  These bands are heuristics, not identifications. The reliable")
    print("  method is to run this twice and compare: a value that CHANGES")
    print("  between two sweeps is a live measurement, one that does not is a")
    print("  setpoint. Better still, use the app's Address Explorer, tick")
    print("  \"Only what is moving\", and watch the rig cycle.")

    # ---- gas analyzer probe (high-D float registers) ---------------------
    # Exact addresses from the PLC tag table the technician supplied.
    print("\n" + "-" * 60)
    print(" GAS ANALYZER PROBE (high-D float registers)")
    print("-" * 60)
    # eServer's labels on these registers are shifted by one: the PLC mirrors
    # the analyzer in COMPONENT order and this VA-5000 has no CO channel.
    # Verified 2026-09-02 against a simultaneous instrument reading.
    gas = [
        (28089, "Component 1 = CH4 (vol%)  D28089  [eServer mislabels it CO]"),
        (28091, "Component 2 = CO2 (vol%)  D28091  <-- needed for Purity"),
        (28093, "Component 3 = O2  (vol%)  D28093  [eServer mislabels it CH4]"),
        (28095, "Component 4 = not equipped  D28095"),
    ]
    any_gas = False
    for addr, label in gas:
        try:
            rr = read_holding(client, addr, 2, UNIT)
        except Exception as exc:        # noqa: BLE001
            print(f"  addr {addr} {label} : ERROR ({exc})")
            continue
        if rr is None or rr.isError() or not hasattr(rr, "registers") \
                or len(rr.registers) < 2:
            print(f"  addr {addr} {label} : NOT READABLE")
        else:
            v = decode_float(rr.registers[0], rr.registers[1], WORD_ORDER)
            print(f"  addr {addr} {label} : {v:.4f}  (READABLE)")
            any_gas = True

    if any_gas:
        print()
        print("  => The PLC DOES mirror the analyzer -- no MOV needs adding.")
        print("     Read them in COMPONENT order, not by eServer's labels:")
        print("     1 = CH4, 2 = CO2, 3 = O2, 4 = not equipped.")

    if not any_gas:
        print("\n  => Expected: the PLC never holds these values.")
        print("     CO/CO2/CH4/O2 come from the HORIBA VA-5000 analyzer, which")
        print("     is its own Modbus/TCP server (192.168.1.100, slave 255).")
        print("     On 192.168.1.5 they ARE readable -- if they are not here,")
        print("     check you are talking to the PLC and not another device.")
        print("     Read the analyzer directly:  python tools/read_horiba.py")

    client.close()
    print("\n" + "=" * 60)
    print(" DONE.  Copy everything above and send it to me.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

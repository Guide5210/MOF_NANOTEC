"""
Find Modbus servers on the LAN and map their registers (discovery tool).

DO NOT POINT THIS AT THE VA-5000 ANYMORE
----------------------------------------
The VA-5000 manual (ch. 8) settled its addressing, and it states plainly:

    "Both of read and write address not specified in the address map are
     prohibited. Do not read and write the prohibited address. That may
     affect the performance of the analyzer."

This script sweeps blindly, so it would read exactly those prohibited
addresses. Use **tools/read_horiba.py** for the analyzer -- it touches only
documented registers. Keep this one for unknown devices (a new PLC, an
unidentified box found on the network).

WHY THIS EXISTS
---------------
CO / CO2 / CH4 / O2 do not come from PLC sensors -- they come from a HORIBA
VA-5000 multi-component gas analyzer. Per HORIBA's own specification the
VA-5000 speaks **Ethernet (Modbus/TCP) as a standard feature**, so we can read
the concentrations straight from the analyzer and skip the PLC entirely (the
high D-registers eServer shows are almost certainly eServer's own tags for a
second device, which is why they were never readable on the PLC).

Two things are unknown and this script finds both:
  1. the analyzer's IP address on the Wi-Fi router's LAN,
  2. its unit ID + which registers hold the four concentrations.

WHAT IT DOES
------------
  Step 1  list what is on the network (ARP) and scan every local /24 for
          hosts with a Modbus port open (502 by default).
  Step 2  for each host found, probe unit IDs 0..16 + 247 + 255 (the VA-5000
          is known to *require* the right unit ID even over TCP).
  Step 3  sweep holding registers (FC3) AND input registers (FC4), decode
          every candidate as float32 (both word orders) and as scaled 16-bit
          ints, and flag values that look like a gas concentration in vol%.

USAGE (on the eServer PC, in PowerShell / cmd):

    python tools/scan_horiba.py                    # auto: scan every local /24
    python tools/scan_horiba.py 192.168.1.0/24     # scan one subnet
    python tools/scan_horiba.py 192.168.1.55       # skip discovery, map this host
    python tools/scan_horiba.py 192.168.1.55 502   # ... with a specific port
    python tools/scan_horiba.py 192.168.1.55 502 1 # ... host port unit
    python tools/scan_horiba.py 192.168.1.55 502 1 0 2000   # ... start end

Requires pymodbus:   pip install pymodbus

Then copy EVERYTHING the script prints and send it back.
"""

from __future__ import annotations

import ipaddress
import socket
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

# Ports worth trying. 502 is Modbus TCP; the others are common serial-to-
# Ethernet gateway ports, in case the analyzer is bridged rather than native.
SCAN_PORTS = (502, 503, 4001, 10001)

# Unit / station IDs to probe. Plain TCP devices answer on 0 or 255; HORIBA is
# reported to honour a real unit ID, so sweep the low range too.
PROBE_UNITS = (1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 247, 255)

# Addresses used to decide "does this unit answer at all".
PROBE_ADDRS = (0, 1, 100, 1000, 30000, 40000)

CONNECT_TIMEOUT = 0.35     # seconds, for the port scan
MODBUS_TIMEOUT = 1.5       # seconds, for register reads
SWEEP_CHUNK = 100          # registers per request (conservative)


# ---------------------------------------------------------------------------
# Network discovery
# ---------------------------------------------------------------------------
def local_ipv4s() -> list[str]:
    """Every IPv4 address this PC holds (it may sit on LAN + Wi-Fi at once)."""
    found: set[str] = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))          # no traffic sent; just picks a NIC
        found.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            found.add(info[4][0])
    except Exception:
        pass
    return sorted(a for a in found if not a.startswith("127."))


def show_arp() -> None:
    """Print the ARP table -- a free list of everything on the network."""
    print("-" * 70)
    print(" DEVICES THIS PC HAS TALKED TO (arp -a)")
    print("-" * 70)
    try:
        out = subprocess.run(["arp", "-a"], capture_output=True, text=True,
                             timeout=15).stdout
        print(out.strip() or "  (empty)")
    except Exception as exc:                # noqa: BLE001
        print(f"  (could not run arp: {exc})")
    print()


def port_open(host: str, port: int, timeout: float = CONNECT_TIMEOUT) -> bool:
    """True if a TCP connect to host:port succeeds."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def scan_subnet(cidr: str, ports: tuple[int, ...]) -> list[tuple[str, int]]:
    """Return [(host, port), ...] for every open Modbus-ish port in ``cidr``."""
    net = ipaddress.ip_network(cidr, strict=False)
    hosts = [str(h) for h in net.hosts()]
    print(f"  scanning {cidr}  ({len(hosts)} addresses x {len(ports)} ports) ...")

    targets = [(h, p) for h in hosts for p in ports]
    hits: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=256) as pool:
        results = pool.map(lambda t: (t, port_open(*t)), targets)
        for (host, port), ok in results:
            if ok:
                print(f"    [OPEN] {host}:{port}")
                hits.append((host, port))
    if not hits:
        print("    (nothing open)")
    return hits


# ---------------------------------------------------------------------------
# Modbus helpers (tolerant of pymodbus 2.x / 3.x API differences)
# ---------------------------------------------------------------------------
def read_regs(client, fc: int, addr: int, count: int, unit: int):
    """Read holding (fc=3) or input (fc=4) registers; None on hard failure."""
    fn = client.read_holding_registers if fc == 3 else client.read_input_registers
    for kw in ("device_id", "slave", "unit"):
        try:
            return fn(addr, count=count, **{kw: unit})
        except TypeError:
            continue
        except Exception:
            return None
    try:
        return fn(addr, count=count)
    except Exception:
        return None


def regs_of(rr) -> list[int] | None:
    """Extract the register list from a pymodbus response, or None on error."""
    if rr is None or not hasattr(rr, "registers"):
        return None
    try:
        if rr.isError():
            return None
    except Exception:
        return None
    return list(rr.registers)


def decode_float(w0: int, w1: int, word_order: str) -> float:
    """Decode two 16-bit registers into a float32."""
    if word_order == "little":
        combined = ((w1 & 0xFFFF) << 16) | (w0 & 0xFFFF)
    else:
        combined = ((w0 & 0xFFFF) << 16) | (w1 & 0xFFFF)
    return struct.unpack(">f", struct.pack(">I", combined))[0]


def find_units(client, host: str, port: int) -> list[int]:
    """Probe unit IDs; return those that answer on any function code."""
    print(f"\n  probing unit IDs on {host}:{port} ...")
    answering: list[int] = []
    for unit in PROBE_UNITS:
        for fc in (3, 4):
            for addr in PROBE_ADDRS:
                if regs_of(read_regs(client, fc, addr, 2, unit)) is not None:
                    print(f"    [OK] unit {unit} answers  (FC{fc} @ addr {addr})")
                    answering.append(unit)
                    break
            else:
                continue
            break
    if not answering:
        print("    (no unit ID answered -- device may not be a Modbus server)")
    return answering


def sweep(client, unit: int, fc: int, start: int, end: int) -> dict[int, int]:
    """Read start..end inclusive in chunks; skip chunks the device refuses."""
    out: dict[int, int] = {}
    addr = start
    while addr <= end:
        count = min(SWEEP_CHUNK, end - addr + 1)
        words = regs_of(read_regs(client, fc, addr, count, unit))
        if words:
            for j, w in enumerate(words):
                out[addr + j] = w
        addr += count
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def plausible(v: float) -> bool:
    """Does this look like a gas concentration in vol%?

    The lower bound also throws away denormals -- decoding a zero word against
    its noisy neighbour yields ~1e-41, which would otherwise flood the list.
    """
    return (v == v) and 1e-4 <= abs(v) <= 105.0


def report_block(host: str, port: int, unit: int, fc: int,
                 start: int, end: int) -> None:
    """Sweep one (unit, function code) and print everything interesting."""
    from pymodbus.client import ModbusTcpClient

    client = ModbusTcpClient(host, port=port, timeout=MODBUS_TIMEOUT)
    if not client.connect():
        print(f"  [ERROR] lost connection to {host}:{port}")
        return

    label = "holding (FC3)" if fc == 3 else "input (FC4)"
    print("\n" + "=" * 70)
    print(f" {host}:{port}  unit {unit}  --  {label} registers {start}..{end}")
    print("=" * 70)

    words = sweep(client, unit, fc, start, end)
    client.close()
    total = end - start + 1
    print(f"  readable words: {len(words)} / {total}")
    if not words:
        print("  (nothing readable in this range)")
        return

    # --- raw non-zero words -------------------------------------------------
    nonzero = [(a, w) for a, w in sorted(words.items()) if w]
    print(f"\n  NON-ZERO RAW WORDS ({len(nonzero)}):")
    if not nonzero:
        print("    (all zero)")
    for a, w in nonzero[:150]:
        # a 16-bit int scaled by 10/100 is a very common concentration format
        print(f"    {a:>6} = {w:>6}   (/10 = {w/10:>8.2f}   /100 = {w/100:>8.2f})")
    if len(nonzero) > 150:
        print(f"    ... and {len(nonzero) - 150} more")

    # --- float32 decode, both word orders -----------------------------------
    for order in ("big", "little"):
        cands: list[tuple[int, float]] = []
        for a in range(start, end):
            if a in words and (a + 1) in words:
                v = decode_float(words[a], words[a + 1], order)
                if plausible(v):
                    cands.append((a, v))
        print(f"\n  FLOAT32 candidates, word order = {order}  ({len(cands)}):")
        if not cands:
            print("    (none)")
        for a, v in cands[:60]:
            hint = ""
            if 15.0 <= v <= 23.0:
                hint = "   <-- could be O2 (air ~20.9)"
            elif 40.0 <= v <= 105.0:
                hint = "   <-- could be CO2 (what Purity needs)"
            print(f"    {a:>6} = {v:>12.4f}{hint}")
        if len(cands) > 60:
            print(f"    ... and {len(cands) - 60} more")


# ---------------------------------------------------------------------------
def main() -> int:
    args = sys.argv[1:]
    print("=" * 70)
    print(" Modbus TCP discovery + blind register map")
    print("=" * 70)
    print(" WARNING: this sweeps undocumented addresses. Never aim it at the")
    print("          HORIBA VA-5000 -- its manual forbids that. For the")
    print("          analyzer use:  python tools/read_horiba.py")
    print("=" * 70)
    print()

    try:
        from pymodbus.client import ModbusTcpClient  # noqa: F401
    except ImportError:
        print("[ERROR] pymodbus is not installed on this PC.")
        print("        Run:  pip install pymodbus")
        return 1

    # ---- work out what to scan --------------------------------------------
    targets: list[tuple[str, int]] = []
    forced_unit: int | None = None
    start, end = 0, 2000

    if args and "/" not in args[0]:
        # explicit host [port [unit [start end]]]
        host = args[0]
        port = int(args[1]) if len(args) > 1 else 502
        targets = [(host, port)]
        if len(args) > 2:
            forced_unit = int(args[2])
        if len(args) > 4:
            start, end = int(args[3]), int(args[4])
        print(f"Target given explicitly: {host}:{port}")
        if not port_open(host, port, timeout=2.0):
            print(f"[WARN] {host}:{port} did not accept a TCP connection.")
            print("       Continuing anyway -- it may just be slow.")
    else:
        show_arp()
        subnets = [args[0]] if args else []
        if not subnets:
            ips = local_ipv4s()
            print(f"This PC's IPv4 addresses: {', '.join(ips) or '(none found)'}")
            subnets = [f"{ip}/24" for ip in ips]
        if not subnets:
            print("[ERROR] Could not work out a subnet to scan.")
            print("        Re-run with one, e.g.:  python tools/scan_horiba.py 192.168.1.0/24")
            return 1
        print("-" * 70)
        print(" PORT SCAN (looking for Modbus servers)")
        print("-" * 70)
        for cidr in subnets:
            targets.extend(scan_subnet(cidr, SCAN_PORTS))

    if not targets:
        print("\n[RESULT] No Modbus server found on this network.")
        print("  Next steps:")
        print("   - check the analyzer's own screen: Ethernet / network settings")
        print("     (it needs an IP on the same subnet as this PC)")
        print("   - confirm its LAN cable goes to the same Wi-Fi router")
        print("   - re-run with the analyzer's IP once you have read it off the screen:")
        print("       python tools/scan_horiba.py <ip>")
        return 0

    # ---- probe + map each target ------------------------------------------
    from pymodbus.client import ModbusTcpClient

    print("\n" + "-" * 70)
    print(f" FOUND {len(targets)} CANDIDATE(S): "
          + ", ".join(f"{h}:{p}" for h, p in targets))
    print("-" * 70)

    for host, port in targets:
        client = ModbusTcpClient(host, port=port, timeout=MODBUS_TIMEOUT)
        if not client.connect():
            print(f"\n[SKIP] cannot open {host}:{port}")
            continue
        units = [forced_unit] if forced_unit is not None \
            else find_units(client, host, port)
        client.close()

        for unit in units:
            for fc in (3, 4):
                report_block(host, port, unit, fc, start, end)

    print("\n" + "=" * 70)
    print(" DONE.  Copy everything above and send it back.")
    print("=" * 70)
    print("\nWhat to look for: four values that move together with the gas")
    print("readings on the VA-5000's own screen. Match them against the")
    print("screen and we know the CO / CO2 / CH4 / O2 addresses for good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
PLC / HMI connectivity probe.

Run this on the PC that is on the same WiFi/LAN as the Weintek HMI to find
out whether we can pull data directly over the network (Modbus TCP) instead
of going through eServer's output files.

    python tools/probe_plc.py            # uses default 192.168.1.5
    python tools/probe_plc.py 192.168.1.5

What it does
------------
1. Raw TCP test on the ports that matter (no extra install needed):
     502   = standard Modbus TCP            -> if OPEN, Python can read directly
     12348 = Weintek eServer protocol       -> proprietary, NOT usable by us
     102 / 44818 = other industrial ports   -> just for information
2. If port 502 is open AND `pymodbus` is installed, it tries to read the
   first 10 holding registers so we can see real values + confirm the map.

Reading nothing here is fine — it just tells us the HMI's Modbus TCP server
is off, and we should use the eServer-output path instead.
"""

from __future__ import annotations

import socket
import sys

HOST = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.5"

PORTS = {
    502:   "Modbus TCP (standard)  -> USABLE by Python if open",
    12348: "Weintek eServer proto  -> proprietary, not usable",
    102:   "S7 / ISO-on-TCP        -> info only",
    44818: "EtherNet/IP            -> info only",
}


def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    """Return True if a TCP connection to host:port succeeds."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def main() -> int:
    print(f"\nProbing HMI/PLC at {HOST} ...\n" + "-" * 56)
    modbus_open = False
    for port, desc in PORTS.items():
        ok = tcp_open(HOST, port)
        mark = "OPEN " if ok else "closed"
        print(f"  port {port:<6} [{mark}]  {desc}")
        if port == 502 and ok:
            modbus_open = True
    print("-" * 56)

    if not modbus_open:
        print(
            "\nResult: Modbus TCP (502) is CLOSED.\n"
            "  -> Direct IP pull is NOT available yet.\n"
            "  -> Either enable 'Modbus TCP/IP Server' on the HMI in\n"
            "     EasyBuilder Pro, or use the eServer-output path.\n"
        )
        return 0

    print("\nResult: Modbus TCP (502) is OPEN -> direct pull is possible!")
    try:
        from pymodbus.client import ModbusTcpClient
    except ImportError:
        print(
            "  Install pymodbus to read actual values:\n"
            "      pip install pymodbus\n"
            "  then re-run this script.\n"
        )
        return 0

    client = ModbusTcpClient(HOST, port=502, timeout=2)
    if not client.connect():
        print("  Could not open a Modbus session despite the port being open.")
        return 1
    try:
        # Try a few common starting addresses; the real map comes from the HMI.
        for start in (0, 1, 100, 4096):
            rr = client.read_holding_registers(start, count=10)
            if rr.isError():
                print(f"  read @ {start:<5}: error ({rr})")
            else:
                print(f"  holding regs @ {start:<5}: {rr.registers}")
    finally:
        client.close()
    print(
        "\nNext step: tell me which of these registers correspond to\n"
        "MFC-01 / MFC-07 / BPR-01 / CO2 vol%, and I'll wire the live reader.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

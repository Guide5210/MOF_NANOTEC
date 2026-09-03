"""
PLC register map — the single source of truth for every VPSA sensor.

Built from the tag table the PLC technician supplied. Edit the addresses HERE
(or, at run time, in the Live Monitor's "Edit all sensor addresses…" dialog) —
no other code needs to change. For the low D-register range the Delta AS series
exposes ``Modbus address == D-number`` (confirmed: D100 reads at Modbus 100).

Only four sensors feed the unchanged PSA pipeline; their ``column`` is set to
the canonical name the analyzer resolves (see ``constants.COLUMN_EXACT``). Every
other sensor is carried for the live display / machine-status view / CSV log.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Canonical analyzer column names — must stay in sync with constants.COLUMN_EXACT
# so the live rows resolve through build_column_map() unchanged.
COL_MFC01 = "MFC-01 (CO2 ) SLPM"
COL_MFC07 = "MFC-07 (AD-GAS ) SMLM"
COL_BPR   = "BPR -01 SLPM"
COL_CO2   = "CO2 ( vol%)"


@dataclass
class SensorDef:
    """One tag: where it lives, how to decode it, how to show it.

    A sensor with ``host`` set is read from its **own** Modbus endpoint rather
    than the PLC connection configured in the dialog — that is how the HORIBA
    VA-5000 gas analyzer is polled alongside the PLC.
    """
    tag: str                       # human label shown in the status view
    address: int                   # Modbus address (== Delta D-number, low range)
    kind: str = "float"            # 'float' (2 words) | 'uint16' | 'int16'
    unit: str = ""                 # engineering unit, for display
    group: str = "Other"           # grouping in the status view
    bus: str = "tcp"               # 'tcp' (polled) | 'rs485' (not over Modbus TCP)
    station: int = 1               # Modbus unit / station id
    poll: bool = True              # include in the live poll loop
    column: str | None = None      # row-dict key; defaults to ``tag``
    host: str | None = None        # own endpoint; None -> the PLC connection
    port: int | None = None        # own endpoint's port
    word_order: str | None = None  # own endpoint's float word order
    # Marks this sensor as a *stand-by source* for a canonical column it does
    # not own. Used only if the column's real owner produced nothing on a
    # poll — e.g. the PLC's copy of CO2 keeps Purity alive when the VA-5000's
    # own endpoint is unreachable.
    mirror_for: str | None = None

    @property
    def key(self) -> str:
        """The dict key this sensor's value is stored under in a live row."""
        return self.column or self.tag


# ---------------------------------------------------------------------------
# HORIBA VA-5000 gas analyzer — its own Modbus/TCP server on the same LAN.
#
# Every value below is from chapter 8 "External Input/Output" of the VA-5000
# instruction manual, not from guesswork:
#   * TCP port 502, protocol Modbus/TCP, function code 03
#   * slave address **fixed to 255** (the analyzer really does enforce it)
#   * data length 2 is stored high word first (Fig. 94) -> word order "big",
#     which is the opposite of the Delta PLC's "little"
#   * "Instantaneous value - Component N concentration value" is a float at
#     HORIBA_CONC_BASE + 2*(N-1), i.e. 10506 / 10508 / 10510 / 10512
#
# Which component is which gas is a per-machine configuration, so it stays
# editable in the Live Monitor's setup dialog. On this rig it was verified
# against the analyzer's own screen: 1 = CH4, 2 = CO2, 3 = O2, 4 = unequipped.
# ---------------------------------------------------------------------------
HORIBA_HOST = "192.168.1.100"
HORIBA_PORT = 502
HORIBA_UNIT = 255
HORIBA_WORD_ORDER = "big"
HORIBA_CONC_BASE = 10506
HORIBA_GROUP = "Gas Analyzer (VA-5000)"
# The same three gases as the PLC mirrors them, kept apart in the status
# view so a disagreement between instrument and PLC is visible at a glance.
GAS_MIRROR_GROUP = "Gas Analyzer (PLC mirror)"


def horiba_address(component: int) -> int:
    """Modbus address of the concentration float for Component 1..4."""
    return HORIBA_CONC_BASE + 2 * (int(component) - 1)


def _horiba(tag: str, component: int, column: str | None = None) -> "SensorDef":
    """A VA-5000 component channel, pointed at the analyzer's own endpoint."""
    return SensorDef(
        tag, horiba_address(component), "float", "vol%", HORIBA_GROUP,
        station=HORIBA_UNIT, column=column,
        host=HORIBA_HOST, port=HORIBA_PORT, word_order=HORIBA_WORD_ORDER,
    )


# ---------------------------------------------------------------------------
# Default map (from the technician's table). Addresses are editable at runtime.
# ---------------------------------------------------------------------------
DEFAULT_SENSORS: list[SensorDef] = [
    # --- Corrected 2026-09-02: address IS the D-number -------------------
    # The whole "compacted window" theory (BPR-01 at 108, half the tags
    # unpublished) came from scanning 192.168.1.20. That is NOT the PLC. The
    # technician's own ISPSoft and Modbus Poll screens both show the PLC is an
    # **AS300N at 192.168.1.5**, and a sweep of it returns 4001/4001 words
    # readable with Modbus address == D-number, exactly as Delta's spec sheet
    # says. Five tags the .20 scan called "not published at all" answer fine
    # there: D110 = 0.068 (the very number eServer shows for MFC-06), D192,
    # D202, D316, plus the MFC flows on D102/D104.
    #
    # So these are the addresses from eServer's own tag table, restored.
    # Anything reading 0.0 below was simply idle when the sweep ran and still
    # needs one confirmation pass with the plant RUNNING.
    SensorDef("MFC-01 (CO2)",     100, "float", "SLPM", "Gas Flow (MFC)", column=COL_MFC01),
    SensorDef("MFC-02 (N2)",      102, "float", "SLPM", "Gas Flow (MFC)"),
    SensorDef("MFC-03 (CH4)",     104, "float", "SLPM", "Gas Flow (MFC)"),
    SensorDef("MFC-04 (HE)",      106, "float", "SLPM", "Gas Flow (MFC)"),
    SensorDef("MFC-05 (MIX)",     108, "float", "SLPM", "Gas Flow (MFC)"),
    SensorDef("MFC-06 (HG GAS)",  110, "float", "NLPM", "Gas Flow (MFC)"),
    # 20 NLPM Brooks SLA5850; the pipeline uses its reading raw, so the number
    # is litres/min. Only the old "SMLM" label was ever wrong. This is the tag
    # that unblocks Recovery and Productivity — and it was never missing, we
    # were asking the wrong device for it.
    SensorDef("MF-07 (AD GAS)",   112, "float", "NLPM", "Gas Flow (MFC)",
              column=COL_MFC07),
    # --- Back-pressure regulator (this is what segments the cycles) -------
    SensorDef("BPR-01",           114, "float", "SLPM", "Back Pressure",  column=COL_BPR),
    # --- Pressure transmitters (PT) -------------------------------------
    SensorDef("PT1-1",            116, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT1-2",            118, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT2-1",            160, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT2-2",            162, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT-10",            164, "float", "mbar", "Pressure (PT)"),
    SensorDef("PT-HG",            190, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT3-1",            192, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT3-2",            194, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT4-1",            196, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT4-2",            198, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT-5",             200, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT-6",             202, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT-8",             204, "float", "bar",  "Pressure (PT)"),
    SensorDef("PT-9",             206, "float", "bar",  "Pressure (PT)"),
    SensorDef("MF-08 (WASTE GAS)",316, "float", "SLPM", "Gas Flow (MFC)"),

    # --- Gas values mirrored INTO the PLC by the program -----------------
    # Confirmed readable on .5 on 2026-09-02, and they were never missing —
    # the .20 device just did not carry them, which is why "ask the technician
    # to add a MOV mirror" was chased for weeks. The mirror already existed.
    #
    # They arrive in the analyzer's own COMPONENT order (1, 2, 3), so eServer's
    # CO/CO2/CH4/O2 labels on these registers are shifted by one exactly as
    # they are on the analyzer itself. Matched against a simultaneous VA-5000
    # reading: D28089 -0.0025 vs CH4 -0.0022, D28091 1.2426 vs CO2 1.2299,
    # D28093 18.1433 vs O2 18.1752. D28095 is Component 4, not equipped.
    #
    # Carried as a cross-check only. The pipeline's CO2 still comes from the
    # VA-5000 direct (below), which is the instrument of record.
    SensorDef("CH4 (vol%) via PLC", 28089, "float", "vol%", GAS_MIRROR_GROUP),
    SensorDef("CO2 (vol%) via PLC", 28091, "float", "vol%", GAS_MIRROR_GROUP,
              mirror_for=COL_CO2),
    SensorDef("O2 (vol%) via PLC",  28093, "float", "vol%", GAS_MIRROR_GROUP),

    # --- Gas analyzer: read from the HORIBA VA-5000 itself, not the PLC ---
    # Component -> gas confirmed on site 2026-08-13 by matching the Modbus
    # values against the analyzer's own MEASUREMENT screen (CH4 -0.0 /
    # CO2 14.5 / O2 9.76). Component 4 is not equipped on this unit: it reads
    # 0.0 with range 0.00 and 0 decimal digits, so it is not polled.
    # NOTE: eServer's old CO/CO2/CH4/O2 labels were shifted by one — this
    # machine has no CO channel at all.
    _horiba("CH4 (vol%)", 1),
    _horiba("CO2 (vol%)", 2, column=COL_CO2),
    _horiba("O2 (vol%)",  3),
    # --- Bed temperatures (TI) — 16-bit words, high D -------------------
    SensorDef("TI1-1",          28122, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI1-2",          28124, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI1-3",          28126, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI1-4",          28128, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI2-1",          28142, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI2-2",          28144, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI2-3",          28146, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI2-4",          28148, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI3-1",          28242, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI3-2",          28244, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI3-3",          28246, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI3-4",          28248, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI4-1",          28250, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI4-2",          28252, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI4-3",          28254, "uint16", "°C", "Temperature (TI)"),
    SensorDef("TI4-4",          28256, "uint16", "°C", "Temperature (TI)"),
    # --- RS-485 controllers — not reachable over Modbus TCP (info only) --
    SensorDef("TIC-PH",  4097, "uint16", "", "Controller (RS-485)", bus="rs485", station=1, poll=False),
    SensorDef("TIC-B1",  4097, "uint16", "", "Controller (RS-485)", bus="rs485", station=2, poll=False),
    SensorDef("TIC-B2",  4097, "uint16", "", "Controller (RS-485)", bus="rs485", station=3, poll=False),
    SensorDef("TIC-B3",  4097, "uint16", "", "Controller (RS-485)", bus="rs485", station=4, poll=False),
    SensorDef("TIC-B4",  4097, "uint16", "", "Controller (RS-485)", bus="rs485", station=5, poll=False),
    SensorDef("TIC-HG",  4097, "uint16", "", "Controller (RS-485)", bus="rs485", station=6, poll=False),
    SensorDef("PIC-PPE", 4097, "int16",  "", "Controller (RS-485)", bus="rs485", station=7, poll=False),
]

# The four canonical column constants, for the quick-edit spinboxes.
CANONICAL_COLUMNS = (COL_MFC01, COL_MFC07, COL_BPR, COL_CO2)


def default_sensors() -> list[SensorDef]:
    """A fresh copy of the default map (so callers can mutate freely)."""
    return [replace(s) for s in DEFAULT_SENSORS]


def with_overrides(sensors: list[SensorDef],
                   addr_by_column: dict[str, int]) -> list[SensorDef]:
    """Return a copy of ``sensors`` with canonical addresses overridden.

    ``addr_by_column`` maps a canonical column constant (e.g. ``COL_CO2``) to a
    new Modbus address — used by the Live Monitor's quick-edit spinboxes.
    """
    out: list[SensorDef] = []
    for s in sensors:
        if s.column in addr_by_column:
            out.append(replace(s, address=int(addr_by_column[s.column])))
        else:
            out.append(replace(s))
    return out


def with_analyzer_endpoint(sensors: list[SensorDef], host: str,
                           port: int, unit: int) -> list[SensorDef]:
    """Point every VA-5000 channel at the analyzer endpoint from the dialog."""
    out: list[SensorDef] = []
    for s in sensors:
        if s.group == HORIBA_GROUP:
            out.append(replace(s, host=(host or "").strip() or HORIBA_HOST,
                               port=int(port), station=int(unit)))
        else:
            out.append(replace(s))
    return out


def pollable(sensors: list[SensorDef]) -> list[SensorDef]:
    """Sensors that should be read in the live poll loop (TCP + poll flag)."""
    return [s for s in sensors if s.bus == "tcp" and s.poll]

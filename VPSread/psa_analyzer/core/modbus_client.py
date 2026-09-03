"""
Thin Modbus-TCP helper for reading the Delta AS PLC through the HMI gateway.

eServer's Address tab tells us the data layout:
  * values are 32-bit IEEE floats (Format = Floating, Read Count = 2 words),
  * station / unit id = 1 (the ``1@`` prefix in ``1@D100``),
  * registers are Delta D-registers read over Modbus TCP.

The exact Delta-D -> Modbus-address offset and the float word order are
confirmed empirically with the Modbus Scanner, then fed to the live worker.

No Qt here — the scanner dialog and worker wrap this.
"""

from __future__ import annotations

from pymodbus.client import ModbusTcpClient

DEFAULT_PORT = 502
DEFAULT_UNIT = 1
DEFAULT_WORD_ORDER = "big"   # Delta AS is usually big-endian word order

# Delta's "Modbus TCP Specifications" sheet: FC03/FC04 accept 1-100 words per
# request (Modbus itself allows 125). Asking for more is outside spec even
# where the PLC tolerates it, and a refused block costs a whole retry round.
MAX_READ_WORDS = 100

# The same sheet caps the server at 8 simultaneous TCP connections, shared with
# eServer and the HMI. Every reader here therefore opens ONE connection and
# reuses it — never one per sensor.
MAX_CONNECTIONS = 8

# Where each AS-series device file starts, as a raw protocol address (the
# published 4xxxx/3xxxx number minus its base). D is what the plant tags use;
# the rest are listed so a hunt can reach them at all.
#   name: (first address, last address, function code)
AS_DEVICE_RANGES: dict[str, tuple[int, int, int]] = {
    "D  (data registers)":      (0,     29999, 3),
    "X  (inputs, word)":        (32768, 32831, 4),   # FC04 only!
    "Y  (outputs, word)":       (40960, 41023, 3),
    "SR (special registers)":   (49152, 51199, 3),
    "T  (timers, word)":        (57344, 57855, 3),
    "C  (counters, word)":      (61440, 61951, 3),
    "E  (index registers)":     (65024, 65038, 3),
}


class ModbusReader:
    """Connect to a Modbus-TCP device and read holding registers / floats."""

    def __init__(self, host: str, port: int = DEFAULT_PORT,
                 device_id: int = DEFAULT_UNIT,
                 word_order: str = DEFAULT_WORD_ORDER,
                 timeout: float = 2.0) -> None:
        self.host = host
        self.port = int(port)
        self.device_id = int(device_id)
        self.word_order = word_order if word_order in ("big", "little") else "big"
        self._client = ModbusTcpClient(host, port=self.port, timeout=timeout)

    # -- connection ---------------------------------------------------------
    def connect(self) -> bool:
        return bool(self._client.connect())

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "ModbusReader":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- raw words ----------------------------------------------------------
    def _request(self, addr: int, count: int, fc: int):
        """Issue one read with the given Modbus function code.

        ==  ==========================  ======================================
        fc  reads                       AS-series devices
        ==  ==========================  ======================================
        01  coils (0x)                  Y, M, SM, S, T, C bits
        02  discrete inputs (1x)        X, Y, M, SM, S, T, C bits
        03  holding registers (4x)      Y, SR, D, T, C, HC, E  <- the plant tags
        04  input registers (3x)        X only
        ==  ==========================  ======================================

        The X device file is published at 3xxxx and is reachable **only**
        through FC04 — FC03 reports it as absent rather than as an error, so
        anything living in X stays invisible until the function code is right.
        """
        fc = int(fc)
        if fc == 1:
            return self._client.read_coils(
                addr, count=count, device_id=self.device_id)
        if fc == 2:
            return self._client.read_discrete_inputs(
                addr, count=count, device_id=self.device_id)
        if fc == 4:
            return self._client.read_input_registers(
                addr, count=count, device_id=self.device_id)
        return self._client.read_holding_registers(
            addr, count=count, device_id=self.device_id)

    @staticmethod
    def _values_of(rr, count: int, fc: int) -> list[int] | None:
        """Pull the payload out of a response — words or bits alike.

        The function code decides which attribute is the payload, not
        ``hasattr``: a pymodbus coil response carries **both** ``bits`` and an
        empty ``registers``, so trusting whichever exists reads every bit
        function as "connected, nothing there".

        Bit responses are padded up to a byte boundary, so a request for 3
        coils comes back carrying 8; the extra ones are not data.
        """
        if rr is None or rr.isError():
            return None
        if int(fc) in (1, 2):
            bits = getattr(rr, "bits", None)
            return None if bits is None else [1 if b else 0 for b in bits[:count]]
        regs = getattr(rr, "registers", None)
        return None if regs is None else list(regs)

    def read_block(self, start: int, count: int, fc: int = 3) -> list[int]:
        """Read ``count`` registers from ``start`` (raw 16-bit words).

        Split into ``MAX_READ_WORDS`` chunks to stay inside Delta's documented
        per-request limit. Returns [] on error.
        """
        out: list[int] = []
        addr = int(start)
        remaining = int(count)
        while remaining > 0:
            chunk = min(MAX_READ_WORDS, remaining)
            vals = self._values_of(self._request(addr, chunk, fc), chunk, fc)
            if vals is None:
                return out  # partial (or empty) — caller treats short as error
            out.extend(vals)
            addr += chunk
            remaining -= chunk
        return out

    def read_range(self, start: int, end: int, fc: int = 3) -> dict[int, int]:
        """Read registers from ``start`` to ``end`` inclusive.

        A device refuses a block *as a whole* if it does not expose even one
        address inside it, so a failed chunk is **subdivided down to single
        registers** rather than abandoned. Without that, one gap would hide up
        to ``MAX_READ_WORDS`` perfectly readable neighbours — which is exactly
        how a sweep across a partly-published map loses the tag it is hunting.

        Returns ``{address: word}`` holding every word the device would give.
        """
        out: dict[int, int] = {}

        def walk(addr: int, count: int) -> None:
            try:
                vals = self._values_of(self._request(addr, count, fc), count, fc)
            except Exception:
                vals = None
            if vals is not None:
                for j, w in enumerate(vals):
                    out[addr + j] = w
                return
            if count == 1:
                return                      # genuinely not exposed
            half = count // 2
            walk(addr, half)
            walk(addr + half, count - half)

        addr, end = int(start), int(end)
        while addr <= end:
            chunk = min(MAX_READ_WORDS, end - addr + 1)
            walk(addr, chunk)
            addr += chunk
        return out

    # -- typed values -------------------------------------------------------
    def floats_from_words(self, words: list[int]) -> list[float]:
        """Decode a word list into float32 values (one per consecutive pair)."""
        vals: list[float] = []
        for i in range(0, len(words) - 1, 2):
            try:
                v = ModbusTcpClient.convert_from_registers(
                    words[i:i + 2],
                    ModbusTcpClient.DATATYPE.FLOAT32,
                    word_order=self.word_order,
                )
                vals.append(float(v))
            except Exception:
                vals.append(float("nan"))
        return vals

    def read_float(self, address: int) -> float | None:
        """Read a single float32 (2 registers) at ``address``; None on error."""
        words = self.read_block(address, 2)
        if len(words) < 2:
            return None
        out = self.floats_from_words(words)
        return out[0] if out else None

    def read_uint16(self, address: int) -> int | None:
        """Read one register as an unsigned 16-bit int; None on error."""
        words = self.read_block(address, 1)
        return int(words[0]) if words else None

    def read_int16(self, address: int) -> int | None:
        """Read one register as a signed 16-bit int; None on error."""
        words = self.read_block(address, 1)
        if not words:
            return None
        v = int(words[0])
        return v - 65536 if v >= 32768 else v

    def read_value(self, address: int, kind: str = "float") -> float | int | None:
        """Read one value decoded as ``kind`` ('float' | 'uint16' | 'int16')."""
        if kind == "uint16":
            return self.read_uint16(address)
        if kind == "int16":
            return self.read_int16(address)
        return self.read_float(address)

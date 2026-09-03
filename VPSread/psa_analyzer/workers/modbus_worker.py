"""
Background worker that polls the plant over Modbus TCP and streams rows to the UI.

Two config shapes are supported:

* ``sensors`` : a list of :class:`SensorDef` (the flexible map from plc_map).
  At connect the worker probes each one once to learn which addresses are
  actually exposed, emits a one-off ``sensors`` report (for the status view),
  then polls only the reachable ones every ``interval_ms``.
* ``registers`` : the legacy ``{column: address}`` dict (all read as float32).

**Two devices, one poll loop.** Most sensors live on the Delta PLC, but the
gas concentrations come from the HORIBA VA-5000 analyzer, which is its own
Modbus/TCP server on the same LAN with a different unit id (255) and the
opposite float word order. A :class:`SensorDef` carrying ``host`` is therefore
read through its own connection; the rest use the PLC connection from the
dialog. One device being unreachable never stops the other.

Each poll emits one batch — a ``list[dict]`` whose keys are the sensor's row
key (canonical analyzer name for the four pipeline sensors, tag for the rest),
plus ``DATE / TIME`` — so it flows straight into the existing LiveBuffer.
"""

from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import QObject, QThread, Signal

from psa_analyzer.core.modbus_client import ModbusReader
from psa_analyzer.core.plc_map import COL_MFC07, SensorDef, pollable

_TS_FMT = "%m/%d/%Y %I:%M:%S %p"


def _sensors_from_config(cfg: dict) -> list[SensorDef]:
    """Normalise either config shape into a list of SensorDef."""
    if cfg.get("sensors"):
        return list(cfg["sensors"])
    # legacy {column: address} dict -> float sensors
    out: list[SensorDef] = []
    for col, addr in (cfg.get("registers") or {}).items():
        out.append(SensorDef(tag=col, address=int(addr), kind="float", column=col))
    return out


def _endpoint(s: SensorDef, cfg: dict) -> tuple[str, int, int, str]:
    """(host, port, unit, word_order) this sensor should be read through."""
    if s.host:                                  # its own device (the VA-5000)
        return (s.host,
                int(s.port or 502),
                int(s.station),
                s.word_order or "big")
    return (cfg["host"],                        # the PLC connection
            int(cfg.get("port", 502)),
            int(cfg.get("unit", 1)),
            cfg.get("word_order", "little"))


class ModbusPollWorker(QObject):
    """Polls a set of Modbus registers every ``interval_ms``."""

    batch    = Signal(list)   # list[dict] of sample rows
    sensors  = Signal(list)   # list[dict] one-off reachability/value report
    status   = Signal(str)
    failed   = Signal(str)
    finished = Signal()

    def __init__(self, config: dict, interval_ms: int = 1000) -> None:
        super().__init__()
        self._cfg = config
        self._interval = max(200, int(interval_ms))
        self._running = False
        self._readers: dict[tuple, ModbusReader | None] = {}

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _report_row(s: SensorDef, reachable: bool, value,
                    host: str = "") -> dict:
        return {
            "tag": s.tag, "key": s.key, "address": s.address, "kind": s.kind,
            "unit": s.unit, "group": s.group, "bus": s.bus, "host": host,
            "reachable": reachable,
            "value": "" if value is None else value,
        }

    def _reader_for(self, ep: tuple) -> "ModbusReader | None":
        """Connected reader for an endpoint, opened once and cached.

        A device that refuses the connection is cached as ``None`` so the poll
        loop keeps serving the other device instead of retrying every tick.
        """
        if ep in self._readers:
            return self._readers[ep]
        host, port, unit, order = ep
        reader = ModbusReader(host, port=port, device_id=unit,
                              word_order=order, timeout=0.7)
        if not reader.connect():
            self.status.emit(f"Cannot connect to {host}:{port} — "
                             f"its sensors will show as not exposed.")
            reader.close()
            self._readers[ep] = None
        else:
            self._readers[ep] = reader
        return self._readers[ep]

    def _close_readers(self) -> None:
        for reader in self._readers.values():
            if reader is not None:
                reader.close()
        self._readers.clear()

    def run(self) -> None:
        try:
            self._running = True
            cfg = self._cfg
            all_sensors = _sensors_from_config(cfg)
            to_poll = pollable(all_sensors)

            # Every sensor's endpoint, resolved once up front.
            eps = {id(s): _endpoint(s, cfg) for s in all_sensors}

            # The PLC must be reachable; the analyzer is allowed to be absent.
            plc_ep = (cfg["host"], int(cfg.get("port", 502)),
                      int(cfg.get("unit", 1)), cfg.get("word_order", "little"))
            if self._reader_for(plc_ep) is None:
                self.failed.emit(
                    f"Cannot connect to {cfg['host']}:{cfg.get('port', 502)}")
                return

            # -- probe each pollable sensor once to learn what's reachable ---
            self.status.emit(f"Probing {len(to_poll)} sensors...")
            reachable: list[SensorDef] = []
            report: list[dict] = []
            for s in all_sensors:
                # only annotate sensors read from their own device (the
                # analyzer); PLC rows stay uncluttered
                host = s.host or ""
                if s.bus != "tcp" or not s.poll:
                    report.append(self._report_row(s, False, None, host))
                    continue
                if not self._running:
                    break
                reader = self._reader_for(eps[id(s)])
                val = None if reader is None \
                    else reader.read_value(s.address, s.kind)
                ok = val is not None
                if ok:
                    reachable.append(s)
                report.append(self._report_row(s, ok, val, host))
            self.sensors.emit(report)

            n_ok = len(reachable)
            n_bad = len(to_poll) - n_ok
            hosts = sorted({eps[id(s)][0] for s in reachable})
            self.status.emit(
                f"Connected to {', '.join(hosts) or cfg['host']} - "
                f"polling {n_ok} sensors"
                + (f" ({n_bad} not exposed)" if n_bad else "") + ".")

            # Canonical analyzer columns must appear in every row (even blank)
            # so build_column_map resolves them and the other cards keep working
            # when, say, the gas analyzer is switched off.
            canon_cols = [s.column for s in all_sensors if s.column]

            # AD-GAS stand-in. The PLC does not publish MFC-07, but it is a
            # flow *controller* holding a setpoint the operator knows, so the
            # dialog can supply that number. Only ever fills a blank — a real
            # reading always wins.
            mirrors = [s for s in reachable if s.mirror_for]
            announced: set[str] = set()

            ad_gas_fixed = float(cfg.get("ad_gas_fixed") or 0.0)
            ad_gas_col = next((s.column for s in all_sensors
                               if s.column == COL_MFC07), None)
            if ad_gas_fixed > 0 and ad_gas_col:
                self.status.emit(
                    f"AD-GAS is a fixed {ad_gas_fixed:g} NLPM stand-in, "
                    f"not a measurement.")

            # -- poll loop ---------------------------------------------------
            while self._running:
                row = {"DATE / TIME": datetime.now().strftime(_TS_FMT)}
                got_any = False
                for s in reachable:
                    reader = self._readers.get(eps[id(s)])
                    if reader is None:
                        continue
                    val = reader.read_value(s.address, s.kind)
                    if val is not None:
                        row[s.key] = val
                        got_any = True
                for c in canon_cols:
                    row.setdefault(c, "")
                # Stand-by sources: only ever fill a column its real owner
                # left blank, so a live instrument always wins.
                for m in mirrors:
                    if row.get(m.mirror_for) == "" and row.get(m.key) != "":
                        row[m.mirror_for] = row[m.key]
                        if m.tag not in announced:
                            announced.add(m.tag)
                            self.status.emit(
                                f"{m.mirror_for} is coming from {m.tag} — the "
                                f"instrument itself is not answering.")
                if ad_gas_fixed > 0 and ad_gas_col and row.get(ad_gas_col) == "":
                    row[ad_gas_col] = ad_gas_fixed
                if got_any:
                    self.batch.emit([row])
                QThread.msleep(self._interval)

            self._close_readers()
            self.finished.emit()
        except Exception as exc:        # noqa: BLE001
            import traceback
            self._close_readers()
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def stop(self) -> None:
        self._running = False


def start_modbus(config: dict, interval_ms: int,
                 on_batch, on_status, on_failed, on_finished,
                 on_sensors=None
                 ) -> tuple[QThread, ModbusPollWorker]:
    """Spin up a ModbusPollWorker on its own QThread and wire signals."""
    thread = QThread()
    worker = ModbusPollWorker(config, interval_ms)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.batch.connect(on_batch)
    worker.status.connect(on_status)
    worker.failed.connect(on_failed)
    worker.finished.connect(on_finished)
    if on_sensors is not None:
        worker.sensors.connect(on_sensors)

    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker

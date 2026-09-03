"""
Background worker that tails eServer's CSV and streams new rows to the UI.

Mirrors the signal style of :mod:`analysis_worker` so the main window wires it
the same way. It polls the file (or the newest ``*.csv`` in a folder) every
``interval_ms`` and emits each fresh batch of rows; the UI feeds them into a
:class:`~psa_analyzer.core.live_buffer.LiveBuffer`.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from psa_analyzer.core.live_csv import CsvTailReader, newest_csv


class LiveCsvWorker(QObject):
    """Polls a growing CSV on a worker thread and emits new rows."""

    batch    = Signal(list)   # list[dict] — newly appended rows
    status   = Signal(str)    # human-readable status line
    failed   = Signal(str)
    finished = Signal()

    def __init__(self, source: str | Path, interval_ms: int = 1000) -> None:
        super().__init__()
        self._source = Path(source)
        self._interval = max(100, int(interval_ms))
        self._running = False

    def _resolve(self) -> Path | None:
        """Return the file to read — the source itself, or newest in a folder."""
        if self._source.is_dir():
            return newest_csv(self._source)
        return self._source if self._source.exists() else None

    def run(self) -> None:
        try:
            self._running = True
            reader: CsvTailReader | None = None
            current: Path | None = None
            self.status.emit(f"Watching {self._source} ...")
            while self._running:
                target = self._resolve()
                if target is None:
                    self.status.emit(f"Waiting for CSV at {self._source} ...")
                elif target != current:
                    # First file, or eServer rolled to a new file.
                    current = target
                    reader = CsvTailReader(target)
                    self.status.emit(f"Reading {target.name} ...")
                if reader is not None:
                    try:
                        rows = reader.read_new()
                        if rows:
                            self.batch.emit(rows)
                    except Exception as exc:        # noqa: BLE001
                        self.status.emit(f"Read error: {exc}")
                QThread.msleep(self._interval)
            self.finished.emit()
        except Exception as exc:                    # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")

    def stop(self) -> None:
        """Ask the polling loop to exit (safe to call from the main thread)."""
        self._running = False


def start_live(source: str | Path, interval_ms: int,
               on_batch, on_status, on_failed, on_finished
               ) -> tuple[QThread, LiveCsvWorker]:
    """Spin up a LiveCsvWorker on its own QThread and wire signals."""
    thread = QThread()
    worker = LiveCsvWorker(source, interval_ms)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.batch.connect(on_batch)
    worker.status.connect(on_status)
    worker.failed.connect(on_failed)
    worker.finished.connect(on_finished)

    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker

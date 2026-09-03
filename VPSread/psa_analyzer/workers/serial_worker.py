"""
Stub for future live-data acquisition over a serial port.

The interface deliberately mirrors :mod:`analysis_worker` so the main
window can connect to either source through the same signal contract:

    sample(dict)   # one timestamped reading, e.g.
                   # {'t': 12.34, 'co2_pct': 5.6, 'flow_Lmin': 0.8, ...}
    failed(str)
    finished()

When you're ready to wire up an Arduino / microcontroller:

    1. ``pip install pyserial``
    2. Replace ``_read_loop`` with real serial reads.
    3. The UI side already knows how to consume ``sample`` signals —
       see :class:`ui.plot_canvas.LivePlotCanvas` (TODO).

Keeping this file in place now means the eventual transition from
post-processing to live monitoring is purely additive.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class SerialWorker(QObject):
    """Placeholder serial worker — does nothing until wired to pyserial."""

    sample   = Signal(dict)
    failed   = Signal(str)
    finished = Signal()

    def __init__(self, port: str = "", baud: int = 115200) -> None:
        super().__init__()
        self._port = port
        self._baud = baud
        self._running = False

    def start(self) -> None:
        """Begin polling. Currently a no-op."""
        self._running = True
        # TODO: open pyserial port, then call _read_loop()
        self.failed.emit("Live serial mode not yet implemented.")

    def stop(self) -> None:
        """Stop polling and close the port."""
        self._running = False
        self.finished.emit()

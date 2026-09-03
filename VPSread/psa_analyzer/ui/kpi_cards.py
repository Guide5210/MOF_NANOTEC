"""
KPI card strip — the row of summary metrics across the top of the workspace.

Each card has three pieces:
    * a small uppercase label (sans-serif)
    * a large numeric value (monospaced — aligns decimals)
    * a small unit suffix (sans-serif, muted)
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)


class KPICard(QFrame):
    """A single KPI card. Use :meth:`set_value` to update at runtime."""

    def __init__(self, label: str, unit: str = "",
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("KPICard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        v = QVBoxLayout(self)
        v.setContentsMargins(16, 12, 16, 12)
        v.setSpacing(2)

        self._label = QLabel(label)
        self._label.setObjectName("KPILabel")
        v.addWidget(self._label)

        row = QHBoxLayout()
        row.setSpacing(6)
        row.setAlignment(Qt.AlignBottom)

        self._value = QLabel("—")
        self._value.setObjectName("KPIValue")
        row.addWidget(self._value, alignment=Qt.AlignBottom)

        self._unit = QLabel(unit)
        self._unit.setObjectName("KPIUnit")
        row.addWidget(self._unit, alignment=Qt.AlignBottom)
        row.addStretch(1)
        v.addLayout(row)

    def set_value(self, value: float | str,
                  fmt: str = "{:.3f}") -> None:
        """Update the displayed value, formatting numerics nicely."""
        if isinstance(value, str):
            self._value.setText(value)
        elif value is None or (isinstance(value, float) and value != value):
            self._value.setText("—")  # NaN
        else:
            self._value.setText(fmt.format(value))


class KPIStrip(QWidget):
    """
    Horizontal strip of KPI cards, fed by an :class:`AnalysisResult`.

    Cards: Purity, Recovery (raw, uncapped), Productivity, Elapsed time, Cycles.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(12)

        self.card_purity       = KPICard("Final Purity", "%")
        self.card_recovery     = KPICard("Final Recovery", "%")
        self.card_productivity = KPICard("Final Productivity", "t CO₂/m³·day")
        self.card_elapsed      = KPICard("Elapsed Time", "hours")
        self.card_cycles       = KPICard("Total Cycles", "")

        for c in (self.card_purity, self.card_recovery, self.card_productivity,
                  self.card_elapsed, self.card_cycles):
            h.addWidget(c)

    def update_from_result(self, result) -> None:
        """Populate every card from an :class:`AnalysisResult`."""
        self.card_purity.set_value(result.final_purity,       fmt="{:.2f}")
        # Recovery is reported RAW (uncapped) to match the researcher's
        # reference. The 100% cap is display-only on the plot's "Corrected"
        # view, not on the headline Final value.
        self.card_recovery.set_value(result.final_recovery,   fmt="{:.2f}")
        self.card_productivity.set_value(result.final_productivity, fmt="{:.4f}")
        self.card_elapsed.set_value(result.total_elapsed_hours,     fmt="{:.2f}")
        self.card_cycles.set_value(len(result.cycles),              fmt="{:.0f}")

    def clear(self) -> None:
        """Reset every card to its empty state."""
        for c in (self.card_purity, self.card_recovery, self.card_productivity,
                  self.card_elapsed, self.card_cycles):
            c.set_value("—")

"""
Data-table viewer / editor for the Live Monitor ("แก้ไขข้อมูล").

Shows a snapshot of everything recorded so far in a grid. The user can refresh,
delete selected rows (e.g. startup noise), and save the result to CSV. Deletes
are applied to the live buffer so the next analysis / save reflects the edit.

Non-modal so live polling keeps running while the table is open. The grid is a
snapshot: hit Refresh to pull in rows captured after it was opened.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QDialog,
)

# Cap the number of rows rendered so a multi-hour run stays responsive.
_MAX_DISPLAY = 5000


class DataTableDialog(QDialog):
    """View, prune, and export the recorded live data."""

    def __init__(self, buffer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Recorded data — view / edit")
        self.resize(960, 640)
        self.setModal(False)
        self._buffer = buffer
        self._index_map: list[int] = []   # display row -> buffer row index

        root = QVBoxLayout(self)

        self._info = QLabel("")
        self._info.setStyleSheet("color:#6b7280;")
        root.addWidget(self._info)

        self._table = QTableWidget(0, 0)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        root.addWidget(self._table, stretch=1)

        btns = QHBoxLayout()
        b_refresh = QPushButton("↻ Refresh")
        b_refresh.clicked.connect(self.refresh)
        b_delete = QPushButton("🗑 Delete selected rows")
        b_delete.clicked.connect(self._delete_selected)
        b_save = QPushButton("💾 Save to CSV…")
        b_save.clicked.connect(self._save_csv)
        btns.addWidget(b_refresh)
        btns.addWidget(b_delete)
        btns.addStretch(1)
        btns.addWidget(b_save)
        root.addLayout(btns)

        self.refresh()

    # -- table build --------------------------------------------------------
    def refresh(self) -> None:
        rows = self._buffer.rows_snapshot()
        total = len(rows)
        # show the most recent _MAX_DISPLAY rows
        start = max(0, total - _MAX_DISPLAY)
        view = rows[start:]
        self._index_map = list(range(start, total))

        # union of keys, preserving first-seen order
        cols: list[str] = []
        seen = set()
        for r in view:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    cols.append(k)

        self._table.clear()
        self._table.setColumnCount(len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setRowCount(len(view))
        for i, r in enumerate(view):
            for j, c in enumerate(cols):
                v = r.get(c, "")
                self._table.setItem(i, j, QTableWidgetItem("" if v is None else str(v)))

        shown = (f"showing last {len(view):,} of {total:,}"
                 if total > len(view) else f"{total:,} rows")
        self._info.setText(f"Recorded data — {shown}.  "
                           f"Select rows and Delete to prune, then Save to CSV.")

    # -- actions ------------------------------------------------------------
    def _delete_selected(self) -> None:
        sel = sorted({idx.row() for idx in self._table.selectionModel().selectedRows()})
        if not sel:
            QMessageBox.information(self, "Delete rows",
                                   "Select one or more rows first.")
            return
        buf_indices = [self._index_map[r] for r in sel if r < len(self._index_map)]
        n = self._buffer.drop_indices(buf_indices)
        self.refresh()
        QMessageBox.information(self, "Delete rows", f"Removed {n} row(s).")

    def _save_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save recorded data", "vpsa_recorded.csv",
            "CSV files (*.csv)")
        if not path:
            return
        try:
            df = pd.DataFrame(self._buffer.rows_snapshot())
            df.to_csv(path, index=False)
        except Exception as exc:        # noqa: BLE001
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved",
                               f"Saved {len(df):,} rows to\n{Path(path).name}")

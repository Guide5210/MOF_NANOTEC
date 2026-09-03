"""
Background workers for the load + analysis pipeline.

Two stages so the UI can interpose a 'crop' step between them:
  1. LoadWorker      — read the workbook, build the column map.
  2. AnalysisWorker  — run the PSA pipeline on a (possibly cropped) DataFrame.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QObject, QThread, Signal

from psa_analyzer.core import (
    AnalysisParams,
    AnalysisResult,
    load_psa_workbook,
    run_analysis,
)


# ---------------------------------------------------------------------------
# Stage 1 — load workbook
# ---------------------------------------------------------------------------
class LoadWorker(QObject):
    """Reads the Excel workbook on a background thread."""

    progress = Signal(int, str)
    finished = Signal(object, object, object)   # df, colmap, path
    failed   = Signal(str)

    def __init__(self, xlsx_path: Path) -> None:
        super().__init__()
        self._xlsx_path = Path(xlsx_path)

    def run(self) -> None:
        try:
            self.progress.emit(0, f"Opening {self._xlsx_path.name}...")
            df, colmap = load_psa_workbook(
                self._xlsx_path,
                progress_cb=lambda p, m: self.progress.emit(p, m),
            )
            self.finished.emit(df, colmap, self._xlsx_path)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


def start_load(xlsx_path: Path,
               on_progress, on_finished, on_failed) -> tuple[QThread, LoadWorker]:
    thread = QThread()
    worker = LoadWorker(xlsx_path)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)

    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker


# ---------------------------------------------------------------------------
# Stage 2 — run analysis on a pre-loaded (and possibly cropped) DataFrame
# ---------------------------------------------------------------------------
class AnalysisWorker(QObject):
    """Runs :func:`run_analysis` on a worker thread."""

    progress = Signal(int, str)
    finished = Signal(object)   # AnalysisResult
    failed   = Signal(str)

    def __init__(self, df: pd.DataFrame, colmap: dict[str, str],
                 params: AnalysisParams) -> None:
        super().__init__()
        self._df = df
        self._colmap = colmap
        self._params = params

    def run(self) -> None:
        try:
            result = run_analysis(
                self._df, self._colmap, self._params,
                progress_cb=lambda p, m: self.progress.emit(p, m),
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


def start_analysis(df: pd.DataFrame,
                   colmap: dict[str, str],
                   params: AnalysisParams,
                   on_progress, on_finished, on_failed
                   ) -> tuple[QThread, AnalysisWorker]:
    """Spin up an AnalysisWorker on its own QThread and wire signals."""
    thread = QThread()
    worker = AnalysisWorker(df, colmap, params)
    worker.moveToThread(thread)

    thread.started.connect(worker.run)
    worker.progress.connect(on_progress)
    worker.finished.connect(on_finished)
    worker.failed.connect(on_failed)

    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    thread.start()
    return thread, worker

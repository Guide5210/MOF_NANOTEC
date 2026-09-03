"""Core PSA analysis package (Qt-free, importable from CLI or tests)."""

from .analyzer import AnalysisResult, run_analysis
from .constants import AnalysisParams, PALETTE, PlotColors
from .data_loader import load_psa_workbook, ColumnResolutionError
from .live_buffer import LiveBuffer
from .live_csv import CsvTailReader, newest_csv
from .exporters import (
    export_cycles_csv,
    export_dashboard,
    export_pdf_report,
    export_single_metric,
    export_xlsx_report,
)

__all__ = [
    "AnalysisParams",
    "AnalysisResult",
    "ColumnResolutionError",
    "CsvTailReader",
    "LiveBuffer",
    "PALETTE",
    "PlotColors",
    "export_cycles_csv",
    "export_dashboard",
    "export_pdf_report",
    "export_single_metric",
    "export_xlsx_report",
    "load_psa_workbook",
    "newest_csv",
    "run_analysis",
]

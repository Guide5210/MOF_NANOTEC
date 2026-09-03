"""
PSA Analyzer — application entry point.

Run with:
    python main.py
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from psa_analyzer.ui import MainWindow


def main() -> int:
    """Construct QApplication, show the main window, run the event loop."""
    app = QApplication(sys.argv)
    app.setApplicationName("PSA Analyzer")
    app.setOrganizationName("KMITL")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

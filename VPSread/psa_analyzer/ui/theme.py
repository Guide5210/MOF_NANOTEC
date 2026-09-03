"""
Light/Dark QSS themes.

Design notes
------------
* Sans-serif (Segoe UI / Inter / Roboto) is used for all chrome —
  labels, buttons, menus, headers.
* A monospaced font (JetBrains Mono / Consolas) is reserved for KPI
  values, table cells, and tooltips so decimal points line up
  vertically — critical for scanning a column of measurements.
* High data-to-ink: borders are subtle, surfaces are flat, no gradients.
"""

from __future__ import annotations

from enum import Enum


SANS_STACK = '"Segoe UI", "Inter", "Roboto", "Helvetica Neue", Arial, sans-serif'
MONO_STACK = '"JetBrains Mono", "Cascadia Mono", "Consolas", "Menlo", monospace'


class Theme(Enum):
    LIGHT = "light"
    DARK  = "dark"


# ---------------------------------------------------------------------------
# Light theme — clean, report-friendly
# ---------------------------------------------------------------------------
LIGHT_QSS = f"""
* {{
    font-family: {SANS_STACK};
    font-size: 10pt;
    color: #1f2937;
}}

QMainWindow, QWidget {{
    background-color: #ffffff;
}}

QFrame#Sidebar {{
    background-color: #f7f7f9;
    border-right: 1px solid #e5e7eb;
}}

QLabel#SectionHeader {{
    font-size: 9pt;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding-top: 12px;
    padding-bottom: 4px;
}}

QLabel#KPILabel {{
    color: #6b7280;
    font-size: 9pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

QLabel#KPIValue {{
    font-family: {MONO_STACK};
    font-size: 22pt;
    font-weight: 600;
    color: #111827;
}}

QLabel#KPIUnit {{
    color: #6b7280;
    font-size: 9pt;
}}

QFrame#KPICard {{
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}}

QPushButton {{
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 7px 14px;
    color: #1f2937;
}}
QPushButton:hover  {{ background-color: #f3f4f6; }}
QPushButton:pressed {{ background-color: #e5e7eb; }}
QPushButton:disabled {{ color: #9ca3af; border-color: #e5e7eb; }}

QPushButton#PrimaryButton {{
    background-color: #1f77b4;
    color: #ffffff;
    border: none;
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover   {{ background-color: #1a6699; }}
QPushButton#PrimaryButton:pressed {{ background-color: #15557f; }}
QPushButton#PrimaryButton:disabled {{ background-color: #9ca3af; }}

QLineEdit, QDoubleSpinBox, QSpinBox {{
    background-color: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    padding: 5px 8px;
    font-family: {MONO_STACK};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: #1f77b4;
}}

/* Styling a spin box's frame without also styling its sub-controls leaves Qt
   drawing the native step buttons at the wrong size and position, which is why
   they looked ragged. Position and paint them explicitly. */
QDoubleSpinBox, QSpinBox {{
    padding-right: 22px;
}}
QDoubleSpinBox::up-button, QSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    background-color: #f3f4f6;
    border-left: 1px solid #d1d5db;
    border-top-right-radius: 5px;
}}
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    background-color: #f3f4f6;
    border-left: 1px solid #d1d5db;
    border-top: 1px solid #d1d5db;
    border-bottom-right-radius: 5px;
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background-color: #e5e7eb;
}}
QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed,
QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{
    background-color: #d1d5db;
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #4b5563;
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #4b5563;
}}
QDoubleSpinBox::up-arrow:disabled, QSpinBox::up-arrow:disabled {{
    border-bottom-color: #c3c8cf;
}}
QDoubleSpinBox::down-arrow:disabled, QSpinBox::down-arrow:disabled {{
    border-top-color: #c3c8cf;
}}

/* A splitter the user can actually find and grab. */
QSplitter::handle:horizontal {{
    background-color: #e5e7eb;
    width: 6px;
    margin: 0 2px;
    border-radius: 3px;
}}
QSplitter::handle:horizontal:hover {{ background-color: #1f77b4; }}

QTableView {{
    background-color: #ffffff;
    alternate-background-color: #f9fafb;
    gridline-color: #e5e7eb;
    font-family: {MONO_STACK};
    selection-background-color: #dbeafe;
    selection-color: #1f2937;
}}
QHeaderView::section {{
    background-color: #f3f4f6;
    color: #6b7280;
    border: none;
    border-bottom: 1px solid #e5e7eb;
    padding: 6px 8px;
    font-weight: 600;
    font-size: 9pt;
}}

QProgressBar {{
    background-color: #f3f4f6;
    border: none;
    border-radius: 3px;
    height: 6px;
    text-align: center;
}}
QProgressBar::chunk {{
    background-color: #1f77b4;
    border-radius: 3px;
}}

QMenuBar {{
    background-color: #f7f7f9;
    border-bottom: 1px solid #e5e7eb;
    color: #1f2937;
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{ background-color: #e5e7eb; }}
QMenuBar::item:pressed {{ background-color: #1f77b4; color: #ffffff; }}
QMenu {{
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 4px;
    color: #1f2937;
}}
QMenu::item {{ padding: 6px 26px 6px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: #1f77b4; color: #ffffff; }}
QMenu::item:disabled {{ color: #9ca3af; }}
QMenu::separator {{ height: 1px; background: #e5e7eb; margin: 4px 8px; }}

QStatusBar {{
    background-color: #f7f7f9;
    border-top: 1px solid #e5e7eb;
    color: #6b7280;
}}
"""


# ---------------------------------------------------------------------------
# Dark theme — for long monitoring sessions
# ---------------------------------------------------------------------------
DARK_QSS = f"""
* {{
    font-family: {SANS_STACK};
    font-size: 10pt;
    color: #e5e7eb;
}}

QMainWindow, QWidget {{
    background-color: #1e1e1e;
}}

QFrame#Sidebar {{
    background-color: #252526;
    border-right: 1px solid #3a3a3a;
}}

QLabel#SectionHeader {{
    font-size: 9pt;
    font-weight: 600;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding-top: 12px;
    padding-bottom: 4px;
}}

QLabel#KPILabel {{
    color: #9ca3af;
    font-size: 9pt;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

QLabel#KPIValue {{
    font-family: {MONO_STACK};
    font-size: 22pt;
    font-weight: 600;
    color: #f9fafb;
}}

QLabel#KPIUnit {{
    color: #9ca3af;
    font-size: 9pt;
}}

QFrame#KPICard {{
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
}}

QPushButton {{
    background-color: #2d2d2d;
    border: 1px solid #4a4a4a;
    border-radius: 6px;
    padding: 7px 14px;
    color: #e5e7eb;
}}
QPushButton:hover  {{ background-color: #383838; }}
QPushButton:pressed {{ background-color: #424242; }}
QPushButton:disabled {{ color: #6b7280; border-color: #3a3a3a; }}

QPushButton#PrimaryButton {{
    background-color: #1f77b4;
    color: #ffffff;
    border: none;
    font-weight: 600;
}}
QPushButton#PrimaryButton:hover   {{ background-color: #2589cc; }}
QPushButton#PrimaryButton:pressed {{ background-color: #1a6699; }}
QPushButton#PrimaryButton:disabled {{ background-color: #4a4a4a; }}

QLineEdit, QDoubleSpinBox, QSpinBox {{
    background-color: #2a2a2a;
    border: 1px solid #4a4a4a;
    border-radius: 6px;
    padding: 5px 8px;
    color: #e5e7eb;
    font-family: {MONO_STACK};
}}
QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
    border-color: #1f77b4;
}}

/* Same sub-control treatment as the light theme — see the note there. */
QDoubleSpinBox, QSpinBox {{
    padding-right: 22px;
}}
QDoubleSpinBox::up-button, QSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    background-color: #3a3a3a;
    border-left: 1px solid #4a4a4a;
    border-top-right-radius: 5px;
}}
QDoubleSpinBox::down-button, QSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    background-color: #3a3a3a;
    border-left: 1px solid #4a4a4a;
    border-top: 1px solid #4a4a4a;
    border-bottom-right-radius: 5px;
}}
QDoubleSpinBox::up-button:hover, QSpinBox::up-button:hover,
QDoubleSpinBox::down-button:hover, QSpinBox::down-button:hover {{
    background-color: #4a4a4a;
}}
QDoubleSpinBox::up-button:pressed, QSpinBox::up-button:pressed,
QDoubleSpinBox::down-button:pressed, QSpinBox::down-button:pressed {{
    background-color: #565656;
}}
QDoubleSpinBox::up-arrow, QSpinBox::up-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid #d1d5db;
}}
QDoubleSpinBox::down-arrow, QSpinBox::down-arrow {{
    width: 0; height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #d1d5db;
}}
QDoubleSpinBox::up-arrow:disabled, QSpinBox::up-arrow:disabled {{
    border-bottom-color: #5a5a5a;
}}
QDoubleSpinBox::down-arrow:disabled, QSpinBox::down-arrow:disabled {{
    border-top-color: #5a5a5a;
}}

QSplitter::handle:horizontal {{
    background-color: #3a3a3a;
    width: 6px;
    margin: 0 2px;
    border-radius: 3px;
}}
QSplitter::handle:horizontal:hover {{ background-color: #1f77b4; }}

QTableView {{
    background-color: #252526;
    alternate-background-color: #2a2a2a;
    gridline-color: #3a3a3a;
    color: #e5e7eb;
    font-family: {MONO_STACK};
    selection-background-color: #1f4368;
}}
QHeaderView::section {{
    background-color: #2d2d2d;
    color: #9ca3af;
    border: none;
    border-bottom: 1px solid #3a3a3a;
    padding: 6px 8px;
    font-weight: 600;
}}

QProgressBar {{
    background-color: #2a2a2a;
    border: none;
    border-radius: 3px;
    height: 6px;
}}
QProgressBar::chunk {{
    background-color: #1f77b4;
    border-radius: 3px;
}}

QMenuBar {{
    background-color: #252526;
    border-bottom: 1px solid #3a3a3a;
    color: #e5e7eb;
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{ background-color: #3a3a3a; }}
QMenuBar::item:pressed {{ background-color: #1f77b4; color: #ffffff; }}
QMenu {{
    background-color: #2d2d30;
    border: 1px solid #3a3a3a;
    padding: 4px;
    color: #e5e7eb;
}}
QMenu::item {{ padding: 6px 26px 6px 20px; border-radius: 4px; }}
QMenu::item:selected {{ background-color: #1f77b4; color: #ffffff; }}
QMenu::item:disabled {{ color: #6b7280; }}
QMenu::separator {{ height: 1px; background: #3a3a3a; margin: 4px 8px; }}

QStatusBar {{
    background-color: #252526;
    border-top: 1px solid #3a3a3a;
    color: #9ca3af;
}}
"""


def get_stylesheet(theme: Theme) -> str:
    """Return the QSS string for the requested theme."""
    return LIGHT_QSS if theme is Theme.LIGHT else DARK_QSS


def pyqtgraph_palette(theme: Theme) -> dict:
    """
    Colours to pass to pyqtgraph at runtime so the plot matches the theme.

    Returned dict keys: ``background``, ``foreground``.
    """
    if theme is Theme.LIGHT:
        return {"background": "#ffffff", "foreground": "#1f2937"}
    return {"background": "#1e1e1e", "foreground": "#e5e7eb"}

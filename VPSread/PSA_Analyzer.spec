# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for PSA Analyzer — Demo 1.

Build:
    pyinstaller PSA_Analyzer.spec --noconfirm --clean

Produces a single-file Windows executable at:
    dist/PSA_Analyzer_Demo1.exe

Notes
-----
* matplotlib's bundled DejaVu fonts (mpl-data/fonts/ttf) must be shipped —
  report_pdf._register_fonts() loads them at runtime so the PDF renders the
  ₂ / ρ / Δ / Σ glyphs instead of empty boxes.
* pyqtgraph imports some submodules dynamically, so collect them explicitly.
* The live-monitor modules (core/live_csv, core/live_buffer, ui/live_window,
  workers/live_worker) are pure-Python inside the package, so PyInstaller
  picks them up automatically by following imports from main.py.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# --- Data files -----------------------------------------------------------
datas = []
datas += collect_data_files("matplotlib")    # mpl-data incl. fonts/ttf (DejaVu)
datas += collect_data_files("reportlab")      # reportlab's built-in resources

# --- Hidden imports -------------------------------------------------------
hiddenimports = []
hiddenimports += collect_submodules("pyqtgraph")
hiddenimports += collect_submodules("pymodbus")   # Modbus-TCP live mode
hiddenimports += ["openpyxl", "openpyxl.cell._writer"]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Trim a few large, unused stacks to keep the exe smaller.
        "tkinter", "PyQt5", "PyQt6", "PySide2",
        "scipy", "IPython", "notebook", "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="PSA_Analyzer_Demo1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # GUI app — no console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="app.ico",       # add an .ico here if you want a custom icon
)

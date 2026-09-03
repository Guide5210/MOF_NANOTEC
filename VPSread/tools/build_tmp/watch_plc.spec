# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the PLC change watcher.

Build (from the project root):
    pyinstaller tools/build_tmp/watch_plc.spec --noconfirm --clean \
        --distpath tools/dist --workpath tools/build_tmp/work

Produces a standalone console exe at:
    tools/dist/watch_plc.exe
"""

import os

from PyInstaller.utils.hooks import collect_submodules

SRC = os.path.join(SPECPATH, "..", "watch_plc.py")

hiddenimports = collect_submodules("pymodbus")

a = Analysis(
    [SRC],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "PySide6", "PyQt5", "PyQt6", "matplotlib",
              "pandas", "numpy", "scipy", "reportlab", "openpyxl"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="watch_plc",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

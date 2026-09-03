# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for the HORIBA VA-5000 reader.

Build (from the project root):
    pyinstaller tools/build_tmp/read_horiba.spec --noconfirm --clean \
        --distpath tools/dist --workpath tools/build_tmp/work

Produces a standalone console exe at:
    tools/dist/read_horiba.exe

No Python install needed on the target PC — same idea as scan_plc.exe.
"""

import os

from PyInstaller.utils.hooks import collect_submodules

SRC = os.path.join(SPECPATH, "..", "read_horiba.py")

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
    name="read_horiba",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,           # console tool — output is the whole point
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

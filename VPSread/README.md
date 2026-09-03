# PSA Analyzer

Desktop application for Pressure Swing Adsorption (PSA) experiment analysis.
Built for the CALF-20 / MOF research workflow at KMITL.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

## Architecture

```
psa_analyzer/
├── main.py                Entry point — boots QApplication
├── core/                  Pure Python, no Qt — testable in isolation
│   ├── constants.py       Defaults, column keywords, palette
│   ├── data_loader.py     Multi-sheet Excel reader + column auto-detect
│   ├── analyzer.py        4-step PSA pipeline → AnalysisResult
│   └── exporters.py       CSV + matplotlib SVG/PNG export
├── workers/               QThread workers (Qt-aware)
│   ├── analysis_worker.py Background runner for the analysis
│   └── serial_worker.py   Stub for future live-data mode
└── ui/                    QWidgets
    ├── theme.py           Light/Dark QSS + pyqtgraph palette
    ├── sidebar.py         File picker + parameter inputs + actions
    ├── kpi_cards.py       Summary metric cards
    ├── plot_canvas.py     Three-panel interactive pyqtgraph view
    └── main_window.py     Composition root
```

### Design rules

1. **`core/` never imports Qt.** You can run the same pipeline from a CLI,
   Jupyter notebook, or pytest. The UI is one of several possible front ends.
2. **All experimental parameters flow through `AnalysisParams`.**
   Nothing is hardcoded; the sidebar fields populate this dataclass and pass
   it by value to the analyser.
3. **Heavy work runs on a `QThread`.** `workers/analysis_worker.py`
   emits `progress`, `finished`, `failed` signals — the main window connects
   them to sidebar/KPI/plot widgets.
4. **Two plot backends, by design.** `pyqtgraph` powers the interactive
   in-app canvas (drag/zoom/hover); `matplotlib` produces publication-grade
   SVG/PNG for export. They never need to share code.
5. **Live mode is plumbed but inert.** `workers/serial_worker.py` mirrors
   the analysis worker's signal shape. Wiring up `pyserial` later is purely
   additive — no UI changes needed beyond a new source toggle.

## Extending

* **New adsorbent / rig geometry** — change values in the sidebar; no code edits.
* **New KPI** — add a column in `analyzer._aggregate_cycles`, a card in
  `kpi_cards.KPIStrip`, and (optionally) a panel in `plot_canvas`.
* **New file format** — write a new loader in `core/` that returns the same
  `(df, colmap)` tuple shape and feed it to `run_analysis`.
* **Live serial mode** — flesh out `workers/serial_worker.py` and wire its
  `sample` signal to `PlotCanvas.append_live_sample`.

## Calculation reference

See the spec in the original brief. The four steps live in
`core/analyzer.py` as `_compute_row_level`, `_aggregate_steps`,
`_aggregate_cycles`, `_final_metrics`. Each is a pure function; the public
`run_analysis` is the only thing the UI needs to know about.

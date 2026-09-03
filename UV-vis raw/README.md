# CALF-20 UV-Vis Linker Analyzer — Developer README

A single-file HTML tool for quantifying residual MOF linkers (1,2,4-triazole and
oxalate) in CALF-20 synthesis wash water by UV-Vis spectroscopy. It builds
Beer–Lambert calibrations from standard spectra and deconvolutes two-component
mixtures by classical least squares (CLS), with an optional light-scattering
correction for turbid samples.

This document is written for another developer (human or AI) who needs to
maintain or extend the tool. It covers the science the code encodes, the file
format it parses, the architecture, every UI mode, the known limitations, and
the design decisions that must not be silently reverted.

---

## 1. What problem this solves

**Chemistry.** CALF-20 is Zn₂(triazolate)₂(oxalate). After synthesis the solid is
washed; the wash water carries residual dissolved linkers — 1,2,4-triazole (TZ,
MW 69.07) and oxalic acid (Ox, weighed as the dihydrate, MW 126.07). The research
question is *how much TZ and Ox remain in each wash*, which indicates whether the
wash is clean.

**Why this is hard by UV-Vis.** Both linkers are weak far-UV chromophores whose
absorption bands overlap near 200–210 nm. A single-wavelength Beer–Lambert
reading gives one equation with two unknowns (c_TZ, c_Ox) → unsolvable. The tool
therefore uses the **whole spectrum**: each wavelength is one equation, so
200–260 nm gives ~60 equations for 2 unknowns (over-determined) and CLS solves it.
This only works because the two ε(λ) shapes differ enough (TZ dies out by ~210 nm,
Ox extends to ~260 nm; condition number of the [ε_TZ, ε_Ox] matrix ≈ 15 — well
conditioned).

**Instrument.** Thermo Scientific Evolution 350, double-beam, run from INSIGHT
software, measured in **scan mode** (full spectrum), NOT fix mode. "Fixed
wavelength" in this project refers only to how calibration *reads* the data
afterward — never to the acquisition mode. Deconvolution needs the full spectrum.

---

## 2. The science the code encodes (do not break these)

These are hard-won empirical findings from real data. Several were mistakes that
cost days; the code encodes the corrections. If you change them, you will
silently reintroduce the bugs.

### 2.1 Beer–Lambert is only linear for A ≈ 0–1
Verified on real data by a **ratio test**: for two standards at c₁ and c₂, the
ratio A(c₂)/A(c₁) must equal c₂/c₁ at every wavelength. On real Ox data the ratio
holds up to A ≈ 1 and collapses above it (detector saturation + stray light).
→ The tool's default absorbance ceiling `CEIL = 1.0`. Wavelengths above the
ceiling are **dropped** from ε and from the CLS fit. The green shaded band in the
charts marks the valid A ≤ CEIL region.

### 2.2 Read calibration at a FIXED wavelength, never at "peak A"
The single most important lesson. The TZ peak wanders between 190 and 192 nm
across concentrations (the true peak is ~194 nm, below the instrument's
guaranteed-flat range). If you build a calibration curve from each scan's *peak*
absorbance, you are reading different wavelengths for different concentrations →
R² collapses (observed: TZ peak-A gave R²=0.95, fixed-200 nm gave R²=0.9997).
→ The **Calibration mode** reads A at one fixed λ (a draggable cursor). Per
species: TZ default 200 nm, Ox default 204 nm. The λ-cursor is the tool's
signature feature precisely because it embodies this lesson: drag it to 194 nm and
R² looks perfect but only 3 points survive the ceiling (a trap); drag to 200 nm
and all 5 points are valid with R²=0.9997.

### 2.3 Stray light rises steeply below 200 nm
Fitted from real data: ≈0.17 % at 205 nm, ≈0.80 % at 200 nm. This is why the fit
window starts at 200 nm by default and why the TZ 194 nm peak is unreliable
(Evolution 350 baseline flatness is only guaranteed 200–800 nm). The stray-light
model lives in `stray(w)` and `trueA(Am,w)` and is used only to *estimate the
dilution* needed for a saturated scan (the "dilute ~N×" hint on sample rows). It
is NOT used to correct absorbances used in the fit.

### 2.4 ε values recovered from real data (sanity anchors)
- TZ @ 200 nm: ε ≈ 508 M⁻¹cm⁻¹, R² = 0.9997, intercept ≈ +0.013
- Ox @ 204 nm: ε ≈ 2122 M⁻¹cm⁻¹, R² = 0.9979, intercept ≈ +0.007
- Ox absorbs ~4× stronger than TZ, so Ox standards use a lower concentration
  range (0.1–0.4 mM) than TZ (0.1–1.0 mM). Any regression that returns ε wildly
  off these is a red flag (usually a wavelength/units bug).

### 2.5 Two-point calibrations are meaningless
Two points always give R²=1. The code flags `n < 3` and shows "R² n/a" rather
than a fake perfect fit. Do not remove this guard.

### 2.6 CLS must skip wavelengths where ε is undefined
`buildEps()` returns `null` (not 0) for wavelengths with fewer than 2 usable
standards. `runCLS()` skips any wavelength where either ε is null. An earlier bug
silently used ε=0 there, biasing the fit. Keep the null-skip.

### 2.7 Light scattering (turbid / unfiltered samples)
Real wash water and any Zn-containing sample scatter light (suspended MOF/ZnCO₃
particles). Two regimes were observed:
- **Rayleigh-like** (fine particles): A ∝ λ⁻⁴, smooth. The CLS model can absorb
  this by adding a `k_s·λ⁻⁴` term — this is the "correct light scattering"
  checkbox, on by default. Validated on synthetic turbid mixtures: without it Ox
  errored +100 %+; with it, error < 1 %. On clear samples the term fits ≈0 and
  does no harm.
- **Mie (coarse particles, e.g. undissolved ZnCO₃):** baseline is *flat* across
  wavelengths, NOT λ⁻⁴. The λ⁻⁴ term does **not** fix this. For those samples the
  physical fix is to centrifuge or filter before measuring. The code cannot
  correct Mie scattering; document this to the user, don't pretend to solve it.

### 2.8 pH / speciation caveat (NOT yet in the code — important)
Oxalic acid speciates with pH (pKa₁=1.25, pKa₂=4.27): H₂Ox / HOx⁻ / Ox²⁻ absorb
differently. Calibration standards made in DI sit at low pH (mostly H₂Ox/HOx⁻);
real wash water measured at pH 5–6 is mostly Ox²⁻. Using DI-pH ε on a pH-5.6
sample biases Ox. TZ (pKa 2.2) is pH-insensitive over 3–6, so TZ stays quant.
The tool does **not** model this yet — it is the top candidate for a future
"matrix-matched calibration" or multi-pH ε feature. Until then, Ox from wash
water should be reported as semi-quantitative.

---

## 3. Input file format

The tool parses **INSIGHT "Spectrum, Comma Separated Values (*.csv)"** exports.
Format is stacked blocks, one per sample:

```
1,4TZ 1                         <- sample title (may contain commas!)
7/9/2026 3:33:26 PM             <- datetime line
Wavelength (nm),Absorbance      <- header row
400.00,0.000359                 <- data rows (descending or ascending nm)
399.00,0.000719
...                             <- then the next block's title, etc.
```

Parsing gotchas the code already handles (regression-tested — keep the tests):
- **Sample names with internal commas** (`1,4TZ 1`). A naïve numeric check breaks
  because `parseFloat("4TZ 1") === 4`. The title-detection uses a strict-numeric
  regex `/^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?$/` on every field; a line is data only
  if *all* fields are strictly numeric.
- **"Weshing"** (a real typo in the data): wash detection regex is broad
  (`/we?sh|wash|rins|filtrat|supernat|eluat|permeat|mother|\brx\b/`).
- **Report CSV** (a different export: `#,Sample ID,User Name,Date` with no
  spectra) is detected and rejected with a message telling the user to re-export
  as Spectrum CSV.
- Also parses wide CSV (nm,Abs,nm,Abs…), 2-column CSV, and Evolution XML
  (SpreadsheetML) as fallbacks.

Auto-detection from the sample name:
- `gRole(name)` → 'tz' | 'ox' | 'wash' | '' (matches "1,4TZ", "14TZ", "TZ",
  "Ox", "oxalate", wash synonyms).
- `gConc(name)` → pulls "0.25 mM" style numbers (also comma-decimal "0,5 mM").
  Names without an explicit mM value leave conc null; the user types it in.

---

## 4. Architecture (single file, no build step)

Plain HTML + CSS + vanilla JS, one external dep: **Chart.js 4.4.1** from cdnjs.
Fonts: Instrument Sans + JetBrains Mono from Google Fonts. No bundler, no
localStorage (artifacts forbid it), no network calls except the CDN. Open the
file in a browser and it runs.

### 4.1 Global state
```
S        array of sample objects (see shape below)
CAL      {tz, ox}  — legacy per-λ calibration cache (used by drawCalib indirectly)
mode     'spectra' | 'calib' | 'deconv'
filter   'all' | 'tz' | 'ox' | 'wash'   (spectra mode legend filter)
species  'tz' | 'ox'                    (calibration mode active species)
LAMBDA   {tz:200, ox:204}               (per-species reading wavelength)
CEIL     1.0                            (Beer–Lambert absorbance ceiling)
LAST     last CLS result (for CSV export)
chSpec/chCalSpec/chCal/chWash           Chart.js instances
```

Sample object shape (built in `load()`):
```
{ name, file, role, conc, color, locked, use,
  nm[], abs[], npts, short,          // short = <50 pts (incomplete export)
  pA, pNm,                           // peak abs + its nm (set by annotate())
  over,                              // peak exceeds CEIL
  adv,                               // {A, D, c} dilution advice if saturated
  dup }                              // duplicate of another loaded scan
```

### 4.2 Function map (all in the one `<script>`)
Parsing: `parseAny → parseDelim | parseXML`, helpers `cells, mk, dedupe`.
Name inference: `gRole, gConc, cleanName`.
Loading: `pick` (file input), `load` (FileReader + report-CSV reject), drag/drop
handlers on `#dz`.
Spectral math: `peak, At` (nearest-wavelength absorbance), `lin` (linear
regression w/ R²), `stray, trueA` (stray-light dilution estimate).
Per-sample annotation & colour: `annotate, autoColor, shade, usable`.
Sample list UI: `renderList, sampHTML`, row callbacks `setUse/setColor/setName/
setConc/setRole/del`.
Modes: `setMode, setFilter, setSpecies, setLambda`.
Charts: `opts, ceilingBand, drawSpectra, drawCalib (+ attachDrag), drawWash`.
Calibration readout is rendered inside `drawCalib`.
Deconvolution: `buildEps` (full ε(λ) over the window, null where undefined),
`washList, drawDeconv, autoRun, runCLS`, solvers `solve3` (3×3) and `solveN`
(general n×n, used when the scattering term makes it 4×4), `scatBasis` (λ⁻⁴).
Export: `exportCSV`.
Orchestration: `drawAll` (draws current mode), `rebuild` (annotate→colour→
list→draw, called after any data change). Init: `rebuild()` at end of script.

### 4.3 The three modes
- **Spectra** — overlays all loaded absorbance curves, filterable by role, green
  Beer–Lambert band + red ceiling line drawn by the `ceilingBand` plugin.
- **Calibration** (the primary workflow) — species toggle (TZ/Ox). Top chart:
  that species' standard spectra with a **draggable red λ-cursor** (drag on the
  canvas or use the slider; `attachDrag` wires pointer/touch → data-x →
  `LAMBDA[species]`). Bottom chart: A-vs-c scatter at the chosen λ with the fit
  line; points above CEIL are drawn as open circles and excluded. The big readout
  shows ε (M⁻¹cm⁻¹ = slope/b ×1000), R² (with "n/a" if <3 pts), and the
  equation. A single contextual message flags λ<200, over-ceiling points, <3
  points, or confirms a good fit.
- **Deconvolute** — pick a wash scan, set fit window (λmin/λmax), dilution, path
  b, and the scattering checkbox. `runCLS` builds the design matrix over
  wavelengths where both ε are defined and A<CEIL, solves, and plots
  measured/fit/residual (+ scattering curve when enabled). Two result cards show
  residual TZ and Ox (mM and mg/L). Residual RMS + a runs test drive a
  trustworthy / structured-residual / negative-conc message. `exportCSV` writes
  the result + per-λ data.

---

## 5. How a user runs it (happy path)
1. Export from INSIGHT → Reports → Export → Spectrum CSV (one file can hold all
   samples; blocks are split automatically).
2. Drag the file onto the drop zone. Samples auto-group into TZ / Ox / Wash /
   Unassigned; role and conc are auto-filled from names where possible.
3. Type the concentration for each standard (mM). Wash rows take no conc.
4. **Calibration mode:** pick TZ, drag the λ-cursor to ~200 nm, read ε and R².
   Switch to Ox, cursor to ~204 nm. Aim for R² ≥ 0.999 with all points under the
   ceiling.
5. **Deconvolute mode:** select the wash scan, set the dilution factor used in
   the lab, keep "correct light scattering" on for turbid samples, click
   Deconvolute. Read residual TZ/Ox; check the residual is flat (a peak near
   260 nm suggests a third absorber — a real observation in this project).

---

## 6. Validation already done (reproduce before shipping changes)
- Parser: all 11 real samples in the stacked CSV classify correctly
  (TZ=5, Ox=5, Wash=1) including the `1,4TZ 1` comma-name case; report-CSV is
  rejected.
- Calibration math on real data: TZ@200 R²=0.9997 ε=508; Ox@204 R²=0.9979 ε=2122.
- CLS on synthetic clean mixtures: recovered c_TZ=0.42→0.4202, c_Ox=0.18→0.1800
  (<0.1 % error).
- Scattering term on synthetic turbid mixtures: OFF → Ox +100 %+ error; ON →
  <1 %; clear sample unaffected.
When you change parsing, regression, or CLS, re-run these as node scripts
(extract the function bodies, feed the real files in /uploads) and confirm the
numbers still hold.

---

## 7. Known limitations / open work (good next tickets)
1. **pH / oxalate speciation (highest value).** Add matrix-matched or multi-pH ε
   so Ox from pH-5–6 wash becomes quantitative (see §2.8). Needs standards at the
   wash pH; currently blocked on lab reagents (only oxalic acid on hand, no NaOH/
   buffer at last report).
2. **Turbidity / %T mode.** The project also measures %T at 400–700 nm to track
   suspended solids across wash rounds. A companion mode could ingest visible-
   range %T, plot the round-by-round turbidity ladder, and (stretch) use the
   visible scattering to subtract Mie baseline from the far-UV linker fit.
3. **Mie scattering** cannot be math-corrected; the tool should more loudly steer
   Zn-containing samples to "centrifuge/filter first" (currently only implied).
4. **Recovery-test helper.** The lab prepared known TZ+Ox mixtures (bottle 5).
   A mode that takes the weighed concentrations and reports % recovery from the
   deconvolution would turn this into a one-click method-validation.
5. **Uncertainty.** No error bars yet — propagate regression SE into ε and into
   the CLS result; report LOD/LOQ (ICH: 3.3·s/slope and 10·s/slope; already
   computed offline as TZ LOD≈24 µM, Ox LOD≈21 µM).
6. **Baseline/intercept handling.** TZ intercept at 200 nm is small but
   statistically non-zero (stray-light background); keep the fitted intercept —
   do NOT force-through-zero (forcing zero shifted ε by ~3.6 %). If you add a
   force-zero option, make it opt-in and warn.

---

## 8. Style / conventions to preserve
- Single self-contained file; no build, no external state, no localStorage.
- Colours encode meaning: TZ blue `#1f6fe0`, Ox amber `#e07b1a`, wash violet
  `#7c45cc`, the movable λ-cursor coral `#e0533d`. Keep species colours stable.
- Numbers/data in JetBrains Mono; UI text in Instrument Sans.
- `rebuild()` after any state change; never mutate charts in place — destroy and
  recreate (the code already does).
- Keep the scientific guards (§2) intact. If a change would relax the A-ceiling,
  the fixed-λ reading, the null-ε skip, the <3-point R² guard, or the peak-inside-
  window rule, that change is almost certainly wrong.

---

## 9. Provenance
Built iteratively (v1→v13) against real Evolution 350 exports from the CALF-20
wash-water study. Each scientific guard corresponds to a specific real failure
observed during development. This README reflects the state at v13 (mode-based UI,
draggable λ-cursor, CLS with optional λ⁻⁴ scattering correction).

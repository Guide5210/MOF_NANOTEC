# XRD Pattern Viewer — CALF-20 / MOF powder diffraction analysis

**Location:** `C:\MOF_NanoTec\XRD_CALF20\`
**Main file:** `xrd_viewer.html` — one self-contained HTML file, ~290 KB, ~7,200 lines
**Version:** v7 (2 Sep 2026) · **Status:** working, 73/73 automated checks passing

---

## อ่านก่อน (สรุปภาษาไทย)

เปิดไฟล์ `xrd_viewer.html` ด้วย Chrome/Edge ได้เลย ไม่ต้องติดตั้งอะไร ไม่ต้องมีเซิร์ฟเวอร์

ลำดับการใช้งานปกติ:

1. แท็บ **Data** → ลากไฟล์ `.brml` จากเครื่อง XRD ลงไป (หรือ `.txt` / `.xy` / `.csv` / `.xlsx`)
2. แท็บ **Reference** → เปิด "Show simulated peaks" ถ้าต้องการเทียบกับเฟสอ้างอิง
3. แท็บ **Region** → ถ้าต้องการวิเคราะห์เฉพาะช่วง 2θ ที่สนใจ
4. แท็บ **Analysis** → กด **Analyse patterns**
5. ดูผลที่ตาราง Peak Analysis ด้านล่าง, แท็บ **Crystallinity**, แท็บ **Measurement**
6. แท็บ **Export** → PNG / CSV / Markdown / PDF report

⚠️ **ก่อนอ้างค่า % ผลึก** ให้เปิด "Show amorphous background" ในแท็บ Analysis แล้วดูว่าเส้นประวิ่งตามโหนกกว้างจริง ไม่ได้กินยอดพีค — ถ้าต้องขยับค่า "peak / halo split" ต้องใช้ค่าเดียวกันทุก batch แล้วเขียนไว้ใน method

⚠️ **ไฟล์ `beta_calf20.xy` ในโฟลเดอร์นี้ไม่ใช่เฟส β** — มันคือสำเนาของ τ ดูหัวข้อ *Reference data in this folder* ด้านล่าง

---

## 1. What this is

A single-file browser application for analysing powder X-ray diffraction (PXRD)
patterns, built for a CALF-20 MOF scale-up thesis at KMITL + NANOTEC but
generalised to work with any crystalline material.

It loads scans, fits peaks, computes crystallite size and a crystalline /
amorphous ratio, overlays simulated reference phases, identifies which phase a
sample is, and writes a publication-ready PDF report.

**Everything is in one HTML file.** No build step, no server, no install.
Three libraries load from CDN:

| Library  | Version | Purpose                  |
|----------|---------|--------------------------|
| Chart.js | 4.4.1   | All plotting             |
| jsPDF    | 2.5.1   | PDF report export        |
| SheetJS  | 0.18.5  | Reading `.xlsx`          |

Because the user runs it from `file:///` on a lab PC, **keep it
dependency-light and inline.** Do not introduce a bundler, a framework, or
split it into modules unless explicitly asked. The `.brml` reader was written
against the browser's native `DecompressionStream` specifically to avoid adding
a fourth CDN dependency.

**Browser requirement:** `DecompressionStream('deflate-raw')` — Chrome/Edge
103+, Firefox 113+, Safari 16.4+. Only `.brml` loading needs it; everything
else works in older browsers. The code raises a clear error rather than
failing silently if it is missing.

---

## 2. Folder contents

```
C:\MOF_NanoTec\XRD_CALF20\
├── xrd_viewer.html          ← THE APPLICATION (open this)
├── README.md                ← this file
│
├── dev\                     ← verification harness (see §12)
│   ├── README-dev.md        how to run the tests
│   ├── check_static.js      syntax + dangling-id + stale-reference check
│   ├── test_viewer.mjs      73 end-to-end checks in headless Chromium
│   └── fixtures\
│       ├── scan_a.brml      Bruker fixture, 1466 pts, 5.0001–34.9797°
│       └── scan_b.brml      Bruker fixture, 1467 pts, real CALF-20 scan
│
├── alpha_calf20_ref.cif     CCDC 2265298 (DUGHEW)  — α source structure
├── gamma_calf20_ref.cif     CCDC 2265299 (DUGBOA)  — γ source structure
├── tau_calf20_ref.cif       CCDC 2370609 (DUGGUL)  — τ source structure
├── run_cif_xrd.jl           Julia + PyCall + pymatgen: CIF → .xy simulation
├── alpha_calf20.xy          simulated profiles produced by run_cif_xrd.jl
├── beta_calf20.xy           ⚠️ NOT β — see §4
├── gamma_calf20.xy
├── tau_calf20.xy
└── Sim_XRD\                 ⚠️ older, MISLABELLED batch — see §4
    └── *.xy
```

The `.cif` / `.xy` / `.jl` files were already here; `xrd_viewer.html`,
`README.md` and `dev\` were added on 2 Sep 2026.

---

## 3. Who the user is (context that matters)

- **Soranan** (handle: Deamon5210). **Communicates in Thai — reply in Thai.** Uses "ผม".
- Senior Nanomaterials Engineering undergrad at **KMITL**, expected grad May 2027.
- Thesis is joint **KMITL + NANOTEC (NCAS group)**. Advisors: Assoc. Prof. Dr. Tosapol
  Maluangnont (KMITL) and Dr. Bunyarat Rungtaweevoranit (NANOTEC). AMCHAM scholar.
- Working on a **12-batch CALF-20 scale-up**, characterising every batch by PXRD.
- **Preferences:** wants sharp, conclusive engineering answers — recommend the best
  option and say why, rather than listing choices. Values scientific integrity highly:
  do **not** hand-wave crystallography, and verify data before building on it.
  Ask before risky or irreversible decisions.

---

## 4. ⚠️ Reference data in this folder is partly mislabelled

This has caused real problems twice. **Verify before you import any `.xy` here
into the phase library.** Evidence, measured on 2 Sep 2026:

| File                     | MD5 (first 8) | Main peak 2θ | Verdict |
|--------------------------|---------------|--------------|---------|
| `alpha_calf20.xy`        | `4b0eb8e5`    | 13.741°      | ✅ correct (α) |
| `beta_calf20.xy`         | `75dd9f3c`    | 14.573°      | ❌ **byte-identical to `tau_calf20.xy`** |
| `gamma_calf20.xy`        | `1ae9b25b`    | 14.965°      | ✅ correct (γ) |
| `tau_calf20.xy`          | `75dd9f3c`    | 14.573°      | ✅ correct (τ) |
| `Sim_XRD\alpha_calf20.xy`| `1b19cd2f`    | 14.040°      | ❌ not α |
| `Sim_XRD\tau_calf20.xy`  | `3c194504`    | 13.828°      | ❌ this is ~α's position — α↔τ swapped |
| `Sim_XRD\beta_calf20.xy` | `9f369989`    | 14.584°      | ⚠️ unverified |
| `Sim_XRD\gamma_calf20.xy`| `1ae9b25b`    | 14.965°      | ✅ same as top level |

**Why β is a duplicate:** `run_cif_xrd.jl` does this deliberately —

```julia
convert_cif_to_xy("tau_calf20_ref.cif", "tau_calf20.xy")
# นำข้อมูลจากไฟล์ tau ไปสร้างเป็นไฟล์ beta ให้โดยอัตโนมัติ
convert_cif_to_xy("tau_calf20_ref.cif", "beta_calf20.xy")
```

There is **no `beta_calf20_ref.cif` in this folder.** The real β structure is
Chen et al. 2023 (`tz3c00930_si_003.cif`), which contains V1/V2 **vanadium
dummy atoms** (disorder placeholders) that make pymatgen and VESTA choke — so
whoever wrote the script sidestepped it by copying τ. That is a reasonable
workaround for a plot, but the *filename lies*, and importing it as β would put
two identical patterns in the phase library and silently corrupt phase ID.

**The good news:** the four phases built into `xrd_viewer.html` (`PHASE_REFS`)
are **not** these files. They were generated separately and cross-verified
against pymatgen `XRDCalculator` output, with β built from a hand-cleaned
minimal CIF (P2₁/c, 11-atom asymmetric unit, V atoms stripped). Their main
peaks are α 13.741 / β 14.573 / γ 14.965 / τ 14.561 — note β and τ differ by
**0.011°**, which is the physically correct near-degeneracy. **Trust the
built-ins, not the loose `.xy` files.**

If you regenerate β: strip the V atoms from Chen's CIF first, and after
generating, `merge_sites(tol=0.02)` on any CCDC mol2-derived structure (they
contain pre-expanded atoms beyond the asymmetric unit).

---

## 5. Architecture

One HTML file, three parts:

1. **`<head>`** — CDN script tags, Google Fonts, and one `<style>` block that
   defines the whole visual system through CSS custom properties on `:root`,
   with a `html.dark` block that overrides them. **Every colour in the UI goes
   through a variable**, which is what makes the dark theme a single class toggle.
2. **`<body>`** — header, stats row, chart panel, the **control deck**
   (carousel of tab panels), peak-analysis table.
3. **One `<script>` at the bottom** — everything else, plain ES2020, no modules.
   Read top to bottom; it is ordered constants → parsers → maths → rendering →
   handlers → init.

### Section map (line numbers, v7)

| Line | Section |
|-----:|---------|
| 14 | CSS `:root` light theme variables |
| 48 | CSS `html.dark` overrides |
| 222 | CSS layout (single full-width column) |
| 915 | CSS control deck / carousel |
| 1058 | CSS crystallinity panel |
| 1116 | CSS measurement panel |
| 1183 | `<div class="container">` — page body starts |
| 1218 | Control deck markup (tabs + 9 panels) |
| 1678 | `PALETTES` / `PALETTES_DARK` — trace colours |
| 1779 | `CHART_THEME` — canvas colours per theme |
| 1853 | `DEFAULTS` + `TT_MIN` / `TT_MAX` (2–50° hard limits) |
| 1933 | `PHASE_REFS` — the phase library seed (4 CALF-20 polymorphs) |
| 2068 | `state` — the single runtime state object |
| 2137 | `parseXRDXlsx`, `parseXRDText`, `estimateBaseline` |
| 2233 | `readZipEntries` — minimal ZIP reader |
| 2302 | `parseBRML` — Bruker file → `{x, y, meta}` |
| 2522 | `readFile` — extension router |
| 2615 | `findCandidatePeaks`, `fitGaussian` |
| 2749 | `roiWindow` / `roiSlice` / `inROI` — **the ROI invariant lives here** |
| 2785 | `detectPeaks` |
| 2855 | `snipBackground`, `computeCrystallinity` |
| 2953 | `scherrerSize` |
| 3020 | `refGeometry`, `generateProfile` — sim overlay geometry |
| 3121 | `buildDatasets` — **central render prep** |
| 3268 | `refStickPlugin` — hkl labels + alignment guides |
| 3401 | `inlineLabelPlugin` — on-chart trace names |
| 3516 | `roiShadePlugin` — ROI band |
| 3553 | `renderChart` |
| 3802 | `renderFileList` + drag/rename/recolour |
| 3925 | Right-click context menu |
| 4099 | `handleFiles` — load pipeline, `.brml` metadata adoption |
| 4192 | `resetView` |
| 4247 | **Phase library** — add/remove/import/export |
| 4737 | View-range handlers + presets |
| 4776 | ROI handlers |
| 4846 | `applyTheme` |
| 4953 | Peak-analysis handlers, **Analyse patterns** button |
| 5019 | `refActive`, `matchToRef` |
| 5069 | `renderPeakTable` |
| 5203 | `flattenPeaksForExport`, CSV + Markdown export |
| 5341 | `renderPdfChart` — publication chart (light-only) |
| 5523 | `pdfText` — Unicode → ASCII for jsPDF |
| 5564 | `renderNotesPages` — Thai text → canvas → PDF |
| 5723 | `computeStatsArray` |
| 5761 | `identifyPhase` — **phase ID algorithm** |
| 5865 | `computeBatchStatistics` |
| 5920 | `buildInterpretation` — auto-written report prose |
| 6219 | `buildPdfReport` |
| 6776 | **Control deck** — carousel logic |
| 6941 | `renderCrystPanel` |
| 7042 | `renderMeasurementPanel` |
| 7204 | `init()` — DOM ↔ state sync on first paint |

---

## 6. State schema

`state` (line 2068) is the only mutable application state. Arrays suffixed
`ByPattern` are **positional and parallel to `state.patterns`** — if you splice
one, splice them all (this was a real bug: deleting or reordering a pattern
left another batch's peaks attached to the wrong sample).

```js
state = {
  patterns: [],          // {name, displayName, x[], y[], baseline, meta, customColor, visible}
  peaksByPattern: [],    // parallel · fitted peaks, or null
  crystByPattern: [],    // parallel · crystallinity result, or null
  phaseIDByPattern: [],  // parallel · phase-ID result, or null

  // --- view (figure zoom only, never affects a number) ---
  intensityMode: 'raw' | 'normalized',
  layoutMode:    'stacked' | 'overlay',
  batchLabels:   'legend' | 'inline',
  offsetMul, rangeMin, rangeMax, smooth, palette, figTitle,
  theme: 'light' | 'dark',

  // --- analysis region (defines what every number means) ---
  roiOn, roiMin, roiMax,

  // --- reference phases ---
  refOn, activePhases: ['alpha','beta',…], hklLabelPhase, hklThresh, simFWHM,

  // --- analysis parameters ---
  peakDetectThreshold,   // % of max
  scherrerInstrFwhm,     // deg
  scherrerK,             // 0.9
  scherrerLambda,        // nm — auto-set from .brml
  crystWin,              // SNIP clipping width, deg
  crystBgShow,
  notes,
}
```

### Invariants worth protecting

1. **The view window never changes a number.** `rangeMin/rangeMax` only set the
   chart axis. All maths goes through `roiWindow()` / `roiSlice()` / `inROI()`.
   `resetView()` deliberately does **not** reset the ROI for this reason.
2. **Exports are always light-themed.** `chartTheme(forExport)` and
   `buildDatasets(forExport)` take a flag; PNG and PDF pass `true`. A thesis
   figure must not depend on what the screen was showing.
3. **Hidden patterns are excluded everywhere** — chart, fitting, crystallinity,
   phase ID, tables, all exports.
4. **`PHASE_REFS` is a mutable registry, not a constant.** Anything that
   iterates phases must iterate `state.activePhases` / `Object.keys(PHASE_REFS)`,
   never a hard-coded list of four.

---

## 7. The science (read before touching these)

### 7.1 Peak fitting — `fitGaussian` (2659)

Gaussian fit by log-linearisation: `ln(y−b)` is a parabola in x, solved by
weighted least squares (weights = `y−b`, so the peak top dominates) via
Cramer's rule. Fit window = points above 30 % of peak height. Returns position,
height, FWHM, area. Raw-maximum FWHM is noisy to ±20 %; the fit gets ~2 %,
which matters because FWHM feeds Scherrer directly.

### 7.2 Crystallite size — `scherrerSize` (2953)

`D = K·λ / (β·cos θ)`, K = 0.9, β = √(FWHM_obs² − FWHM_instr²) in radians.
Returns `null` when the peak is narrower than the instrument (can't deconvolve)
— callers must handle `null` rather than printing a fake number. Sizes > 500 nm
are discarded from batch statistics as beyond Scherrer's validity.

Batch statistics use **the main (strongest) peak only** — the convention in
materials-science reproducibility papers, because mixing sharp and broad peaks
inside one batch inflates the SD artificially. Sample SD (Bessel n−1).

### 7.3 Crystallinity — `snipBackground` + `computeCrystallinity` (2855)

The Bruker file carries raw counts but its `EvaluationContainer.xml` is **empty**
unless someone runs an evaluation in DIFFRAC.EVA — there is no vendor
crystallinity number to import. So it is computed here:

```
%Crystallinity = A_cryst / (A_cryst + A_amorph) × 100
```

separated by a **SNIP** background (Ryan et al., NIM B 34 (1988) 396):

1. LLS transform `v = log(log(√(y+1)+1)+1)` compresses dynamic range so tall
   Bragg peaks don't drag the background up.
2. Iterative clipping `v[i] = min(v[i], (v[i−w]+v[i+w])/2)` for w = 1…W.
   Anything narrower than W is clipped away; broader features survive.
3. Inverse LLS → B(x).

W comes from `state.crystWin` in **degrees**, converted to points via the scan
step. Default 1.5° sits in a wide gap between a MOF reflection (0.1–0.5° FWHM)
and an amorphous halo (5–10°).

The flat instrument/air floor is removed from **both** areas first
(`floor = min(B)` over the ROI) so it counts as neither crystalline nor amorphous.

> **Honesty requirement — do not remove this.** The UI, the crystallinity panel,
> the Markdown export and the PDF all state that this is a **relative index**,
> valid for comparing samples measured with identical scan settings over the same
> window, and **not** an absolute crystalline weight fraction (that needs an
> internal standard or Rietveld refinement). The "Show amorphous background"
> toggle exists so the user can visually verify the split before trusting it.

### 7.4 Phase identification — `identifyPhase` (5761)

Intensity-weighted peak matching:

```
Score(phase) = Σ(wᵢ · matchedᵢ) / Σ(wᵢ) × 100 %
  wᵢ       = reference peak relative intensity (0–100) → main peak dominates
  matchedᵢ = 1 if a detected peak lies within tol of ref peak i
  tol      = max(0.20°, detected peak FWHM)
```

Only reference peaks **inside the ROI** are scored — otherwise a phase is
penalised for reflections the user deliberately excluded, and the score would
depend on the reference's 2θ extent rather than on the sample.

Confidence, in this priority order (do not reshuffle):

```
bestScore < 30             → 'low'        (probably not this material)
lead < 5                   → 'ambiguous'  (top two tied)
bestScore ≥ 60 AND lead≥15 → 'high'
bestScore ≥ 40 OR lead ≥ 5 → 'moderate'
else                       → 'low'
```

### 7.5 ⚠️ β and τ are nearly indistinguishable by PXRD — this is physics

Their unit cells differ by < 0.03 Å. Main peaks: β 14.573° vs τ 14.561° →
**0.011° apart**, far below standard instrument resolution (~0.05°). Drwęska
et al. separated them by **single-crystal** XRD, which powder cannot do.

The tool reports `ambiguous` when the top two scores are within 5 %.
**Do not "fix" this by forcing a pick.** The user explicitly chose to keep all
four phases separate and report the ambiguity. That is the scientifically
correct behaviour and removing it would be a regression.

---

## 8. The `.brml` reader

A `.brml` is a plain **ZIP of XML**. `readZipEntries` (2233) walks the
end-of-central-directory record and inflates only the two parts that matter,
using `DecompressionStream('deflate-raw')` — which is why loading a 430 KB file
is instant despite the archive containing two ~2.7 MB XML blobs.

| Part | Read? | Contains |
|------|-------|----------|
| `Experiment0/RawData0.xml` | ✅ | data points + the **as-measured** instrument record |
| `Experiment0/DataContainer.xml` | ✅ | instrument identity, operator, sample header |
| `Experiment0/MeasurementContainer.xml` | ❌ | 2.9 MB, not needed |
| `Experiment0/InstructionContainer.xml` | ❌ | 2.7 MB — the instrument's **full component catalogue** |
| `Experiment0/EvaluationContainer.xml` | ❌ | empty unless someone ran DIFFRAC.EVA |

> **Read optics from `RawData0.xml` → `<FixedInformation>`, never from
> `InstructionContainer.xml`.** The latter lists every optic the instrument
> *owns*, not what was in the beam — you would report hardware sitting in a cabinet.

Data rows look like:

```xml
<Datum>37.8,1,5.0001,2.5001,381</Datum>
<!--    time, absorption, 2Theta, Theta, counts -->
```

Column indices are read from the `<DataViews>` descriptors rather than assumed,
so an unusual scan layout still lands on the right columns.

### Two XML traps already hit

- `xsi:type="SollerInfoData"` is an **attribute value**, not a tag name — the tag
  is `<InfoData>`. A `querySelector('SollerInfoData > …')` silently matches
  nothing. Query the child element (`AxialDivergence`) directly.
- `Instrument` has a descendant `<Orientation>` on the Soller optic ("Axial")
  *before* the goniometer's own ("Vertical"). Use `:scope >` for direct children.

### Metadata surfaced in the Measurement tab

Sample (name, position, operator) · Scan (type, mode, start/stop/step, points,
time-per-step **set vs effective**, start time, duration, status) · Source
(anode, kV/mA, λ Kα₁/Kα₂/avg/Kβ, ratio, focus, tube serial) · Optics (Kβ filter
+ thickness, axial Soller, mounted list, goniometer type/radius/orientation) ·
Detector (model, mode, angular opening, discriminator) · Counting statistics
(mean, RMS, max, min, **σ = √N** on the strongest point) · Instrument (site,
system, serial, firmware).

Two of those are editorial additions worth keeping:

- **Effective time/step.** On a PSD fast scan every point is swept by all 189
  LYNXEYE strips, so the effective counting time (37.8 s) is ~190× the set step
  time (0.2 s). Reporting only the latter badly understates counting statistics.
- **σ = √N.** On a scan with max 13,606 counts that is ±117 counts (0.86 %) —
  the honest precision floor on any intensity ratio taken from that scan.

Patterns loaded from `.txt` / `.xy` / `.csv` / `.xlsx` have `meta === null`; the
Measurement tab says so plainly and shows only what can be derived from the data.

---

## 9. The control deck (UI)

Controls used to live in a 300 px left rail that required scrolling past eight
stacked panels. v7 replaced it with a horizontal carousel below the chart, which
also gave the diffractogram the full page width.

```
.deck
├── .deck-nav
│   ├── #deckPrev  ‹
│   ├── #deckTabs   [scrollable strip of .deck-tab]
│   └── #deckNext  ›
└── #deckBody      [9 × section.deck-panel, one .active at a time]
```

Panels: `data · display · reference · roi · analysis · cryst · measurement ·
notes · export` (`DECK_PANELS`, line 6786).

Driven four ways: click a tab · the ‹ › arrows · ← / → keys (suppressed when a
form control has focus, so they don't fight the range sliders) · drag the tab
strip or swipe the panel body.

**Panels are shown/hidden, never re-created** — every control keeps its DOM node
and its listeners. That is why moving the entire sidebar in here required no
change to any handler.

### Adding a panel

1. Add the id to `DECK_PANELS` (6786).
2. Add a `<button class="deck-tab" data-panel="yourid">` in `#deckTabs`.
3. Add `<section class="deck-panel" data-panel="yourid">` in `#deckBody`.
4. Lay it out with `.deck-cols` / `.deck-col` (auto-fit columns), `<h3>` headings
   and `.deck-hint` for explanatory text.
5. If it renders from state, render it in `setDeckPanel()` on entry and add it to
   `refreshDeckDerived()` (6824) so it updates while visible.

### Two derived panels

- **`renderCrystPanel` (6941)** — one stacked bar per pattern. Crystalline share
  is solid; amorphous is **hatched**, deliberately: it is an estimate from a
  fitted background and should not read with the same visual authority as the
  integrated Bragg area. Shows areas, mean ± SD, range, spread in percentage
  points, the window used and the SNIP width.
- **`renderMeasurementPanel` (7042)** — the instrument record, grouped like a
  methods section. Empty fields and empty groups are dropped rather than shown
  as dashes. Row labels must stay unique across groups (the test enforces this).

---

## 10. Theming

- Light is the default and matches printed figures.
- `applyTheme()` (4846) toggles `html.dark`, which swaps the CSS variables, and
  re-renders the chart so the canvas follows.
- Chart.js cannot read CSS variables, so canvas colours come from `CHART_THEME`
  (1779) via `chartTheme(forExport)`.
- Near-black trace colours are invisible on dark, so `PALETTES_DARK` (1742)
  provides lifted variants in the same order, and `phaseColorForTheme()` lifts
  any reference colour whose luminance is too low.
- **Exports ignore the theme entirely.** `renderPdfChart()` is hard-wired to
  black-on-white; PNG export re-renders through that same path at 3× rather than
  screenshotting the on-screen canvas.

---

## 11. Exports and the jsPDF gotchas

| Button | Path |
|--------|------|
| Save figure (PNG) | `renderPdfChart(3)` → canvas → PNG. Always light. |
| Save patterns (CSV) | Raw 2θ / intensity column pairs, full measured range |
| Peak table CSV / Markdown | Fitted peaks + size + crystallinity + matched hkl, with a settings header line |
| Export PDF report | Chart, auto-written summary, per-sample table, phase ID table, notes, all peaks |

**jsPDF's built-in Helvetica is WinAnsi, not Unicode.** Greek letters render as
garbage. Two mechanisms handle this:

- Every phase carries a `pdfName` (`alpha-CALF-20`, …) — ASCII only. Any new PDF
  text containing a phase name **must** use `pdfName`, and `asciiPhaseName()`
  derives one automatically for imported phases.
- `pdf.text` and `pdf.splitTextToSize` are wrapped so all strings pass through
  `pdfText()` (5523), which converts θ→theta, ±→+/-, °→deg, etc.
- **Thai notes** cannot go through jsPDF text at all. `renderNotesPages()` (5564)
  draws them to a canvas in the browser's own font stack and embeds the result as
  an image, with pagination and character-level wrapping (Thai has no spaces).

`jsPDF.save` is an **instance** property, not on the prototype — patching the
prototype to spy on it silently never fires. The test listens for the browser's
download event instead.

---

## 12. How to validate a change

**Never ship an edit to this file without running both checks.** Every session
so far has caught real bugs this way — parallel arrays desynchronising on
delete/reorder, a silently-swallowed PDF exception, two XML selectors matching
nothing.

```powershell
cd C:\MOF_NanoTec\XRD_CALF20\dev

# 1. Static: syntax, dangling getElementById targets, stale references
node check_static.js

# 2. End-to-end: 73 checks in headless Chromium
npm install playwright chart.js@4.4.1 jspdf@2.5.1 xlsx@0.18.5
npx playwright install chromium
node test_viewer.mjs
```

The CDN is stubbed with local npm copies of the same library versions, so the
harness runs offline. Env overrides: `XRD_HTML`, `XRD_LIB`, `XRD_BRML`,
`XRD_BRML2`, `CHROMIUM_PATH`. See `dev/README-dev.md`.

**What the 73 checks cover:** `.brml` parsing and metadata · wavelength
auto-adoption · peak fitting · crystallinity (including a synthetic pattern with
a known crystalline:amorphous split) · ROI restricting every derived number while
leaving the view alone · phase ID · phase library import/merge/delete rules ·
dark mode + the export-stays-light rule · the carousel (tab↔panel pairing, arrow
bounds, keyboard, slider guard) · both derived panels · PNG and PDF generation
end-to-end · CSV/Markdown content.

After a bulk or scripted edit, also spot-check: exactly one definition each of
`renderChart`, `inlineLabelPlugin`, `refStickPlugin`, `roiShadePlugin`,
`buildDatasets` (a de-dup script once deleted function headers and broke the
file). `check_static.js` does this for you.

---

## 13. Deliberate non-features

Do not "fix" these without asking — each was a decision:

- **No `localStorage`.** The tool is opened from `file://` on different machines;
  a silently-remembered phase library is a reproducibility hazard in a thesis
  workflow. The library is saved and loaded as an explicit JSON file instead.
- **No d-spacing anywhere in the UI or exports.** Removed on request — it is
  redundant with 2θ, and crystallite size is the reported structural quantity.
  `peak.dSpacing` is still computed internally; leave it, it costs nothing.
- **Sim traces are solid, not dashed.** Explicitly requested. They are
  distinguished by colour and the "(sim.)" label.
- **`resetView()` does not reset the ROI.** See §6.
- **β/τ ambiguity is reported, not resolved.** See §7.5.
- **The batch palette must stay distinguishable from itself *and* from the four
  phase hues.** This was over-corrected once into all-dark neutrals that made
  batches look identical. Verify both properties if you touch `PALETTES`.

---

## 14. Likely next work

- Applying phase ID across all 12 batches and summarising the phase distribution.
- Pinning down the SNIP width against the best and worst batches, then fixing one
  value for the whole series and recording it in the methods section.
- Adding the ROI band and hkl guides to the PDF chart.
- Exporting the phase-ID table to xlsx.
- Per-instrument instrument-broadening calibration (a LaB₆ or Si standard scan
  would let `scherrerInstrFwhm` be measured rather than assumed at 0.10°).

---

## 15. Change log

**v7 — 2 Sep 2026**
Left sidebar → horizontal carousel of 9 tab panels under the chart; chart now
full width. New **Crystallinity** panel (stacked bars, areas, mean ± SD).
New **Measurement** panel (full instrument record from `.brml`). `.brml` parser
extended to ~30 metadata fields incl. effective counting time and √N precision.
Fixed: Soller selector matched nothing; goniometer orientation was read off the
Soller; duplicate "Mode" row label.

**v6 — 15 Aug 2026**
d-spacing removed from all outputs. Crystallinity (SNIP area ratio) added.
Analysis ROI separated from the view range. `PHASE_REFS` became an editable
phase library with CSV/`.xy` import and JSON save/load. 2θ ceiling raised to 50°.
Dark mode. Direct `.brml` reading. Fixed: parallel arrays not spliced on
delete/reorder; PDF export failing silently.

**v5.x and earlier** — core viewer, Gaussian fitting, Scherrer, 4-phase CALF-20
reference data with hkl indexing, multi-phase ID, intensity×layout refactor,
PDF report with auto-interpretation, multilingual notes.

---

*When in doubt: verify the data first, keep it single-file, respect the β/τ
ambiguity, use `pdfName` in PDFs, run both checks before shipping, and reply in Thai.*

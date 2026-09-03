# dev — verification harness

Two checks. Run both before shipping any edit to `../xrd_viewer.html`.
See §12 of the main `README.md` for why.

## 1. Static check (fast, no dependencies)

```powershell
node check_static.js
node check_static.js path\to\other.html    # optional explicit target
```

Extracts the inline `<script>` from the HTML, runs `node --check` on it, then:

- cross-checks every `getElementById('…')` target against the ids present in the
  markup — catches "moved the markup, forgot a handler";
- warns if any of `renderChart`, `inlineLabelPlugin`, `refStickPlugin`,
  `roiShadePlugin`, `buildDatasets`, `renderCrystPanel`, `renderMeasurementPanel`,
  `setDeckPanel` is defined anything other than exactly once;
- flags stale identifiers from past refactors (`state.mode`, `state.refPeaks`,
  `filesPanel`, `getPatternColor(`, `REF_CALF20`), ignoring comments.

Exit code is non-zero on syntax failure or a missing id.

## 2. End-to-end check (headless Chromium, 73 assertions)

One-time setup in this folder:

```powershell
npm install playwright chart.js@4.4.1 jspdf@2.5.1 xlsx@0.18.5
npx playwright install chromium
```

Then:

```powershell
node test_viewer.mjs
```

The three CDN `<script src>` tags are intercepted and fulfilled from the local
`node_modules` copies of the same versions, and Google Fonts is stubbed, so the
run works offline and is not affected by CDN outages.

### Environment overrides

| Variable | Default | Purpose |
|----------|---------|---------|
| `XRD_HTML` | `../xrd_viewer.html` | the file under test |
| `XRD_LIB` | `./node_modules` | where chart.js / jspdf / xlsx live |
| `XRD_BRML` | `./fixtures/scan_a.brml` | first Bruker fixture |
| `XRD_BRML2` | `./fixtures/scan_b.brml` | second Bruker fixture |
| `CHROMIUM_PATH` | Playwright's own | explicit browser binary |

### Fixtures

`fixtures/scan_a.brml` and `fixtures/scan_b.brml` are two real Bruker D8 Advance
scans. **Some assertions are bound to their exact contents** — swap the fixtures
and you must update these numbers (they are the point of the test: they prove
the parser read the real file rather than falling back to a default):

- `scan_a`: 1466 points · 5.0001–34.9797° · step 0.02046° · `Continuous PSD fast`
  · Cu 40 kV / 40 mA · countsMax 8662
- `scan_b`: 1467 points · stop 35.0001° · countsMax 13606

Everything else — deck behaviour, ROI semantics, crystallinity maths, the phase
library, theming, exports — is fixture-independent.

### What it covers

| Group | Checks |
|-------|-------:|
| Load + `.brml` parsing + metadata | 7 |
| Peak fitting, crystallinity, phase ID, peak table | 7 |
| Analysis ROI semantics | 4 |
| Background overlay, 2θ ceiling | 2 |
| Dark mode + export-stays-light | 4 |
| Phase library (import, merge, delete rules) | 6 |
| Report: interpretation, PDF, CSV, Markdown | 10 |
| Control deck / carousel | 8 |
| Crystallinity panel | 6 |
| Measurement panel | 9 |
| Second real `.brml`, badges, no-JS-errors | 5 |

Screenshots are written to `/tmp/shot_light.png` and `/tmp/shot_dark.png` (or
the OS temp dir) — useful for eyeballing a layout change.

## Adding a check

Append near the end, before the final screenshot, using the `check(name, ok,
detail)` helper. Controls now live inside carousel panels, so open the owning
tab first:

```js
await openTab('analysis');
await page.click('#detectPeaks');
```

Anything driven through `page.evaluate()` works regardless of which panel is
visible; only real clicks need the tab opened, because Playwright refuses to
click a hidden element.

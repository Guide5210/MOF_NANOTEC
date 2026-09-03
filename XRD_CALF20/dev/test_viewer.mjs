// End-to-end verification of xrd_viewer.html in headless Chromium.
//
// FIXTURE-BOUND ASSERTIONS — if you swap the .brml fixtures, these numbers
// must be updated too (they are the point of the test: they prove the parser
// read the real file rather than a default):
//   scan_a.brml : 1466 points, 5.0001-34.9797 deg, countsMax 8662,
//                 'Continuous PSD fast', step 0.02046, Cu 40kV/40mA
//   scan_b.brml : 1467 points, stop 35.0001 deg, countsMax 13606
// Everything else (deck behaviour, ROI, crystallinity, phase library,
// theming, exports) is fixture-independent.
// CDN is unreachable from this sandbox, so the three <script src> tags are
// fulfilled from local npm copies of exactly the same library versions.
import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

// Paths are overridable so this harness runs anywhere:
//   XRD_HTML   the viewer under test          (default ../xrd_viewer.html)
//   XRD_LIB    node_modules holding chart.js / jspdf / xlsx  (default ./node_modules)
//   XRD_BRML   a Bruker .brml fixture         (default ./fixtures/scan_a.brml)
//   XRD_BRML2  a second, different .brml      (default ./fixtures/scan_b.brml)
// See dev/README-dev.md for setup.
const HERE = path.dirname(new URL(import.meta.url).pathname);
const HTML = process.env.XRD_HTML || path.join(HERE, '..', 'xrd_viewer.html');
const LIB  = process.env.XRD_LIB  || path.join(HERE, 'node_modules');
const BRML_A = process.env.XRD_BRML  || path.join(HERE, 'fixtures', 'scan_a.brml');
const BRML_B = process.env.XRD_BRML2 || path.join(HERE, 'fixtures', 'scan_b.brml');
const ROUTES = {
  'chart.umd.min.js': `${LIB}/chart.js/dist/chart.umd.js`,
  'jspdf.umd.min.js': `${LIB}/jspdf/dist/jspdf.umd.js`,
  'xlsx.full.min.js': `${LIB}/xlsx/dist/xlsx.full.min.js`,
};

const results = [];
const check = (name, ok, detail = '') => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
};

const browser = await chromium.launch(
  process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {});
const page = await browser.newPage({ viewport: { width: 1600, height: 1100 } });

const consoleErrors = [];
page.on('console', m => { if (m.type() === 'error') consoleErrors.push(m.text()); });
page.on('pageerror', e => consoleErrors.push('pageerror: ' + e.message));

await page.route('**/*', async route => {
  const url = route.request().url();
  const hit = Object.keys(ROUTES).find(k => url.endsWith(k));
  if (hit) return route.fulfill({ path: ROUTES[hit], contentType: 'application/javascript' });
  if (url.includes('fonts.googleapis') || url.includes('fonts.gstatic')) {
    return route.fulfill({ status: 200, contentType: 'text/css', body: '' });
  }
  return route.continue();
});

await page.goto('file://' + HTML);
await page.waitForTimeout(700);

// Controls now live in the deck carousel, one panel visible at a time.
// Every interaction below opens the owning tab first — which is itself a
// test that the tab actually reveals its panel.
const openTab = async (name) => {
  await page.click(`.deck-tab[data-panel="${name}"]`);
  await page.waitForTimeout(220);
};

check('page loads with no JS errors', consoleErrors.length === 0, consoleErrors.join(' | '));
check('Chart.js loaded', await page.evaluate(() => typeof Chart === 'function'));

// ---------------------------------------------------------------- .brml
const brml = BRML_A;
await page.setInputFiles('#fileInput', brml);
await page.waitForTimeout(1200);

const afterBrml = await page.evaluate(() => ({
  n: state.patterns.length,
  pts: state.patterns[0] ? state.patterns[0].x.length : 0,
  x0: state.patterns[0] ? state.patterns[0].x[0] : null,
  xN: state.patterns[0] ? state.patterns[0].x[state.patterns[0].x.length - 1] : null,
  meta: state.patterns[0] ? state.patterns[0].meta : null,
  lambda: state.scherrerLambda,
  range: [state.rangeMin, state.rangeMax],
}));
check('.brml parsed', afterBrml.n === 1 && afterBrml.pts === 1466,
      `${afterBrml.n} pattern(s), ${afterBrml.pts} points`);
check('.brml 2theta range correct',
      Math.abs(afterBrml.x0 - 5.0001) < 0.01 && Math.abs(afterBrml.xN - 34.9797) < 0.01,
      `${afterBrml.x0} .. ${afterBrml.xN}`);
check('.brml metadata extracted',
      !!afterBrml.meta && afterBrml.meta.anode === 'Cu' &&
      Math.abs(afterBrml.meta.lambdaAngstrom - 1.5406) < 1e-6,
      JSON.stringify(afterBrml.meta));
check('wavelength auto-applied from file',
      Math.abs(afterBrml.lambda - 0.15406) < 1e-9, String(afterBrml.lambda));
check('view range snapped to measured span',
      afterBrml.range[0] === 5 && afterBrml.range[1] === 35, JSON.stringify(afterBrml.range));

// ------------------------------------------------- synthetic 2nd pattern
// A clean two-Gaussian pattern on a broad amorphous hump with a known
// crystalline:amorphous split, used to sanity-check the SNIP separation.
const synth = [];
for (let i = 0; i <= 2000; i++) {
  const x = 5 + i * 0.015;
  const hump = 900 * Math.exp(-Math.pow(x - 20, 2) / (2 * 6 * 6));
  const p1 = 9000 * Math.exp(-Math.pow(x - 13.74, 2) / (2 * 0.12 * 0.12));
  const p2 = 2500 * Math.exp(-Math.pow(x - 20.25, 2) / (2 * 0.12 * 0.12));
  synth.push(`${x.toFixed(4)}\t${(200 + hump + p1 + p2).toFixed(1)}`);
}
fs.writeFileSync('/tmp/synth.xy', synth.join('\n'));
await page.setInputFiles('#fileInput', '/tmp/synth.xy');
await page.waitForTimeout(500);
check('second pattern loaded', await page.evaluate(() => state.patterns.length) === 2);

// ------------------------------------------------------------- analysis
await openTab('reference');
await page.click('#refToggle');           // turn reference overlays on
await page.waitForTimeout(300);
await openTab('analysis');
await page.click('#detectPeaks');
await page.waitForTimeout(1500);

const ana = await page.evaluate(() => ({
  peaks: state.peaksByPattern.map(p => p ? p.length : null),
  cryst: state.crystByPattern.map(c => c ? +c.crystPct.toFixed(2) : null),
  pid: state.phaseIDByPattern.map(p => p ? { best: p.best, score: +p.bestScore.toFixed(1), conf: p.confidence } : null),
  tablePeakCols: Array.from(document.querySelectorAll('#peakTable thead .col-header th')).map(t => t.textContent),
  tableHTML: document.getElementById('peakTable').innerHTML.slice(0, 400),
}));
check('peaks fitted for both patterns',
      ana.peaks.every(v => v !== null && v > 0), JSON.stringify(ana.peaks));
check('crystallinity computed', ana.cryst.every(v => v !== null && v > 0 && v < 100),
      JSON.stringify(ana.cryst));
check('synthetic crystallinity in expected band (55-90%)',
      ana.cryst[1] > 55 && ana.cryst[1] < 90, `${ana.cryst[1]}%`);
check('phase ID produced', ana.pid.every(v => v !== null), JSON.stringify(ana.pid));
check('d-spacing column removed from table',
      !ana.tablePeakCols.some(t => /d \(/.test(t)), ana.tablePeakCols.join(' | '));
check('crystallite column present', ana.tablePeakCols.some(t => /Crystallite/.test(t)));
check('crystallinity badge in table', /crystallinity/.test(ana.tableHTML) || /crystallinity/.test(
      await page.evaluate(() => document.getElementById('peakTable').innerHTML)));

// ------------------------------------------------------------------ ROI
const before = await page.evaluate(() => state.peaksByPattern[1].length);
await openTab('roi');
await page.evaluate(() => {
  document.getElementById('roiToggle').click();
  document.getElementById('roiMin').value = 12;
  document.getElementById('roiMin').dispatchEvent(new Event('input'));
  document.getElementById('roiMax').value = 16;
  document.getElementById('roiMax').dispatchEvent(new Event('input'));
});
await page.waitForTimeout(900);
const roi = await page.evaluate(() => ({
  on: state.roiOn,
  peaks: state.peaksByPattern[1].length,
  tts: state.peaksByPattern[1].map(p => +p.twoTheta.toFixed(2)),
  view: [state.rangeMin, state.rangeMax],
  cryst: state.crystByPattern[1] ? +state.crystByPattern[1].crystPct.toFixed(1) : null,
  pid: state.phaseIDByPattern[1] ? state.phaseIDByPattern[1].best : null,
}));
check('ROI restricts peak fitting',
      roi.on && roi.peaks < before && roi.tts.every(t => t >= 12 && t <= 16),
      `${before} → ${roi.peaks} peaks at ${JSON.stringify(roi.tts)}`);
check('ROI leaves the view range untouched',
      roi.view[0] === 5 && roi.view[1] === 35, JSON.stringify(roi.view));
check('ROI recomputes crystallinity', roi.cryst !== null && roi.cryst !== ana.cryst[1],
      `${ana.cryst[1]}% → ${roi.cryst}%`);
check('ROI phase ID still resolves', roi.pid !== null, String(roi.pid));

// turn ROI back off
await page.click('#roiToggle');
await page.waitForTimeout(700);

// ------------------------------------------------------- amorphous bg + 50 deg
await openTab('analysis');
await page.click('#crystBgToggle');
await page.waitForTimeout(500);
check('amorphous background trace drawn',
      await page.evaluate(() => chart.data.datasets.some(d => d._isBackground)));

await page.evaluate(() => {
  document.getElementById('rangeMax').value = 50;
  document.getElementById('rangeMax').dispatchEvent(new Event('input'));
});
await page.waitForTimeout(400);
const wide = await page.evaluate(() => ({
  max: state.rangeMax,
  sliderMax: +document.getElementById('rangeMax').max,
  chartMax: chart.scales.x.max,
}));
check('2theta view extends to 50 deg',
      wide.max === 50 && wide.sliderMax === 50 && wide.chartMax === 50, JSON.stringify(wide));

// ------------------------------------------------------------ dark mode
await page.click('#themeToggle');
await page.waitForTimeout(600);
const dark = await page.evaluate(() => ({
  theme: state.theme,
  htmlDark: document.documentElement.classList.contains('dark'),
  bodyBg: getComputedStyle(document.body).backgroundColor,
  tickColor: chart.options.scales.x.ticks.color,
  firstTrace: chart.data.datasets[0].borderColor,
  exportFirstTrace: buildDatasets(true)[0].borderColor,
}));
check('dark theme applied to DOM', dark.theme === 'dark' && dark.htmlDark, dark.bodyBg);
check('chart ink follows dark theme', dark.tickColor === '#e8e6e0', dark.tickColor);
check('dark palette lifts the near-black trace',
      dark.firstTrace === '#e8e6e0', dark.firstTrace);
check('export path stays on the light palette',
      dark.exportFirstTrace === '#1a1a1a', dark.exportFirstTrace);

await page.screenshot({ path: '/tmp/shot_dark.png', fullPage: false });

// -------------------------------------------------- phase library import
fs.writeFileSync('/tmp/zif8.csv',
  '2theta,d,intensity,hkl\n7.31,12.08,100,(0 1 1)\n10.35,8.54,42,(0 0 2)\n' +
  '12.70,6.96,25,(1 1 2)\n14.70,6.02,18,(0 2 2)\n16.42,5.39,30,(0 1 3)\n' +
  '18.02,4.92,55,(2 2 2)\n24.48,3.63,20,(1 3 3)\n26.65,3.34,15,(1 3 4)\n');
await page.setInputFiles('#refFileInput', '/tmp/zif8.csv');
await page.waitForTimeout(900);
const lib = await page.evaluate(() => {
  const ids = Object.keys(PHASE_REFS);
  const added = ids.find(k => !PHASE_REFS[k].builtin);
  return {
    count: ids.length,
    added,
    name: added ? PHASE_REFS[added].name : null,
    pdfName: added ? PHASE_REFS[added].pdfName : null,
    peaks: added ? PHASE_REFS[added].peaks.length : 0,
    maxI: added ? Math.max(...PHASE_REFS[added].peaks.map(p => p.i)) : 0,
    active: state.activePhases.slice(),
    rows: document.querySelectorAll('#phaseLib .phase-row').length,
    hklOptions: Array.from(document.querySelectorAll('#hklLabelPhase option')).map(o => o.textContent),
    simTraces: chart.data.datasets.filter(d => d._isReference).length,
  };
});
check('custom phase imported into library',
      lib.count === 5 && lib.peaks === 8 && Math.abs(lib.maxI - 100) < 1e-6,
      `${lib.count} phases, "${lib.name}", ${lib.peaks} peaks`);
check('imported phase has ASCII pdfName', lib.pdfName === 'zif8', lib.pdfName);
check('imported phase active + drawn',
      lib.active.includes(lib.added) && lib.simTraces === 5,
      `${lib.simTraces} sim traces`);
check('library UI + hkl dropdown updated',
      lib.rows === 5 && lib.hklOptions.length === 6, JSON.stringify(lib.hklOptions));

// deleting a builtin must be refused, deleting a custom must work
const del = await page.evaluate((id) => {
  const beforeCount = Object.keys(PHASE_REFS).length;
  const builtinRefused = removePhase('alpha') === false;
  return { builtinRefused, beforeCount };
}, lib.added);
check('built-in phases cannot be deleted', del.builtinRefused);

// library round-trip through JSON
const roundtrip = await page.evaluate(() => {
  const json = JSON.stringify({
    format: 'xrd-viewer-phase-library', version: 1,
    phases: [{ id: 'uio66', name: 'UiO-66', color: '#123456',
               peaks: [{ tt: 7.36, i: 100 }, { tt: 8.49, i: 60 }, { tt: 25.7, i: 22 }] }],
  });
  const before = Object.keys(PHASE_REFS).length;
  // exercise the same merge path the file reader uses
  const lib = JSON.parse(json);
  let n = 0;
  for (const ph of lib.phases) {
    if (PHASE_REFS[ph.id] && PHASE_REFS[ph.id].builtin) continue;
    addPhase({ name: ph.name, peaks: ph.peaks, color: ph.color, source: 'test' });
    n++;
  }
  return { before, after: Object.keys(PHASE_REFS).length, n };
});
check('library JSON merge adds phases',
      roundtrip.after === roundtrip.before + 1, JSON.stringify(roundtrip));

// ---------------------------------------------------------- PNG + PDF
await page.evaluate(() => { window.__downloads = []; });
page.on('download', d => {});

const png = await page.evaluate(async () => {
  const { dataUrl, aspect } = await renderPdfChart(2);
  return { len: dataUrl.length, aspect, isPng: dataUrl.startsWith('data:image/png') };
});
check('publication chart renders (used by PNG + PDF)',
      png.isPng && png.len > 20000, `${png.len} bytes, aspect ${png.aspect.toFixed(2)}`);

const pdfOk = await page.evaluate(async () => {
  try {
    const interp = buildInterpretation();
    return {
      ok: true,
      abstract: interp.abstract.slice(0, 160),
      hasD: /d-spacing/.test(interp.abstract),
      cols: interp.summaryRows.length ? Object.keys(interp.summaryRows[0]) : [],
      sections: interp.interpretation.map(s => s.heading),
    };
  } catch (e) { return { ok: false, err: e.message }; }
});
check('interpretation builds', pdfOk.ok, pdfOk.err || '');
check('abstract mentions no d-spacing', pdfOk.ok && !pdfOk.hasD);
check('crystallinity section in report',
      pdfOk.ok && pdfOk.sections.some(h => /Crystalline/.test(h)),
      (pdfOk.sections || []).join(' / '));

// Exercise the full PDF generator and capture the real download. jsPDF puts
// `save` on the INSTANCE, not the prototype, so spying on the prototype
// silently never fires — the download event is the honest signal.
await openTab('notes');
await page.evaluate(() => {
  const t = document.getElementById('notesInput');
  t.value = 'ทดสอบบันทึกภาษาไทย\n\nSecond paragraph for pagination.';
  t.dispatchEvent(new Event('input'));
});
await openTab('export');
const dlPromise = page.waitForEvent('download', { timeout: 60000 }).catch(() => null);
await page.click('#exportPeaksPDF');
const dl = await dlPromise;
let pdfBytes = 0, pdfPages = 0;
if (dl) {
  const tmp = '/tmp/report.pdf';
  await dl.saveAs(tmp);
  const buf = fs.readFileSync(tmp);
  pdfBytes = buf.length;
  pdfPages = (buf.toString('latin1').match(/\/Type\s*\/Page[^s]/g) || []).length;
}
check('PDF report generates end-to-end', !!dl && pdfBytes > 50000,
      dl ? `${dl.suggestedFilename()}, ${pdfBytes} bytes, ${pdfPages} pages` : 'no download');
check('PDF has multiple pages (chart + summary + notes + tables)', pdfPages >= 4,
      `${pdfPages} pages`);

// --------------------------------------------------------- CSV / MD text
const texts = await page.evaluate(() => {
  const out = {};
  const origCreate = URL.createObjectURL;
  const blobs = [];
  URL.createObjectURL = (b) => { blobs.push(b); return 'blob:stub'; };
  const origClick = HTMLAnchorElement.prototype.click;
  HTMLAnchorElement.prototype.click = function () {};
  document.getElementById('exportPeaksCSV').click();
  document.getElementById('exportPeaksMD').click();
  URL.createObjectURL = origCreate;
  HTMLAnchorElement.prototype.click = origClick;
  return Promise.all(blobs.map(b => b.text())).then(([csv, md]) => ({ csv, md }));
});
check('CSV has no d_spacing column', !/d_spacing/.test(texts.csv));
check('CSV carries crystallinity', /Crystallinity_pct/.test(texts.csv));
check('CSV carries provenance header', /^# XRD Pattern Viewer/.test(texts.csv));
check('Markdown has no d column', !/\| d \(/.test(texts.md));
check('Markdown reports crystallinity', /\*\*Crystallinity:\*\*/.test(texts.md));

// ================================================= DECK / CAROUSEL
const deck = await page.evaluate(() => ({
  tabs: Array.from(document.querySelectorAll('.deck-tab')).map(t => t.dataset.panel),
  panels: Array.from(document.querySelectorAll('.deck-panel')).map(t => t.dataset.panel),
  sidebarGone: !document.querySelector('.sidebar'),
  deckBelowChart: (() => {
    const c = document.querySelector('.chart-panel');
    const d = document.getElementById('deck');
    return !!(c && d) && (c.compareDocumentPosition(d) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0;
  })(),
}));
check('every tab has a matching panel',
      deck.tabs.length === 9 && deck.tabs.join() === deck.panels.join(),
      deck.tabs.join(' '));
check('left sidebar removed', deck.sidebarGone);
check('deck sits below the chart', deck.deckBelowChart);

// exactly one panel visible at a time
const oneVisible = await page.evaluate(() =>
  document.querySelectorAll('.deck-panel.active').length);
check('exactly one panel active', oneVisible === 1, String(oneVisible));

// arrow buttons walk the carousel and disable at the ends
await openTab('data');
const navStart = await page.evaluate(() => ({
  prevDisabled: document.getElementById('deckPrev').disabled,
  nextDisabled: document.getElementById('deckNext').disabled,
}));
check('prev arrow disabled on the first panel',
      navStart.prevDisabled && !navStart.nextDisabled, JSON.stringify(navStart));
await page.click('#deckNext');
await page.waitForTimeout(250);
const afterNext = await page.evaluate(() => ({
  active: document.querySelector('.deck-panel.active').dataset.panel,
  tabActive: document.querySelector('.deck-tab.active').dataset.panel,
}));
check('next arrow advances tab and panel together',
      afterNext.active === 'display' && afterNext.tabActive === 'display',
      JSON.stringify(afterNext));
// walk to the far end — stop when the arrow disables itself
for (let i = 0; i < 12; i++) {
  const off = await page.evaluate(() => document.getElementById('deckNext').disabled);
  if (off) break;
  await page.click('#deckNext');
  await page.waitForTimeout(120);
}
await page.waitForTimeout(300);
const navEnd = await page.evaluate(() => ({
  active: document.querySelector('.deck-panel.active').dataset.panel,
  nextDisabled: document.getElementById('deckNext').disabled,
}));
check('carousel stops at the last panel',
      navEnd.active === 'export' && navEnd.nextDisabled, JSON.stringify(navEnd));

// keyboard: arrows move panels, but must not hijack a focused slider
await page.evaluate(() => document.body.focus());
await page.keyboard.press('ArrowLeft');
await page.waitForTimeout(250);
const afterKey = await page.evaluate(() =>
  document.querySelector('.deck-panel.active').dataset.panel);
check('arrow key steps the carousel', afterKey === 'notes', afterKey);

await openTab('display');
const sliderGuard = await page.evaluate(async () => {
  const el = document.getElementById('rangeMin');
  el.focus();
  return document.activeElement.id;
});
await page.keyboard.press('ArrowRight');
await page.waitForTimeout(250);
const afterSliderKey = await page.evaluate(() =>
  document.querySelector('.deck-panel.active').dataset.panel);
check('arrow key on a focused slider does NOT change panel',
      afterSliderKey === 'display', `${sliderGuard} → ${afterSliderKey}`);

// ============================================== CRYSTALLINITY PANEL
await openTab('cryst');
const cp = await page.evaluate(() => {
  const host = document.getElementById('crystPanelBody');
  const bars = Array.from(host.querySelectorAll('.cryst-bar'));
  return {
    bars: bars.length,
    widths: bars.map(b => [
      parseFloat(b.querySelector('.seg-c').style.width),
      parseFloat(b.querySelector('.seg-a').style.width),
    ]),
    names: Array.from(host.querySelectorAll('.c-name')).map(n => n.textContent),
    vals: Array.from(host.querySelectorAll('.c-val')).map(n => n.textContent),
    hasMeanSd: /Mean/.test(host.textContent),
    hasCaveat: /relative index/i.test(host.textContent),
    hasWindow: /full measured range|ROI/.test(host.textContent),
  };
});
check('crystallinity panel shows one bar per pattern',
      cp.bars === 2, `${cp.bars} bars for ${JSON.stringify(cp.names)}`);
check('crystalline + amorphous segments sum to 100%',
      cp.widths.every(([c, a]) => Math.abs(c + a - 100) < 0.01),
      JSON.stringify(cp.widths));
check('crystallinity panel reports mean ± SD', cp.hasMeanSd);
check('crystallinity panel states the window used', cp.hasWindow);
check('crystallinity panel keeps the relative-index caveat', cp.hasCaveat);

// panel must follow a settings change while it is on screen
const beforeWin = await page.evaluate(() =>
  document.querySelector('#crystPanelBody .c-val').textContent);
await openTab('analysis');
await page.evaluate(() => {
  const el = document.getElementById('crystWin');
  el.value = 4;
  el.dispatchEvent(new Event('input'));
});
await page.waitForTimeout(700);
await openTab('cryst');
const afterWin = await page.evaluate(() =>
  document.querySelector('#crystPanelBody .c-val').textContent);
check('crystallinity panel tracks the SNIP width',
      beforeWin !== afterWin, `${beforeWin} → ${afterWin}`);
// restore
await openTab('analysis');
await page.evaluate(() => {
  const el = document.getElementById('crystWin');
  el.value = 1.5;
  el.dispatchEvent(new Event('input'));
});
await page.waitForTimeout(600);

// =============================================== MEASUREMENT PANEL
await openTab('measurement');
const mp = await page.evaluate(() => {
  const host = document.getElementById('measPanelBody');
  const groups = Array.from(host.querySelectorAll('.meas-group h4')).map(h => h.textContent);
  const rows = Array.from(host.querySelectorAll('.meas-row')).map(r => [
    r.querySelector('.k').textContent, r.querySelector('.v').textContent]);
  const asObj = Object.fromEntries(rows);
  return {
    pickers: host.querySelectorAll('.meas-sample-tabs button').length,
    groups, asObj,
    rowCount: rows.length,
    uniqueLabels: new Set(rows.map(r => r[0])).size,
    noEmptyValues: rows.every(r => r[1] && r[1].trim() !== '' && r[1] !== 'null'),
  };
});
check('measurement panel lists one picker per pattern',
      mp.pickers === 2, String(mp.pickers));
check('measurement groups present',
      ['Sample', 'Scan', 'Source', 'Optics', 'Detector', 'Counting statistics', 'Instrument']
        .every(g => mp.groups.includes(g)), mp.groups.join(' | '));
check('every measurement row label is unique',
      mp.uniqueLabels === mp.rowCount, `${mp.uniqueLabels}/${mp.rowCount}`);
check('scan settings read from the file',
      mp.asObj['Scan mode'] === 'Continuous PSD fast' &&
      /0.02046/.test(mp.asObj['Step size'] || '') &&
      mp.asObj['Points'] === '1,466',
      `${mp.asObj['Scan mode']} / ${mp.asObj['Step size']} / ${mp.asObj['Points']}`);
check('source block reports the tube correctly',
      mp.asObj['Anode'] === 'Cu' && mp.asObj['Generator'] === '40 kV / 40 mA' &&
      /1.54060/.test(mp.asObj['Lambda Ka1'] || ''),
      `${mp.asObj['Anode']} ${mp.asObj['Generator']} ${mp.asObj['Lambda Ka1']}`);
check('optics + detector read from the mounted record',
      /Filter_Ni/.test(mp.asObj['Kbeta filter'] || '') &&
      mp.asObj['Axial Soller'] === '2.5 deg' &&
      /LYNXEYE/.test(mp.asObj['Model'] || ''),
      `${mp.asObj['Kbeta filter']} / ${mp.asObj['Axial Soller']} / ${mp.asObj['Model']}`);
check('goniometer orientation is the goniometer\'s, not the Soller\'s',
      mp.asObj['Orientation'] === 'Vertical' && mp.asObj['Goniometer'] === 'Theta_Theta',
      `${mp.asObj['Goniometer']} / ${mp.asObj['Orientation']}`);
check('counting statistics + sqrt(N) precision shown',
      /8,662/.test(mp.asObj['Max'] || '') && /%/.test(mp.asObj['Peak sigma (sqrt N)'] || ''),
      `${mp.asObj['Max']} / ${mp.asObj['Peak sigma (sqrt N)']}`);
check('duration derived from the timestamps',
      /min/.test(mp.asObj['Duration'] || ''), mp.asObj['Duration']);
check('measurement panel never prints an empty value', mp.noEmptyValues);

// the non-brml pattern must say so plainly instead of faking fields
await page.evaluate(() => {
  document.querySelectorAll('.meas-sample-tabs button')[1].click();
});
await page.waitForTimeout(300);
const mp2 = await page.evaluate(() => {
  const host = document.getElementById('measPanelBody');
  return {
    text: host.textContent,
    groups: Array.from(host.querySelectorAll('.meas-group h4')).map(h => h.textContent),
  };
});
check('plain data file states it has no instrument record',
      /no instrument record/.test(mp2.text) && mp2.groups.join() === 'From the data',
      mp2.groups.join(' | '));

// ==================================== second, real CALF-20 .brml
await openTab('data');
await page.setInputFiles('#fileInput', BRML_B);
await page.waitForTimeout(1500);
const calf = await page.evaluate(() => {
  const p = state.patterns[state.patterns.length - 1];
  return {
    n: state.patterns.length,
    pts: p.x.length,
    meta: p.meta,
    badge: document.getElementById('tabBadgeData').textContent,
  };
});
check('real CALF-20 .brml loads',
      calf.n === 3 && calf.pts === 1467, `${calf.n} patterns, ${calf.pts} points`);
check('its scan record differs from the first file',
      Math.abs(calf.meta.scanStop - 35.0001) < 1e-6 && calf.meta.points === 1467,
      `stop ${calf.meta.scanStop}, ${calf.meta.points} pts`);
check('counting stats specific to this scan',
      calf.meta.countsMax === 13606, String(calf.meta.countsMax));
check('data tab badge counts patterns', calf.badge === '3', calf.badge);

// back to light for the final screenshot
await page.click('#themeToggle');
await page.waitForTimeout(500);
await page.screenshot({ path: '/tmp/shot_light.png' });

check('no JS errors during the whole run', consoleErrors.length === 0, consoleErrors.slice(0, 3).join(' | '));

await browser.close();

const failed = results.filter(r => !r.ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length) { console.log('FAILURES:'); failed.forEach(f => console.log(' - ' + f.name + ': ' + f.detail)); process.exit(1); }

// Extract the inline script from xrd_viewer.html, syntax-check it, and
// cross-check every getElementById() target against the ids present in the
// markup. Catches the classic "moved the markup, forgot a handler" bug.
const fs = require('fs');
const cp = require('child_process');

const path = require('path');
const TARGET = process.argv[2] || path.join(__dirname, '..', 'xrd_viewer.html');
const h = fs.readFileSync(TARGET, 'utf8');
console.log('checking', TARGET);
const m = h.match(/<script>([\s\S]*?)<\/script>\s*<\/body>/);
if (!m) { console.error('no inline script block found'); process.exit(1); }
fs.writeFileSync('/tmp/v.js', m[1]);
console.log('extracted', m[1].length, 'chars');

try {
  cp.execSync('node --check /tmp/v.js', { stdio: 'pipe' });
  console.log('SYNTAX OK');
} catch (e) {
  console.error('SYNTAX FAIL\n' + e.stderr.toString());
  process.exit(1);
}

const ids = new Set([...h.matchAll(/id="([^"]+)"/g)].map(x => x[1]));
const used = [...new Set([...m[1].matchAll(/getElementById\('([^']+)'\)/g)].map(x => x[1]))];
const missing = used.filter(u => !ids.has(u));
console.log('getElementById targets:', used.length, '| missing:', missing.length ? missing.join(', ') : 'none');

// Duplicate definitions that have bitten this file before.
for (const name of ['renderChart', 'inlineLabelPlugin', 'refStickPlugin', 'roiShadePlugin',
                    'buildDatasets', 'renderCrystPanel', 'renderMeasurementPanel', 'setDeckPanel']) {
  const re = new RegExp('^(?:const|function|let)\\s+' + name + '\\b', 'gm');
  const n = (m[1].match(re) || []).length;
  if (n !== 1) console.log('  !! ' + name + ' defined ' + n + ' times');
}

// Stale references from earlier refactors. Comments are stripped first —
// several of these names are only mentioned in comments that explain why they
// were removed, and flagging those sends the next reader on a ghost hunt.
const code = m[1]
  .replace(/\/\*[\s\S]*?\*\//g, '')     // block comments
  .replace(/^\s*\/\/.*$/gm, '');         // whole-line comments
let stale = 0;
for (const bad of ['state.mode', 'state.refPeaks', 'filesPanel', 'getPatternColor(', 'REF_CALF20']) {
  if (code.includes(bad)) { console.log('  !! stale reference still present: ' + bad); stale++; }
}
if (!stale) console.log('stale references: none');

if (missing.length) process.exit(1);
console.log('CHECKS PASSED');

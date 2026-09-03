#!/usr/bin/env python3
"""
============================================================================
 web_dashboard.py  v2 — ESP32 Water Monitor : Web dashboard (Plotly)
============================================================================
 อ่าน CSV ที่ logger.py เขียน (ไม่แตะ serial) แล้วเสิร์ฟหน้าเว็บ:
   - การ์ดค่าปัจจุบัน 6 ค่า + สถานะ live
   - กราฟ EC / pH แบบ Plotly: ลากกรอบซูม, scroll ซูม, pan, ดับเบิลคลิกรีเซ็ต
   - ย้อนดูข้อมูลเก่าได้ทุกวันที่มีไฟล์ (เลือกช่วง จาก-ถึง ได้เอง)
   - ความละเอียดปรับอัตโนมัติ: ซูมช่วงแคบ = ได้ข้อมูล raw จริง
   - เปิดจากมือถือผ่าน Tailscale: http://100.84.225.79:8080

 รัน:  python3 web_dashboard.py            (port 8080)
 หรือเป็น service ผ่าน water-web.service
============================================================================
"""

import argparse
import csv
import glob
import io
import os
import re
import tempfile
from datetime import datetime, timedelta

from flask import Flask, jsonify, request, Response, send_file, send_file

# สำหรับ export PDF (ต้องมี report.py ในโฟลเดอร์เดียวกัน)
try:
    import report as report_module
    HAVE_REPORT = True
except Exception:
    HAVE_REPORT = False

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "water_data")
MAX_POINTS = 4000          # จุดสูงสุดที่ส่งให้กราฟต่อครั้ง (ซูมแคบ -> ได้ raw)

app = Flask(__name__)


# ============================================================================
#  อ่านข้อมูล
# ============================================================================
def list_dates():
    """วันที่ทั้งหมดที่มีไฟล์ log"""
    out = []
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "water_log_*.csv"))):
        m = re.search(r"water_log_(\d{4}-\d{2}-\d{2})\.csv$", f)
        if m:
            out.append(m.group(1))
    return out


def files_in_range(start, end):
    """เลือกเฉพาะไฟล์วันที่คร่อมช่วง [start,end]"""
    files = []
    for d in list_dates():
        day = datetime.strptime(d, "%Y-%m-%d")
        if day.date() < start.date() or day.date() > end.date():
            continue
        files.append(os.path.join(DATA_DIR, f"water_log_{d}.csv"))
    return files


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_range(start, end):
    """อ่านแถวในช่วงเวลา คืน list ของ dict (เรียงเวลา)"""
    rows = []
    for f in files_in_range(start, end):
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    ts = r.get("timestamp", "")
                    try:
                        t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    if t < start or t > end:
                        continue
                    rows.append({
                        "t": ts,
                        "ec": _f(r.get("EC_uScm")),
                        "tw": _f(r.get("Tw_C")),
                        "sal": _f(r.get("Salinity_ppm")),
                        "tds": _f(r.get("TDS_ppm")),
                        "ph": _f(r.get("pH")),
                        "mv": _f(r.get("pH_mV")),
                        "ok": r.get("rs485_ok", "0"),
                    })
        except Exception:
            continue
    rows.sort(key=lambda r: r["t"])
    return rows


def downsample(rows, max_pts=MAX_POINTS):
    """stride downsampling — ช่วงแคบ (แถวน้อย) = ได้ raw ครบ"""
    step = max(1, len(rows) // max_pts)
    return rows[::step]


# ============================================================================
#  API
# ============================================================================
@app.route("/api/now")
def api_now():
    end = datetime.now()
    rows = read_range(end - timedelta(minutes=5), end)
    if not rows:
        return jsonify({"ok": False, "msg": "no data"})
    last = rows[-1]
    try:
        age = (datetime.now() - datetime.strptime(last["t"], "%Y-%m-%d %H:%M:%S")).total_seconds()
    except Exception:
        age = None
    last["age_s"] = age
    last["ok"] = (last["ok"] == "1") and (age is not None and age < 15)
    return jsonify(last)


@app.route("/api/dates")
def api_dates():
    return jsonify({"dates": list_dates()})


@app.route("/api/history")
def api_history():
    """
    ช่วงเวลา: ?start=YYYY-MM-DDTHH:MM&end=... (datetime-local)
    หรือย่อ:  ?minutes=60  (นับถอยจากตอนนี้)
    """
    now = datetime.now()
    if request.args.get("start") and request.args.get("end"):
        try:
            start = datetime.fromisoformat(request.args["start"])
            end = datetime.fromisoformat(request.args["end"])
        except ValueError:
            return jsonify({"error": "bad datetime"}), 400
    else:
        minutes = int(request.args.get("minutes", 60))
        start, end = now - timedelta(minutes=minutes), now

    if end <= start:
        return jsonify({"error": "end <= start"}), 400

    rows = downsample(read_range(start, end))
    return jsonify({
        "t":   [r["t"] for r in rows],
        "ec":  [r["ec"] for r in rows],
        "ph":  [r["ph"] for r in rows],
        "tw":  [r["tw"] for r in rows],
        "n":   len(rows),
        "raw": len(rows) > 0 and (len(rows) < MAX_POINTS),   # true = raw ครบไม่ถูกย่อ
    })


def _parse_range():
    """อ่านช่วงเวลาจาก query (?start&end หรือ ?minutes) คืน (start,end) หรือ None"""
    now = datetime.now()
    if request.args.get("start") and request.args.get("end"):
        try:
            return (datetime.fromisoformat(request.args["start"]),
                    datetime.fromisoformat(request.args["end"]))
        except ValueError:
            return None
    minutes = int(request.args.get("minutes", 60))
    return (now - timedelta(minutes=minutes), now)


@app.route("/api/export.csv")
def export_csv():
    """ดาวน์โหลด CSV (raw ครบทุกแถว) ตามช่วงที่เลือก"""
    rng = _parse_range()
    if not rng:
        return jsonify({"error": "bad datetime"}), 400
    start, end = rng
    rows = read_range(start, end)          # raw ทั้งหมด ไม่ downsample
    if not rows:
        return jsonify({"error": "no data in range"}), 404

    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp", "EC_uScm", "Tw_C", "Salinity_ppm",
                "TDS_ppm", "pH", "pH_mV", "rs485_ok"])
    for r in rows:
        w.writerow([r["t"], r["ec"], r["tw"], r["sal"], r["tds"],
                    r["ph"], r["mv"], r["ok"]])
    fname = f"water_{start:%Y%m%d_%H%M}-{end:%Y%m%d_%H%M}.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.route("/api/export.pdf")
def export_pdf():
    """สร้าง scientific PDF ของช่วงที่เลือก (ใช้ report.py) แล้วส่งให้ดาวน์โหลด"""
    if not HAVE_REPORT:
        return jsonify({"error": "report.py not found"}), 500
    rng = _parse_range()
    if not rng:
        return jsonify({"error": "bad datetime"}), 400
    start, end = rng

    rep_dir = os.path.join(DATA_DIR, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    base = os.path.join(rep_dir, f"export_{start:%Y%m%d_%H%M}-{end:%Y%m%d_%H%M}")
    try:
        res = report_module.generate_report(
            inputs=[os.path.join(DATA_DIR, "water_log_*.csv")],
            output=base, since=start, until=end,
            auto_open=False, want_excel=False,
            meta={"sample": request.args.get("sample", "-")})
    except Exception as e:
        return jsonify({"error": f"report failed: {e}"}), 500
    if not res:
        return jsonify({"error": "no data in range"}), 404
    return send_file(res[0], as_attachment=True)


@app.route("/reports/<path:fname>")
def serve_report(fname):
    """เสิร์ฟไฟล์รายงานอัตโนมัติ (ที่ monitor.py สร้าง) ให้กดจากลิงก์ใน LINE ได้"""
    rep_dir = os.path.join(DATA_DIR, "reports")
    path = os.path.normpath(os.path.join(rep_dir, fname))
    if not path.startswith(rep_dir) or not os.path.isfile(path):
        return jsonify({"error": "not found"}), 404
    return send_file(path)



# ============================================================================
#  หน้าเว็บ — Plotly (zoom/pan/รีเซ็ต ในตัว)
# ============================================================================
PAGE = """<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Water Monitor</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" charset="utf-8"></script>
<style>
 :root{--bg:#0f1720;--card:#1a2430;--txt:#e6edf3;--mut:#8b98a5;--line:#2b3948;
       --ec:#4d9fff;--ph:#ff6b6b;--ok:#2ecc71;--bad:#e74c3c}
 *{box-sizing:border-box;margin:0;padding:0}
 body{background:var(--bg);color:var(--txt);font-family:'Segoe UI',system-ui,sans-serif;padding:14px}
 h1{font-size:1.1rem;font-weight:600}
 .sub{color:var(--mut);font-size:.8rem;margin:2px 0 12px}
 .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:9px;margin-bottom:14px}
 .card{background:var(--card);border-radius:10px;padding:10px 13px}
 .card .lbl{color:var(--mut);font-size:.72rem;text-transform:uppercase;letter-spacing:.04em}
 .card .val{font-size:1.5rem;font-weight:600;margin-top:2px}
 .card .unit{font-size:.75rem;color:var(--mut);margin-left:3px}
 .bar{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin-bottom:12px}
 .bar button,.bar input{background:var(--card);color:var(--txt);border:1px solid var(--line);
   border-radius:7px;padding:6px 11px;font-size:.8rem;cursor:pointer}
 .bar button.on{border-color:var(--ec);color:var(--ec)}
 .bar input{color-scheme:dark}
 .bar .lab{color:var(--mut);font-size:.78rem}
 .chartbox{background:var(--card);border-radius:10px;padding:8px;margin-bottom:12px}
 .chartbox h2{font-size:.85rem;color:var(--mut);padding:4px 8px 0}
 .meta{color:var(--mut);font-size:.72rem;padding:0 8px 4px}
 .plot{width:100%;height:330px}
 .hint{color:var(--mut);font-size:.72rem;margin:-6px 0 12px}
</style>
</head>
<body>
<h1><span id="dot" class="dot" style="background:var(--mut)"></span>ESP32 Water Monitor</h1>
<div class="sub" id="sub">connecting…</div>

<div class="grid">
 <div class="card"><div class="lbl">EC</div><div class="val"><span id="ec">--</span><span class="unit">µS/cm</span></div></div>
 <div class="card"><div class="lbl">pH</div><div class="val"><span id="ph">--</span></div></div>
 <div class="card"><div class="lbl">Water T</div><div class="val"><span id="tw">--</span><span class="unit">°C</span></div></div>
 <div class="card"><div class="lbl">Salinity</div><div class="val"><span id="sal">--</span><span class="unit">ppm</span></div></div>
 <div class="card"><div class="lbl">TDS</div><div class="val"><span id="tds">--</span><span class="unit">ppm</span></div></div>
 <div class="card"><div class="lbl">pH raw</div><div class="val"><span id="mv">--</span><span class="unit">mV</span></div></div>
</div>

<div class="bar">
 <span class="lab">ด่วน:</span>
 <button data-m="10">10 นาที</button>
 <button data-m="60" class="on">1 ชม.</button>
 <button data-m="360">6 ชม.</button>
 <button data-m="1440">24 ชม.</button>
 <button id="allBtn">ทั้งหมด</button>
 <span class="lab">| กำหนดเอง:</span>
 <input type="datetime-local" id="tFrom" step="1">
 <span class="lab">ถึง</span>
 <input type="datetime-local" id="tTo" step="1">
 <button id="applyBtn">ดูช่วงนี้</button>
 <button id="liveBtn" class="on">● live</button>
 <span class="lab">| export:</span>
 <button id="csvBtn">⬇ CSV</button>
 <button id="pdfBtn">⬇ PDF</button>
</div>
<div class="hint">ซูม: ลากกรอบบนกราฟ / scroll เมาส์ • เลื่อน: ลากแกน • รีเซ็ต: ดับเบิลคลิก • ซูมแคบพอจะเห็นข้อมูล raw ทุกจุด</div>

<div class="chartbox"><h2>EC (µS/cm)</h2><div id="pEC" class="plot"></div><div class="meta" id="mEC"></div></div>
<div class="chartbox"><h2>pH</h2><div id="pPH" class="plot"></div><div class="meta" id="mPH"></div></div>

<script>
let minutes = 60;          // โหมดด่วน (null = ช่วงกำหนดเอง)
let live = true;           // auto-refresh กราฟ
const fmt=(v,d=1)=>v==null?'--':Number(v).toFixed(d);

const LAYOUT = {
  paper_bgcolor:'#1a2430', plot_bgcolor:'#1a2430',
  font:{color:'#8b98a5',size:11}, margin:{l:48,r:12,t:8,b:40},
  xaxis:{gridcolor:'#223041',linecolor:'#2b3948'},
  yaxis:{gridcolor:'#223041',linecolor:'#2b3948'},
  dragmode:'zoom', hovermode:'x unified', showlegend:false, autosize:true};
const CONFIG = {responsive:true, scrollZoom:true, displaylogo:false,
  modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d']};

Plotly.newPlot('pEC',[{x:[],y:[],type:'scattergl',mode:'lines',line:{color:'#4d9fff',width:1.4}}],LAYOUT,CONFIG);
Plotly.newPlot('pPH',[{x:[],y:[],type:'scattergl',mode:'lines',line:{color:'#ff6b6b',width:1.4}}],LAYOUT,CONFIG);

// ผู้ใช้ซูมเอง -> หยุด live เพื่อไม่ให้กราฟเด้งรีเซ็ต
for(const id of ['pEC','pPH'])
  document.getElementById(id).on('plotly_relayout',ev=>{
    if(ev['xaxis.range[0]']||ev['xaxis.range']) setLive(false);
  });

function setLive(v){
  live=v;
  const b=document.getElementById('liveBtn');
  b.classList.toggle('on',v); b.textContent=v?'● live':'○ live';
}

async function tickNow(){
  try{
    const d=await (await fetch('/api/now')).json();
    for(const k of ['ec','ph','tw','sal','tds','mv'])
      document.getElementById(k).textContent=fmt(d[k],k==='ph'?2:(k==='sal'||k==='tds'||k==='mv'?0:1));
    const ok=d.ok===true;
    document.getElementById('dot').style.background=ok?'var(--ok)':'var(--bad)';
    document.getElementById('sub').textContent=ok
      ?`live • อัปเดต ${d.t} (${Math.round(d.age_s)}s ago)`
      :(d.t?`ข้อมูลล่าสุด ${d.t} — logger อาจหยุด/เซนเซอร์หลุด`:'ยังไม่มีข้อมูล');
  }catch(e){document.getElementById('sub').textContent='เชื่อมต่อ server ไม่ได้';}
}

function isoLocal(dt){ // datetime -> "YYYY-MM-DDTHH:MM:SS" (เวลาท้องถิ่น)
  const p=n=>String(n).padStart(2,'0');
  return `${dt.getFullYear()}-${p(dt.getMonth()+1)}-${p(dt.getDate())}T${p(dt.getHours())}:${p(dt.getMinutes())}:${p(dt.getSeconds())}`;
}

async function loadHist(){
  let url;
  if(minutes!=null) url='/api/history?minutes='+minutes;
  else{
    const f=document.getElementById('tFrom').value, t=document.getElementById('tTo').value;
    if(!f||!t) return;
    url=`/api/history?start=${f}&end=${t}`;
  }
  try{
    const d=await (await fetch(url)).json();
    if(d.error) return;
    Plotly.react('pEC',[{x:d.t,y:d.ec,type:'scattergl',mode:'lines',line:{color:'#4d9fff',width:1.4}}],LAYOUT,CONFIG);
    Plotly.react('pPH',[{x:d.t,y:d.ph,type:'scattergl',mode:'lines',line:{color:'#ff6b6b',width:1.4}}],LAYOUT,CONFIG);
    const info=`${d.n} จุด ${d.raw?'(raw ครบทุกจุด)':'(ย่อจากข้อมูลจริง — ซูม/เลือกช่วงแคบลงเพื่อดู raw)'}`;
    document.getElementById('mEC').textContent=info;
    document.getElementById('mPH').textContent=info;
  }catch(e){}
}

// ปุ่มช่วงด่วน
document.querySelectorAll('.bar button[data-m]').forEach(b=>{
  b.onclick=()=>{
    minutes=+b.dataset.m;
    document.querySelectorAll('.bar button[data-m],#allBtn').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); setLive(true); loadHist();
  };
});
// ทั้งหมด: ตั้ง from/to ครอบทุกวันที่มีข้อมูล
document.getElementById('allBtn').onclick=async()=>{
  const d=await (await fetch('/api/dates')).json();
  if(!d.dates.length) return;
  document.getElementById('tFrom').value=d.dates[0]+'T00:00:00';
  document.getElementById('tTo').value=d.dates[d.dates.length-1]+'T23:59:59';
  minutes=null;
  document.querySelectorAll('.bar button[data-m]').forEach(x=>x.classList.remove('on'));
  document.getElementById('allBtn').classList.add('on');
  setLive(false); loadHist();
};
// ช่วงกำหนดเอง
document.getElementById('applyBtn').onclick=()=>{
  minutes=null;
  document.querySelectorAll('.bar button[data-m],#allBtn').forEach(x=>x.classList.remove('on'));
  setLive(false); loadHist();
};
// live toggle
document.getElementById('liveBtn').onclick=()=>{ setLive(!live); if(live&&minutes!=null) loadHist(); };

// export CSV/PDF ตามช่วงที่กำลังดูอยู่ (โหมดด่วนหรือกำหนดเอง)
function exportURL(kind){
  if(minutes!=null) return `/api/export.${kind}?minutes=${minutes}`;
  const f=document.getElementById('tFrom').value, t=document.getElementById('tTo').value;
  if(!f||!t){ alert('เลือกช่วง จาก-ถึง ก่อน'); return null; }
  return `/api/export.${kind}?start=${f}&end=${t}`;
}
document.getElementById('csvBtn').onclick=()=>{ const u=exportURL('csv'); if(u) window.open(u,'_blank'); };
document.getElementById('pdfBtn').onclick=()=>{
  const u=exportURL('pdf'); if(!u) return;
  const b=document.getElementById('pdfBtn'); b.textContent='กำลังสร้าง…'; b.disabled=true;
  fetch(u).then(r=>{ if(!r.ok) throw 0; return r.blob(); })
    .then(bl=>{ const a=document.createElement('a');
      a.href=URL.createObjectURL(bl); a.download='water_report.pdf'; a.click(); })
    .catch(()=>alert('สร้าง PDF ไม่สำเร็จ (ช่วงนั้นอาจไม่มีข้อมูล)'))
    .finally(()=>{ b.textContent='⬇ PDF'; b.disabled=false; });
};

// ค่า default ช่อง from/to = ชั่วโมงล่าสุด
(()=>{const n=new Date();
  document.getElementById('tTo').value=isoLocal(n);
  document.getElementById('tFrom').value=isoLocal(new Date(n-3600e3));})();

tickNow(); loadHist();
setInterval(tickNow,2000);
setInterval(()=>{ if(live&&minutes!=null) loadHist(); },10000);
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Water Monitor web dashboard v2")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"[web] v2 (Plotly) เปิดที่ http://<IP>:{args.port}")
    print(f"[web] อ่านข้อมูลจาก: {DATA_DIR}")
    app.run(host=args.host, port=args.port, debug=False)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 check_install.py — ตรวจว่าเครื่องนี้พร้อมรันหรือยัง ก่อนจะไปเจอปัญหาหน้างาน
============================================================================
   python tools\\check_install.py

 ⚠️ ไม่เปิดพอร์ต ไม่เขียนอะไรลง legacy  ตรวจอย่างเดียว
    ปลอดภัยที่จะรันขณะ logger เดิมทำงานอยู่

 ⚠️ โฟลเดอร์นี้ไม่ได้อยู่ลำพัง
    มันเป็น "ผู้อ่าน" ของระบบเดิม  ถ้าไม่มี test_realtime รันอยู่บนเครื่องนั้น
    จะไม่มี CSV ให้อ่าน ไม่มี rec_status.json ให้ดูสถานะ และ hw_test ขั้น A
    จะตกทันที — ไม่ใช่เพราะแอปนี้พัง แต่เพราะไม่มีอะไรให้มันดู
============================================================================
"""

import io
import os
import sys
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

OK, WARN, FAIL = "ok  ", "เตือน", "ขาด "
rows = []
problems = []
warnings = []


def add(level, name, detail, fix=None):
    rows.append((level, name, detail))
    if level == FAIL:
        problems.append((name, detail, fix))
    elif level == WARN:
        warnings.append((name, detail, fix))


# ---------------------------------------------------------------- python
def check_python():
    v = sys.version_info
    txt = "%d.%d.%d  (%s)" % (v.major, v.minor, v.micro,
                              os.path.basename(sys.executable))
    if v < (3, 8):
        add(FAIL, "python", txt, "ต้อง 3.8 ขึ้นไป")
    else:
        add(OK, "python", txt)


def check_module(mod, why, fix, hard=True):
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "")
        add(OK, mod, ("%s  %s" % (ver, why)).strip())
        return True
    except Exception as e:                                  # noqa
        add(FAIL if hard else WARN, mod, "%s — %s" % (why, e), fix)
        return False


# ---------------------------------------------------------------- init
def init_config(legacy_root=None):
    """สร้าง config/app_config.json โดยเดา legacy จากโฟลเดอร์พี่น้อง

    ⚠️ ขั้นตอนที่คนพลาดบ่อยที่สุดตอนย้ายเครื่องคือแก้ path ในไฟล์ตัวอย่างไม่ครบ
       (มี 5 บรรทัดที่ต้องตรงกันหมด)  แก้ไม่ครบแล้วโปรแกรมจะเปิดได้ปกติ
       แต่แสดง OFFLINE ทุกอย่าง ซึ่งดูเหมือนของพัง ทั้งที่แค่ชี้ผิดที่
    """
    import json
    exp = os.path.join(ROOT, "config", "app_config.example.json")
    out = os.path.join(ROOT, "config", "app_config.json")
    root = legacy_root or os.path.join(os.path.dirname(ROOT), "test_realtime")
    root = root.replace("\\", "/").rstrip("/")
    if not os.path.isdir(root):
        print("  ไม่พบโฟลเดอร์ระบบเดิมที่ %s" % root)
        print("  ระบุเองด้วย:  python tools\\check_install.py --init <path ของ test_realtime>")
        return 1
    with io.open(exp, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg["legacy"].update({
        "enabled": True, "root": root,
        "data_dir": root + "/water_data",
        "rec_status": root + "/rec_status.json",
        "sessions": root + "/sessions_3ec.json",
        "reports_dir": root + "/reports",
        "read_only": True})
    if os.path.exists(out):
        bak = out + ".bak"
        os.replace(out, bak)
        print("  ของเดิมสำรองไว้ที่ %s" % os.path.basename(bak))
    with io.open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    print("  เขียน config/app_config.json แล้ว  legacy.root = %s" % root)
    return 0


# ---------------------------------------------------------------- config
def check_config():
    cfgp = os.path.join(ROOT, "config", "app_config.json")
    exp = os.path.join(ROOT, "config", "app_config.example.json")
    if not os.path.exists(cfgp):
        add(FAIL, "app_config.json", "ยังไม่มี",
            "copy config\\app_config.example.json config\\app_config.json "
            "แล้วแก้ legacy.* ให้ชี้ไปที่ test_realtime บนเครื่องนี้")
        return None
    try:
        from ecstation.core import config as CFG
        cfg = CFG.load(cfgp)
    except Exception as e:                                  # noqa
        add(FAIL, "app_config.json", "อ่านไม่ได้: %s" % e,
            "เทียบกับ %s" % os.path.basename(exp))
        return None
    add(OK, "app_config.json", "อ่านได้")
    return cfg


def check_data_dir(cfg):
    d = cfg["data_dir"]
    try:
        os.makedirs(d, exist_ok=True)
        t = os.path.join(d, ".writetest")
        with open(t, "w") as fh:
            fh.write("x")
        os.remove(t)
        add(OK, "data/ เขียนได้", d)
    except Exception as e:                                  # noqa
        add(FAIL, "data/ เขียนได้", "%s — %s" % (d, e),
            "ให้สิทธิ์เขียนโฟลเดอร์นี้ หรือย้ายโปรเจกต์ออกจาก Program Files")


def check_legacy(cfg):
    lg = cfg.get("legacy") or {}
    root = lg.get("root") or ""
    if not lg.get("enabled") or not root:
        add(FAIL, "legacy.root", "ยังไม่ได้ตั้ง",
            "แอปนี้เป็นผู้อ่านของระบบเดิม ถ้าไม่ชี้ไปที่ test_realtime "
            "จะไม่มีข้อมูลให้แสดงเลย")
        return
    if not os.path.isdir(root):
        add(FAIL, "legacy.root", "ไม่มีโฟลเดอร์นี้: %s" % root,
            "คัดลอก test_realtime มาที่เครื่องนี้ด้วย แล้วแก้ path ให้ตรง")
        return
    add(OK, "legacy.root", root)

    from ecstation.core import config as CFG
    if CFG.is_inside(cfg["data_dir"], root):
        add(FAIL, "data_dir ไม่ทับ legacy", cfg["data_dir"],
            "data_dir ต้องอยู่นอก test_realtime")
    else:
        add(OK, "data_dir ไม่ทับ legacy", "แยกกันแล้ว")

    # ---- ตัวรันของระบบเดิม ----
    for n in ("logger_3ec.py", "report_3ec.py"):
        p = os.path.join(root, n)
        add(OK if os.path.exists(p) else WARN, n,
            "พบ" if os.path.exists(p) else "ไม่พบใน legacy.root",
            "ตรวจว่าคัดลอก test_realtime มาครบ")

    # ---- โฟลเดอร์ข้อมูล ----
    dd = lg.get("data_dir") or ""
    if not os.path.isdir(dd):
        add(FAIL, "legacy.data_dir", "ไม่มี: %s" % dd,
            "ต้องชี้ไปที่ test_realtime\\water_data")
    else:
        import glob
        files = sorted(glob.glob(os.path.join(dd, "water_log_*.csv")))
        if not files:
            add(WARN, "ไฟล์ CSV", "ยังไม่มีสักไฟล์ใน %s" % dd,
                "ปกติถ้าเครื่องนี้เพิ่งติดตั้ง — จะมีเมื่อ logger เดิมเริ่มเก็บ")
        else:
            # ⚠️ ห้ามใช้ mtime ตัดสินความสดของข้อมูล
            #    โฟลเดอร์ที่เพิ่งคัดลอกมาจะมี mtime = เวลาที่คัดลอก ทุกไฟล์
            #    ดูเผิน ๆ เหมือนเพิ่งเก็บข้อมูลไปเมื่อครู่ ทั้งที่ข้างในเป็นของเก่า
            #    วันที่ในชื่อไฟล์เป็นของ logger เอง คัดลอกแล้วไม่เปลี่ยน
            newest = os.path.basename(files[-1])
            today = "water_log_{:%Y-%m-%d}.csv".format(datetime.now())
            fresh = (newest == today)
            add(OK if fresh else WARN, "ไฟล์ CSV ล่าสุด",
                "%s%s" % (newest, "" if fresh else "  (ไม่ใช่ของวันนี้)"),
                None if fresh else
                "ยังไม่มีไฟล์ของวันนี้ (%s) — logger เดิมยังไม่ได้เก็บข้อมูลวันนี้ "
                "เวลาแก้ไขไฟล์เชื่อไม่ได้ถ้าเพิ่งคัดลอกโฟลเดอร์มา" % today)

    # ---- logger เดิมกำลังรันอยู่หรือเปล่า ----
    # ⚠️ ตัดสินจาก "แถวล่าสุดใน CSV" ไม่ใช่จากอายุ rec_status.json
    #    logger เดิมเขียนไฟล์สถานะเฉพาะตอนเริ่ม/จบ session ไม่ได้เขียนเป็นจังหวะ
    from ecstation.core import legacy_read as LR
    counter = LR.CsvRowCounter(dd)
    age = counter.row_age_s()
    state = LR.pc_liveness(age)
    add(OK if state == LR.PC_ONLINE else WARN, "logger เดิมกำลังเก็บข้อมูล",
        "%s  (แถวล่าสุดเก่า %s)"
        % (LR.PC_TEXT[state], "— ยังไม่มีข้อมูล" if age is None
           else "%.0f s" % age),
        None if state == LR.PC_ONLINE
        else "เปิด run_logger.bat ใน test_realtime ค้างไว้ "
             "— ไม่งั้น hw_test ขั้น A จะตกและ soak จะไม่พิสูจน์อะไร")
    rp = lg.get("rec_status") or ""
    if os.path.exists(rp):
        _rec, rage = LR.read_rec_status(rp)
        sv = LR.session_view(_rec)
        add(OK, "session ที่เปิดอยู่",
            "mask=%d  (rec_status เก่า %s — ปกติ อัปเดตเฉพาะตอนเริ่ม/จบ session)"
            % (sv["mask"], "—" if rage is None else "%.0f s" % rage))


def check_ports():
    try:
        from ecstation.core import ports as P
        au = P.audit()
    except Exception as e:                                  # noqa
        add(FAIL, "ตรวจพอร์ต", str(e), "ติดตั้ง pyserial")
        return
    n = len(au["ports"])
    add(OK if n else WARN, "พอร์ตที่มองเห็น", "%d ตัว" % n,
        None if n else "เสียบสายทั้งสองบอร์ด แล้วตรวจไดรเวอร์ CH340/CH343")
    for role, label, fix in (
            (P.ROLE_CONTROL, "พอร์ต CONTROL",
             "logger เดิมเป็นคนใช้พอร์ตนี้ ไม่ใช่แอปนี้ "
             "— ถ้าไม่เจอ ให้ตรวจไดรเวอร์ CH340"),
            (P.ROLE_P4_BRIDGE, "พอร์ตจอ P4 (NDJSON)",
             "ตรวจว่าเสียบเส้น USB-Serial-JTAG ไม่ใช่เส้น CH343 ที่ใช้แฟลช")):
        info = au["roles"][role]
        if info["reason"] == P.PICK_OK:
            add(OK, label, info["device"])
        elif info["reason"] == P.PICK_AMBIGUOUS:
            add(WARN, label, "กำกวม %d ตัว: %s"
                % (len(info["candidates"]),
                   ", ".join(c["device"] for c in info["candidates"])),
                "ระบุเองด้วย --bridge-port COMn")
        else:
            add(WARN, label, "ไม่พบ", fix)


# ----------------------------------------------------------------
def main():
    if "--init" in sys.argv:
        i = sys.argv.index("--init")
        arg = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
        print("=" * 78)
        print(" สร้างไฟล์ตั้งค่าให้เครื่องนี้")
        print("=" * 78)
        rc = init_config(arg)
        if rc:
            return rc
        print()
    print("=" * 78)
    print(" ตรวจความพร้อมก่อนใช้งาน — %s" % ROOT)
    print("=" * 78)

    check_python()
    check_module("serial", "ต่อพอร์ตอนุกรม", "pip install pyserial")
    check_module("tkinter", "หน้าจอ viewer",
                 "ติดตั้ง Python ใหม่แล้วติ๊ก 'tcl/tk and IDLE'")
    check_module("matplotlib", "กราฟ", "pip install matplotlib")

    cfg = check_config()
    if cfg:
        check_data_dir(cfg)
        check_legacy(cfg)
    check_ports()

    print()
    for level, name, detail in rows:
        print("  [%s] %-26s %s" % (level, name, detail))

    print("\n" + "-" * 78)
    if problems:
        print(" ยังใช้งานไม่ได้ — ต้องแก้ %d ข้อ\n" % len(problems))
        for i, (name, detail, fix) in enumerate(problems, 1):
            print("  %d. %s — %s" % (i, name, detail))
            if fix:
                print("     วิธีแก้: %s" % fix)
    else:
        print(" ส่วนที่จำเป็นครบแล้ว")
    if warnings:
        print("\n เตือน %d ข้อ (รันได้ แต่การทดสอบอาจไม่ได้พิสูจน์อะไร):\n"
              % len(warnings))
        for i, (name, detail, fix) in enumerate(warnings, 1):
            print("  %d. %s — %s" % (i, name, detail))
            if fix:
                print("     %s" % fix)
    print("=" * 78)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

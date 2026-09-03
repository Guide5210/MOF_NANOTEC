#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""config.py — โหลดค่าตั้งของแอป พร้อมค่าเริ่มต้นที่ปลอดภัย

⚠️ ค่าเริ่มต้นทุกตัวต้องปลอดภัยเมื่อไม่มีไฟล์ config
   ผู้ใช้ที่เพิ่งคัดลอกโปรเจกต์มาต้องเปิดได้เลย ไม่ใช่เจอ traceback
"""

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT = {
    "legacy": {
        "enabled": False, "root": "", "data_dir": "", "rec_status": "",
        "sessions": "", "reports_dir": "", "read_only": True,
    },
    "bridge": {
        "enabled": True, "mode": "auto", "port": None,
        "vid_pid": "303A:1001", "state_interval_s": 3.0,
        # ⚠️ จอส่ง heartbeat ทุก 5 วินาที (pc_bridge.c:340 `t - last_hb > 5000`)
        #    ค่านี้จึงต้องทน heartbeat หายอย่างน้อย 3 ครั้ง = 15 s + margin
        #    ถ้าตั้ง 10 s จะทนได้แค่ 2 ครั้ง แล้วสะดุดครั้งเดียว (เขียน SD,
        #    เล่นเสียง, task ถูกแย่ง) ลิงก์จะเด้ง OFFLINE/ONLINE ไปมา
        #    ทุกครั้งที่เด้งจะลาก mask จาก FOLLOWING ไป STALE และอาจถึง CONFLICT
        #    ทั้งที่ไม่มีอะไรผิดจริง — ปรัชญาเดียวกับ missed_sample_limit=3
        "offline_after_s": 16.0, "max_line_bytes": 512,
        "raw_capture": False, "raw_capture_max_mb": 32,
    },
    "pc_liveness": {"online_within_s": 10.0, "stale_within_s": 30.0},
    "ui": {"window": None, "chart_mode": "split", "ec_decimals": 1},
    "data_dir": os.path.join(HERE, "data"),
}


def is_inside(child, parent):
    """child อยู่ใต้ parent หรือไม่ (เทียบแบบ normalize แล้ว ไม่ใช่เทียบสตริงดิบ)

    เทียบสตริงดิบพลาดได้ง่ายมากบน Windows: ตัวพิมพ์ใหญ่เล็กต่างกัน,
    / กับ \\ ปนกัน, มี .. คั่นกลาง — จึงต้องผ่าน realpath ก่อนเสมอ
    """
    if not child or not parent:
        return False
    try:
        c = os.path.realpath(os.path.abspath(child))
        p = os.path.realpath(os.path.abspath(parent))
    except Exception:
        return False
    if os.name == "nt":
        c, p = c.lower(), p.lower()
    return c == p or c.startswith(p.rstrip(os.sep) + os.sep)


def _guard_data_dir(cfg):
    """กัน data_dir ไปทับโฟลเดอร์ของ legacy

    ⚠️ นี่คือด่านสุดท้ายของกฎ "ห้ามเขียนอะไรลงโฟลเดอร์ legacy"
       config ที่พิมพ์ผิดบรรทัดเดียวสามารถทำให้ event log ไปตกในโฟลเดอร์
       ของระบบเดิมได้ — ตรงนั้นเป็นข้อมูลจริงของห้องแล็บ ไม่ใช่ที่ทดลอง
       เจอแล้วให้ถอยไปใช้ค่าเริ่มต้นและบอกให้ชัด ไม่ใช่เขียนต่อเงียบ ๆ
    """
    root = (cfg.get("legacy") or {}).get("root") or ""
    if root and is_inside(cfg["data_dir"], root):
        print("[config] data_dir อยู่ในโฟลเดอร์ของ legacy ({}) — "
              "ไม่อนุญาต ใช้ {} แทน".format(root, DEFAULT["data_dir"]))
        cfg["data_dir"] = os.path.abspath(DEFAULT["data_dir"])
    return cfg


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        elif k in out:
            out[k] = v
    return out


def load(path=None):
    path = path or os.path.join(HERE, "config", "app_config.json")
    user = {}
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                user = json.load(fh)
        except Exception as e:
            print("[config] อ่าน {} ไม่ได้ ({}) — ใช้ค่าเริ่มต้น".format(path, e))
    cfg = _merge(DEFAULT, user)
    # ⚠️ "./data" ต้องเทียบกับรากโปรเจกต์ ไม่ใช่ cwd
    #    os.path.abspath() เทียบกับ cwd — ถ้าใครรัน `python -m ecstation.app`
    #    จากโฟลเดอร์อื่น จะได้ data/ ไปโผล่ที่นั่นแทน แล้ว event log กับ
    #    diag ของรอบนั้นจะกระจัดกระจายหาไม่เจอ (ไฟล์ .bat ปลอดภัยอยู่แล้ว
    #    เพราะ cd /d "%~dp0" ก่อน แต่ห้ามพึ่งเรื่องนั้นอย่างเดียว)
    d = cfg["data_dir"]
    if not os.path.isabs(d):
        d = os.path.join(HERE, d)
    cfg["data_dir"] = os.path.abspath(d)
    cfg = _guard_data_dir(cfg)
    os.makedirs(cfg["data_dir"], exist_ok=True)
    return cfg

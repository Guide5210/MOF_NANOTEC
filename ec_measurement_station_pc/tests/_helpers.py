# -*- coding: utf-8 -*-
"""ตัวช่วยสำหรับเทสต์ — ทำให้เทสต์รันได้โดยไม่ต้องมีฮาร์ดแวร์และไม่ต้องมี pytest"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LEGACY_ROOT = os.environ.get("EC_LEGACY_ROOT", r"C:/MOF_NanoTec/test_realtime")


def tmp_cfg(tmpdir, legacy_root=None):
    return {
        "legacy": {"enabled": bool(legacy_root), "root": legacy_root or "",
                   "data_dir": os.path.join(legacy_root or "", "water_data"),
                   "rec_status": os.path.join(legacy_root or "", "rec_status.json"),
                   "sessions": "", "reports_dir": "", "read_only": True},
        "bridge": {"enabled": True, "mode": "mock", "port": None,
                   "vid_pid": "303A:1001", "state_interval_s": 0.2,
                   "offline_after_s": 1.0, "max_line_bytes": 512},
        "pc_liveness": {"online_within_s": 10.0, "stale_within_s": 30.0},
        "ui": {}, "data_dir": tmpdir,
    }


class FakePort(object):
    def __init__(self, device, description="", hwid=""):
        self.device, self.description, self.hwid = device, description, hwid

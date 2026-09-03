# -*- coding: utf-8 -*-
"""
test_no_legacy_mutation — ระบบใหม่ต้องไม่แตะโฟลเดอร์ของระบบเดิมเลย

ข้อกำหนด: viewer/bridge ใหม่ห้ามแก้ source, docs, config, CSV, session JSON,
รายงาน หรือโฟลเดอร์รายงานของ legacy  (logger เดิมยังเขียนไฟล์สถานะของตัวเอง
ได้ตามปกติ — นั่นเป็นงานของมัน ไม่ใช่ของเรา)

ตรวจสามชั้น เพราะชั้นเดียวไม่พอ
  A. static  — ไล่ AST ทุกไฟล์ ห้ามมีเส้นทางเขียนไฟล์นอกโมดูลที่ได้รับอนุญาต
  B. dynamic — สร้าง legacy จำลอง ถ่าย manifest (sha256+size+mtime) รัน bridge
               เต็มรูปแบบ แล้วถ่าย manifest ใหม่ ต้องเหมือนกันทุกไบต์
  C. guard   — config ที่พิมพ์ผิดจนชี้ data_dir เข้าไปใน legacy ต้องถูกปัดตก

⚠️ ชั้น A จำเป็นเพราะชั้น B พิสูจน์ได้แค่ "เส้นทางที่เทสต์เดินผ่าน"
   ชั้น B จำเป็นเพราะชั้น A อ่านแค่รูปทรงของโค้ด ไม่รู้ค่าที่รันจริง
"""
import ast
import hashlib
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime

from _helpers import ROOT, tmp_cfg  # noqa
from ecstation.bridge import p4_bridge as B
from ecstation.bridge.event_log import EventLog
from ecstation.bridge.pc_state import PcStateSource
from ecstation.core import config as CFG

# โมดูลที่ได้รับอนุญาตให้เขียนไฟล์ — และเขียนได้เฉพาะใน data/ ของโปรเจกต์นี้
#   event_log.py   -> data/events/*.jsonl   (หลักฐานการวัด)
#   lab_theme.py   -> data/ui_state.json    (ค่าตั้งหน้าจอ)
#   raw_capture.py -> data/raw/*.log        (วินิจฉัย ปิดโดยปริยาย มีเพดานขนาด)
#   snapshot.py    -> data/diag/*.json      (วินิจฉัย ตัดข้อมูลห้องแล็บออกแล้ว)
# ทั้งสี่ตัวมีเทสต์เฉพาะของตัวเองยืนยันปลายทางอีกชั้น
# (test_event_dedup · test_no_legacy_mutation · test_p1c_tools)
WRITERS_ALLOWED = {"event_log.py", "lab_theme.py", "raw_capture.py",
                   "snapshot.py"}

# ไลบรารีวาดภาพใช้ได้เฉพาะชั้น UI
#
# ⚠️ ห้ามให้ ecstation/core หรือ ecstation/bridge แตะ matplotlib เด็ดขาด
#    เพราะสองชั้นนั้นทำงานอยู่ในเธรดที่ต้องไม่หน่วง และการเผลอ import
#    ตัววาดภาพเข้าไปคือก้าวแรกของการ "เผลอสร้างรายงาน" ในที่ที่ไม่ควรมี
UI_ONLY_IMPORTS = ("matplotlib", "cycler", "tkinter")
# ไฟล์ที่เป็นเครื่องมือ/เทสต์ ไม่ได้ถูกโหลดตอนใช้งานจริง
TOOLING = {"mock_p4.py"}

WRITE_MODES = ("w", "a", "x", "+")

# ⚠️ ต้องดู "เจ้าของเมธอด" ไม่ใช่แค่ชื่อเมธอด
#    matplotlib มี Line2D.remove() ซึ่งเป็นการลบเส้นออกจากกราฟ ไม่ใช่ลบไฟล์
#    สแกนเนอร์ที่จับแค่ชื่อ "remove" จะฟ้องผิดจนคนเลิกเชื่อเทสต์ตัวนี้
#    ซึ่งอันตรายกว่าไม่มีเทสต์ เพราะจะมีคนไปปิดมันทิ้ง
FS_MODULES = ("os", "shutil", "pathlib", "Path")
FS_MUTATORS = ("remove", "unlink", "rmdir", "rename", "replace", "removedirs",
               "truncate", "chmod", "utime", "makedirs", "mkdir", "rmtree",
               "copy", "copy2", "copyfile", "copytree", "move",
               "write_text", "write_bytes", "touch")


def code_only(src, path):
    """คืนซอร์สที่ตัด comment และ docstring ออก

    ⚠️ การค้นสตริงในไฟล์ทั้งดุ้นจะไปเจอคำในคอมเมนต์ เช่นบรรทัดที่เขียนว่า
       "การสร้างรายงานยังเป็นของ report_3ec.py ในระบบเดิม" ซึ่งเป็น
       *เอกสารยืนยันขอบเขต* ไม่ใช่การเรียกใช้ — ถ้าฟ้องตรงนั้นด้วย
       คนจะถูกบังคับให้ลบคอมเมนต์ที่ดีทิ้งเพื่อให้เทสต์ผ่าน
    """
    tree = ast.parse(src, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(tree)


def py_files(*dirs):
    for d in dirs:
        for base, _, names in os.walk(os.path.join(ROOT, d)):
            if "__pycache__" in base:
                continue
            for n in sorted(names):
                if n.endswith(".py"):
                    yield os.path.join(base, n)


def manifest(root):
    """ลายนิ้วมือของทั้งต้นไม้ — เนื้อไฟล์ ขนาด และเวลาแก้ไข"""
    out = {}
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for n in sorted(names):
            p = os.path.join(base, n)
            rel = os.path.relpath(p, root).replace("\\", "/")
            try:
                with open(p, "rb") as fh:
                    data = fh.read()
                st = os.stat(p)
                out[rel] = (hashlib.sha256(data).hexdigest(), st.st_size,
                            round(st.st_mtime, 3))
            except OSError as e:
                out[rel] = ("UNREADABLE", str(e), 0)
    return out


# ============================================================== A. static
class TestStaticNoWritePaths(unittest.TestCase):
    def _writes_in(self, path):
        with open(path, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)

        # ชื่อที่ import ตรงมาจากโมดูลไฟล์ระบบ — พวกนี้เรียกแบบเปล่า ๆ ได้
        bare_fs = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in FS_MODULES:
                bare_fs |= {a.asname or a.name for a in node.names}

        found = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id == "open":
                mode = None
                if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                    mode = node.args[1].value
                for kw in node.keywords:
                    if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                        mode = kw.value.value
                if isinstance(mode, str) and any(m in mode for m in WRITE_MODES):
                    found.append("open(mode=%r) @L%d" % (mode, node.lineno))
            elif isinstance(fn, ast.Attribute) and fn.attr in FS_MUTATORS:
                owner = fn.value
                root = owner
                while isinstance(root, ast.Attribute):
                    root = root.value
                owner_name = root.id if isinstance(root, ast.Name) else ""
                if owner_name in FS_MODULES:
                    found.append("%s.%s() @L%d"
                                 % (owner_name, fn.attr, node.lineno))
            elif isinstance(fn, ast.Name) and fn.id in bare_fs:
                found.append("%s() @L%d" % (fn.id, node.lineno))
        return found

    def test_legacy_read_never_opens_anything_for_writing(self):
        """ไฟล์ที่อ่านของ legacy ต้องไม่มีเส้นทางเขียนแม้แต่เส้นเดียว"""
        p = os.path.join(ROOT, "ecstation", "core", "legacy_read.py")
        self.assertEqual(self._writes_in(p), [],
                         "legacy_read.py ต้องเป็น read-only บริสุทธิ์")

    def test_only_event_log_writes_files(self):
        offenders = {}
        for p in py_files("ecstation"):
            n = os.path.basename(p)
            if n in WRITERS_ALLOWED:
                continue
            w = self._writes_in(p)
            if n == "config.py":
                # config สร้าง data_dir ของตัวเองได้ แต่ห้ามทำอย่างอื่น
                w = [x for x in w if "makedirs" not in x]
            if w:
                offenders[os.path.relpath(p, ROOT)] = w
        self.assertEqual(offenders, {},
                         "มีโมดูลที่เขียนไฟล์นอกเหนือจากที่อนุญาต")

    def test_event_log_writes_only_under_its_own_events_dir(self):
        p = os.path.join(ROOT, "ecstation", "bridge", "event_log.py")
        with open(p, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn('os.path.join(data_dir, "events")', src)
        for bad in ("test_realtime", "water_data", "reports", "sessions_3ec"):
            self.assertNotIn(bad, src)

    def test_every_writer_targets_only_our_own_data_dir(self):
        """โมดูลที่เขียนไฟล์ได้ ต้องประกอบ path จาก data_dir ที่ส่งเข้ามาเท่านั้น

        ⚠️ นี่คือด่านที่กันไม่ให้มีใครเผลอเติม path ของ legacy เข้าไปทีหลัง
           รายชื่อ WRITERS_ALLOWED ขยายได้ แต่ทุกตัวต้องผ่านข้อนี้
        """
        targets = {"event_log.py": '"events"', "raw_capture.py": '"raw"',
                   "snapshot.py": '"diag"'}
        for p in py_files("ecstation"):
            n = os.path.basename(p)
            if n not in WRITERS_ALLOWED:
                continue
            with open(p, encoding="utf-8") as fh:
                src = fh.read()
            for bad in ("test_realtime", "water_data", "reports",
                        "sessions_3ec", "rec_status"):
                self.assertNotIn(bad, src, "%s อ้างถึง %s" % (n, bad))
            if n in targets:
                self.assertIn("os.path.join(data_dir, %s)" % targets[n], src,
                              "%s ไม่ได้ประกอบ path จาก data_dir" % n)

    def test_no_module_hardcodes_a_legacy_path(self):
        """เส้นทางของ legacy ต้องมาจาก config เท่านั้น ห้ามฝังในโค้ด"""
        offenders = []
        for p in py_files("ecstation"):
            with open(p, encoding="utf-8") as fh:
                src = fh.read()
            src = code_only(src, p)
            for bad in ("test_realtime", "MOF_NanoTec"):
                if bad in src:
                    offenders.append((os.path.relpath(p, ROOT), bad))
        self.assertEqual(offenders, [])

    def test_no_subprocess_or_shell_escape_hatch(self):
        """ห้ามมีทางอ้อมไปแตะไฟล์ผ่าน shell"""
        for p in py_files("ecstation"):
            with open(p, encoding="utf-8") as fh:
                tree = ast.parse(fh.read(), filename=p)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod = getattr(node, "module", "") or ""
                    names = [a.name for a in node.names] + [mod]
                    for n in names:
                        self.assertNotIn(n.split(".")[0],
                                         ("subprocess", "os.system", "shutil"),
                                         "%s import %s" % (p, n))


# ============================================================= B. dynamic
class TestLegacyTreeUntouched(unittest.TestCase):
    def setUp(self):
        self.legacy = tempfile.mkdtemp(prefix="ec_legacy_fake_")
        self.mine = tempfile.mkdtemp(prefix="ec_new_")
        self._build_legacy()
        self.cfg = tmp_cfg(self.mine, self.legacy)

    def tearDown(self):
        shutil.rmtree(self.legacy, ignore_errors=True)
        shutil.rmtree(self.mine, ignore_errors=True)

    def _build_legacy(self):
        """legacy จำลอง — ครบทุกชนิดไฟล์ที่ข้อกำหนดสั่งห้ามแตะ"""
        L = self.legacy
        os.makedirs(os.path.join(L, "water_data"))
        os.makedirs(os.path.join(L, "reports"))
        os.makedirs(os.path.join(L, "docs"))

        for n in ("logger_3ec.py", "desktop_ui.py", "report_3ec.py",
                  "calibration.py", "lab_theme.py", "report_jobs.py"):
            with open(os.path.join(L, n), "w", encoding="utf-8") as fh:
                fh.write("# legacy source — ห้ามแตะ\n")
        with open(os.path.join(L, "docs", "SYSTEM.md"), "w",
                  encoding="utf-8") as fh:
            fh.write("# legacy docs\n")
        with open(os.path.join(L, "ec_ui_config.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"active_mask": 6}, fh)
        with open(os.path.join(L, "sessions_3ec.json"), "w",
                  encoding="utf-8") as fh:
            json.dump([{"sensor": 2, "sample": "CALF-20 B3"}], fh)
        with open(os.path.join(L, "rec_status.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"active": [False, True, True], "mask": 6,
                       "sample": [None, "CALF-20 B3", None],
                       "updated": datetime.now().isoformat()}, fh)
        p = os.path.join(L, "water_data",
                         "water_log_{:%Y-%m-%d}.csv".format(datetime.now()))
        with open(p, "w", encoding="utf-8") as fh:
            fh.write("timestamp,sensor,ec,temp\n")
            for i in range(50):
                fh.write("2026-08-28 12:00:00,1,1146.0,20.6\n")
        with open(os.path.join(L, "reports", "session_02.pdf"), "wb") as fh:
            fh.write(b"%PDF-1.4 fake\n")
        with open(os.path.join(L, "reports", "session_02.xlsx"), "wb") as fh:
            fh.write(b"PK\x03\x04fake\n")

    def _exercise_everything(self):
        """เดินทุกเส้นทางที่ระบบใหม่มี — ไม่ใช่แค่ทางที่สวยงาม"""
        log = EventLog(self.mine)
        state = PcStateSource(self.cfg)
        tr = B.LoopbackTransport()
        br = B.P4Bridge(self.cfg, log, state, transport=tr)

        def j(d):
            return json.dumps(d).encode()

        frames = [
            j({"v": 1, "type": "hb", "boot_id": "p4-a", "ts_ms": 1,
               "queued": 0, "link": "online", "heap": 1, "heap_big": 1,
               "display_mask": 7}),
            j({"v": 1, "type": "event", "event_id": "p4-a-000001",
               "boot_id": "p4-a", "event": "reading_saved", "sensor": 2,
               "ec_us_cm": 1146.0, "temperature_c": 20.6,
               "tolerance_us_cm": 11.5, "stable_for_ms": 15000,
               "after_link_error": False, "ts_ms": 2}),
            j({"v": 1, "type": "cmd", "request_id": 1, "boot_id": "p4-a",
               "action": "start_session", "sensor": 2, "ts_ms": 3}),
            j({"v": 1, "type": "cmd", "request_id": 2, "boot_id": "p4-a",
               "action": "calibrate", "sensor": 1, "ts_ms": 4}),
            u"{ขยะ".encode("utf-8"), b"[]", b"x" * 900,
            j({"v": 1, "type": "hb", "boot_id": "p4-b", "ts_ms": 5,
               "queued": 3, "link": "offline", "heap": 1, "heap_big": 1,
               "display_mask": 255}),
        ]
        for f in frames:
            br._handle(f)
        for _ in range(5):
            br._send_state()
            state.snapshot()
        log.close()
        return br

    def test_legacy_tree_is_byte_identical_after_a_full_bridge_run(self):
        before = manifest(self.legacy)
        br = self._exercise_everything()
        after = manifest(self.legacy)

        changed = {k: (before.get(k), after.get(k))
                   for k in set(before) | set(after)
                   if before.get(k) != after.get(k)}
        self.assertEqual(changed, {}, "legacy เปลี่ยนไป: %s" % changed)
        # และต้องได้ทำงานจริง ไม่ใช่ผ่านเพราะไม่ได้ทำอะไรเลย
        self.assertEqual(br.counters["events"], 1)
        self.assertEqual(br.counters["cmds_nacked"], 2)
        self.assertGreaterEqual(br.counters["state_sent"], 5)

    def test_no_new_file_or_folder_appears_in_legacy(self):
        before = set(manifest(self.legacy))
        before_dirs = {d for b, ds, _ in os.walk(self.legacy) for d in
                       [os.path.relpath(os.path.join(b, x), self.legacy)
                        for x in ds]}
        self._exercise_everything()
        after = set(manifest(self.legacy))
        after_dirs = {d for b, ds, _ in os.walk(self.legacy) for d in
                      [os.path.relpath(os.path.join(b, x), self.legacy)
                       for x in ds]}
        self.assertEqual(after - before, set())
        self.assertEqual(after_dirs - before_dirs, set())

    def test_everything_written_landed_in_our_own_data_dir(self):
        self._exercise_everything()
        written = sorted(manifest(self.mine))
        self.assertTrue(written, "ต้องมีการเขียน event log จริง")
        for rel in written:
            self.assertTrue(rel.startswith("events/"),
                            "เขียนนอก data/events/: %s" % rel)
        self.assertTrue(any(r.startswith("events/p4_events_") for r in written))

    def test_reports_button_can_only_open_the_folder(self):
        """ปุ่ม Reports ต้องเรียก os.startfile อย่างเดียว ห้ามมีเส้นทางสร้างไฟล์"""
        p = os.path.join(ROOT, "ecstation", "ui", "dashboard.py")
        with open(p, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("os.startfile", src)
        self.assertEqual(TestStaticNoWritePaths()._writes_in(p), [],
                         "dashboard.py มีเส้นทางเขียนไฟล์")

    def test_reports_folder_is_never_opened_for_writing(self):
        """ปุ่ม Reports ในระบบใหม่เปิดโฟลเดอร์ได้อย่างเดียว ห้ามสร้างรายงาน"""
        rep = os.path.join(self.legacy, "reports")
        before = manifest(rep)
        self._exercise_everything()
        self.assertEqual(manifest(rep), before)

    def test_no_report_generation_is_reachable_from_the_new_code(self):
        """ห้าม import / เรียก / ทำซ้ำตัวสร้างรายงานของระบบเดิม

        matplotlib ไม่อยู่ในรายการนี้เพราะ P1-B ต้องวาดกราฟสด — แต่ถูกจำกัด
        ให้อยู่ได้เฉพาะ ecstation/ui โดย test_plot_library_stays_in_the_ui_layer
        ส่วนตัวเขียนไฟล์รายงาน (reportlab/openpyxl/…) ยังห้ามเด็ดขาดทั้งโปรเจกต์
        """
        banned = ("report_3ec", "report_jobs", "calibration",
                  "reportlab", "openpyxl", "xlsxwriter", "pandas", "fpdf",
                  "savefig", "to_excel", "export_sensor_session")
        for p in py_files("ecstation"):
            with open(p, encoding="utf-8") as fh:
                src = fh.read()
            code = code_only(src, p)
            for b in banned:
                self.assertNotIn(b, code,
                                 "%s อ้างถึง %s" % (os.path.relpath(p, ROOT), b))

    def test_plot_library_stays_in_the_ui_layer(self):
        for sub in ("core", "bridge"):
            for p in py_files(os.path.join("ecstation", sub)):
                with open(p, encoding="utf-8") as fh:
                    src = fh.read()
                src = code_only(src, p)
                for b in UI_ONLY_IMPORTS:
                    self.assertNotIn(b, src, "%s แตะ %s ทั้งที่อยู่ชั้น %s"
                                     % (os.path.relpath(p, ROOT), b, sub))

    def test_ui_config_is_written_inside_our_own_data_dir(self):
        """ค่าตั้งหน้าจอต้องไม่ไปตกในโฟลเดอร์ของระบบเดิม"""
        from ecstation.ui import lab_theme as LT
        from ecstation.core.config import is_inside
        proj = os.path.dirname(ROOT) if os.path.basename(ROOT) == "" else ROOT
        self.assertTrue(is_inside(LT.UI_CONFIG_FILE,
                                  os.path.join(proj, "data")),
                        "ui_state.json ไม่ได้อยู่ใน data/ : %s"
                        % LT.UI_CONFIG_FILE)
        self.assertNotIn("test_realtime", LT.UI_CONFIG_FILE)


# ============================================================== C. guard
class TestConfigGuard(unittest.TestCase):
    def setUp(self):
        self.legacy = tempfile.mkdtemp(prefix="ec_legacy_guard_")
        self.tmp = tempfile.mkdtemp(prefix="ec_cfgdir_")

    def tearDown(self):
        shutil.rmtree(self.legacy, ignore_errors=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _load_with(self, data_dir):
        p = os.path.join(self.tmp, "app_config.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"legacy": {"enabled": True, "root": self.legacy},
                       "data_dir": data_dir}, fh)
        return CFG.load(p)

    def test_data_dir_inside_legacy_is_rejected(self):
        cfg = self._load_with(os.path.join(self.legacy, "data"))
        self.assertFalse(CFG.is_inside(cfg["data_dir"], self.legacy))
        self.assertFalse(os.path.exists(os.path.join(self.legacy, "data")))

    def test_data_dir_equal_to_legacy_root_is_rejected(self):
        cfg = self._load_with(self.legacy)
        self.assertFalse(CFG.is_inside(cfg["data_dir"], self.legacy))

    def test_data_dir_with_dotdot_escape_is_still_caught(self):
        sneaky = os.path.join(self.legacy, "sub", "..", "data")
        cfg = self._load_with(sneaky)
        self.assertFalse(CFG.is_inside(cfg["data_dir"], self.legacy))

    def test_a_normal_data_dir_is_left_alone(self):
        want = os.path.join(self.tmp, "data")
        cfg = self._load_with(want)
        self.assertEqual(os.path.realpath(cfg["data_dir"]),
                         os.path.realpath(want))

    def test_sibling_directory_is_not_mistaken_for_a_child(self):
        self.assertFalse(CFG.is_inside(self.legacy + "_other", self.legacy))


if __name__ == "__main__":
    unittest.main()

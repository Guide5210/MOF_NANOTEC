#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 mock_p4.py — จอ ESP32-P4 จำลอง สำหรับทดสอบ bridge โดยไม่ต้องมีฮาร์ดแวร์
============================================================================
 ต่อผ่าน TCP บน localhost แทนพอร์ตอนุกรม เพื่อให้รันได้ทั้ง Windows และ Linux
 โดยไม่ต้องมี com0com หรือ pty

 ใช้:
     python -m ecstation.app --bridge-mode mock      (ฝั่งรับ)
     python tools\mock_p4.py --scenario normal       (ฝั่งจอจำลอง)

 รายการฉาก:
     normal        heartbeat ปกติ + event ตามจังหวะ
     mask-change   display_mask 7 -> 6 -> 3 -> 7
     mask-boot     heartbeat แรกส่ง display_mask 255 (ค่าตั้งต้นของเฟิร์มแวร์)
     mask-invalid  ส่ง 0 และบิตนอกช่วง
     reboot        boot_id เปลี่ยนกลางทาง
     disconnect    เงียบ 15 วินาทีแล้วกลับมา
     dup           ส่ง event_id ซ้ำ 3 ครั้ง
     malformed     JSON เสีย / ไม่ใช่ UTF-8 / บรรทัดยาวเกิน / v:99 / type แปลก
     queue-full    queued ไต่ขึ้นจนถึง 32
     heap-leak     heap ลดลงเรื่อย ๆ
     cmd           ส่งคำสั่ง ต้องได้ NACK COMMANDS_DISABLED กลับมา
     flood         ยิงบรรทัดถี่มาก ทดสอบว่า logger ไม่กระทบ
     soak          รันยาว สำหรับทดสอบหน่วยความจำ
============================================================================
"""

import argparse
import json
import random
import socket
import sys
import time

BOOT = "p4-c96e-a820"

# ย่อเวลารอทุกจุดลงเท่านี้เท่า — ใช้ตอนรันสวีปอัตโนมัติ
# ⚠️ ย่อเวลาแล้วต้องย่อ offline_after ฝั่ง bridge ตามด้วย ไม่งั้นฉาก
#    disconnect จะไม่ได้พิสูจน์อะไรเลย (ดู tools/e2e_mock.py)
TIME_SCALE = 1.0


def frame(**kw):
    d = {"v": 1}
    d.update(kw)
    return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


class Mock(object):
    def __init__(self, sock, boot=BOOT):
        self.s = sock
        self.boot = boot
        self.n = 0
        self.mono = 100000
        self.queued = 0
        self.heap = 328399
        self.heap_big = 253952
        self.mask = 7
        self.rx = b""

    # ---- ส่ง ----
    def send_raw(self, text):
        try:
            self.s.sendall((text + "\n").encode("utf-8"))
        except OSError:
            raise SystemExit("[mock] ฝั่งรับปิดการเชื่อมต่อ")
        print("  ->", text[:110])

    def hb(self, mask=None, link="online"):
        self.mono += 2000
        self.send_raw(frame(type="hb", boot_id=self.boot, ts_ms=self.mono,
                            queued=self.queued, link=link, heap=self.heap,
                            heap_big=self.heap_big,
                            display_mask=self.mask if mask is None else mask))

    def _eid(self):
        self.n += 1
        return "%s-%06d" % (self.boot, self.n)

    def reading(self, sensor=2, ec=1146.0, eid=None):
        self.mono += 500
        self.send_raw(frame(
            type="event", event_id=eid or self._eid(), boot_id=self.boot,
            event="reading_saved", sensor=sensor,
            ec_us_cm=round(ec + random.gauss(0, 0.4), 1),
            temperature_c=20.6, tolerance_us_cm=11.5, stable_for_ms=15000,
            after_link_error=False, ts_ms=self.mono,
            wall=time.strftime("%d %b %Y  %H:%M:%S")))

    def context(self, event, sensor=2, ec=None):
        self.mono += 300
        kw = dict(type="event", event_id=self._eid(), boot_id=self.boot,
                  event=event, sensor=sensor, ts_ms=self.mono)
        if ec is not None:
            kw["ec_us_cm"] = ec
        self.send_raw(frame(**kw))

    def cmd(self, action="recording_start", rid=7):
        self.mono += 100
        self.send_raw(frame(type="cmd", request_id=rid, boot_id=self.boot,
                            action=action, sensor=0, ts_ms=self.mono,
                            payload={}))

    # ---- รับ ----
    def poll(self, seconds):
        end = time.time() + (seconds / TIME_SCALE)
        self.s.settimeout(0.2)
        while time.time() < end:
            try:
                chunk = self.s.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                return
            if not chunk:
                return
            self.rx += chunk
            while b"\n" in self.rx:
                line, self.rx = self.rx.split(b"\n", 1)
                try:
                    d = json.loads(line.decode("utf-8"))
                except Exception:
                    print("  <- [อ่านไม่ออก]", line[:80]); continue
                t = d.get("type")
                if t == "state":
                    print("  <- state seq=%s pc_online=%s rec=%s mask=%s rows=%s sample=%r%s"
                          % (d.get("seq"), d.get("pc_online"), d.get("recording"),
                             d.get("session_mask"), d.get("csv_rows"),
                             d.get("sample_id"),
                             "  [active_mask สมมติ]" if d.get("active_mask_assumed") else ""))
                elif t == "ack":
                    print("  <- ack rid=%s ok=%s code=%s %s"
                          % (d.get("request_id"), d.get("ok"),
                             d.get("code"), d.get("message", "")))
                else:
                    print("  <-", line[:100])


# ---------------------------------------------------------------- ฉาก
def sc_normal(m, dur):
    end = time.time() + dur
    k = 0
    while time.time() < end:
        m.hb(); m.poll(2.0); k += 1
        if k % 3 == 0:
            m.context("STABILITY_REACHED", 2, 84.6)
            m.reading(2, 1146.0)


def sc_mask_change(m, dur):
    for mask in (7, 6, 3, 7):
        m.mask = mask
        for _ in range(2):
            m.hb(); m.poll(2.0)


def sc_mask_boot(m, dur):
    m.hb(mask=255); m.poll(2.0)       # เฟิร์มแวร์ยังไม่ได้ตั้งค่า
    m.hb(mask=255); m.poll(2.0)
    m.mask = 6
    for _ in range(3):
        m.hb(); m.poll(2.0)


def sc_mask_invalid(m, dur):
    m.hb(mask=0);    m.poll(2.0)      # ต้องถูกปฏิเสธ ใช้ค่าเดิม
    m.hb(mask=0b1111); m.poll(2.0)    # บิตนอกช่วง ต้องถูกตัด
    m.hb(mask=6);    m.poll(2.0)


def sc_reboot(m, dur):
    for _ in range(3):
        m.hb(); m.poll(2.0)
    m.reading(2); m.poll(1.0)
    m.boot = "p4-11ff-3c02"; m.n = 0; m.queued = 0; m.mono = 500
    print("[mock] จอรีบูต boot_id ->", m.boot)
    for _ in range(3):
        m.hb(); m.poll(2.0)
    m.reading(2)


def sc_disconnect(m, dur):
    for _ in range(3):
        m.hb(); m.poll(2.0)
    print("[mock] เงียบ 15 วินาที (จอควรกลายเป็น OFFLINE ฝั่ง PC)")
    m.poll(15.0)
    for _ in range(3):
        m.hb(); m.poll(2.0)


def sc_dup(m, dur):
    m.hb(); m.poll(1.0)
    eid = m._eid()
    for _ in range(3):
        m.reading(2, 1146.0, eid=eid)   # ต้องถูกเก็บครั้งเดียว
        m.poll(0.6)
    m.hb(); m.poll(2.0)


def sc_malformed(m, dur):
    m.hb(); m.poll(1.0)
    m.send_raw("{ไม่ใช่ json เลย")
    m.send_raw('{"v":99,"type":"hb","boot_id":"x","ts_ms":1,"queued":0,"link":"online"}')
    m.send_raw('{"v":1,"type":"ดาวอังคาร"}')
    m.send_raw('{"v":1,"type":"event","event_id":"x","boot_id":"y","event":"reading_saved","sensor":2,"ec":99}')
    m.send_raw('{"v":1,"type":"event","event_id":"z","boot_id":"y","event":"reading_saved","sensor":2,"stable_ec_us_cm":10,"ec_us_cm":99,"temperature_c":20,"tolerance_us_cm":1,"stable_for_ms":1,"ts_ms":1}')
    try:
        m.s.sendall(b'{"v":1,"type":"hb","pad":"' + b"A" * 5000 + b'"}\n')
        print("  -> [บรรทัดยาว 5,000 ไบต์]")
        m.s.sendall(b'\xff\xfe not utf-8\n')
        print("  -> [ไม่ใช่ UTF-8]")
    except OSError:
        pass
    m.poll(1.0)
    m.hb(); m.poll(2.0)               # ต้องยังทำงานต่อได้ตามปกติ


def sc_queue_full(m, dur):
    for q in (0, 4, 12, 24, 31, 32, 32):
        m.queued = q; m.hb(); m.poll(1.5)


def sc_heap_leak(m, dur):
    end = time.time() + dur
    while time.time() < end:
        m.heap = max(40000, m.heap - 4000)
        m.heap_big = max(20000, m.heap_big - 3500)
        m.hb(); m.poll(2.0)


def sc_cmd(m, dur):
    m.hb(); m.poll(1.0)
    for rid, act in ((7, "recording_start"), (8, "recording_stop"),
                     (9, "calibration_request")):
        m.cmd(act, rid); m.poll(1.5)


def sc_flood(m, dur):
    end = time.time() + dur
    n = 0
    while time.time() < end:
        for _ in range(50):
            m.mono += 5
            try:
                m.s.sendall((frame(type="event", event_id=m._eid(),
                                   boot_id=m.boot, event="STABILITY_LOST",
                                   sensor=2, ts_ms=m.mono) + "\n").encode())
            except OSError:
                return
            n += 1
        m.hb()
        m.poll(0.2)
    print("[mock] ยิงไป %d เฟรม" % n)


def sc_soak(m, dur):
    end = time.time() + dur
    k = 0
    while time.time() < end:
        m.hb(); m.poll(2.0); k += 1
        if k % 5 == 0:
            m.reading(random.choice([2, 3]))
        if k % 60 == 0:
            m.mask = random.choice([3, 6, 7])
            print("[mock] เปลี่ยน display_mask ->", m.mask)


SCENARIOS = {
    "normal": sc_normal, "mask-change": sc_mask_change, "mask-boot": sc_mask_boot,
    "mask-invalid": sc_mask_invalid, "reboot": sc_reboot,
    "disconnect": sc_disconnect, "dup": sc_dup, "malformed": sc_malformed,
    "queue-full": sc_queue_full, "heap-leak": sc_heap_leak, "cmd": sc_cmd,
    "flood": sc_flood, "soak": sc_soak,
}


def main():
    ap = argparse.ArgumentParser(description="จอ ESP32-P4 จำลอง")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8781)
    ap.add_argument("--scenario", default="normal", choices=sorted(SCENARIOS))
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--boot-id", default=BOOT)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="ย่อเวลารอลงกี่เท่า (1 = เวลาจริง)")
    a = ap.parse_args()
    global TIME_SCALE
    TIME_SCALE = max(1.0, a.speed)

    print("[mock] ต่อไปที่ %s:%d ..." % (a.host, a.port))
    s = socket.create_connection((a.host, a.port), timeout=5)
    print("[mock] ต่อแล้ว · ฉาก '%s' · %.0f วินาที\n" % (a.scenario, a.seconds))
    m = Mock(s, a.boot_id)
    try:
        SCENARIOS[a.scenario](m, a.seconds)
    except (KeyboardInterrupt, SystemExit) as e:
        print("\n[mock]", e or "หยุด")
    finally:
        s.close()
    print("\n[mock] จบฉาก")


if __name__ == "__main__":
    main()

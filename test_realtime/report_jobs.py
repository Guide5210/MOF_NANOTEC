#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
 report_jobs.py — คิวงานสร้างรายงาน แยกออกจากเธรดที่อ่าน serial
============================================================================

 ทำไมต้องมีไฟล์นี้
 -----------------
 เดิม logger_3ec.py เรียก report_3ec.export_sensor_session() ตรง ๆ ใน main loop
 ระหว่างนั้น "ไม่มีการเรียก ser.readline() เลย" ซึ่งพังสามอย่างพร้อมกัน:

   1. timestamp ของ CSV เพี้ยน
      logger ใส่เวลาด้วย datetime.now() ตอน *อ่าน* ไม่ใช่ตอน *วัด*
      ข้อมูลที่บอร์ดพ่นออกมาระหว่างสร้างรายงานจะค้างใน buffer ของ OS
      แล้วถูกอ่านรวดเดียว ได้หลายแถวที่เวลากระจุกอยู่ท้ายสุด
      -> ข้อมูลไม่หาย แต่แกนเวลาผิด ซึ่งมองไม่ออกว่าผิด จึงอันตรายกว่าหาย

   2. จอสัมผัสจะ timeout
      pc_bridge ฝั่งจอตั้ง CMD_TIMEOUT_MS = 5000
      วัดจริงบนข้อมูล 24 ชั่วโมง: หยุด session 3 ตัวติดกัน = 5.9 วินาที
      จอจะขึ้น "Request status unknown" ทั้งที่ PC ทำสำเร็จแล้ว
      แล้วผู้ใช้จะกดซ้ำ -> ได้ session ซ้อน

   3. จอจะเห็น PC เป็น OFFLINE
      pc_bridge ตั้ง PC_STALE_MS = 10000
      export_combined_report() บนข้อมูล 24 ชั่วโมง = 5.7 วินาที
      ถ้าข้อมูลมากกว่านี้ก็เกิน 10 วินาทีได้ไม่ยาก

 ทั้งสามข้อเกิดขึ้นแล้ววันนี้ตอนกดคีย์ 1/2/3 เพียงแต่ยังไม่มีใครเห็น

 ข้อตกลงของไฟล์นี้
 -----------------
   - worker หนึ่งตัวเท่านั้น  รายงานสร้างทีละใบตามลำดับที่สั่ง
     (report_3ec ใช้ pyplot ซึ่งมี global state ไม่ปลอดภัยกับหลายเธรด)
   - ห้ามแตะ serial ห้ามแตะ CSV  worker รู้จักแค่ฟังก์ชันที่ถูกส่งเข้ามา
   - worker พิมพ์ออกคอนโซลเองได้  บรรทัดไม่ปนกันกลางบรรทัดเพราะ GIL
     ทำให้ print() หนึ่งครั้งเป็น atomic ในทางปฏิบัติ
     (ยอมรับว่าลำดับบรรทัดอาจสลับกับ main loop บ้าง — แลกกับการที่
      CSV ไม่ขาดช่วง ซึ่งสำคัญกว่ามาก)
============================================================================
"""

import queue
import threading
import time
import traceback
from datetime import datetime


class Job(object):
    __slots__ = ("id", "name", "state", "submitted", "started",
                 "finished", "error", "result")

    def __init__(self, jid, name):
        self.id = jid
        self.name = name
        self.state = "queued"      # queued -> running -> done | failed
        self.submitted = datetime.now()
        self.started = None
        self.finished = None
        self.error = None
        self.result = None

    def duration_s(self):
        if not self.started:
            return 0.0
        end = self.finished or datetime.now()
        return (end - self.started).total_seconds()


class ReportJobs(object):
    """คิวงานหนัก + worker หนึ่งตัว

    ใช้:
        jobs = ReportJobs()
        jobs.submit("SENSOR 02 session report", fn, arg1, kw=...)
        ...
        jobs.shutdown(wait=True)      # ตอนปิดโปรแกรม ต้องรอให้เขียนไฟล์ครบ
    """

    def __init__(self, tag="jobs", quiet=False):
        self.tag = tag
        self.quiet = quiet
        self._q = queue.Queue()
        self._lock = threading.Lock()
        self._jobs = []              # ประวัติทั้งหมด (จำกัดความยาว)
        self._running = None
        self._next_id = 1
        self._stop = threading.Event()
        # daemon=False โดยตั้งใจ — ถ้า process ตายกลางคันระหว่างเขียน PDF
        # จะได้ไฟล์ครึ่ง ๆ กลาง ๆ  ให้ shutdown() เป็นคนคุมการจบเท่านั้น
        self._worker = threading.Thread(target=self._loop, name="report-jobs",
                                        daemon=False)
        self._worker.start()

    # ------------------------------------------------------------------
    def _say(self, msg):
        if not self.quiet:
            print("[{}] {}".format(self.tag, msg), flush=True)

    def _loop(self):
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            job, fn, args, kw = item
            with self._lock:
                job.state = "running"
                job.started = datetime.now()
                self._running = job
            self._say("{} — เริ่มสร้าง (ยังเก็บ CSV ต่อตามปกติ)".format(job.name))
            try:
                job.result = fn(*args, **kw)
                job.state = "done"
                self._say("{} — เสร็จใน {:.1f} วินาที".format(
                    job.name, job.duration_s()))
            except Exception as e:
                job.state = "failed"
                job.error = "{}: {}".format(type(e).__name__, e)
                self._say("!! {} — ล้มเหลว: {}".format(job.name, job.error))
                self._say("   ข้อมูลดิบใน CSV ยังครบ สร้างใหม่ทีหลังได้ด้วย "
                          "report_3ec.py")
                if not self.quiet:
                    traceback.print_exc()
            finally:
                job.finished = datetime.now()
                with self._lock:
                    self._running = None
                # กันรูปค้างจากเส้นทางที่ raise กลางคัน ก่อน plt.close() ถูกเรียก
                try:
                    import matplotlib.pyplot as plt
                    plt.close("all")
                except Exception:
                    pass
                self._q.task_done()

    # ------------------------------------------------------------------
    def submit(self, name, fn, *args, **kw):
        """เข้าคิวงานหนึ่งชิ้น คืน Job ทันที ไม่บล็อก"""
        if self._stop.is_set():
            raise RuntimeError("ReportJobs ถูกปิดไปแล้ว")
        with self._lock:
            job = Job(self._next_id, name)
            self._next_id += 1
            self._jobs.append(job)
            del self._jobs[:-64]
        self._q.put((job, fn, args, kw))
        return job

    # ------------------------------------------------------------------
    def busy(self):
        with self._lock:
            return self._running is not None or not self._q.empty()

    def queued(self):
        return self._q.qsize()

    def running_name(self):
        with self._lock:
            return self._running.name if self._running else None

    def snapshot(self):
        """สถานะย่อ — ใช้ส่งให้จอสัมผัสและเขียนลง pc_state.json"""
        with self._lock:
            done = sum(1 for j in self._jobs if j.state == "done")
            failed = sum(1 for j in self._jobs if j.state == "failed")
            run = self._running
            last_err = None
            for j in reversed(self._jobs):
                if j.state == "failed":
                    last_err = j.error
                    break
            return {
                "busy": run is not None,
                "running": run.name if run else None,
                "running_for_s": round(run.duration_s(), 1) if run else 0.0,
                "queued": self._q.qsize(),
                "done": done,
                "failed": failed,
                "last_error": last_err,
            }

    # ------------------------------------------------------------------
    def wait_idle(self, timeout=None):
        """รอจนคิวว่าง คืน True ถ้าว่างจริง"""
        deadline = None if timeout is None else time.time() + timeout
        while self.busy():
            if deadline and time.time() > deadline:
                return False
            time.sleep(0.1)
        return True

    def shutdown(self, wait=True, timeout=300):
        """ปิด worker

        wait=True จะรอให้รายงานที่ค้างอยู่เขียนไฟล์จนจบก่อน
        ถ้าไม่รอ ผู้ใช้จะได้ PDF ที่เขียนไม่จบ ซึ่งแย่กว่าไม่ได้ไฟล์เลย
        """
        if self._stop.is_set():
            return True
        self._stop.set()
        if wait and self.busy():
            n = self._q.qsize() + (1 if self.running_name() else 0)
            self._say("รอสร้างรายงานที่ค้างอยู่ {} งานให้เสร็จก่อนปิด "
                      "(Ctrl+C ซ้ำเพื่อทิ้ง)".format(n))
        self._q.put(None)
        self._worker.join(timeout if wait else 1.0)
        if self._worker.is_alive():
            self._say("!! รายงานยังสร้างไม่เสร็จภายใน {} วินาที "
                      "— ปิดโปรแกรมโดยไม่รอต่อ".format(timeout))
            return False
        return True

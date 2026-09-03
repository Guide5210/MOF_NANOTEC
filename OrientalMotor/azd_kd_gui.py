"""
azd_kd_gui.py — หน้าควบคุม AZM46AK-TS30 + AZD-KD ผ่าน Modbus RTU
====================================================================
ต้องมี azd_kd_controller.py อยู่โฟลเดอร์เดียวกัน
ติดตั้ง:  pip install PySide6 pymodbus pyserial
รัน:      python azd_kd_gui.py

ความสามารถ:
  - เลือก COM port / Connect / Disconnect
  - หมุน CW / CCW, ตั้งความเร็ว (Hz) + accel/decel/current
  - เปลี่ยนความเร็ว real-time ขณะหมุน (Apply)
  - STOP ใหญ่ + Reset Alarm
  - Monitor สด 4 Hz: ตำแหน่ง, ความเร็ว (Hz / r/min มอเตอร์ / r/min เพลาออกหลังเกียร์ 30:1),
    อุณหภูมิไดรเวอร์/มอเตอร์, odometer/tripmeter, status word + ไฟ READY
"""

import sys
from serial.tools import list_ports
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
    QGroupBox, QLabel, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
    QMessageBox, QFrame, QSizePolicy,
)

from azd_kd_controller import AZDKD, GEAR_RATIO

POLL_MS = 250           # 4 Hz
STATUS_READY_BIT = 0x20  # ยืนยันจากการทดสอบจริง (0x00000020 ตอน READY)


# ---------- widget ช่วยแสดงค่า ----------
class ValueTile(QFrame):
    def __init__(self, title, unit=""):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("tile")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 8)
        t = QLabel(title)
        t.setObjectName("tileTitle")
        self.val = QLabel("—")
        self.val.setObjectName("tileValue")
        self.unit = QLabel(unit)
        self.unit.setObjectName("tileUnit")
        row = QHBoxLayout()
        row.addWidget(self.val)
        row.addWidget(self.unit, alignment=Qt.AlignBottom)
        row.addStretch()
        lay.addWidget(t)
        lay.addLayout(row)

    def set(self, text):
        self.val.setText(str(text))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AZD-KD Motor Console — AZM46AK-TS30 (Modbus RTU)")
        self.az = None
        self.running_dir = None

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        # ================= ซ้าย: การเชื่อมต่อ + ควบคุม =================
        left = QVBoxLayout()
        main.addLayout(left, 0)

        # --- connection ---
        gb_conn = QGroupBox("การเชื่อมต่อ")
        g = QGridLayout(gb_conn)
        self.cb_port = QComboBox()
        self.btn_refresh = QPushButton("↻")
        self.btn_refresh.setFixedWidth(36)
        self.btn_connect = QPushButton("Connect")
        self.lbl_conn = QLabel("● ยังไม่เชื่อมต่อ")
        self.lbl_conn.setObjectName("connBad")
        g.addWidget(QLabel("COM port"), 0, 0)
        g.addWidget(self.cb_port, 0, 1)
        g.addWidget(self.btn_refresh, 0, 2)
        g.addWidget(self.btn_connect, 1, 0, 1, 3)
        g.addWidget(self.lbl_conn, 2, 0, 1, 3)
        left.addWidget(gb_conn)

        # --- motion ---
        gb_run = QGroupBox("ควบคุมการหมุน")
        g2 = QGridLayout(gb_run)

        self.sp_speed = QSpinBox()
        self.sp_speed.setRange(1, 500_000)
        self.sp_speed.setValue(3000)
        self.sp_speed.setSuffix(" Hz")
        self.sp_speed.setSingleStep(100)

        self.sp_accel = QSpinBox()
        self.sp_accel.setRange(1, 1_000_000_000)
        self.sp_accel.setValue(1_000_000)
        self.sp_accel.setToolTip("หน่วย 0.001 kHz/s (1,000,000 = 1000 kHz/s)")

        self.sp_current = QDoubleSpinBox()
        self.sp_current.setRange(1.0, 100.0)
        self.sp_current.setValue(100.0)
        self.sp_current.setSuffix(" %")

        self.lbl_out_speed = QLabel("เพลาออก (÷30): 6.0 r/min")
        self.sp_speed.valueChanged.connect(self._update_out_speed)

        self.btn_ccw = QPushButton("◀ CCW")
        self.btn_cw = QPushButton("CW ▶")
        for b in (self.btn_ccw, self.btn_cw):
            b.setMinimumHeight(46)
        self.btn_apply = QPushButton("Apply speed (ขณะหมุน)")
        self.btn_stop = QPushButton("■ STOP")
        self.btn_stop.setObjectName("stopBtn")
        self.btn_stop.setMinimumHeight(56)
        self.btn_alarm = QPushButton("Reset Alarm")

        g2.addWidget(QLabel("ความเร็ว (ที่มอเตอร์)"), 0, 0)
        g2.addWidget(self.sp_speed, 0, 1)
        g2.addWidget(self.lbl_out_speed, 1, 0, 1, 2)
        g2.addWidget(QLabel("Accel/Decel"), 2, 0)
        g2.addWidget(self.sp_accel, 2, 1)
        g2.addWidget(QLabel("กระแสรัน"), 3, 0)
        g2.addWidget(self.sp_current, 3, 1)
        g2.addWidget(self.btn_ccw, 4, 0)
        g2.addWidget(self.btn_cw, 4, 1)
        g2.addWidget(self.btn_apply, 5, 0, 1, 2)
        g2.addWidget(self.btn_stop, 6, 0, 1, 2)
        g2.addWidget(self.btn_alarm, 7, 0, 1, 2)
        left.addWidget(gb_run)
        left.addStretch()

        # ================= ขวา: monitor =================
        right = QVBoxLayout()
        main.addLayout(right, 1)

        gb_mon = QGroupBox("Monitor (อ่านสดจากไดรเวอร์)")
        mg = QGridLayout(gb_mon)

        self.tile_hz = ValueTile("ความเร็วจริง (มอเตอร์)", "Hz")
        self.tile_rpm = ValueTile("ความเร็วจริง (มอเตอร์)", "r/min")
        self.tile_out = ValueTile("ความเร็วเพลาออก (หลังเกียร์ 30:1)", "r/min")
        self.tile_fbpos = ValueTile("ตำแหน่งจริง (feedback)", "step")
        self.tile_cmdpos = ValueTile("ตำแหน่งคำสั่ง (command)", "step")
        self.tile_tdrv = ValueTile("อุณหภูมิไดรเวอร์", "°C")
        self.tile_tmot = ValueTile("อุณหภูมิมอเตอร์", "°C")
        self.tile_odo = ValueTile("Odometer สะสม", "kRev")
        self.tile_trip = ValueTile("Tripmeter (ตั้งแต่เปิดไฟ)", "kRev")
        self.tile_status = ValueTile("Status word", "")

        tiles = [
            self.tile_hz, self.tile_rpm, self.tile_out,
            self.tile_fbpos, self.tile_cmdpos, self.tile_status,
            self.tile_tdrv, self.tile_tmot, self.tile_odo, self.tile_trip,
        ]
        for i, t in enumerate(tiles):
            mg.addWidget(t, i // 3, i % 3)
        right.addWidget(gb_mon)

        self.lbl_state = QLabel("สถานะ: —")
        self.lbl_state.setObjectName("stateLine")
        right.addWidget(self.lbl_state)
        right.addStretch()

        # ---------- events ----------
        self.btn_refresh.clicked.connect(self.refresh_ports)
        self.btn_connect.clicked.connect(self.toggle_connect)
        self.btn_cw.clicked.connect(lambda: self.run_dir("cw"))
        self.btn_ccw.clicked.connect(lambda: self.run_dir("ccw"))
        self.btn_apply.clicked.connect(self.apply_speed)
        self.btn_stop.clicked.connect(self.stop_motor)
        self.btn_alarm.clicked.connect(self.reset_alarm)

        self.timer = QTimer(self)
        self.timer.setInterval(POLL_MS)
        self.timer.timeout.connect(self.poll)

        self.refresh_ports()
        self.set_controls_enabled(False)
        self.apply_style()

    # ---------- helpers ----------
    def _update_out_speed(self):
        rpm_out = self.sp_speed.value() / 1000.0 * 60.0 / GEAR_RATIO
        self.lbl_out_speed.setText(f"เพลาออก (÷30): {rpm_out:.2f} r/min")

    def set_controls_enabled(self, en):
        for w in (self.btn_cw, self.btn_ccw, self.btn_apply,
                  self.btn_stop, self.btn_alarm):
            w.setEnabled(en)

    def refresh_ports(self):
        self.cb_port.clear()
        for p in list_ports.comports():
            desc = f"{p.device} — {p.description}"
            self.cb_port.addItem(desc, p.device)
            # กันเลือก mini-USB (MEXE02) ผิด
            if "oriental" in desc.lower() or "common virtual" in desc.lower():
                idx = self.cb_port.count() - 1
                self.cb_port.setItemText(idx, desc + "  [MEXE02 — ห้ามใช้]")

    # ---------- connection ----------
    def toggle_connect(self):
        if self.az is None:
            port = self.cb_port.currentData()
            if not port:
                QMessageBox.warning(self, "ไม่มีพอร์ต", "ไม่พบ COM port — เสียบ converter แล้วกด ↻")
                return
            try:
                az = AZDKD(port=port, slave_id=1)
                az.connect()
                az.read_monitor_fast()  # ยิงทดสอบหนึ่งรอบก่อนประกาศว่าเชื่อมต่อได้
            except Exception as e:
                QMessageBox.critical(self, "เชื่อมต่อไม่ได้", str(e))
                return
            self.az = az
            self.lbl_conn.setText(f"● เชื่อมต่อแล้ว: {port}")
            self.lbl_conn.setObjectName("connGood")
            self.btn_connect.setText("Disconnect")
            self.set_controls_enabled(True)
            self.timer.start()
        else:
            self.disconnect_driver()
        self.apply_style()

    def disconnect_driver(self):
        self.timer.stop()
        try:
            if self.az:
                self.az.stop()
                self.az.close()
        except Exception:
            pass
        self.az = None
        self.running_dir = None
        self.lbl_conn.setText("● ยังไม่เชื่อมต่อ")
        self.lbl_conn.setObjectName("connBad")
        self.btn_connect.setText("Connect")
        self.set_controls_enabled(False)

    # ---------- motion ----------
    def _motion_params(self):
        return dict(
            speed_hz=self.sp_speed.value(),
            accel=self.sp_accel.value(),
            decel=self.sp_accel.value(),
            current_pct=self.sp_current.value(),
        )

    def run_dir(self, direction):
        if not self.az:
            return
        try:
            self.az.run_continuous(direction, **self._motion_params())
            self.running_dir = direction
        except Exception as e:
            QMessageBox.critical(self, "สั่งหมุนไม่สำเร็จ", str(e))

    def apply_speed(self):
        if not self.az or not self.running_dir:
            return
        try:
            self.az.run_continuous(self.running_dir, **self._motion_params())
        except Exception as e:
            QMessageBox.critical(self, "เปลี่ยนความเร็วไม่สำเร็จ", str(e))

    def stop_motor(self):
        if not self.az:
            return
        try:
            self.az.stop()
            self.running_dir = None
        except Exception as e:
            QMessageBox.critical(self, "หยุดไม่สำเร็จ", str(e))

    def reset_alarm(self):
        if not self.az:
            return
        try:
            self.az.reset_alarm()
        except Exception as e:
            QMessageBox.critical(self, "รีเซ็ตอะลาร์มไม่สำเร็จ", str(e))

    # ---------- polling ----------
    def poll(self):
        if not self.az:
            return
        try:
            m = self.az.read_monitor_fast()
        except Exception as e:
            self.lbl_state.setText(f"สถานะ: อ่านค่าล้มเหลว — {e}")
            return
        self.tile_hz.set(m["speed_hz_motor"])
        self.tile_rpm.set(m["speed_rpm_motor"])
        self.tile_out.set(f'{m["output_rpm_after_gear"]:.2f}')
        self.tile_fbpos.set(m["position_step"])
        self.tile_cmdpos.set(m["command_position_step"])
        self.tile_tdrv.set(f'{m["driver_temp_c"]:.1f}')
        self.tile_tmot.set(f'{m["motor_temp_c"]:.1f}')
        self.tile_odo.set(m["odometer_krev"])
        self.tile_trip.set(m["tripmeter_krev"])
        sw = m["status_word"]
        self.tile_status.set(f"0x{sw:08X}")

        ready = bool(sw & STATUS_READY_BIT)
        moving = m["speed_hz_motor"] != 0
        state = []
        state.append("READY" if ready else "BUSY")
        if moving:
            state.append(f"หมุน {self.running_dir or ''} @ {m['speed_hz_motor']} Hz")
        else:
            state.append("หยุดนิ่ง")
        self.lbl_state.setText("สถานะ: " + "  |  ".join(state))

    # ---------- style / close ----------
    def apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #14171c; color: #e8e8e8;
                font-family: 'Segoe UI', sans-serif; font-size: 13px; }
            QGroupBox { border: 1px solid #2c313a; border-radius: 8px;
                margin-top: 10px; padding-top: 14px; font-weight: 600; }
            QGroupBox::title { left: 12px; padding: 0 4px; color: #9aa4b2; }
            QPushButton { background: #232833; border: 1px solid #333a46;
                border-radius: 6px; padding: 8px 12px; }
            QPushButton:hover { background: #2c3340; }
            QPushButton:disabled { color: #555; }
            QPushButton#stopBtn { background: #7a1f1f; border-color: #9c2a2a;
                font-size: 16px; font-weight: 700; }
            QPushButton#stopBtn:hover { background: #932525; }
            QComboBox, QSpinBox, QDoubleSpinBox { background: #1b1f26;
                border: 1px solid #333a46; border-radius: 6px; padding: 5px; }
            QFrame#tile { background: #1b1f26; border: 1px solid #2c313a;
                border-radius: 8px; }
            QLabel#tileTitle { color: #9aa4b2; font-size: 11px; }
            QLabel#tileValue { font-size: 22px; font-weight: 700; color: #6fc3ff; }
            QLabel#tileUnit { color: #9aa4b2; font-size: 11px; }
            QLabel#connGood { color: #58d68d; }
            QLabel#connBad { color: #e77e7e; }
            QLabel#stateLine { color: #cfd6df; padding: 6px 2px; font-size: 14px; }
        """)

    def closeEvent(self, ev):
        self.disconnect_driver()   # หยุดมอเตอร์ + ปิดพอร์ตก่อนออกเสมอ
        ev.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(940, 520)
    w.show()
    sys.exit(app.exec())

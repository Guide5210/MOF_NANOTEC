# ESP32 Water Monitor — คู่มือติดตั้งบน Ubuntu (run 24/7)

ระบบนี้แยกหน้าที่: **ESP32 อ่าน sensor** ส่งผ่าน USB → **Ubuntu รับ + เก็บ CSV ตลอดเวลา** แล้วสร้างรายงาน PDF/Excel เมื่อต้องการ ไม่พึ่ง WiFi/Blynk บนตัว ESP32 (เสถียรกว่า และเลี่ยงปัญหา eduroam)

---

## สิ่งที่ต้องมีในโฟลเดอร์เดียวกัน

```
logger.py              ← ตัวเก็บข้อมูล
report.py              ← ตัวสร้างรายงาน
water-logger.service   ← systemd unit
setup_ubuntu.sh        ← สคริปต์ติดตั้งอัตโนมัติ
```

---

## ฝั่ง ESP32 — burn firmware แบบไม่มี WiFi

ในไฟล์ `water_monitor_v7.ino` แก้บรรทัดเดียว:
```cpp
#define ENABLE_BLYNK  0     // 0 = ตัด WiFi/Blynk ออก (sensor + serial เท่านั้น)
```
แล้ว burn — ESP32 จะเบา เสถียร ไม่มี RF รบกวน ADC (pH นิ่งขึ้นอีก) และ OLED ยังโชว์ version ปกติ

---

## ติดตั้งบน Ubuntu — วิธีอัตโนมัติ (แนะนำ)

```bash
cd /path/to/โฟลเดอร์ที่มีไฟล์
chmod +x setup_ubuntu.sh
./setup_ubuntu.sh "CALF-20 wash batch 3"
```

สคริปต์จะทำให้ครบ: ติดตั้ง Python deps → เพิ่มสิทธิ์ serial → ปิด sleep → ติดตั้ง+เริ่ม service

เสร็จแล้วดู log สดว่าเก็บข้อมูลอยู่:
```bash
journalctl -u water-logger -f
```
ควรเห็นบรรทัด `[HH:MM:SS] rows=... EC=... pH=...` เดินเรื่อย ๆ

---

## ติดตั้งแบบ manual (ถ้าอยากเข้าใจทีละขั้น)

### 1. Dependencies
```bash
sudo apt update && sudo apt install -y python3-pip
pip3 install pyserial pandas matplotlib openpyxl scipy --break-system-packages
```

### 2. สิทธิ์ serial
```bash
sudo usermod -a -G dialout $USER
# logout/login ใหม่ (service ไม่ต้องรอ เพราะรันผ่าน systemd)
```

### 3. ปิด sleep/suspend
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```
ถ้าเป็น laptop เปิดฝาทิ้ง: Settings → Power → ตั้ง "เมื่อปิดฝา = ไม่ทำอะไร" ด้วย

### 4. ติดตั้ง service
แก้ `water-logger.service` แทน `__USER__`, `__WORKDIR__`, `__SAMPLE__` ด้วยค่าจริง แล้ว:
```bash
sudo cp water-logger.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable water-logger
sudo systemctl start water-logger
```

---

## คำสั่งที่ใช้บ่อย

```bash
journalctl -u water-logger -f          # ดู log สด
systemctl status water-logger          # สถานะ
sudo systemctl stop water-logger       # หยุด (ปิด CSV สะอาด)
sudo systemctl start water-logger      # เริ่ม
sudo systemctl restart water-logger    # เริ่มใหม่
```

ข้อมูล CSV อยู่ที่ `<โฟลเดอร์>/water_data/water_log_YYYY-MM-DD.csv`

---

## สร้างรายงาน (ทำเมื่อต้องการ — service ยังรันอยู่ได้)

```bash
cd /path/to/โฟลเดอร์
python3 report.py --sample "CALF-20 wash batch 3"        # รวมทุกวัน
python3 report.py --input water_data/water_log_2026-07-03.csv   # วันเดียว
python3 report.py --open                                  # เปิด PDF ด้วย
```

รายงานไม่รบกวน service — logger เขียน CSV ต่อ, report แค่อ่าน CSV มาทำ PDF/Excel

---

## (แนะนำ) ตั้งชื่อ USB ให้คงที่ด้วย udev

ปัญหา: `/dev/ttyUSB0` อาจเปลี่ยนเป็น `ttyUSB1` เมื่อเสียบใหม่ ทำให้ auto-detect อาจสับสนถ้ามีอุปกรณ์ serial หลายตัว แก้ด้วยการสร้างชื่อคงที่ `/dev/water-monitor`:

หา VID:PID ของ ESP32:
```bash
lsusb
# มองหา Silicon Labs CP210x (10c4:ea60) หรือ QinHeng CH340 (1a86:7523)
```

สร้าง udev rule (แทน idVendor/idProduct ตามที่เจอ):
```bash
sudo tee /etc/udev/rules.d/99-water-monitor.rules <<'EOF'
SUBSYSTEM=="tty", ATTRS{idVendor}=="10c4", ATTRS{idProduct}=="ea60", SYMLINK+="water-monitor"
EOF
sudo udevadm control --reload-rules && sudo udevadm trigger
```

แล้วแก้ service ให้ fix port:
```
ExecStart=/usr/bin/python3 __WORKDIR__/logger.py --service --port /dev/water-monitor --sample "..."
```
เสถียรกว่า auto-detect เมื่อมีอุปกรณ์หลายตัว

---

## (ตัวเลือก) SSH เข้าดูข้อมูลจากที่อื่น

ติดตั้ง SSH server บน Ubuntu:
```bash
sudo apt install -y openssh-server
sudo systemctl enable --now ssh
ip addr | grep inet          # ดู IP ของเครื่อง
```

จากเครื่องอื่นในเน็ตเดียวกัน:
```bash
ssh ชื่อuser@<IP-ของ-ubuntu>
journalctl -u water-logger -f              # ดู log สด
cd /path/to/โฟลเดอร์ && python3 report.py  # สร้างรายงานระยะไกล
```

ดึงไฟล์รายงานกลับมาเครื่องตัวเอง:
```bash
scp ชื่อuser@<IP>:/path/to/โฟลเดอร์/report_*.pdf .
```

> ถ้าเน็ตเป็น eduroam: Ubuntu ต่อ eduroam ได้ปกติผ่าน NetworkManager (ต่างจาก ESP32) — แต่ SSH ข้ามเครื่องบน eduroam อาจถูก firewall กัน client-isolation ถ้าติดปัญหา ให้ SSH ผ่านเน็ตวงเดียวกัน (เช่น hotspot/router ส่วนตัว) หรือใช้ VPN ของมหาลัย

---

## Checklist แก้ปัญหา

| อาการ | สาเหตุ | แก้ |
|---|---|---|
| service ไม่เก็บข้อมูล | หา port ไม่เจอ | `ls /dev/ttyUSB*` เช็ก ESP32 เสียบอยู่ไหม; ใช้ udev fix port |
| `Permission denied` port | ไม่ได้อยู่กลุ่ม dialout | `sudo usermod -a -G dialout $USER` แล้ว reboot |
| ข้อมูลหยุดกลางดึก | เครื่อง suspend | `systemctl mask sleep.target ...` + ปิด sleep ใน Settings |
| CSV ว่าง/ค่าเป็น NaN | RS485/pH ไม่มา | เช็กสายฝั่ง ESP32, ดู `journalctl -u water-logger -f` |
| service ตายแล้วไม่ฟื้น | (ไม่ควรเกิด) | `Restart=always` ตั้งไว้แล้ว; ดู `systemctl status` |

---

## หลักการออกแบบ (ทำไมทำแบบนี้)

- **fsync ทุกแถว** → ไฟดับ/kill กลางคัน เสียอย่างมากแค่แถวสุดท้าย
- **Restart=always** → crash/USB หลุดยาว → systemd ฟื้นเอง
- **SIGTERM handling** → `systemctl stop` ปิด CSV สะอาด
- **แยก logger (service) กับ report (manual)** → สร้างรายงานไม่กระทบการเก็บข้อมูล
- **ESP32 ไม่ต่อ WiFi** → ตัดจุดล้มเหลว (eduroam/RF/reconnect) ออกหมด

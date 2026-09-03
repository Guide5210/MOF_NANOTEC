#!/usr/bin/env bash
# ============================================================================
#  setup_ubuntu.sh  —  ติดตั้งระบบ ESP32 Water Monitor logger บน Ubuntu ครบชุด
# ----------------------------------------------------------------------------
#  ทำให้ครบในคำสั่งเดียว:
#    1) ติดตั้ง Python dependencies
#    2) เพิ่ม user เข้ากลุ่ม dialout (สิทธิ์ serial)
#    3) ปิด sleep/suspend (กัน USB หลุดตอน run ยาว)
#    4) ติดตั้ง systemd service (auto-start ตอนบูต + auto-restart)
#    5) เริ่ม service ทันที
#
#  วิธีใช้:
#    cd ไปยังโฟลเดอร์ที่มี logger.py, report.py, water-logger.service
#    chmod +x setup_ubuntu.sh
#    ./setup_ubuntu.sh                         # sample = "-"
#    ./setup_ubuntu.sh "CALF-20 wash batch 3"  # ระบุชื่อตัวอย่าง
#
#  หมายเหตุ: จะถาม sudo password ระหว่างทาง (ติดตั้ง service ต้องใช้ root)
# ============================================================================
set -e

SAMPLE="${1:--}"
WORKDIR="$(cd "$(dirname "$0")" && pwd)"
RUN_USER="$(whoami)"
SERVICE_NAME="water-logger"

echo "=============================================="
echo " ESP32 Water Monitor — Ubuntu setup"
echo "=============================================="
echo " ผู้ใช้      : $RUN_USER"
echo " โฟลเดอร์    : $WORKDIR"
echo " ตัวอย่าง    : $SAMPLE"
echo "=============================================="
echo ""

# ---- ตรวจว่ามีไฟล์ที่จำเป็นครบ ----
for f in logger.py report.py water-logger.service; do
    if [ ! -f "$WORKDIR/$f" ]; then
        echo "!! ไม่พบ $f ในโฟลเดอร์นี้ — วางไฟล์ให้ครบก่อนรัน"
        exit 1
    fi
done

# ---- 1. Python dependencies ----
echo "[1/5] ติดตั้ง Python dependencies..."
sudo apt update -qq
sudo apt install -y python3-pip
# Ubuntu 23+ ต้องใช้ --break-system-packages
pip3 install pyserial pandas matplotlib openpyxl scipy --break-system-packages 2>/dev/null \
    || pip3 install pyserial pandas matplotlib openpyxl scipy
echo "    ✓ dependencies พร้อม"

# ---- 2. สิทธิ์ serial (dialout) ----
echo "[2/5] เพิ่ม $RUN_USER เข้ากลุ่ม dialout..."
sudo usermod -a -G dialout "$RUN_USER"
echo "    ✓ เพิ่มแล้ว (มีผลเต็มที่หลัง logout/reboot — service ทำงานได้เลยเพราะรันผ่าน systemd)"

# ---- 3. ปิด sleep/suspend ----
echo "[3/5] ปิด sleep/suspend (กัน USB หลุดตอน run ยาว)..."
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target 2>/dev/null || true
echo "    ✓ ปิด suspend แล้ว"
echo "    (ถ้าเป็น laptop: ตั้งค่าเพิ่มให้ 'ปิดฝาแล้วไม่ sleep' ใน Settings > Power ด้วย)"

# ---- 4. ติดตั้ง systemd service ----
echo "[4/5] ติดตั้ง systemd service..."
# แทนค่า placeholder ในไฟล์ service
TMP_SVC="/tmp/${SERVICE_NAME}.service"
sed -e "s|__USER__|$RUN_USER|g" \
    -e "s|__WORKDIR__|$WORKDIR|g" \
    -e "s|__SAMPLE__|$SAMPLE|g" \
    "$WORKDIR/water-logger.service" > "$TMP_SVC"
sudo cp "$TMP_SVC" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
echo "    ✓ ติดตั้ง + enable (จะเริ่มเองตอนบูต)"

# ---- 5. เริ่ม service ----
echo "[5/5] เริ่ม service..."
sudo systemctl restart "${SERVICE_NAME}.service"
sleep 3
echo ""
echo "=============================================="
echo " เสร็จสิ้น! สถานะ service:"
echo "=============================================="
sudo systemctl status "${SERVICE_NAME}.service" --no-pager -l || true
echo ""
echo "=============================================="
echo " คำสั่งที่ใช้บ่อย:"
echo "=============================================="
echo "  ดู log สด        : journalctl -u $SERVICE_NAME -f"
echo "  หยุด             : sudo systemctl stop $SERVICE_NAME"
echo "  เริ่ม            : sudo systemctl start $SERVICE_NAME"
echo "  สถานะ            : systemctl status $SERVICE_NAME"
echo "  สร้างรายงาน      : cd $WORKDIR && python3 report.py --sample \"$SAMPLE\""
echo "  ข้อมูล CSV อยู่ที่ : $WORKDIR/water_data/"
echo "=============================================="

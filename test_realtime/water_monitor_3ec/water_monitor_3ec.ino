/*
 * ============================================================================
 *  water_monitor_3ec.ino  v8.3 — อ่าน EC 3 ตัว (SEN0706 RS485) ส่ง serial + ESP-NOW
 *  บอร์ด: ESP32 CONTROL CIRCUIT V1.2
 * ----------------------------------------------------------------------------
 *  - EC 3 ตัวบน RS485 bus เดียวกัน (addr 1, 2, 3)
 *  - ส่ง serial 2 แบบ: อ่านง่าย + บรรทัด DATA สำหรับ logger ฝั่ง Ubuntu
 *  - ส่ง ESP-NOW ไปแสดงผลที่จอ Waveshare ESP32-P4-WIFI6-Touch-LCD-7B
 *  - OLED 128x64 แบบหลายหน้า สลับอัตโนมัติ + กดปุ่ม BOOT เปลี่ยนหน้าเองได้
 * ----------------------------------------------------------------------------
 *  DATA format:  DATA,<ec1>,<t1>,<ec2>,<t2>,<ec3>,<t3>,<ok1><ok2><ok3>
 *  คำสั่ง serial:  T<hhmmss>  ตั้งนาฬิกา เช่น  T143005  = 14:30:05
 *  ไลบรารี: ModbusMaster | SSD1306 (ThingPulse)
 * ----------------------------------------------------------------------------
 *  เปลี่ยนจาก v8.2:
 *    [เพิ่ม] OLED หลายหน้า: OVERVIEW / SENSOR 1-3 / LINK / SYSTEM
 *    [เพิ่ม] แถบสถานะด้านบนทุกหน้า: นาฬิกา + heartbeat + จุดบอกหน้า
 *    [เพิ่ม] กราฟย่อ (sparkline) ในหน้ารายตัว
 *    [เพิ่ม] นาฬิกา ตั้งผ่าน serial ได้ (ก่อนตั้งจะแสดงเป็นเวลาที่เปิดเครื่อง)
 *    [เพิ่ม] ปุ่ม BOOT (GPIO0) เปลี่ยนหน้า — กดแล้วหยุดสลับอัตโนมัติ 20 วินาที
 * ============================================================================
 */

#include <Arduino.h>
#include <Wire.h>
#include <ModbusMaster.h>
#include "SSD1306.h"

#define FIRMWARE_VERSION "8.3-3EC"

// ====== ESP-NOW (ส่งค่า EC ไปจอ ESP32-P4 แบบ one-way) ======
#define ENABLE_ESPNOW 1

#if ENABLE_ESPNOW
  #include <WiFi.h>
  #include <esp_now.h>
  #include <esp_wifi.h>

  uint8_t PEER_MAC[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
  #define ESPNOW_CHANNEL 1

  // ต้องตรงกับ ec_packet.h ฝั่ง P4 เป๊ะ — 41 ไบต์
  typedef struct __attribute__((packed)) {
    uint32_t seq;
    float    ec[3];
    float    tw[3];
    int16_t  sal[3];
    int16_t  tds[3];
    uint8_t  ok;
  } EcPacket;

  static_assert(sizeof(EcPacket) == 41, "EcPacket ต้องเป็น 41 ไบต์ ให้ตรงกับฝั่งรับ");

  EcPacket txPkt;
  bool espnowOK = false;
  uint32_t espnowSeq = 0, espnowSent = 0, espnowFail = 0, espnowTxErr = 0;
  uint8_t  espnowChannel = 0;
  String   myMac = "--";

  extern float ecVal[], twVal[];
  extern int   salVal[], tdsVal[];
  extern bool  okVal[];

  // signature เปลี่ยนที่ ESP-IDF v5.4 (Arduino core 3.2+) จึงเช็คที่ IDF ไม่ใช่ core
  #if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
  void onEspNowSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) espnowSent++; else espnowFail++;
  }
  #else
  void onEspNowSent(const uint8_t *mac, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) espnowSent++; else espnowFail++;
  }
  #endif

  void espnowInit() {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    WiFi.setSleep(false);

    // ย้ายตัวเองไปช่องเดียวกับฝั่งรับ — ตั้งแค่ peer.channel ไม่พอ
    esp_wifi_set_promiscuous(true);
    esp_wifi_set_channel(ESPNOW_CHANNEL, WIFI_SECOND_CHAN_NONE);
    esp_wifi_set_promiscuous(false);

    if (esp_now_init() != ESP_OK) {
      Serial.println(F("[espnow] init ล้มเหลว — ทำงานต่อ (serial ปกติ)"));
      return;
    }
    esp_now_register_send_cb(onEspNowSent);

    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, PEER_MAC, 6);
    peer.channel = ESPNOW_CHANNEL;
    peer.ifidx   = WIFI_IF_STA;
    peer.encrypt = false;
    if (esp_now_add_peer(&peer) != ESP_OK) {
      Serial.println(F("[espnow] add peer ล้มเหลว"));
      return;
    }

    espnowOK = true;
    wifi_second_chan_t sec;
    esp_wifi_get_channel(&espnowChannel, &sec);
    myMac = WiFi.macAddress();

    Serial.printf("[espnow] พร้อม | ช่องจริง %u | packet %u ไบต์ | MAC ตัวเอง %s\n",
                  espnowChannel, (unsigned)sizeof(EcPacket), myMac.c_str());
    if (espnowChannel != ESPNOW_CHANNEL) {
      Serial.printf("[espnow] เตือน: ช่องไม่ตรงกับที่ตั้งไว้ (%d)\n", ESPNOW_CHANNEL);
    }
  }

  void espnowSend() {
    if (!espnowOK) return;
    txPkt.seq = ++espnowSeq;
    for (int i = 0; i < 3; i++) {
      txPkt.ec[i]  = ecVal[i];
      txPkt.tw[i]  = twVal[i];
      txPkt.sal[i] = (int16_t)constrain(salVal[i], -1, 32767);
      txPkt.tds[i] = (int16_t)constrain(tdsVal[i], -1, 32767);
    }
    txPkt.ok = (okVal[0] ? 1 : 0) | (okVal[1] ? 2 : 0) | (okVal[2] ? 4 : 0);

    esp_err_t e = esp_now_send(PEER_MAC, (uint8_t *)&txPkt, sizeof(txPkt));
    if (e != ESP_OK) {
      espnowTxErr++;
      Serial.printf("[espnow] send error: %s\n", esp_err_to_name(e));
    }
  }
#endif


// ---------------- พินแมป (V1.2) ----------------
const int RS485_RX  = 16;
const int RS485_TX  = 17;
const int I2C_SDA   = 21;
const int I2C_SCL   = 22;
const int BTN_PIN   = 0;      // ปุ่ม BOOT บนบอร์ด — กดแล้วเป็น LOW

// ---------------- ค่าคงที่ ----------------
#define SEN_BAUD       4800
#define N_SENSORS      3
const uint8_t ADDRS[N_SENSORS] = {1, 2, 3};

#define MB_GAP_MS      220
#define POLL_EVERY_MS  2500
#define SERIAL_OUT_MS  2500
#define OLED_MS        200          // วาดถี่ขึ้นเพื่อให้ animation ลื่น
#define PAGE_AUTO_MS   5000         // สลับหน้าอัตโนมัติทุก 5 วินาที
#define PAGE_HOLD_MS   20000        // กดปุ่มแล้วหยุดสลับอัตโนมัติ 20 วินาที

// ---------------- อ็อบเจกต์ ----------------
ModbusMaster   node;
SSD1306Wire    display(0x3C, I2C_SDA, I2C_SCL);
HardwareSerial RS485Serial(2);
bool oledOK = false;

// ---------------- สถานะต่อ sensor ----------------
float ecVal[N_SENSORS];
float twVal[N_SENSORS];
int   salVal[N_SENSORS];
int   tdsVal[N_SENSORS];
bool  okVal[N_SENSORS];
uint32_t okCount[N_SENSORS], errCount[N_SENSORS];

uint32_t bootMs = 0, loopCount = 0, loopRate = 0;
bool heartbeat = false;

// ---------------- นาฬิกา ----------------
// มีโมดูล RTC DS3231 อยู่บน I2C เส้นเดียวกับ OLED
// ถ้าตรวจไม่เจอจะถอยไปนับเวลาเองจาก millis() เหมือนเดิม
// ตั้งค่าผ่าน serial ได้ 2 แบบ:
//    T143005          -> ตั้งเฉพาะเวลา เก็บใน RAM หายเมื่อถอดไฟ
//    T20260806143005  -> ตั้งวันที่+เวลาลง DS3231 อยู่ได้แม้ถอดไฟ
bool     clockSet = false;
uint32_t clockBaseSec = 0;      // วินาทีของวันตอนที่ตั้ง
uint32_t clockBaseMs  = 0;      // millis() ตอนที่ตั้ง

#define DS3231_ADDR 0x68

struct RtcTime { uint16_t year; uint8_t mon, day, hour, min, sec; };

bool    rtcPresent = false;     // ตรวจเจอชิปบน I2C
bool    rtcValid   = false;     // อ่านออกมาแล้วค่าสมเหตุสมผล
RtcTime rtcNow;

static uint8_t bcd2dec(uint8_t b) { return (uint8_t)((b >> 4) * 10 + (b & 0x0F)); }
static uint8_t dec2bcd(uint8_t d) { return (uint8_t)(((d / 10) << 4) | (d % 10)); }

bool rtcDetect() {
  Wire.beginTransmission(DS3231_ADDR);
  return Wire.endTransmission() == 0;
}

bool rtcRead(RtcTime &t) {
  Wire.beginTransmission(DS3231_ADDR);
  Wire.write((uint8_t)0x00);
  if (Wire.endTransmission() != 0) return false;
  if (Wire.requestFrom(DS3231_ADDR, 7) != 7) return false;

  uint8_t s = Wire.read(), mi = Wire.read(), h = Wire.read();
  Wire.read();                            // วันในสัปดาห์ ไม่ได้ใช้
  uint8_t d = Wire.read(), mo = Wire.read(), y = Wire.read();

  t.sec = bcd2dec(s & 0x7F);
  t.min = bcd2dec(mi & 0x7F);

  // bit6 = 1 คือโหมด 12 ชั่วโมง ต้องแปลงเป็น 24 ชั่วโมงเอง
  if (h & 0x40) {
    uint8_t hr = bcd2dec(h & 0x1F) % 12;
    t.hour = (h & 0x20) ? hr + 12 : hr;   // bit5 = PM
  } else {
    t.hour = bcd2dec(h & 0x3F);
  }

  t.day  = bcd2dec(d & 0x3F);
  t.mon  = bcd2dec(mo & 0x1F);
  t.year = 2000 + bcd2dec(y);

  return (t.mon >= 1 && t.mon <= 12 && t.day >= 1 && t.day <= 31 &&
          t.hour < 24 && t.min < 60 && t.sec < 60);
}

bool rtcWrite(const RtcTime &t) {
  Wire.beginTransmission(DS3231_ADDR);
  Wire.write((uint8_t)0x00);
  Wire.write(dec2bcd(t.sec));
  Wire.write(dec2bcd(t.min));
  Wire.write(dec2bcd(t.hour));          // bit6 = 0 คือโหมด 24 ชั่วโมง
  Wire.write((uint8_t)1);               // วันในสัปดาห์ ไม่ได้ใช้
  Wire.write(dec2bcd(t.day));
  Wire.write(dec2bcd(t.mon));
  Wire.write(dec2bcd((uint8_t)(t.year - 2000)));
  return Wire.endTransmission() == 0;
}

uint32_t nowOfDaySec() {
  if (rtcValid) return rtcNow.hour * 3600UL + rtcNow.min * 60UL + rtcNow.sec;
  if (!clockSet) return (millis() - bootMs) / 1000;
  return (clockBaseSec + (millis() - clockBaseMs) / 1000) % 86400UL;
}

// ---------------- ส่งเวลาออกบัส RS485 ให้จอ P4 ----------------
// ยิงเป็นเฟรม Modbus "write multiple registers" ไปยัง address 0x20
// ซึ่งไม่มีเซนเซอร์ตัวไหนใช้ จึงไม่มีใครตอบและไม่รบกวนการอ่านเซนเซอร์เลย
// จอ P4 แค่ดักฟังอยู่บนบัสเดียวกัน จึงเก็บเฟรมนี้ไปตั้งนาฬิกาได้
#define TIME_FRAME_ADDR  0x20

uint16_t mbCRC(const uint8_t *d, size_t n) {
  uint16_t c = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    c ^= d[i];
    for (int b = 0; b < 8; b++) c = (c & 1) ? (uint16_t)((c >> 1) ^ 0xA001)
                                            : (uint16_t)(c >> 1);
  }
  return c;
}

void sendTimeFrame() {
  if (!rtcValid) return;

  uint8_t f[15];
  f[0]  = TIME_FRAME_ADDR;
  f[1]  = 0x10;                 // write multiple registers
  f[2]  = 0x00; f[3] = 0x00;    // เริ่มที่ register 0
  f[4]  = 0x00; f[5] = 0x03;    // 3 register
  f[6]  = 0x06;                 // 6 ไบต์
  f[7]  = (uint8_t)(rtcNow.year - 2000);
  f[8]  = rtcNow.mon;
  f[9]  = rtcNow.day;
  f[10] = rtcNow.hour;
  f[11] = rtcNow.min;
  f[12] = rtcNow.sec;
  uint16_t c = mbCRC(f, 13);
  f[13] = (uint8_t)(c & 0xFF);
  f[14] = (uint8_t)(c >> 8);

  RS485Serial.write(f, sizeof(f));
  RS485Serial.flush();
}

void formatClock(char *out, size_t n) {
  uint32_t s = nowOfDaySec();
  snprintf(out, n, "%02u:%02u:%02u",
           (unsigned)(s / 3600), (unsigned)((s / 60) % 60), (unsigned)(s % 60));
}

// ---------------- ประวัติสำหรับ sparkline ----------------
#define SPARK_N 42
float sparkBuf[N_SENSORS][SPARK_N];
uint8_t sparkCount = 0;
uint8_t sparkHead = 0;

void sparkPush() {
  for (int i = 0; i < N_SENSORS; i++) {
    sparkBuf[i][sparkHead] = okVal[i] ? ecVal[i] : NAN;
  }
  sparkHead = (sparkHead + 1) % SPARK_N;
  if (sparkCount < SPARK_N) sparkCount++;
}

// ---------------- หน้าจอ ----------------
enum {
  PAGE_OVERVIEW = 0,
  PAGE_S1, PAGE_S2, PAGE_S3,
  PAGE_LINK,
  PAGE_SYSTEM,
  PAGE_COUNT
};
int      page = PAGE_OVERVIEW;
uint32_t pageChangedMs = 0;
uint32_t holdUntilMs = 0;     // หยุดสลับอัตโนมัติจนถึงเวลานี้

// ============================================================================
//  อ่านเซนเซอร์ 1 ตัว
// ============================================================================
void readOne(int i) {
  node.begin(ADDRS[i], RS485Serial);

  uint8_t r = node.readHoldingRegisters(0x0000, 2);   // EC + Temp
  if (r == node.ku8MBSuccess) {
    ecVal[i] = node.getResponseBuffer(0) / 10.0f;
    twVal[i] = node.getResponseBuffer(1) / 10.0f;
    okVal[i] = true;
    okCount[i]++;
  } else {
    okVal[i] = false;
    errCount[i]++;
    delay(MB_GAP_MS);
    return;
  }
  delay(MB_GAP_MS);

  r = node.readHoldingRegisters(0x0002, 2);           // Salinity + TDS
  if (r == node.ku8MBSuccess) {
    salVal[i] = node.getResponseBuffer(0);
    tdsVal[i] = node.getResponseBuffer(1);
  } else {
    salVal[i] = -1; tdsVal[i] = -1;
  }
  delay(MB_GAP_MS);
}

// ============================================================================
//  คาลิเบรตเซนเซอร์ EC
// ----------------------------------------------------------------------------
//  SEN0706 คาลิเบรตที่ตัวเซนเซอร์เองผ่าน Modbus ค่าจึงติดอยู่กับเซนเซอร์
//  ใช้ได้ทั้งกับไฟล์ CSV ฝั่ง PC และจอ ESP32-P4 พร้อมกัน ไม่ต้องแก้สองที่
//
//     register 0x0110 = 0x0004            รหัสสั่งคาลิเบรต EC
//     register 0x0111 = ค่ามาตรฐาน x 10    (1413 uS/cm -> 14130)
//     เขียนทีเดียวสองตัวด้วย function 0x10
//
//  สั่งจาก PC ผ่าน serial:  C<n>,<uS>   เช่น  C1,1413
// ============================================================================
#define CAL_REG_CMD      0x0110
#define CAL_CMD_EC       0x0004

#define CAL_RC_BAD_RANGE   0xFF     /* ค่ามาตรฐานนอกช่วงที่หัววัดรองรับ */
#define CAL_RC_NOT_IN_SOL  0xF0     /* ค่าที่อ่านได้ห่างจากมาตรฐานเกินเหตุ */
#define CAL_SANITY_RATIO   0.5f     /* ห่างได้ไม่เกิน 50% */

// คืนรหัสผลของ ModbusMaster (ku8MBSuccess = สำเร็จ) เพื่อให้ผู้เรียกส่งต่อได้
uint8_t calibrateSensor(int i, float standardUS) {
  // SEN0706 (K=1) วัดได้ 1~2000 uS/cm การ cal นอกช่วงนี้ไม่มีความหมาย
  if (standardUS < 1 || standardUS > 2000) {
    Serial.printf("[cal] fail n=%d เหตุ=ค่ามาตรฐาน %.0f อยู่นอกช่วง 1-2000 uS/cm\n",
                  i + 1, standardUS);
    return CAL_RC_BAD_RANGE;
  }

  /*
   * ⚠️ กันคาลิเบรตตอนหัววัดไม่ได้จุ่มน้ำยาอยู่จริง
   *
   * การคาลิเบรตคือการบอกหัววัดว่า "ที่เจ้าวัดได้ตอนนี้คือค่ามาตรฐาน"
   * ถ้าหัววัดอยู่ในอากาศหรือเพิ่งถูกยกขึ้นมา ค่าจะผิดไปมาก แล้วคำสั่งนี้
   * จะทำให้หัววัดเพี้ยนถาวรทันที โดยที่ยังตอบกลับว่า "สำเร็จ"
   *
   * ด่านนี้อยู่ที่จุดลงมือ จึงคุ้มครองทุกทาง ทั้งปุ่มบนจอ คำสั่งผ่านบัส
   * และคำสั่ง serial ที่ใช้ตอน debug
   */
  if (!okVal[i] || isnan(ecVal[i]) ||
      fabsf(ecVal[i] - standardUS) / standardUS > CAL_SANITY_RATIO) {
    Serial.printf("[cal] fail n=%d ค่าตอนนี้ %.1f ห่างจาก %.0f uS/cm เกินไป "
                  "— จุ่มหัววัดในน้ำยาแล้วรอให้นิ่งก่อน\n",
                  i + 1, okVal[i] ? ecVal[i] : NAN, standardUS);
    return CAL_RC_NOT_IN_SOL;
  }

  uint16_t raw = (uint16_t)(standardUS * 10.0f + 0.5f);

  node.begin(ADDRS[i], RS485Serial);
  node.setTransmitBuffer(0, CAL_CMD_EC);
  node.setTransmitBuffer(1, raw);
  uint8_t r = node.writeMultipleRegisters(CAL_REG_CMD, 2);
  delay(MB_GAP_MS);

  if (r == node.ku8MBSuccess) {
    Serial.printf("[cal] ok n=%d std=%.0f uS/cm (reg=%u)\n", i + 1, standardUS, raw);
  } else {
    Serial.printf("[cal] fail n=%d rc=0x%02X (เซนเซอร์ไม่ตอบ/ปฏิเสธคำสั่ง)\n",
                  i + 1, r);
  }
  return r;
}

// ============================================================================
//  รับคำสั่งจากจอ ESP32-P4 ผ่านบัส RS485
// ----------------------------------------------------------------------------
//  จอเป็นฝ่ายฟังอย่างเดียวไม่ได้เป็น master  เวลาผู้ใช้กดปุ่มบนจอ จอจะส่งเฟรม
//  สั้น ๆ มาที่ address 0x21 (ไม่มีเซนเซอร์ตัวไหนใช้) แล้ว "บอร์ดนี้" เป็นคน
//  ยิงคำสั่ง Modbus ให้เซนเซอร์อีกที  บัสจึงยังมี master ตัวเดียวเหมือนเดิม
//  ไม่มีทางชนกันเชิงโครงสร้าง
//
//    รับ  21 10 00 00 00 03 06 <cmd> <A> <B_hi> <B_lo> <seq> <rsv> crc  (15 ไบต์)
//    ตอบ 22 10 00 00 00 03 06 <cmd> <A> <status> <rc> <seq> <rsv> crc  (15 ไบต์)
//         status: 0 = สำเร็จ, 1 = ล้มเหลว, 2 = รับคำสั่งแล้วกำลังทำ
// ============================================================================
#define CMD_FRAME_ADDR   0x21
#define ACK_FRAME_ADDR   0x22
#define BUS_FRAME_LEN    15

#define CMD_CAL          0x01     // A = เซนเซอร์ 1-3, B = ค่ามาตรฐาน uS/cm
#define CMD_OLED_PAGE    0x10     // A = หมายเลขหน้า
#define CMD_OLED_NEXT    0x11     // เปลี่ยนไปหน้าถัดไป
#define CMD_OLED_AUTO    0x12     // A = 0 หยุดสลับ / 1 สลับอัตโนมัติ

bool    oledAuto   = true;        // สลับหน้าอัตโนมัติอยู่มั้ย
uint8_t lastCmdSeq = 0xFF;        // กันทำซ้ำเมื่อได้เฟรมเดิมสองรอบ

// หน้าต่างเลื่อนขนาดเท่าเฟรม — ไบต์ใหม่เข้าท้าย ไบต์เก่าหลุดหัว
// วิธีนี้ไม่ต้องจัดการบัฟเฟอร์ให้ยุ่ง และทนขยะที่ปนมาบนบัสได้เอง
uint8_t cmdWin[BUS_FRAME_LEN];

bool frameCrcOK(const uint8_t *f, size_t n) {
  uint16_t got = (uint16_t)f[n - 2] | ((uint16_t)f[n - 1] << 8);
  return got == mbCRC(f, n - 2);
}

void sendAckFrame(uint8_t cmd, uint8_t a, uint8_t status, uint8_t rc, uint8_t seq) {
  uint8_t f[BUS_FRAME_LEN];
  f[0]  = ACK_FRAME_ADDR;
  f[1]  = 0x10;
  f[2]  = 0x00; f[3] = 0x00;
  f[4]  = 0x00; f[5] = 0x03;
  f[6]  = 0x06;
  f[7]  = cmd;
  f[8]  = a;
  f[9]  = status;
  f[10] = rc;
  f[11] = seq;
  f[12] = 0;
  uint16_t c = mbCRC(f, 13);
  f[13] = (uint8_t)(c & 0xFF);
  f[14] = (uint8_t)(c >> 8);

  RS485Serial.write(f, sizeof(f));
  RS485Serial.flush();
}

void touchOledPage(int newPage) {
  page = newPage % PAGE_COUNT;
  pageChangedMs = millis();
  holdUntilMs   = millis() + PAGE_HOLD_MS;
}

void handleBusCommand(const uint8_t *d) {
  uint8_t  cmd = d[0];
  uint8_t  a   = d[1];
  uint16_t b   = ((uint16_t)d[2] << 8) | d[3];
  uint8_t  seq = d[4];

  if (seq == lastCmdSeq) return;        // เฟรมซ้ำ ไม่ทำซ้ำ
  lastCmdSeq = seq;

  uint8_t status = 0, rc = 0;

  switch (cmd) {
    case CMD_CAL:
      if (a >= 1 && a <= N_SENSORS && b >= 1 && b <= 2000) {
        Serial.printf("[bus] จอสั่งคาลิเบรต EC#%u ที่ %u uS/cm\n", a, b);

        /*
         * ⚠️ ต้องตอบรับ "ก่อน" ลงมือคาลิเบรต
         *
         * calibrateSensor() ใช้เวลานานได้ถึง 2 วินาที เพราะ ModbusMaster
         * รอคำตอบจากหัววัดนานเท่านั้น (ku16MBResponseTimeout = 2000 ms)
         * ถ้าไม่รีบบอกจอว่าได้รับคำสั่งแล้ว จอจะนึกว่าเฟรมหาย แล้วส่งซ้ำ
         * เข้ามากลางคัน  ModbusMaster ที่กำลังรอคำตอบอยู่จะอ่านเฟรมนั้น
         * เป็นคำตอบ เห็นไบต์แรกเป็น 0x21 ไม่ใช่ 0x01 แล้วคืน
         * ku8MBInvalidSlaveID (0xE0) ทั้งที่หัววัดไม่ได้ผิดอะไรเลย
         */
        delay(30);
        sendAckFrame(cmd, a, 2, 0, seq);   /* 2 = รับคำสั่งแล้ว กำลังทำ */
        delay(30);

        rc = calibrateSensor(a - 1, (float)b);
        status = (rc == node.ku8MBSuccess) ? 0 : 1;
      } else {
        Serial.printf("[bus] คำสั่งคาลิเบรตไม่ถูกต้อง (n=%u std=%u)\n", a, b);
        status = 1;
      }
      break;

    case CMD_OLED_PAGE:
      if (a < PAGE_COUNT) {
        touchOledPage(a);
        Serial.printf("[bus] จอสั่งเปลี่ยนหน้า OLED เป็น %u\n", a);
      } else {
        status = 1;
      }
      break;

    case CMD_OLED_NEXT:
      touchOledPage(page + 1);
      Serial.printf("[bus] จอสั่งเปลี่ยนหน้า OLED -> %d\n", page);
      break;

    case CMD_OLED_AUTO:
      oledAuto = (a != 0);
      if (oledAuto) holdUntilMs = 0;    // ให้กลับมาสลับได้ทันที
      Serial.printf("[bus] จอสั่งสลับหน้าอัตโนมัติ = %s\n", oledAuto ? "เปิด" : "ปิด");
      break;

    default:
      Serial.printf("[bus] ไม่รู้จักคำสั่ง 0x%02X\n", cmd);
      status = 1;
      break;
  }

  delay(30);                            // เว้นช่องก่อนตอบ ตามสเปก Modbus
  sendAckFrame(cmd, a, status, rc, seq);
}

// เรียกจาก loop เฉพาะตอนที่ไม่ได้อยู่ระหว่างคุยกับเซนเซอร์
// (loop เป็น single thread และ readOne() ทำงานจนจบก่อนคืนค่า จึงปลอดภัยเอง)
void pollBusCommands() {
  while (RS485Serial.available()) {
    memmove(cmdWin, cmdWin + 1, BUS_FRAME_LEN - 1);
    cmdWin[BUS_FRAME_LEN - 1] = (uint8_t)RS485Serial.read();

    if (cmdWin[0] == CMD_FRAME_ADDR && cmdWin[1] == 0x10 &&
        frameCrcOK(cmdWin, BUS_FRAME_LEN)) {
      handleBusCommand(cmdWin + 7);
      memset(cmdWin, 0, sizeof(cmdWin));   // กันจับเฟรมเดิมซ้ำจากไบต์ค้าง
    }
  }
}

// ============================================================================
//  OLED — ส่วนประกอบร่วม
// ============================================================================

// แถบบนสุด: นาฬิกา + heartbeat + จุดบอกหน้า
void drawStatusBar(const char *title) {
  display.setFont(ArialMT_Plain_10);
  display.setTextAlignment(TEXT_ALIGN_LEFT);

  char clk[12];
  formatClock(clk, sizeof(clk));
  display.drawString(0, 0, clk);

  if (!clockSet) {           // ยังไม่ได้ตั้งนาฬิกา = แสดงเวลาที่เปิดเครื่อง
    display.drawString(52, 0, "up");
  }

  display.setTextAlignment(TEXT_ALIGN_RIGHT);
  display.drawString(112, 0, title);
  display.setTextAlignment(TEXT_ALIGN_LEFT);

  if (heartbeat) display.fillCircle(122, 5, 3); else display.drawCircle(122, 5, 3);

  display.drawHorizontalLine(0, 12, 128);
}

// จุดบอกหน้าที่ขอบล่าง
void drawPageDots() {
  const int n = PAGE_COUNT;
  const int gap = 9;
  int x0 = 64 - (n * gap) / 2 + gap / 2;
  for (int i = 0; i < n; i++) {
    if (i == page) display.fillCircle(x0 + i * gap, 61, 2);
    else           display.drawCircle(x0 + i * gap, 61, 1);
  }
}

// กราฟย่อของเซนเซอร์ตัวหนึ่ง วางในกรอบ (x,y,w,h)
void drawSparkline(int s, int x, int y, int w, int h) {
  if (sparkCount < 2) {
    display.setFont(ArialMT_Plain_10);
    display.drawString(x, y, "collecting...");
    return;
  }

  float mn = NAN, mx = NAN;
  for (int k = 0; k < sparkCount; k++) {
    float v = sparkBuf[s][k];
    if (isnan(v)) continue;
    if (isnan(mn) || v < mn) mn = v;
    if (isnan(mx) || v > mx) mx = v;
  }
  if (isnan(mn)) return;
  float span = mx - mn;
  if (span < 1.0f) span = 1.0f;       // กันหารศูนย์เมื่อค่านิ่ง

  int prevX = -1, prevY = -1;
  for (int k = 0; k < sparkCount; k++) {
    // ไล่จากเก่าสุดไปใหม่สุด
    int idx = (sparkHead + SPARK_N - sparkCount + k) % SPARK_N;
    float v = sparkBuf[s][idx];
    int px = x + (w - 1) * k / (sparkCount - 1);
    if (isnan(v)) { prevX = -1; continue; }
    int py = y + h - 1 - (int)((v - mn) / span * (h - 1));
    if (prevX >= 0) display.drawLine(prevX, prevY, px, py);
    else            display.setPixel(px, py);
    prevX = px; prevY = py;
  }
}

// ============================================================================
//  OLED — แต่ละหน้า
// ============================================================================

void pageOverview() {
  drawStatusBar("ALL");
  display.setFont(ArialMT_Plain_10);
  char b[26];
  for (int i = 0; i < N_SENSORS; i++) {
    int y = 15 + i * 14;
    if (okVal[i]) {
      snprintf(b, sizeof(b), "#%d %5.0f uS  %4.1fC", i + 1, ecVal[i], twVal[i]);
      display.drawString(0, y, b);
    } else {
      snprintf(b, sizeof(b), "#%d   --- ERR ---", i + 1);
      display.drawString(0, y, b);
    }
  }
  drawPageDots();
}

void pageSensor(int s) {
  char title[8];
  snprintf(title, sizeof(title), "EC #%d", s + 1);
  drawStatusBar(title);

  char b[26];
  if (!okVal[s]) {
    display.setFont(ArialMT_Plain_16);
    display.drawString(0, 26, "SENSOR ERROR");
    display.setFont(ArialMT_Plain_10);
    snprintf(b, sizeof(b), "err %lu / ok %lu",
             (unsigned long)errCount[s], (unsigned long)okCount[s]);
    display.drawString(0, 46, b);
    drawPageDots();
    return;
  }

  // ค่า EC ตัวใหญ่
  display.setFont(ArialMT_Plain_24);
  snprintf(b, sizeof(b), "%.0f", ecVal[s]);
  display.drawString(0, 14, b);

  display.setFont(ArialMT_Plain_10);
  display.drawString(0, 38, "uS/cm");

  // อุณหภูมิ + TDS + Salinity ด้านขวา
  snprintf(b, sizeof(b), "%.1f C", twVal[s]);
  display.setTextAlignment(TEXT_ALIGN_RIGHT);
  display.drawString(128, 16, b);
  if (tdsVal[s] >= 0) snprintf(b, sizeof(b), "TDS %d", tdsVal[s]);
  else                snprintf(b, sizeof(b), "TDS --");
  display.drawString(128, 27, b);
  if (salVal[s] >= 0) snprintf(b, sizeof(b), "SAL %d", salVal[s]);
  else                snprintf(b, sizeof(b), "SAL --");
  display.drawString(128, 38, b);
  display.setTextAlignment(TEXT_ALIGN_LEFT);

  drawSparkline(s, 0, 49, 128, 10);
  drawPageDots();
}

void pageLink() {
  drawStatusBar("LINK");
  display.setFont(ArialMT_Plain_10);
  char b[30];

#if ENABLE_ESPNOW
  if (espnowOK) {
    snprintf(b, sizeof(b), "ESP-NOW  ch %u  OK", espnowChannel);
    display.drawString(0, 15, b);
    display.drawString(0, 26, myMac.c_str());
    snprintf(b, sizeof(b), "seq %lu  sent %lu",
             (unsigned long)espnowSeq, (unsigned long)espnowSent);
    display.drawString(0, 37, b);
    snprintf(b, sizeof(b), "fail %lu  txerr %lu",
             (unsigned long)espnowFail, (unsigned long)espnowTxErr);
    display.drawString(0, 48, b);
  } else {
    display.drawString(0, 15, "ESP-NOW  ไม่พร้อม");
    display.drawString(0, 30, "ส่งผ่าน serial อย่างเดียว");
  }
#else
  display.drawString(0, 15, "ESP-NOW ปิดอยู่");
#endif
  drawPageDots();
}

void pageSystem() {
  drawStatusBar("SYS");
  display.setFont(ArialMT_Plain_10);
  char b[30];

  uint32_t up = (millis() - bootMs) / 1000;
  snprintf(b, sizeof(b), "uptime %02u:%02u:%02u",
           (unsigned)(up / 3600), (unsigned)((up / 60) % 60), (unsigned)(up % 60));
  display.drawString(0, 15, b);

  snprintf(b, sizeof(b), "loop %lu/s  v%s", (unsigned long)loopRate, FIRMWARE_VERSION);
  display.drawString(0, 26, b);

  snprintf(b, sizeof(b), "ok  %lu/%lu/%lu",
           (unsigned long)okCount[0], (unsigned long)okCount[1], (unsigned long)okCount[2]);
  display.drawString(0, 37, b);

  snprintf(b, sizeof(b), "err %lu/%lu/%lu",
           (unsigned long)errCount[0], (unsigned long)errCount[1], (unsigned long)errCount[2]);
  display.drawString(0, 48, b);

  drawPageDots();
}

void drawOLED() {
  if (!oledOK) return;
  display.clear();
  switch (page) {
    case PAGE_OVERVIEW: pageOverview();       break;
    case PAGE_S1:       pageSensor(0);        break;
    case PAGE_S2:       pageSensor(1);        break;
    case PAGE_S3:       pageSensor(2);        break;
    case PAGE_LINK:     pageLink();           break;
    case PAGE_SYSTEM:   pageSystem();         break;
    default:            pageOverview();       break;
  }
  display.display();
}

// ============================================================================
//  ปุ่มเปลี่ยนหน้า (BOOT / GPIO0)
// ============================================================================
void handleButton() {
  static bool lastLow = false;
  static uint32_t lastChangeMs = 0;

  bool low = (digitalRead(BTN_PIN) == LOW);
  uint32_t now = millis();

  if (low != lastLow && now - lastChangeMs > 40) {   // debounce 40ms
    lastChangeMs = now;
    lastLow = low;
    if (low) {                                        // ขอบขาลง = กด
      page = (page + 1) % PAGE_COUNT;
      pageChangedMs = now;
      holdUntilMs = now + PAGE_HOLD_MS;
    }
  }
}

// ============================================================================
//  คำสั่งทาง serial :  T<hhmmss>  ตั้งนาฬิกา
// ============================================================================
static int digits2(const char *p) { return (p[0] - '0') * 10 + (p[1] - '0'); }

void handleSerialCmd() {
  static char buf[24];
  static uint8_t len = 0;

  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      buf[len] = '\0';
      // C<n>,<uS>  -> คาลิเบรตเซนเซอร์ตัวที่ n เช่น  C1,1413
      if (buf[0] == 'C' || buf[0] == 'c') {
        int n = 0;
        float us = 0;
        if (sscanf(buf + 1, "%d,%f", &n, &us) == 2 && n >= 1 && n <= N_SENSORS) {
          calibrateSensor(n - 1, us);
        } else {
          Serial.println(F("[cal] fail รูปแบบคำสั่งไม่ถูก ใช้  C1,1413"));
        }
        len = 0;
        continue;
      }

      bool isT = (buf[0] == 'T' || buf[0] == 't');

      // T20260806143005 -> ตั้งวันที่+เวลาลง DS3231 (อยู่ได้แม้ถอดไฟ)
      if (isT && len == 15) {
        RtcTime t;
        t.year = (uint16_t)(digits2(buf + 1) * 100 + digits2(buf + 3));
        t.mon  = (uint8_t)digits2(buf + 5);
        t.day  = (uint8_t)digits2(buf + 7);
        t.hour = (uint8_t)digits2(buf + 9);
        t.min  = (uint8_t)digits2(buf + 11);
        t.sec  = (uint8_t)digits2(buf + 13);

        if (t.year < 2000 || t.mon < 1 || t.mon > 12 || t.day < 1 || t.day > 31 ||
            t.hour > 23 || t.min > 59 || t.sec > 59) {
          Serial.println(F("[rtc] ค่าวันที่/เวลาไม่ถูกต้อง"));
        } else if (!rtcPresent) {
          Serial.println(F("[rtc] ไม่พบ DS3231 — ใช้ T143005 ตั้งเฉพาะเวลาแทน"));
        } else if (rtcWrite(t)) {
          rtcNow = t;
          rtcValid = true;
          Serial.printf("[rtc] ตั้งเป็น %04u-%02u-%02u %02u:%02u:%02u\n",
                        t.year, t.mon, t.day, t.hour, t.min, t.sec);
          sendTimeFrame();                 // ส่งให้จอ P4 ทันที ไม่ต้องรอรอบ
        } else {
          Serial.println(F("[rtc] เขียน DS3231 ไม่สำเร็จ"));
        }
      }
      // T143005 -> ตั้งเฉพาะเวลา เก็บใน RAM หายเมื่อถอดไฟ
      else if (isT && len == 7) {
        int hh = digits2(buf + 1);
        int mm = digits2(buf + 3);
        int ss = digits2(buf + 5);
        if (hh < 24 && mm < 60 && ss < 60) {
          clockBaseSec = hh * 3600UL + mm * 60UL + ss;
          clockBaseMs  = millis();
          clockSet = true;
          Serial.printf("[clock] ตั้งเป็น %02d:%02d:%02d\n", hh, mm, ss);
        } else {
          Serial.println(F("[clock] ค่าเวลาไม่ถูกต้อง"));
        }
      }
      len = 0;
    } else if (len < sizeof(buf) - 1) {
      buf[len++] = c;
    }
  }
}

// ============================================================================
//  setup
// ============================================================================
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.print(F("\n=== Water Monitor 3xEC  v"));
  Serial.print(F(FIRMWARE_VERSION));
  Serial.println(F(" ==="));

  pinMode(BTN_PIN, INPUT_PULLUP);
  RS485Serial.begin(SEN_BAUD, SERIAL_8N1, RS485_RX, RS485_TX);

  Wire.begin(I2C_SDA, I2C_SCL);

  // ไล่ดูว่ามีอะไรอยู่บน I2C บ้าง เผื่อโมดูลหลุดจะได้รู้ทันทีจาก log
  Serial.print(F("[i2c] พบอุปกรณ์:"));
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) Serial.printf(" 0x%02X", a);
  }
  Serial.println();

  rtcPresent = rtcDetect();
  if (rtcPresent) {
    rtcValid = rtcRead(rtcNow);
    if (rtcValid) {
      Serial.printf("[rtc] DS3231 พร้อม  %04u-%02u-%02u %02u:%02u:%02u\n",
                    rtcNow.year, rtcNow.mon, rtcNow.day,
                    rtcNow.hour, rtcNow.min, rtcNow.sec);
    } else {
      Serial.println(F("[rtc] เจอ DS3231 แต่ค่าเวลาเพี้ยน — ตั้งด้วย T20260806143005"));
    }
  } else {
    Serial.println(F("[rtc] ไม่พบ DS3231 — ใช้นาฬิกานับเองจาก millis()"));
  }

  oledOK = display.init();
  if (oledOK) {
    display.flipScreenVertically();
    display.clear();
    display.setFont(ArialMT_Plain_16);
    display.drawString(0, 8, "EC Monitor x3");
    display.setFont(ArialMT_Plain_24);
    display.drawString(0, 28, "v" FIRMWARE_VERSION);
    display.display();
    delay(1200);
  }

  for (int i = 0; i < N_SENSORS; i++) {
    ecVal[i] = NAN; twVal[i] = NAN; salVal[i] = -1; tdsVal[i] = -1;
    okVal[i] = false; okCount[i] = 0; errCount[i] = 0;
    for (int k = 0; k < SPARK_N; k++) sparkBuf[i][k] = NAN;
  }

  Serial.printf("OLED %s | RS485 %d baud | sensors addr 1,2,3\n",
                oledOK ? "OK" : "X", SEN_BAUD);

#if ENABLE_ESPNOW
  espnowInit();
#endif

  Serial.println(F("รูปแบบ DATA,ec1,t1,ec2,t2,ec3,t3,ok1ok2ok3"));
  Serial.println(F("ตั้งนาฬิกา: พิมพ์  T143005  แล้ว Enter (= 14:30:05)"));
  Serial.println(F("ปุ่ม BOOT = เปลี่ยนหน้า OLED\n"));

  bootMs = millis();
  pageChangedMs = bootMs;
}

// ============================================================================
//  loop
// ============================================================================
void loop() {
  uint32_t now = millis();
  loopCount++;

  handleButton();
  handleSerialCmd();

  static uint32_t tRate = 0;
  if (now - tRate >= 1000) {
    tRate = now;
    loopRate = loopCount;
    loopCount = 0;
    heartbeat = !heartbeat;
  }

  // สลับหน้าอัตโนมัติ (ถ้าไม่ได้กดปุ่มค้างไว้ และจอสัมผัสไม่ได้สั่งให้หยุด)
  if (oledAuto && now > holdUntilMs && now - pageChangedMs >= PAGE_AUTO_MS) {
    page = (page + 1) % PAGE_COUNT;
    pageChangedMs = now;
  }

  // รับคำสั่งจากจอสัมผัส — ต้องอยู่นอกช่วงที่กำลังคุยกับเซนเซอร์
  pollBusCommands();

  // อ่าน 3 ตัวตามรอบ
  static uint32_t tPoll = 0;
  if (now - tPoll >= POLL_EVERY_MS) {
    tPoll = now;
    for (int i = 0; i < N_SENSORS; i++) readOne(i);
    sparkPush();
#if ENABLE_ESPNOW
    espnowSend();
#endif

    // ส่งเวลาให้จอ P4 ทุก 4 รอบ (~10 วินาที)
    // ยิงตรงนี้เพราะ readOne() เพิ่งคุยกับเซนเซอร์ครบทุกตัวแล้ว บัสจึงว่าง
    // หน่วงก่อนเล็กน้อยให้พ้นช่วงเว้นระหว่างเฟรมตามสเปก Modbus
    static uint8_t timeTick = 0;
    if (rtcValid && ++timeTick >= 4) {
      timeTick = 0;
      delay(30);
      sendTimeFrame();
    }
  }

  // อ่านนาฬิกาใหม่ทุกวินาที
  static uint32_t tRtc = 0;
  if (rtcPresent && now - tRtc >= 1000) {
    tRtc = now;
    RtcTime t;
    if (rtcRead(t)) { rtcNow = t; rtcValid = true; }
  }

  static uint32_t tDraw = 0;
  if (now - tDraw >= OLED_MS) { tDraw = now; drawOLED(); }

  // ส่ง serial
  static uint32_t tOut = 0;
  if (now - tOut >= SERIAL_OUT_MS) {
    tOut = now;

    for (int i = 0; i < N_SENSORS; i++) {
      Serial.printf("#%d ", i + 1);
      if (okVal[i]) Serial.printf("EC:%.1f T:%.1f  ", ecVal[i], twVal[i]);
      else          Serial.print("ERR  ");
    }
    Serial.println();

    Serial.print(F("DATA,"));
    for (int i = 0; i < N_SENSORS; i++) {
      if (okVal[i]) { Serial.print(ecVal[i], 1); Serial.print(','); Serial.print(twVal[i], 1); }
      else          { Serial.print(F("NaN,NaN")); }
      Serial.print(',');
    }
    for (int i = 0; i < N_SENSORS; i++) Serial.print(okVal[i] ? 1 : 0);
    Serial.println();

#if ENABLE_ESPNOW
    Serial.printf("[espnow] seq=%lu sent=%lu fail=%lu txerr=%lu\n",
                  (unsigned long)espnowSeq, (unsigned long)espnowSent,
                  (unsigned long)espnowFail, (unsigned long)espnowTxErr);
#endif
  }
}

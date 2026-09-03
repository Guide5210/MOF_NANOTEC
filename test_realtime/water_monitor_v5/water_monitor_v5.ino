/*
 * ============================================================================
 *  ESP32 Water Monitor v2  —  SEN0706 (4-in-1) + pH + RTC + OLED(health)
 *  บอร์ด: ESP32 CONTROL CIRCUIT V1.2
 * ----------------------------------------------------------------------------
 *  ค่าที่อ่าน/แสดง:
 *    SEN0706 (RS485 Modbus, ขา 16/17, 4800 baud, addr 1):
 *       - EC        µS/cm   (register 0x0000, /10)
 *       - Tw น้ำ    °C      (register 0x0001, /10)
 *       - Salinity  PPM     (register 0x0002)
 *       - TDS       PPM     (register 0x0003)
 *    pH  : analog GPIO35 (2-point cal + ชดเชยอุณหภูมิน้ำ)
 *    RTC : DS3231 -> วัน/เวลา
 *
 *  OLED เน้น "health/status ของบอร์ด":
 *    - RS485 ยังคุยอยู่ไหม + นับ error / success
 *    - RAM ว่าง, uptime, loop rate
 *    - วัน/เดือน/ปี + เวลา
 *    (ค่าวัดจริงดูใน Serial monitor ซึ่งมี timestamp แล้ว)
 * ----------------------------------------------------------------------------
 *  ไลบรารี: ModbusMaster (Doc Walker) | ESP8266+ESP32 OLED SSD1306 (ThingPulse)
 *           | RTClib (Adafruit)
 *  คำสั่ง Serial (115200):
 *    cal ph1 <pH> / cal ph2 <pH> / cal show / cal reset  (เติม ' force' ข้าม gate)
 *    time YYYY MM DD HH MM SS   ตั้งเวลา RTC
 *    debug on|off               ดู raw Modbus
 * ============================================================================
 */

#include <Arduino.h>
#include <EEPROM.h>
#include <math.h>
#include <Wire.h>
#include <ModbusMaster.h>
#include "SSD1306.h"
#include <RTClib.h>

// ---------------- พินแมป (schematic V1.2) ----------------
const int PH_PIN     = 35;
const int RS485_RX   = 16;
const int RS485_TX   = 17;
const int SW_SET_PIN = 13;
const int I2C_SDA    = 21;
const int I2C_SCL    = 22;

// ---------------- ค่าคงที่ปรับได้ ----------------
// pH ต่อตรงเข้า GPIO35 (ไม่ผ่าน divider) -> ตั้ง 1.0
// ถ้าค่า mV ออกมาต่ำกว่าจริงครึ่งหนึ่ง ค่อยเปลี่ยนเป็น 1.5
#define PH_DIVIDER     1.0f

#define SEN0706_ADDR   1
#define SEN0706_BAUD   4800

// EC software correction: จาก KCl 1413 µS/cm @ 20.6°C อ่านได้ 1363 -> 1413/1363
// ปรับเลขนี้ได้ถ้าคาลิเบรตใหม่ (ตั้ง 1.0 = ปิด correction)
#define EC_CAL_FACTOR  1.0367f

#define NUM_SAMPLES    30
#define TRIM_EACH_SIDE 5
#define PH_STABLE_MV   8.0f

#define MB_POLL_MS     1000
#define OLED_PAGE_MS   4000
#define SERIAL_OUT_MS  2000

// ---------------- อ็อบเจกต์ ----------------
ModbusMaster   ec_node;
SSD1306Wire    display(0x3C, I2C_SDA, I2C_SCL);
RTC_DS3231     rtc;
HardwareSerial RS485Serial(2);

bool oledOK = false, rtcOK = false, debugRaw = false;

// ---------------- pH cal ----------------
struct CalData { uint32_t magic; float phV[2]; float phVal[2]; };
const uint32_t CAL_MAGIC = 0x50484B32u;
CalData cal;

// ---------------- สถานะ ----------------
int   phBuf[NUM_SAMPLES]; int sampleIdx = 0;
float phMv = 0, phSpread = 999; bool havePH = false;

float ecValue = NAN, ecTempC = NAN, phValue = NAN;
int   salPPM  = -1,  tdsPPM  = -1;
bool  ecOK = false;

// health counters
uint32_t bootMs = 0, loopCount = 0, loopRate = 0;
uint32_t mbOKcount = 0, mbErrCount = 0;
uint32_t lastOKms = 0;
bool heartbeat = false;
int  oledPage = 0;

// ============================================================================
//  EEPROM
// ============================================================================
void setDefaults(bool save) {
  cal.magic = CAL_MAGIC;
  cal.phV[0]=1500; cal.phVal[0]=7.0f;
  cal.phV[1]=2000; cal.phVal[1]=4.0f;
  if (save){ EEPROM.put(0,cal); EEPROM.commit(); }
}
bool calSane(){
  if(cal.magic!=CAL_MAGIC) return false;
  if(cal.phV[0]==cal.phV[1]) return false;
  for(int i=0;i<2;i++) if(isnan(cal.phV[i])||isnan(cal.phVal[i])) return false;
  return true;
}
void loadCal(){ EEPROM.get(0,cal); if(!calSane()){ setDefaults(true); Serial.println(F("[EEPROM] cal เสีย -> ค่าโรงงาน")); } }
void saveCal(){ EEPROM.put(0,cal); EEPROM.commit(); }

// ============================================================================
//  median-trimmed mean
// ============================================================================
float trimmedMean(int*buf,int n,float*sp){
  static int t[NUM_SAMPLES];
  for(int i=0;i<n;i++) t[i]=buf[i];
  for(int i=1;i<n;i++){int k=t[i],j=i-1;while(j>=0&&t[j]>k){t[j+1]=t[j];j--;}t[j+1]=k;}
  long s=0;int c=0,lo=TRIM_EACH_SIDE,hi=n-TRIM_EACH_SIDE-1;
  for(int i=lo;i<=hi;i++){s+=t[i];c++;}
  if(sp)*sp=(float)(t[hi]-t[lo]);
  return c>0?(float)s/c:0;
}

// ============================================================================
//  pH conversion
// ============================================================================
float computePH(float mv,float tC){
  float dv=cal.phV[1]-cal.phV[0];
  float slope=(dv!=0)?(cal.phVal[1]-cal.phVal[0])/dv:0;
  float off=cal.phVal[0]-slope*cal.phV[0];
  float raw=slope*mv+off;
  float tk=(isnan(tC)?25.0f:tC)+273.15f;
  float ph=7.0f+(raw-7.0f)*(298.15f/tk);
  return ph<0?0:(ph>14?14:ph);
}

// ============================================================================
//  อ่าน SEN0706 : 2 ธุรกรรม (EC/Temp แล้ว Salinity/TDS)
//  scale ตามโค้ดทางการ: EC /10, Temp /10, Sal/TDS = ตรง
// ============================================================================
void readSEN0706(){
  bool ok1=false, ok2=false;

  // --- register 0-1 : EC, Temp ---
  uint8_t r = ec_node.readHoldingRegisters(0x0000, 2);
  if(r==ec_node.ku8MBSuccess){
    uint16_t rEC=ec_node.getResponseBuffer(0), rT=ec_node.getResponseBuffer(1);
    ecValue=((float)rEC/10.0f)*EC_CAL_FACTOR;   // µS/cm (แก้ด้วย correction factor)
    ecTempC=(float)rT/10.0f;       // °C
    ok1=true;
    if(debugRaw){Serial.print(F("[MB0] rawEC="));Serial.print(rEC);Serial.print(F(" rawT="));Serial.println(rT);}
  } else if(debugRaw){Serial.print(F("[MB0 err] 0x"));Serial.println(r,HEX);}

  delay(20); // เว้นจังหวะระหว่างธุรกรรม

  // --- register 2-3 : Salinity, TDS ---
  uint8_t r2 = ec_node.readHoldingRegisters(0x0002, 2);
  if(r2==ec_node.ku8MBSuccess){
    salPPM=(int)ec_node.getResponseBuffer(0);   // PPM
    tdsPPM=(int)ec_node.getResponseBuffer(1);   // PPM
    ok2=true;
    if(debugRaw){Serial.print(F("[MB1] sal="));Serial.print(salPPM);Serial.print(F(" tds="));Serial.println(tdsPPM);}
  } else if(debugRaw){Serial.print(F("[MB1 err] 0x"));Serial.println(r2,HEX);}

  ecOK = ok1;                 // ใช้ EC เป็นตัวชี้สถานะหลัก
  if(ok1){ mbOKcount++; lastOKms=millis(); } else mbErrCount++;
}

// ============================================================================
//  คาลิเบรต pH
// ============================================================================
void calibratePH(int p,float t,bool force){
  if(!havePH){Serial.println(F("!! ยังไม่มีค่า pH"));return;}
  if(!force&&phSpread>PH_STABLE_MV){
    Serial.print(F("!! pH ไม่นิ่ง spread "));Serial.print(phSpread,1);Serial.println(F(" mV (เติม ' force')"));return;
  }
  cal.phV[p]=phMv; cal.phVal[p]=t; saveCal();
  Serial.print(F(">> pH p"));Serial.print(p+1);Serial.print(F(" : "));Serial.print(phMv,1);
  Serial.print(F(" mV -> "));Serial.print(t,2);Serial.println(F(" [saved]"));
}
void showCal(){
  Serial.println(F("------- Calibration (pH) -------"));
  Serial.print(F("p1: "));Serial.print(cal.phV[0],1);Serial.print(F(" mV -> "));Serial.println(cal.phVal[0],2);
  Serial.print(F("p2: "));Serial.print(cal.phV[1],1);Serial.print(F(" mV -> "));Serial.println(cal.phVal[1],2);
  Serial.print(F("PH_DIVIDER="));Serial.println(PH_DIVIDER,3);
  Serial.println(F("--------------------------------"));
}

// ============================================================================
//  timestamp helper
// ============================================================================
void printTimestamp(){
  if(rtcOK){
    DateTime n=rtc.now(); char b[24];
    snprintf(b,sizeof(b),"[%02d:%02d:%02d] ",n.hour(),n.minute(),n.second());
    Serial.print(b);
  } else {
    Serial.print('['); Serial.print(millis()/1000); Serial.print(F("s] "));
  }
}

// ============================================================================
//  Serial parser
// ============================================================================
void handleSerial(){
  static char line[64]; static uint8_t idx=0;
  while(Serial.available()){
    char c=Serial.read();
    if(c=='\n'||c=='\r'){
      if(idx==0)continue;
      line[idx]=0; idx=0;
      String s=String(line); s.trim();
      bool force=s.endsWith(" force"); if(force)s=s.substring(0,s.length()-6);
      if(s=="cal show") showCal();
      else if(s=="cal reset"){ setDefaults(true); Serial.println(F(">> คืนค่าโรงงาน")); showCal(); }
      else if(s.startsWith("cal ph1 ")) calibratePH(0,s.substring(8).toFloat(),force);
      else if(s.startsWith("cal ph2 ")) calibratePH(1,s.substring(8).toFloat(),force);
      else if(s=="debug on"){ debugRaw=true; Serial.println(F(">> debug ON")); }
      else if(s=="debug off"){ debugRaw=false; Serial.println(F(">> debug OFF")); }
      else if(s.startsWith("time ")){
        int y,mo,d,h,mi,se;
        if(sscanf(s.c_str(),"time %d %d %d %d %d %d",&y,&mo,&d,&h,&mi,&se)==6){
          rtc.adjust(DateTime(y,mo,d,h,mi,se)); Serial.println(F(">> ตั้งเวลาแล้ว"));
        } else Serial.println(F("!! time YYYY MM DD HH MM SS"));
      }
      else Serial.println(F("?? ไม่รู้จักคำสั่ง (cal show)"));
    } else if(idx<sizeof(line)-1) line[idx++]=c;
  }
}

// ============================================================================
//  OLED : เน้น health/status
// ============================================================================
void drawOLED(){
  if(!oledOK) return;
  display.clear();
  display.setTextAlignment(TEXT_ALIGN_LEFT);

  // แถวหัว: ชื่อ + heartbeat (กระพริบ = ลูปยังวิ่ง)
  display.setFont(ArialMT_Plain_10);
  display.drawString(0,0,"SYSTEM STATUS");
  if(heartbeat) display.fillCircle(123,4,3); else display.drawCircle(123,4,3);

  if(oledPage==0){
    // หน้า 1: สถานะการสื่อสาร + หน่วยความจำ
    char b[26];
    // RS485 link: ถ้าอ่านสำเร็จภายใน 3 วิ = LIVE
    bool live = (millis()-lastOKms) < 3000;
    snprintf(b,sizeof(b),"RS485: %s", live?"LIVE":"NO DATA");
    display.setFont(ArialMT_Plain_16);
    display.drawString(0,12,b);

    display.setFont(ArialMT_Plain_10);
    snprintf(b,sizeof(b),"ok:%lu err:%lu",(unsigned long)mbOKcount,(unsigned long)mbErrCount);
    display.drawString(0,32,b);
    snprintf(b,sizeof(b),"RAM:%uK  %lu lp/s",
             (unsigned)(ESP.getFreeHeap()/1024),(unsigned long)loopRate);
    display.drawString(0,44,b);

    // uptime
    uint32_t up=(millis()-bootMs)/1000;
    snprintf(b,sizeof(b),"up %02lu:%02lu:%02lu",
             (unsigned long)(up/3600),(unsigned long)((up/60)%60),(unsigned long)(up%60));
    display.drawString(0,54,b);
  } else {
    // หน้า 2: วัน/เวลา + อุณหภูมิน้ำ
    if(rtcOK){
      DateTime n=rtc.now(); char d[24],t[16];
      snprintf(d,sizeof(d),"%04d-%02d-%02d",n.year(),n.month(),n.day());
      snprintf(t,sizeof(t),"%02d:%02d:%02d",n.hour(),n.minute(),n.second());
      display.setFont(ArialMT_Plain_10); display.drawString(0,14,d);
      display.setFont(ArialMT_Plain_24); display.drawString(0,26,t);
    } else { display.setFont(ArialMT_Plain_16); display.drawString(0,20,"RTC --"); }
    display.setFont(ArialMT_Plain_10);
    char b[24];
    if(!isnan(ecTempC)) snprintf(b,sizeof(b),"Water %.1f C",ecTempC);
    else                snprintf(b,sizeof(b),"Water --");
    display.drawString(0,54,b);
  }
  display.display();
}

// ============================================================================
//  setup
// ============================================================================
void setup(){
  Serial.begin(115200); delay(200);
  Serial.println(F("\n=== ESP32 Water Monitor v2 ==="));

  EEPROM.begin(64); loadCal();

  RS485Serial.begin(SEN0706_BAUD,SERIAL_8N1,RS485_RX,RS485_TX);
  ec_node.begin(SEN0706_ADDR,RS485Serial);

  analogReadResolution(12);
  analogSetPinAttenuation(PH_PIN,ADC_11db);

  Wire.begin(I2C_SDA,I2C_SCL);
  oledOK=display.init();
  if(oledOK){ display.flipScreenVertically(); display.clear(); display.display(); }
  rtcOK=rtc.begin();
  if(rtcOK&&rtc.lostPower()){ rtc.adjust(DateTime(F(__DATE__),F(__TIME__))); Serial.println(F("[RTC] ตั้งเวลา compile")); }

  pinMode(SW_SET_PIN,INPUT);

  Serial.printf("OLED %s | RTC %s | RS485 %d baud addr %d\n",
    oledOK?"OK":"X", rtcOK?"OK":"X", SEN0706_BAUD, SEN0706_ADDR);
  Serial.println(F("'debug on' ดู raw | 'cal show' ดูคาลิเบรต\n"));

  bootMs=millis(); lastOKms=0;
}

// ============================================================================
//  loop
// ============================================================================
void loop(){
  uint32_t now=millis(); loopCount++;

  static uint32_t tRate=0;
  if(now-tRate>=1000){ tRate=now; loopRate=loopCount; loopCount=0; heartbeat=!heartbeat; }

  // pH sampling
  phBuf[sampleIdx]=analogReadMilliVolts(PH_PIN); sampleIdx++;
  if(sampleIdx>=NUM_SAMPLES){
    sampleIdx=0; float sp; float mv=trimmedMean(phBuf,NUM_SAMPLES,&sp);
    phMv=mv*PH_DIVIDER; phSpread=sp*PH_DIVIDER; havePH=true;
    phValue=computePH(phMv,ecTempC);
  }

  static uint32_t tMB=0;
  if(now-tMB>=MB_POLL_MS){ tMB=now; readSEN0706(); }

  static uint32_t tPage=0;
  if(now-tPage>=OLED_PAGE_MS){ tPage=now; oledPage=(oledPage+1)%2; }
  static uint32_t tDraw=0;
  if(now-tDraw>=250){ tDraw=now; drawOLED(); }

  // Serial สรุป + timestamp
  static uint32_t tOut=0;
  if(now-tOut>=SERIAL_OUT_MS){
    tOut=now;
    printTimestamp();
    Serial.print(F("EC:")); if(ecOK&&!isnan(ecValue))Serial.print(ecValue,1);else Serial.print(F("--"));
    Serial.print(F("uS Tw:")); if(!isnan(ecTempC))Serial.print(ecTempC,1);else Serial.print(F("--"));
    Serial.print(F("C Sal:")); if(salPPM>=0)Serial.print(salPPM);else Serial.print(F("--"));
    Serial.print(F(" TDS:")); if(tdsPPM>=0)Serial.print(tdsPPM);else Serial.print(F("--"));
    Serial.print(F("ppm pH:")); if(!isnan(phValue))Serial.print(phValue,2);else Serial.print(F("--"));
    Serial.print(F("(")); Serial.print(phMv,0); Serial.print(F("mV)"));
    Serial.print(F(" RS485:")); Serial.println(ecOK?F("OK"):F("X"));

    // บรรทัด machine-readable สำหรับ Python parser
    // รูปแบบ: DATA,<EC>,<Tw>,<Sal>,<TDS>,<pH>,<mV>,<rs485ok>
    // ค่าที่อ่านไม่ได้ = NaN (Python จัดการเอง), Python ใส่ timestamp เวลาคอมเอง
    Serial.print(F("DATA,"));
    if(ecOK&&!isnan(ecValue)) Serial.print(ecValue,1); else Serial.print(F("NaN")); Serial.print(',');
    if(!isnan(ecTempC))       Serial.print(ecTempC,1); else Serial.print(F("NaN")); Serial.print(',');
    if(salPPM>=0)             Serial.print(salPPM);     else Serial.print(F("NaN")); Serial.print(',');
    if(tdsPPM>=0)             Serial.print(tdsPPM);     else Serial.print(F("NaN")); Serial.print(',');
    if(!isnan(phValue))       Serial.print(phValue,2);  else Serial.print(F("NaN")); Serial.print(',');
    Serial.print(phMv,0); Serial.print(',');
    Serial.println(ecOK?1:0);
  }

  handleSerial();
}

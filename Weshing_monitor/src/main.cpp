/*
 * Hydroponic Monitor (based on electroniclinic.com by Engr. Fahad)
 * EC, TDS, pH, Temperature
 * + Two-point calibration for pH and EC via Serial commands (saved to NVS)
 * Blynk/WiFi removed - standalone serial output.
 */
#include <Arduino.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <Preferences.h> // NVS storage (built-in ESP32) for saving calibration

unsigned long int avgval;
int buffer_arr[10], temp;

float ph_act;

// --- Reading scheduler (replaces BlynkTimer) ---
unsigned long lastReadMs = 0;
const unsigned long READ_INTERVAL_MS = 5000;


namespace pin {
const byte tds_sensor = A0; // 36 (EC/TDS probe analog pin)
const byte one_wire_bus = 5; // Dallas Temperature Sensor
const byte ph_sensor = 35;   // pH probe analog pin
}

namespace device {
float aref = 3.3; // Vref, this is for 3.3v compatible controller boards, for arduino use 5.0v.
}

namespace sensor {
float ec = 0;
unsigned int tds = 0;
float waterTemp = 0;
}

// ---------------- Two-point calibration ----------------
// Linear model: value = slope * voltage + offset
// pH  : voltage = median-filtered probe voltage
// EC  : voltage = temperature-compensated probe voltage (Vc)
Preferences prefs;

// Defaults reproduce the original hard-coded behaviour
float phSlope  = -5.70;          // original: ph_act = -5.70 * volt + 14.95
float phOffset = 15.65 - 0.7;    // = 14.95
float ecSlope  = 1.0;            // original: ec = Vc * 1.0
float ecOffset = 0.0;

// Captured calibration points: [0] = point 1, [1] = point 2
float phCalVolt[2], phCalRef[2];
bool  phCalSet[2] = {false, false};
float ecCalVolt[2], ecCalRef[2];
bool  ecCalSet[2] = {false, false};
// -------------------------------------------------------

OneWire oneWire(pin::one_wire_bus);
DallasTemperature dallasTemperature(&oneWire);

// Forward declarations (required for .cpp / PlatformIO)
void EC_and_ph();
void readEC();
void ph_Sensor();
float readPhVoltage();
float readEcVoltage();
void loadCalibration();
void handleSerialCommands();
void printHelp();
void printCalStatus();


void setup() {
  Serial.begin(115200); // Debugging on hardware Serial 0
  dallasTemperature.begin();

  loadCalibration(); // restore saved calibration from NVS
  printHelp();
}


void loop() {
  handleSerialCommands(); // listen for calibration commands

  if (millis() - lastReadMs >= READ_INTERVAL_MS) {
    lastReadMs = millis();
    EC_and_ph();
  }
}

// Returns temperature-compensated EC probe voltage (Vc)
float readEcVoltage() {
  dallasTemperature.requestTemperatures();
  sensor::waterTemp = dallasTemperature.getTempCByIndex(0);
  float rawEc = analogRead(pin::tds_sensor) * device::aref / 4096; // analog value -> voltage
  float temperatureCoefficient = 1.0 + 0.02 * (sensor::waterTemp - 25.0); // temperature compensation
  return rawEc / temperatureCoefficient;
}

void readEC() {
  float Vc = readEcVoltage();
  sensor::ec = ecSlope * Vc + ecOffset; // two-point calibrated EC
  sensor::tds = (133.42 * pow(sensor::ec, 3) - 255.86 * sensor::ec * sensor::ec + 857.39 * sensor::ec) * 0.5; // EC -> TDS
  Serial.print(F("TDS:")); Serial.println(sensor::tds);
  Serial.print(F("EC:")); Serial.println(sensor::ec, 2);
  Serial.print(F("Temperature:")); Serial.println(sensor::waterTemp, 2);
}

// Returns median-filtered pH probe voltage
float readPhVoltage()
{
  for (int i = 0; i < 10; i++)
  {
    buffer_arr[i] = analogRead(pin::ph_sensor);
    delay(30);
  }
  for (int i = 0; i < 9; i++)
  {
    for (int j = i + 1; j < 10; j++)
    {
      if (buffer_arr[i] > buffer_arr[j])
      {
        temp = buffer_arr[i];
        buffer_arr[i] = buffer_arr[j];
        buffer_arr[j] = temp;
      }
    }
  }
  avgval = 0;
  for (int i = 2; i < 8; i++)
    avgval += buffer_arr[i];
  return (float)avgval * 3.3 / 4096.0 / 6;
}

void ph_Sensor()
{
  float volt = readPhVoltage();
  ph_act = phSlope * volt + phOffset; // two-point calibrated pH
  Serial.print("pH Val: ");
  Serial.println(ph_act);
}

void EC_and_ph()
{
  readEC();
  ph_Sensor();
}

// ---------------- Calibration helpers ----------------

void loadCalibration() {
  prefs.begin("cal", true); // read-only
  phSlope  = prefs.getFloat("phS", phSlope);
  phOffset = prefs.getFloat("phO", phOffset);
  ecSlope  = prefs.getFloat("ecS", ecSlope);
  ecOffset = prefs.getFloat("ecO", ecOffset);
  prefs.end();
}

void savePhCalibration() {
  prefs.begin("cal", false);
  prefs.putFloat("phS", phSlope);
  prefs.putFloat("phO", phOffset);
  prefs.end();
}

void saveEcCalibration() {
  prefs.begin("cal", false);
  prefs.putFloat("ecS", ecSlope);
  prefs.putFloat("ecO", ecOffset);
  prefs.end();
}

// Compute slope/offset once both pH points are captured
void computePhCalibration() {
  if (!(phCalSet[0] && phCalSet[1])) return;
  float dv = phCalVolt[0] - phCalVolt[1];
  if (fabs(dv) < 1e-5) {
    Serial.println(F("[pH] ERROR: two points too close, re-do calibration."));
    phCalSet[0] = phCalSet[1] = false;
    return;
  }
  phSlope  = (phCalRef[0] - phCalRef[1]) / dv;
  phOffset = phCalRef[0] - phSlope * phCalVolt[0];
  savePhCalibration();
  phCalSet[0] = phCalSet[1] = false;
  Serial.print(F("[pH] CALIBRATED -> slope=")); Serial.print(phSlope, 4);
  Serial.print(F(" offset=")); Serial.println(phOffset, 4);
}

// Compute slope/offset once both EC points are captured
void computeEcCalibration() {
  if (!(ecCalSet[0] && ecCalSet[1])) return;
  float dv = ecCalVolt[0] - ecCalVolt[1];
  if (fabs(dv) < 1e-5) {
    Serial.println(F("[EC] ERROR: two points too close, re-do calibration."));
    ecCalSet[0] = ecCalSet[1] = false;
    return;
  }
  ecSlope  = (ecCalRef[0] - ecCalRef[1]) / dv;
  ecOffset = ecCalRef[0] - ecSlope * ecCalVolt[0];
  saveEcCalibration();
  ecCalSet[0] = ecCalSet[1] = false;
  Serial.print(F("[EC] CALIBRATED -> slope=")); Serial.print(ecSlope, 4);
  Serial.print(F(" offset=")); Serial.println(ecOffset, 4);
}

void printCalStatus() {
  Serial.println(F("----- Calibration status -----"));
  Serial.print(F("pH: slope=")); Serial.print(phSlope, 4);
  Serial.print(F(" offset=")); Serial.println(phOffset, 4);
  Serial.print(F("EC: slope=")); Serial.print(ecSlope, 4);
  Serial.print(F(" offset=")); Serial.println(ecOffset, 4);
  Serial.println(F("------------------------------"));
}

void printHelp() {
  Serial.println(F("\n=== Two-point calibration (Serial commands) ==="));
  Serial.println(F("Dip probe in standard #1, wait stable, then send:"));
  Serial.println(F("  PH_CAL1 <pH>    e.g. PH_CAL1 7.0"));
  Serial.println(F("  PH_CAL2 <pH>    e.g. PH_CAL2 4.0   (auto-computes & saves)"));
  Serial.println(F("  EC_CAL1 <mS/cm> e.g. EC_CAL1 1.413"));
  Serial.println(F("  EC_CAL2 <mS/cm> e.g. EC_CAL2 12.88 (auto-computes & saves)"));
  Serial.println(F("  CAL_STATUS      show current coefficients"));
  Serial.println(F("  CAL_RESET       restore default calibration"));
  Serial.println(F("  HELP            show this help"));
  Serial.println(F("==============================================\n"));
}

void handleSerialCommands() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() == 0) return;

  int sp = line.indexOf(' ');
  String cmd = (sp < 0) ? line : line.substring(0, sp);
  String arg = (sp < 0) ? ""   : line.substring(sp + 1);
  cmd.toUpperCase();
  float val = arg.toFloat();

  if (cmd == "PH_CAL1" || cmd == "PH_CAL2") {
    int idx = (cmd == "PH_CAL1") ? 0 : 1;
    phCalVolt[idx] = readPhVoltage();
    phCalRef[idx]  = val;
    phCalSet[idx]  = true;
    Serial.print(F("[pH] point ")); Serial.print(idx + 1);
    Serial.print(F(" captured: ref=")); Serial.print(val, 2);
    Serial.print(F(" volt=")); Serial.println(phCalVolt[idx], 4);
    computePhCalibration();
  }
  else if (cmd == "EC_CAL1" || cmd == "EC_CAL2") {
    int idx = (cmd == "EC_CAL1") ? 0 : 1;
    ecCalVolt[idx] = readEcVoltage();
    ecCalRef[idx]  = val;
    ecCalSet[idx]  = true;
    Serial.print(F("[EC] point ")); Serial.print(idx + 1);
    Serial.print(F(" captured: ref=")); Serial.print(val, 3);
    Serial.print(F(" volt=")); Serial.println(ecCalVolt[idx], 4);
    computeEcCalibration();
  }
  else if (cmd == "CAL_STATUS") {
    printCalStatus();
  }
  else if (cmd == "CAL_RESET") {
    phSlope = -5.70; phOffset = 15.65 - 0.7;
    ecSlope = 1.0;   ecOffset = 0.0;
    savePhCalibration();
    saveEcCalibration();
    phCalSet[0] = phCalSet[1] = ecCalSet[0] = ecCalSet[1] = false;
    Serial.println(F("[CAL] reset to defaults."));
    printCalStatus();
  }
  else if (cmd == "HELP") {
    printHelp();
  }
  else {
    Serial.print(F("[CAL] unknown command: ")); Serial.println(cmd);
    Serial.println(F("Type HELP for the list of commands."));
  }
}

// PSL-iPPG2C -> ESP32-D0WD DevKit / DWKIT V4
//
// Wiring
//   iPPG2C 5V   -> ESP32 VIN / 5V
//   iPPG2C GND  -> ESP32 GND
//   iPPG2C PPG  -> ESP32 GPIO34 (ADC1_CH6)
//   iPPG2C Beat -> ESP32 GPIO35 (ADC1_CH7)
//
// Notes
//   - PSL-iPPG2C is powered from 5V, but its signal outputs are specified as 0~3.3V.
//   - ADC1 pins are used so Wi-Fi activity does not interfere with sampling.
//
// Serial monitor: 1000000 baud
// Output:
//   RAW,<ms>,<sample>,<ppg_raw>,<beat_raw>,<ppg_v>,<beat_v>
//   BEAT,<ms>,<sample>,<bpm>,<ibi_ms>

#include <Arduino.h>

constexpr uint8_t PIN_PPG = 34;
constexpr uint8_t PIN_BEAT = 35;
constexpr uint8_t PIN_HEART_LED = 2;

constexpr uint32_t SERIAL_BAUD = 1000000;
constexpr uint32_t SAMPLE_RATE_HZ = 200;
constexpr uint32_t SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

constexpr uint8_t ADC_BITS = 12;
constexpr uint32_t ADC_MAX_COUNTS = (1UL << ADC_BITS) - 1UL;
constexpr float ADC_REF_V = 3.3f;

constexpr float BEAT_THRESHOLD_HIGH_V = 1.75f;
constexpr float BEAT_THRESHOLD_LOW_V = 1.55f;
constexpr uint32_t BEAT_THRESHOLD_HIGH = (uint32_t)((BEAT_THRESHOLD_HIGH_V / ADC_REF_V) * ADC_MAX_COUNTS + 0.5f);
constexpr uint32_t BEAT_THRESHOLD_LOW = (uint32_t)((BEAT_THRESHOLD_LOW_V / ADC_REF_V) * ADC_MAX_COUNTS + 0.5f);

static uint32_t g_sample_index = 0;
static uint32_t g_last_beat_ms = 0;
static uint32_t g_last_ibi_ms = 0;
static float g_last_bpm = 0.0f;
static bool g_beat_state = false;
static uint32_t g_next_sample_us = 0;

static inline float countsToVolts(uint32_t counts) {
  return (counts * ADC_REF_V) / (float)ADC_MAX_COUNTS;
}

static void printHeader() {
  Serial.println(F("type,ms,sample,ppg_raw,beat_raw,ppg_v,beat_v"));
}

static void configureAdc() {
  analogReadResolution(ADC_BITS);
#if defined(ADC_11db)
  analogSetPinAttenuation(PIN_PPG, ADC_11db);
  analogSetPinAttenuation(PIN_BEAT, ADC_11db);
#endif
}

static void processOneSample(uint32_t now_us) {
  const uint32_t now_ms = now_us / 1000UL;
  const uint32_t ppg_raw = (uint32_t)analogRead(PIN_PPG);
  const uint32_t beat_raw = (uint32_t)analogRead(PIN_BEAT);
  const float ppg_v = countsToVolts(ppg_raw);
  const float beat_v = countsToVolts(beat_raw);

  bool new_beat = false;
  if (!g_beat_state && beat_raw >= BEAT_THRESHOLD_HIGH) {
    g_beat_state = true;
    new_beat = true;
  } else if (g_beat_state && beat_raw <= BEAT_THRESHOLD_LOW) {
    g_beat_state = false;
  }

  digitalWrite(PIN_HEART_LED, g_beat_state ? HIGH : LOW);

  if (new_beat) {
    if (g_last_beat_ms != 0) {
      g_last_ibi_ms = now_ms - g_last_beat_ms;
      if (g_last_ibi_ms > 0) {
        g_last_bpm = 60000.0f / (float)g_last_ibi_ms;
      }
    }
    g_last_beat_ms = now_ms;
  }

  Serial.print(F("RAW,"));
  Serial.print(now_ms);
  Serial.print(',');
  Serial.print(g_sample_index);
  Serial.print(',');
  Serial.print(ppg_raw);
  Serial.print(',');
  Serial.print(beat_raw);
  Serial.print(',');
  Serial.print(ppg_v, 4);
  Serial.print(',');
  Serial.println(beat_v, 4);

  if (new_beat) {
    Serial.print(F("BEAT,"));
    Serial.print(now_ms);
    Serial.print(',');
    Serial.print(g_sample_index);
    Serial.print(',');
    Serial.print(g_last_bpm, 2);
    Serial.print(',');
    Serial.println(g_last_ibi_ms);
  }

  g_sample_index++;
}

void setup() {
  pinMode(PIN_HEART_LED, OUTPUT);
  digitalWrite(PIN_HEART_LED, LOW);

  configureAdc();

  Serial.begin(SERIAL_BAUD);
  delay(500);

  printHeader();
  Serial.println(F("INFO,ESP32_D0WD,200Hz,ADC12bit"));

  g_next_sample_us = micros() + SAMPLE_PERIOD_US;
}

void loop() {
  uint32_t now = micros();

  while ((int32_t)(now - g_next_sample_us) >= 0) {
    processOneSample(g_next_sample_us);
    g_next_sample_us += SAMPLE_PERIOD_US;
    now = micros();
  }
}

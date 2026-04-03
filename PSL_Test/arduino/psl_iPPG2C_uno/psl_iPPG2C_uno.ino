// PSL-iPPG2C -> Arduino UNO
//
// Wiring
//   iPPG2C 5V   -> UNO 5V
//   iPPG2C GND  -> UNO GND
//   iPPG2C PPG  -> UNO A0
//   iPPG2C Beat -> UNO A1
//
// Serial monitor: 1000000 baud
// Output:
//   RAW,<ms>,<sample>,<ppg_raw>,<beat_raw>,<ppg_v>,<beat_v>
//   BEAT,<ms>,<sample>,<bpm>,<ibi_ms>
//
// This sketch follows the signal structure shown in PSL_Data Arduino examples,
// but uses a micros()-based scheduler so it can compile on classic Arduino UNO
// without extra timer libraries.

constexpr uint8_t PIN_PPG = A0;
constexpr uint8_t PIN_BEAT = A1;
constexpr uint8_t PIN_HEART_LED = 13;

constexpr unsigned long SERIAL_BAUD = 1000000UL;
constexpr unsigned long SAMPLE_RATE_HZ = 250UL;
constexpr unsigned long SAMPLE_PERIOD_US = 1000000UL / SAMPLE_RATE_HZ;

constexpr float ADC_REF_V = 5.0f;
constexpr unsigned int ADC_MAX_COUNTS = 1023U;

constexpr float BEAT_THRESHOLD_HIGH_V = 1.65f;
constexpr float BEAT_THRESHOLD_LOW_V = 1.55f;
constexpr unsigned int BEAT_THRESHOLD_HIGH = (unsigned int)((BEAT_THRESHOLD_HIGH_V / ADC_REF_V) * ADC_MAX_COUNTS + 0.5f);
constexpr unsigned int BEAT_THRESHOLD_LOW = (unsigned int)((BEAT_THRESHOLD_LOW_V / ADC_REF_V) * ADC_MAX_COUNTS + 0.5f);

unsigned long g_sample_index = 0;
unsigned long g_last_beat_ms = 0;
unsigned long g_last_ibi_ms = 0;
float g_last_bpm = 0.0f;
bool g_beat_state = false;
unsigned long g_next_sample_us = 0;

static float countsToVolts(unsigned int counts) {
  return (counts * ADC_REF_V) / (float)ADC_MAX_COUNTS;
}

static void printHeader() {
  Serial.println(F("type,ms,sample,ppg_raw,beat_raw,ppg_v,beat_v"));
}

static void processOneSample(unsigned long now_us) {
  unsigned long now_ms = now_us / 1000UL;
  unsigned int ppg_raw = (unsigned int)analogRead(PIN_PPG);
  unsigned int beat_raw = (unsigned int)analogRead(PIN_BEAT);
  float ppg_v = countsToVolts(ppg_raw);
  float beat_v = countsToVolts(beat_raw);

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

  Serial.begin(SERIAL_BAUD);
  delay(300);

  printHeader();
  Serial.println(F("INFO,UNO,250Hz,ADC10bit"));
  g_next_sample_us = micros() + SAMPLE_PERIOD_US;
}

void loop() {
  unsigned long now = micros();
  while ((long)(now - g_next_sample_us) >= 0) {
    processOneSample(g_next_sample_us);
    g_next_sample_us += SAMPLE_PERIOD_US;
    now = micros();
  }
}

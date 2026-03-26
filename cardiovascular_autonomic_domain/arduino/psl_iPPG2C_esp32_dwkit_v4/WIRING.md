# PSL-iPPG2C to ESP32-DWVKIT V4 Wiring

This setup preserves the same serial CSV format already consumed by the Python capture scripts.

## Pin map

| PSL-iPPG2C | ESP32-DWVKIT V4 | Notes |
| --- | --- | --- |
| 5V | VIN / 5V | Sensor module spec says DC 5V input |
| GND | GND | Common ground required |
| PPG | GPIO34 | ADC1 input, analog only |
| Beat | GPIO35 | ADC1 input, analog only |

## ASCII schematic

```text
PSL-iPPG2C                       ESP32-DWVKIT V4
-----------                      ----------------
5V      -----------------------> VIN / 5V
GND     -----------------------> GND
PPG     -----------------------> GPIO34 (ADC1_CH6)
Beat    -----------------------> GPIO35 (ADC1_CH7)
```

## Why this mapping

- The module is powered from 5V, but its published signal output range is 0~3.3V, which matches ESP32 ADC input range.
- GPIO34 and GPIO35 are ADC1 inputs, so they avoid the ADC2/Wi-Fi conflict on classic ESP32 boards.
- The sketch keeps the existing `RAW,...` and `BEAT,...` serial lines so `capture_and_analyze.py` and `sequential_measurement_session.py` can be reused without changes.

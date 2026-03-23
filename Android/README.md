# Android Viewer

`Android` contains only the Android receiver app.

- App role: connect to the laptop hotspot and receive WebRTC over `GET /health` and `POST /offer`
- Server location: `C:\AI_HealthCare\BackEnd`
- Default port: `8080`

## Android Studio

1. Open `C:\AI_HealthCare\Android` in Android Studio.
2. Run the `app` module after Gradle sync.
3. The app connects automatically to the Wi-Fi gateway address.

## Connection behavior

- Android 7.1.2 target
- auto connect on app launch
- no manual server input
- current default requested track count: `1`

## Run the FastAPI backend

```powershell
cd C:\AI_HealthCare\BackEnd
C:\AI_HealthCare\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
C:\AI_HealthCare\.venv\Scripts\python.exe .\main.py --camera 1
```

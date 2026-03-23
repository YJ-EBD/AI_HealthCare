# AI Healthcare BackEnd

`BackEnd` is now a FastAPI backend for the Android viewer.

It provides:

- `GET /health`
- `POST /offer`
- `GET /` simple status page

The laptop camera capture and WebRTC answer generation are handled inside the same FastAPI process.

## Run

```powershell
cd C:\AI_HealthCare\BackEnd
C:\AI_HealthCare\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
C:\AI_HealthCare\.venv\Scripts\python.exe .\main.py --camera 1
```

## Camera selection

Single camera:

```powershell
C:\AI_HealthCare\.venv\Scripts\python.exe .\main.py --camera 1
```

Multiple cameras:

```powershell
C:\AI_HealthCare\.venv\Scripts\python.exe .\main.py --camera 0 --camera 1
```

Probe camera indexes:

```powershell
C:\AI_HealthCare\.venv\Scripts\python.exe .\main.py --list-cameras
```

## Environment example

You can also use `.env` values based on `.env.example`.

## Defaults

- Host: `0.0.0.0`
- Port: `8080`
- Camera size: `640x480`
- FPS: `15`

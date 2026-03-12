# YOLO Model Manager

Ein schlanker Flask-Server mit Web-UI zum Verwalten und Testen von Ultralytics YOLO-Modellen (Detection, Segmentation, Pose).

![Flask](https://img.shields.io/badge/Flask-2.x-black) ![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-8a2be2) ![License](https://img.shields.io/badge/License-MIT-green)

## Inhalt
- [Features](#features)
- [Schnellstart](#schnellstart)
- [Quellen-Optionen](#quellen-optionen)
- [API-Endpunkte](#api-endpunkte)
- [Konfiguration](#konfiguration)
- [Limits & Sicherheit](#limits--sicherheit)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## Features
- **Model Hub:** Modelle auflisten, laden/entladen, aktives Modell hervorheben.
- **Model Testing:** Bild- und Video-Inferenz, Drag&Drop, Preview, Status/Logs, annotierte Ergebnisse im UI.
- **Video-Streaming:** Fortschritt via SSE, annotierte Videos werden als MP4 zurückgegeben.
- **Batch-Images:** Mehrere Bilder hochladen, Ergebnisse + annotierte Dateien erhalten.
- **Webcam/URL/Stream:** Quellen nach Docs (siehe `yolo_prediction_docs.md`) auswählbar.
- **History & Export:** Letzte Ergebnisse mit Detections, CSV/JSON/summary-Export.

## Schnellstart
1. Abhängigkeiten installieren (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
2. Modelle (.pt) in `models/` legen.
3. Server starten:
   ```bash
   python app.py
   ```
4. Browser öffnen: `http://127.0.0.1:5000`

## Quellen-Optionen
| Modus  | UI-Tab | Backend-Route | Hinweis |
|--------|--------|---------------|---------|
| Bild   | Bild   | `POST /test` | Einzelbild, annotiertes PNG |
| Video  | Video  | `POST /test_video_stream` | SSE-Progress, MP4-Output |
| URL    | URL    | `POST /test_url_stream` | YouTube/RTSP/HTTP; speichert optional |
| Webcam | Webcam | `GET/POST /test_webcam` | MJPEG-Stream per Kamera-ID |
| Batch  | Batch  | `POST /test_batch` | Mehrere Bilder, JSON + erste annotierte Vorschau |
| Stream | Stream | `POST /test_url_stream` | Gleich wie URL, für Live-Streams gedacht |

## API-Endpunkte
- `GET /` – Web-UI
- `GET /models` – Modell-Liste
- `POST /load/<model>` – Modell laden
- `POST /test` – Bild-Inferenz
- `POST /test_video` – Sync-Video mit annotated MP4
- `POST /test_video_stream` – Video mit SSE-Progress, annotated MP4
- `POST /test_batch` – Mehrere Bilder
- `GET/POST /test_webcam` – MJPEG-Webcam-Stream
- `POST /test_url_stream` – URL/Stream mit SSE
- `GET /history` – History abrufen (Filter `source_type`, Limit)
- `POST /history/clear` – History leeren
- `GET /exports/results/{json|csv|summary}` – Exporte
- `GET /uploads/<file>` – Ergebnisse/Assets laden

### Beispiel: Bild-Inferenz per cURL
```bash
curl -X POST http://127.0.0.1:5000/test \
  -F "image=@/path/to/image.jpg" \
  -F "conf=0.25" -F "iou=0.45" -F "imgsz=640" -F "save=true"
```

## Konfiguration
Alle zentral in `app.py`:
- `MAX_VIDEO_MB`, `MAX_VIDEO_DURATION_S` – Upload-Limits
- `MODEL_DIR`, `UPLOAD_FOLDER`, `UPLOAD_FOLDER_ABS` – Pfade
- `results_history` – speichert letzte Runs (maxlen 100)

## Limits & Sicherheit
- FFmpeg wird fürs Transkodieren bevorzugt; ohne ffmpeg fallback zu OpenCV.
- Für lange Quellen nutzt der Code `stream=True`, um RAM-Leaks zu vermeiden.
- GPU wird nur verwendet, wenn `torch.cuda.is_available()` true ist.
- `MAX_CONTENT_LENGTH` auf 500 MB gesetzt, Videos zusätzlich durch Größe/Dauer begrenzt.

## Troubleshooting
- **400 bei `/test_url_stream`**: Sicherstellen, dass ein Modell geladen ist (UI zeigt aktives Modell, sonst zuerst `POST /load/<model>`). In der UI wird das Modell jetzt automatisch nachgeladen.
- **Keine Modelle in der Liste**: `.pt`-Dateien in `models/` legen und `↻` klicken.
- **Video wird nicht abgespielt**: Prüfen, ob ffmpeg installiert ist; sonst wird OpenCV-Fallback genutzt.
- **Webcam schwarz**: Kamera-ID prüfen (`0`, `1`, ...); Browser lädt MJPEG von `/test_webcam`.

## Development
- Stack: Flask, reines JS/CSS (keine Bundler).
- Frontend: `templates/index.html`, `static/app.js`, `static/styles.css`
- Backend: `app.py`
- Tests manuell via UI oder cURL; kein automatisches Test-Setup vorhanden.

# YOLO Model Manager

Ein schlanker Flask-Server mit Web-UI zum Verwalten und Testen von Ultralytics YOLO-Modellen (Detection, Segmentation, Pose).

## Features
- **Model Hub:** Modelle auflisten, laden/entladen, aktives Modell hervorheben.
- **Model Testing:** Bild- und Video-Inferenz, Drag&Drop, Preview, Status/Logs, annotierte Ergebnisse im UI.
- **Video-Streaming:** Fortschritt via SSE, annotierte Videos werden als MP4 zurückgegeben.
- **Batch-Images:** Mehrere Bilder hochladen, Ergebnisse + annotierte Dateien erhalten.
- **Webcam/URL/Stream:** Quellen nach Docs (siehe `yolo_prediction_docs.md`) auswählbar.
- **Limits & Safety:** Dateigröße/Dauer-Limits für Videos, kein `show=True`, `stream=True` für lange Quellen.

## Quickstart
1) Abhängigkeiten installieren (Python 3.10+):
```bash
pip install -r requirements.txt
```
2) Modelle in `models/` legen (`.pt`).
3) Server starten:
```bash
python app.py
```
4) Browser: `http://127.0.0.1:5000`

## Wichtige Endpunkte
- `GET /` – Web-UI
- `GET /models` – Modell-Liste
- `POST /load/<model>` – Modell laden
- `POST /test` – Bild/URL/Webcam/Verzeichnis/Glob/Stream (JSON-Resultate, optional Annotate)
- `POST /test_video` – Sync-Video mit annotated MP4
- `POST /test_video_stream` – Video mit SSE-Progress, annotated MP4
- `POST /test_batch` – Mehrere Bilder
- `GET/POST /test_webcam` – MJPEG-Webcam-Stream
- `GET /uploads/<file>` – Ergebnisse/Assets

## Hinweise aus den Ultralytics-Docs
- Für lange Quellen immer `stream=True` nutzen, um RAM-Leaks zu vermeiden (im Code umgesetzt).
- Annotierte Outputs werden in `uploads/` abgelegt; Videos werden bei Bedarf nach H.264 MP4 transkodiert.

## Konfiguration
Im `app.py`:
- `MAX_VIDEO_MB`, `MAX_VIDEO_DURATION_S` – Upload-Limits
- `MODEL_DIR`, `UPLOAD_FOLDER` – Pfade

## Known Constraints
- FFmpeg wird bevorzugt fürs Transkodieren; ohne ffmpeg fällt der Code auf OpenCV zurück.
- Für GPU-Inferenz muss `torch.cuda.is_available()` true sein und das Modell kompatibel.

## Development
Code-Stil: Flask + reines JS/CSS (keine Bundler).  
Front-End Assets: `templates/index.html`, `static/app.js`, `static/styles.css`.  
Server: `app.py`.

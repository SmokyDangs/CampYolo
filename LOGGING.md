# Logging & Debugging Dokumentation

## Übersicht

CampYolo verfügt über ein umfassendes Logging-System für einfaches Debugging und Überwachung.

## Log-Konfiguration

### Log-Verzeichnis
- **Pfad**: `logs/`
- **App-Logs**: `logs/app_YYYYMMDD_HHMMSS.log` (zeitgestempelt)
- **Error-Logs**: `logs/errors.log` (nur Fehler, persistent)

### Log-Level
| Level | Beschreibung |
|-------|-------------|
| DEBUG | Detaillierte Informationen für Entwicklung |
| INFO | Allgemeine Betriebsinformationen |
| WARNING | Warnungen, keine Funktionsbeeinträchtigung |
| ERROR | Fehler, die Funktionalität beeinträchtigen |
| CRITICAL | Kritische Fehler |

### Log-Format
```
YYYY-MM-DD HH:MM:SS | LEVEL | Logger | Funktion:Zeile | Nachricht
```

**Beispiel:**
```
2026-03-20 13:45:29 | INFO | CampYolo | test_webcam:702 | 📷 Webcam-Stream angefordert: ID=0, conf=0.25, imgsz=640
```

## Debug-Endpoints

### 1. Health-Check
```bash
GET /debug/health
```

**Antwort:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-20T13:44:20.565900",
  "checks": {
    "model_loaded": true,
    "active_model": "yolov8n-pose.pt",
    "cuda_available": true,
    "upload_folder_exists": true,
    "model_folder_exists": true,
    "upload_folder_writable": true
  }
}
```

### 2. System-Informationen
```bash
GET /debug/info
```

**Antwort enthält:**
- System-Informationen (Platform, Python-Version)
- Bibliotheks-Versionen (PyTorch, OpenCV, NumPy)
- CUDA-Informationen (GPU-Name, Speicher)
- Modell-Informationen (verfügbare Modelle, aktives Modell)
- Konfiguration (Upload-Limits, Video-Limits)

### 3. Log-Einträge anzeigen
```bash
GET /debug/logs?lines=100&type=app
```

**Parameter:**
- `lines` (optional): Anzahl der Zeilen (Default: 100)
- `type` (optional): `app` für App-Logs, `error` für Error-Logs

## Log-Beispiele

### Modell laden
```
2026-03-20 13:45:16 | INFO | CampYolo | load_model:408 | 📦 Modell-Laden angefordert: 'yolov8n-pose.pt'
2026-03-20 13:45:16 | DEBUG | CampYolo | load_model:416 | ✅ Modell-Datei existiert: models/yolov8n-pose.pt (6.51 MB)
2026-03-20 13:45:16 | INFO | CampYolo | load_model:426 | ⏳ Lade Modell 'yolov8n-pose.pt'...
2026-03-20 13:45:16 | DEBUG | CampYolo | log_gpu_stats:148 | GPU Speicher: 0.00 MB allokiert, 0.00 MB reserviert
2026-03-20 13:45:16 | INFO | CampYolo | load_model:433 | ✅ Modell 'yolov8n-pose.pt' erfolgreich geladen in 0.05s
2026-03-20 13:45:16 | DEBUG | CampYolo | log_model_info:158 | Modell-Parameter: 3,295,470 (3.30M)
```

### Webcam-Stream
```
2026-03-20 13:45:29 | INFO | CampYolo | test_webcam:702 | 📷 Webcam-Stream angefordert: ID=0, conf=0.25, imgsz=640
2026-03-20 13:45:29 | INFO | CampYolo | test_webcam:715 | ✅ Webcam 0 geöffnet: 640x480 @ 30.0 FPS
2026-03-20 13:45:29 | DEBUG | CampYolo | generate:729 | 🎬 Starte Webcam-Stream für Kamera 0
2026-03-20 13:45:30 | DEBUG | CampYolo | generate:737 | 🔥 Warmup-Inferenz für ersten Frame...
2026-03-20 13:45:30 | DEBUG | CampYolo | generate:741 | ✅ Warmup abgeschlossen
2026-03-20 13:45:30 | DEBUG | CampYolo | generate:771 | 📊 Frame 0: 205.9ms (Ø: 205.9ms)
```

### Request-Logging
```
2026-03-20 13:45:29 | DEBUG | requests | before_request:105 | ▶▶▶ GET /test_webcam | IP: 127.0.0.1
2026-03-20 13:45:29 | DEBUG | requests | before_request:107 |     Query-Params: {'cam_id': '0', 'conf': '0.25', 'imgsz': '640'}
2026-03-20 13:45:29 | INFO | requests | after_request:122 | ◀◀◀ GET /test_webcam | Status: 200 | Dauer: 1.26ms
```

## Fehlerbehandlung

### Globaler Exception-Handler
Alle unbehandelten Exceptions werden automatisch geloggt mit:
- Eindeutiger Error-ID
- Vollständigem Stacktrace
- JSON-Response für API-Clients

**Error-Log Beispiel:**
```
2026-03-20 14:00:00 | ERROR | CampYolo | handle_exception:130 | ❗ UNBEHANDELTER FEHLER [20260320140000123456]: CUDA error
2026-03-20 14:00:00 | ERROR | CampYolo | handle_exception:131 | Stacktrace:
Traceback (most recent call last):
  File "/app/app.py", line 123, in function
    problematic_code()
  ...
```

## Nützliche Commands

### Logs in Echtzeit anzeigen
```bash
tail -f logs/app_*.log
```

### Nur Fehler anzeigen
```bash
tail -f logs/errors.log
```

### Logs nach特定em Fehler durchsuchen
```bash
grep "ERROR" logs/app_*.log | tail -20
```

### Performance-Analyse (langsame Requests)
```bash
grep "Dauer:" logs/app_*.log | awk -F'Dauer: ' '{print $2}' | awk '{if ($1 > 1000) print}'
```

## Emoji-Legende

| Emoji | Bedeutung |
|-------|-----------|
| 📦 | Modell-Operation |
| ✅ | Erfolg |
| ❌ | Fehler |
| ⏳ | Laufende Operation |
| 🔄 | Neustart/Reload |
| 📷 | Webcam |
| 🎬 | Stream-Start |
| 🔥 | Warmup |
| 📊 | Statistik/Metriken |
| 🚀 | Server-Start |
| 📁 | Verzeichnis/Datei |
| 🔧 | Konfiguration/Debug |
| ⚠️ | Warnung |
| ❗ | Kritischer Fehler |

## Tipps für effektives Debugging

1. **Health-Check zuerst**: Prüfen Sie mit `/debug/health`, ob das System gesund ist
2. **System-Info**: Verwenden Sie `/debug/info` für Umgebungsinformationen
3. **Live-Logs**: `tail -f logs/app_*.log` für Echtzeit-Überwachung
4. **Error-Only**: `logs/errors.log` enthält nur Fehler
5. **Request-Dauer**: Achten Sie auf `Dauer:` im Log für Performance-Probleme
6. **GPU-Speicher**: Debug-Logs zeigen GPU-Speichernutzung vor/nach Operationen

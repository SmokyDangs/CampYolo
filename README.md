# 🚀 CampYolo - YOLO Vision 2026

Ein moderner Flask-Server mit Web-UI zum Verwalten und Testen von Ultralytics YOLO-Modellen für Objekterkennung, Pose-Estimation und Segmentierung.

![Flask](https://img.shields.io/badge/Flask-3.x-black)
![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-8a2be2)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Inhaltsverzeichnis

- [Features](#-features)
- [Schnellstart](#-schnellstart)
- [Installation](#-installation)
- [Benutzeroberfläche](#-benutzeroberfläche)
- [API-Endpunkte](#-api-endpunkte)
- [Projektstruktur](#-projektstruktur)
- [Konfiguration](#-konfiguration)
- [Logging](#-logging)
- [Troubleshooting](#-troubleshooting)
- [Development](#-development)

---

## ✨ Features

### Modell-Verwaltung
- **Model Hub**: Unterstützt YOLOv5, YOLOv8, YOLOv10, YOLOv11, YOLO12
- **Auto-Download**: Modell-Katalog mit direktem Download
- **Modell-Typen**: Detection, Pose, Segmentation, Classification

### Testing & Inferenz
- **Bild-Analyse**: Drag & Drop, Einzelbild-Inferenz
- **Video-Verarbeitung**: Mit SSE-Fortschrittsanzeige
- **Webcam-Stream**: Echtzeit-Objekterkennung
- **Batch-Verarbeitung**: Mehrere Bilder gleichzeitig
- **URL/Stream**: YouTube, RTSP, HTTP-Streams

### Dataset Verwaltung
- **Few-Shot Learning**: Automatische Annotation mit 2-3 Beispielen
- **Auto-Annotation**: Grounding DINO Integration
- **Export**: JSON, CSV, Summary-Reports

### Weitere Features
- **History**: Letzte Inferenzen mit Export-Funktion
- **Modell-Vergleich**: Mehrere Modelle parallel testen
- **Dark/Light Mode**: Anpassbares UI
- **Responsive Design**: Funktioniert auf Desktop und Tablet

---

## 🚀 Schnellstart

### 1. Installation

```bash
# Repository klonen
git clone <repository-url>
cd CampYolo-main

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate     # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 2. Modelle vorbereiten

```bash
# Modelle automatisch herunterladen oder
# .pt Dateien in models/ Verzeichnis legen
ls models/
# yolov8n.pt  yolov8n-pose.pt
```

### 3. Server starten

```bash
python run.py
```

### 4. Browser öffnen

```
http://127.0.0.1:5000
```

---

## 💻 Installation

### System-Voraussetzungen

- **Python**: 3.12 oder höher
- **GPU**: Optional (CUDA 11.8+ für GPU-Beschleunigung)
- **RAM**: Mindestens 4GB (8GB+ empfohlen)
- **Speicher**: 1GB+ für Modelle

### Abhängigkeiten

```bash
# Core
pip install flask ultralytics torch opencv-python numpy

# Optional für erweiterte Features
pip install requests tqdm pillow
```

### GPU Support (Optional)

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 🎨 Benutzeroberfläche

### Navigation

| Bereich | Beschreibung |
|---------|-------------|
| **Modelle** | Verfügbare YOLO-Modelle verwalten und laden |
| **Testing** | Bilder/Videos analysieren mit verschiedenen Quellen |
| **Dataset verwalten** | Automatische Annotation mit Few-Shot Learning |
| **Training** | Modell-Training konfigurieren und starten |
| **Vergleich** | Mehrere Modelle parallel vergleichen |
| **Verlauf** | Letzte Inferenzen anzeigen und exportieren |
| **Statistik** | Detaillierte Auswertungen und Metriken |
| **Einstellungen** | Anwendungskonfiguration |

### Tastenkürzel

| Taste | Aktion |
|-------|--------|
| `T` | Dark/Light Mode umschalten |
| `?` | Shortcuts anzeigen |

---

## 🔌 API-Endpunkte

### Modell-Management

| Methode | Endpunkt | Beschreibung |
|---------|----------|-------------|
| `GET` | `/models` | Verfügbare Modelle auflisten |
| `GET` | `/models/catalog` | Modell-Katalog für Downloads |
| `POST` | `/models/download/<model>` | Modell herunterladen |
| `POST` | `/load/<model>` | Modell laden |

### Inferenz

| Methode | Endpunkt | Beschreibung |
|---------|----------|-------------|
| `POST` | `/test` | Bild-Inferenz |
| `POST` | `/test_video_stream` | Video mit SSE-Progress |
| `POST` | `/test_batch` | Batch-Bilder Inferenz |
| `GET` | `/test_webcam` | Webcam MJPEG-Stream |
| `POST` | `/test_url_stream` | URL/Stream Inferenz |

### Dataset

| Methode | Endpunkt | Beschreibung |
|---------|----------|-------------|
| `POST` | `/dataset/annotate` | Auto-Annotation mit Prompt |
| `GET` | `/dataset/results` | Annotationsergebnisse |

### History & Export

| Methode | Endpunkt | Beschreibung |
|---------|----------|-------------|
| `GET` | `/history` | Verlauf abrufen |
| `POST` | `/history/clear` | Verlauf löschen |
| `GET` | `/exports/results/<format>` | Export (json/csv/summary) |

### Beispiel: Bild-Inferenz per cURL

```bash
curl -X POST http://127.0.0.1:5000/test \
  -F "image=@/path/to/image.jpg" \
  -F "conf=0.25" \
  -F "iou=0.45" \
  -F "imgsz=640" \
  -F "save=true"
```

---

## 📁 Projektstruktur

```
CampYolo-main/
├── app.py                 # Haupt-Flask-Anwendung
├── run.py                 # Server-Startskript
├── requirements.txt       # Python-Abhängigkeiten
├── README.md             # Diese Dokumentation
├── LOGGING.md            # Logging-Dokumentation
├── .gitignore            # Git Ignore Regeln
│
├── models/               # YOLO Modell-Dateien (.pt)
│   ├── yolov8n.pt
│   └── yolov8n-pose.pt
│
├── uploads/              # Hochgeladene Dateien (temporär)
│   └── .gitkeep
│
├── logs/                 # Anwendungs-Logs
│   ├── app_*.log
│   └── errors.log
│
├── templates/            # HTML Templates
│   └── index.html
│
└── static/               # Frontend-Ressourcen
    ├── app.js           # Haupt-JavaScript
    ├── training.js      # Trainings-Logik
    └── styles.css       # Stylesheet
```

---

## ⚙️ Konfiguration

### Zentrale Einstellungen (app.py)

```python
# Upload-Limits
MAX_VIDEO_MB = 150
MAX_VIDEO_DURATION_S = 60
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Pfade
MODEL_DIR = 'models'
UPLOAD_FOLDER = 'uploads'
LOG_DIR = 'logs'

# Gerät
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Umgebungsvariablen

```bash
# Optional in .env Datei
FLASK_ENV=development
FLASK_DEBUG=1
CUDA_VISIBLE_DEVICES=0
```

---

## 📝 Logging

CampYolo verwendet ein umfassendes Logging-System:

### Log-Level

| Level | Beschreibung |
|-------|-------------|
| `DEBUG` | Detaillierte Entwicklungsinformationen |
| `INFO` | Allgemeine Betriebsinformationen |
| `WARNING` | Warnungen, keine Funktionsbeeinträchtigung |
| `ERROR` | Fehler, die Funktionalität beeinträchtigen |
| `CRITICAL` | Schwere Fehler, Anwendung kann nicht weiterlaufen |

### Log-Dateien

- `logs/app_YYYYMMDD_HHMMSS.log` - Haupt-Log
- `logs/errors.log` - Nur Fehler

### Debug-Endpoints

```bash
GET /debug/health      # System-Gesundheitscheck
GET /debug/info        # System-Informationen
GET /debug/logs        # Logs anzeigen
```

---

## 🔧 Troubleshooting

### Häufige Probleme

| Problem | Lösung |
|---------|--------|
| **Keine Modelle sichtbar** | `.pt` Dateien in `models/` legen, Aktualisieren klicken |
| **Video wird nicht verarbeitet** | FFmpeg Installation prüfen |
| **Webcam zeigt schwarzes Bild** | Kamera-ID ändern (0, 1, 2...) |
| **GPU wird nicht genutzt** | CUDA Installation prüfen, `torch.cuda.is_available()` testen |
| **400 Fehler bei URL-Stream** | Modell muss vorher geladen werden |

### GPU Probleme

```bash
# CUDA Verfügbarkeit prüfen
python -c "import torch; print(torch.cuda.is_available())"

# GPU Info
nvidia-smi
```

---

## 🛠️ Development

### Tech Stack

- **Backend**: Flask 3.x, Python 3.12+
- **ML**: Ultralytics YOLO, PyTorch, OpenCV
- **Frontend**: Vanilla JS, CSS3 (kein Framework)
- **Styling**: Custom CSS mit Variablen

### Code-Struktur

```
app.py
├── Logging Konfiguration
├── Flask App Setup
├── Device Configuration
├── Model Loading
├── Routes
│   ├── Model Management
│   ├── Testing/Inference
│   ├── Dataset
│   ├── History & Export
│   └── Utilities
└── Helper Functions
```

### Best Practices

1. **Logging verwenden**: Alle wichtigen Operationen loggen
2. **Error Handling**: Try/Except mit sinnvollen Fehlermeldungen
3. **GPU Memory**: Nach Inferenz `torch.cuda.empty_cache()` aufrufen
4. **File Cleanup**: Temporäre Dateien regelmäßig löschen

---

## 📄 Lizenz

MIT License - Siehe [LICENSE](LICENSE) Datei für Details.

---

## 🤝 Contributing

Beiträge sind willkommen! Bitte erstellen Sie einen Pull Request oder öffnen Sie ein Issue für Feature-Requests und Bug-Reports.

---

## 📞 Support

Bei Fragen oder Problemen:
1. [Troubleshooting](#-troubleshooting) konsultieren
2. [Logging](#-logging) prüfen
3. Issue auf GitHub erstellen

---

**Erstellt mit ❤️ für die YOLO Community**

# 📐 CampYolo - Projekt Dokumentation

Interne Dokumentation für Entwickler und Maintainer.

---

## 🎯 Projekt-Übersicht

**Name**: CampYolo - YOLO Vision 2026  
**Version**: 2026.1  
**Typ**: Flask Web-Anwendung für YOLO-Modell-Management  
**Zielgruppe**: Entwickler, Data Scientists, ML-Engineer

---

## 🏗️ Architektur

### System-Design

```
┌─────────────────────────────────────────────────────────────┐
│                      Browser Client                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   HTML UI   │  │  JavaScript │  │     CSS     │         │
│  │  (Templates)│  │   (app.js)  │  │ (styles.css)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/HTTPS
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Flask Web Server                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   app.py                             │    │
│  │  ┌───────────┐  ┌───────────┐  ┌──────────────┐    │    │
│  │  │   Routes  │  │  Helpers  │  │   Logging    │    │    │
│  │  └───────────┘  └───────────┘  └──────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Python API
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   ML Backend                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Ultralytics│  │   PyTorch   │  │   OpenCV    │         │
│  │   (YOLO)    │  │  (Inferenz) │  │ (Processing)│         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Datenfluss

1. **User Input** → Web UI (Drag & Drop, Upload, URL)
2. **Request** → Flask Route Handler
3. **Preprocessing** → OpenCV Bildverarbeitung
4. **Inferenz** → YOLO Modell (GPU/CPU)
5. **Postprocessing** → Ergebnis-Extraktion
6. **Response** → JSON + annotierte Medien
7. **History** → Speicherung für Export

---

## 📂 Dateibeschreibung

### Kern-Dateien

| Datei | Größe | Beschreibung |
|-------|-------|-------------|
| `app.py` | ~2000 Zeilen | Flask Hauptanwendung mit allen Routes |
| `run.py` | ~50 Zeilen | Server-Startskript mit CUDA-Setup |
| `templates/index.html` | ~900 Zeilen | Single-Page Web-UI |
| `static/app.js` | ~1950 Zeilen | Client-seitige Logik |
| `static/styles.css` | ~3600 Zeilen | Styling mit Design-System |
| `static/training.js` | Variabel | Trainings-UI Logik |

### Konfigurations-Dateien

| Datei | Zweck |
|-------|-------|
| `requirements.txt` | Python-Paket-Abhängigkeiten |
| `.gitignore` | Git Ignore Regeln |
| `logs/*.log` | Anwendungs-Logs |

---

## 🔑 Kern-Komponenten

### 1. Model Loading (`app.py`)

```python
active_model = {"name": None, "instance": None}

@app.route('/load/<model_name>', methods=['POST'])
def load_model(model_name):
    """Lädt ein YOLO-Modell mit GPU-Support"""
    global active_model
    # 1. Vorheriges Modell entladen (GPU-Speicher freigeben)
    # 2. Neues Modell laden
    # 3. GPU-Statistiken loggen
    # 4. Response mit Status
```

### 2. Inference Pipeline

```
Bild Upload
    ↓
Preprocessing (OpenCV)
    ↓
Model Inferenz (YOLO)
    ↓
Ergebnis-Extraktion
    ↓
Annotation (Bounding Boxes)
    ↓
Response (JSON + Bild)
```

### 3. Few-Shot Auto-Detection

```
Sample-Bild hochladen
    ↓
Manuelle/Auto-Annotation
    ↓
Als Sample speichern
    ↓
Auto-Detection auf Zielbildern
    ↓
Ergebnis-Review & Export
```

---

## 🛠️ Entwicklung

### Lokales Setup

```bash
# 1. Repository klonen
git clone <repository-url>
cd CampYolo-main

# 2. Virtuelle Umgebung
python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows

# 3. Dependencies
pip install -r requirements.txt

# 4. Modelle vorbereiten
# Modelle in models/ Verzeichnis legen

# 5. Starten
python run.py
```

### Code-Qualität

```bash
# Linting (optional)
pip install flake8
flake8 app.py --max-line-length=120

# Type Checking (optional)
pip install mypy
mypy app.py --ignore-missing-imports
```

### Testing

Manuelle Tests über die Web-UI:
1. Modell laden
2. Testbild hochladen
3. Inferenz ausführen
4. Ergebnis prüfen

Automatisierte Tests können mit pytest hinzugefügt werden:
```bash
pip install pytest
pytest tests/
```

---

## 📊 Performance-Optimierung

### GPU Memory Management

```python
# Nach jeder Inferenz
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
```

### Batch Processing

```python
# Für mehrere Bilder
for batch in batches:
    results = model(batch, stream=True)  # RAM-effizient
```

### Caching Strategien

- **Modell-Caching**: Einmal geladenes Modell im Speicher halten
- **History-Caching**: Letzte 100 Ergebnisse im RAM
- **File-Caching**: Temporäre Dateien nach Use löschen

---

## 🔐 Sicherheit

### Upload-Limits

```python
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB
MAX_VIDEO_MB = 150
MAX_VIDEO_DURATION_S = 60
```

### Input Validation

- Dateityp-Prüfung (MIME-Type)
- Dateigrößen-Prüfung
- Pfad-Sanitization (Path Traversal Schutz)

### Best Practices

1. Keine sensiblen Daten in Logs
2. Upload-Verzeichnis außerhalb von Web-Root
3. Rate Limiting für Production erwägen

---

## 📈 Monitoring

### Health Check Endpoints

```bash
GET /debug/health    # System-Status
GET /debug/info      # System-Informationen
GET /debug/logs      # Aktuelle Logs
```

### Wichtige Metriken

- **Inferenz-Zeit**: ms pro Bild/Frame
- **GPU-Auslastung**: % VRAM-Verbrauch
- **Request-Dauer**: ms pro API-Aufruf
- **Fehlerrate**: Errors / Total Requests

---

## 🔄 Changelog

### Version 2026.1 (Aktuell)
- ✅ GUI Refactoring (Inline-Styles entfernt)
- ✅ Manuellen Annotation-Editor entfernt
- ✅ Dataset verwalten Sektion optimiert
- ✅ CSS-Klassen systematisch extrahiert
- ✅ README.md modernisiert
- ✅ .gitignore erweitert
- ✅ Projekt bereinigt

### Version 2026.0 (Initial)
- Flask Server mit YOLO-Integration
- Web-UI für Modell-Management
- Testing für Bilder/Videos
- Few-Shot Auto-Detection
- History & Export-Funktionen

---

## 📝 TODO / Roadmap

### Kurzfristig
- [ ] Unit Tests für Kern-Funktionen
- [ ] API-Dokumentation mit Swagger/OpenAPI
- [ ] Docker-Container für einfache Deployment

### Mittelfristig
- [ ] Benutzer-Authentifizierung
- [ ] Multi-User Support
- [ ] Datenbank-Integration für History

### Langfristig
- [ ] REST API V2
- [ ] WebSocket für Echtzeit-Updates
- [ ] Mobile App Integration

---

## 🤝 Contributing Guidelines

### Pull Request Prozess

1. Feature Branch erstellen (`git checkout -b feature/my-feature`)
2. Änderungen commiten (`git commit -m 'Add feature'`)
3. Branch pushen (`git push origin feature/my-feature`)
4. Pull Request öffnen

### Commit Convention

```
feat:     Neues Feature
fix:      Bugfix
docs:     Dokumentation
style:    Formatierung
refactor: Refactoring
test:     Tests
chore:    Wartung
```

---

## 📞 Support & Kontakt

- **Issues**: GitHub Issues verwenden
- **Fragen**: README Troubleshooting prüfen
- **Logs**: `logs/` Verzeichnis analysieren

---

**Letzte Aktualisierung**: März 2026

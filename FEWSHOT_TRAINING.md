# Few-Shot Inkrementelles Training

## Übersicht

Das Few-Shot Training ermöglicht es, ein YOLO-Modell mit nur **wenigen Beispielen** (3-5 Bilder pro Klasse) zu trainieren und das Modell schrittweise mit neuen Beispielen zu verbessern.

## Features

### 🎯 Few-Shot Training
- **Wenige Beispiele erforderlich**: Starte mit nur 3-5 annotierten Bildern
- **Einfache Annotation**: Integrierter Bounding-Box-Editor im Browser
- **Schnelles Training**: Optimierte Parameter für kleine Datensätze

### 📈 Inkrementelles Training
- **Modell schrittweise verbessern**: Füge neue Beispiele hinzu und trainiere weiter
- **Vorheriges Modell verwenden**: Setze auf bereits trainierten Modellen auf
- **Kein Datenverlust**: Alle bisherigen Samples bleiben erhalten

### 🔮 Modell testen
- **Echtzeit-Vorhersage**: Teste das trainierte Modell direkt im Browser
- **Visuelle Ergebnisse**: Annotierte Bilder mit Detection-Boxen

## Verwendung

### 1. Few-Shot Tab öffnen

Klicke in der Navigation auf **"Few-Shot Training"**.

### 2. Beispiele sammeln

1. **Bilder hochladen**: Ziehe Bilder in die Dropzone oder klicke zum Auswählen
2. **Annotationen erstellen**:
   - Zeichne Bounding-Boxen per Mausklick auf dem Bild
   - Gib einen Klassennamen ein (z.B. "person", "car", "dog")
   - Klicke "Box hinzufügen" für jede Klasse
   - Speichere die Annotationen

### 3. Training starten

1. **Mindestens 3 Samples** mit Annotationen hinzufügen
2. **Training-Parameter** einstellen:
   - **Epochs**: 50 (empfohlen für Few-Shot)
   - **Batch Size**: 8
   - **Image Size**: 640
3. **Basis-Modell** auswählen (YOLOv8n empfohlen)
4. Auf **"Training starten"** klicken

### 4. Inkrementell weitertrainieren

Nach dem ersten Training:
1. **Neue Samples hinzufügen** (zusätzliche Bilder annotieren)
2. Auf **"Inkrementell weitertrainieren"** klicken
3. Das Modell lernt aus den neuen Beispielen und verbessert sich

### 5. Modell testen

1. **Testbild** in die Dropzone "Modell testen" ziehen
2. **Vorhersage anzeigen**: Das annotierte Ergebnis wird angezeigt
3. **Detection-Ergebnisse** werden aufgelistet

### 6. Modell herunterladen

- Auf **"Modell herunterladen"** klicken
- Die `.pt` Datei kann lokal gespeichert oder weiterverwendet werden

## API Endpoints

### Status
```
GET /fewshot/train/status
```
Gibt den aktuellen Status zurück (Samples, Klassen, Trainingsstatus)

### Sample hinzufügen
```
POST /fewshot/train/add_sample
Content-Type: multipart/form-data

image: <Bild>
annotations: JSON Array von Annotationen
```

### Training starten
```
POST /fewshot/train/start
Content-Type: multipart/form-data

epochs: 50
batch: 8
imgsz: 640
model: yolov8n.pt
```

### Inkrementelles Training
```
POST /fewshot/train/incremental
Content-Type: multipart/form-data

epochs: 30
batch: 8
imgsz: 640
```

### Modell herunterladen
```
GET /fewshot/train/download
```

### Vorhersage
```
POST /fewshot/predict
Content-Type: multipart/form-data

image: <Bild>
```

## Tipps für beste Ergebnisse

### Gute Praxis
- **Verschiedene Perspektiven**: Bilder aus unterschiedlichen Winkeln
- **Verschiedene Beleuchtung**: Tagsüber, nachts, innen, außen
- **Verschiedene Größen**: Objekte nah und fern
- **Klare Annotationen**: Boxen sollten das Objekt genau umschließen

### Empfohlene Parameter
| Szenario | Epochs | Batch | Notes |
|----------|--------|-------|-------|
| Erstes Training | 50 | 8 | Basis-Modell anpassen |
| Inkrementell | 30 | 8 | Feinabstimmung |
| Viele Klassen | 75 | 8 | Mehr Training nötig |

### Häufige Probleme

**Problem**: Modell erkennt Objekte nicht
- **Lösung**: Mehr Trainingsbilder hinzufügen, verschiedene Perspektiven

**Problem**: Viele False Positives
- **Lösung**: Epochs reduzieren, Confidence Threshold erhöhen

**Problem**: Training dauert zu lange
- **Lösung**: Batch Size erhöhen, Image Size reduzieren

## Technische Details

### Dataset Format
Das System erstellt automatisch ein YOLO-format Dataset:
```
dataset/
├── data.yaml
├── images/
│   ├── sample_0001_image1.jpg
│   └── ...
└── labels/
    ├── sample_0001_image1.txt
    └── ...
```

### data.yaml Format
```yaml
path: /path/to/dataset
train: images
val: images
names:
  - klasse1
  - klasse2
```

### Label Format (YOLO)
```
<class_id> <x_center> <y_center> <width> <height>
```
Alle Werte normalisiert (0-1).

## Beispiele

### Beispiel 1: Hundeerkennung
1. 5 Bilder von Hunden hochladen
2. Bounding-Boxen um jeden Hund zeichnen
3. Klasse "dog" eingeben
4. Training mit 50 Epochs starten
5. Modell kann nun Hunde in neuen Bildern erkennen

### Beispiel 2: Fahrzeugerkennung (inkrementell)
1. Erstes Training mit 3 Auto-Bildern
2. Modell trainiert sich "car" zu erkennen
3. 5 weitere Bilder mit LKWs hinzufügen
4. Klasse "truck" annotieren
5. Inkrementell weitertrainieren
6. Modell erkennt nun sowohl Autos als auch LKWs

## Limitationen

- **Mindestens 3 Samples** erforderlich für Training
- **Maximale Klassenanzahl**: Hängt von der Modellkapazität ab
- **GPU empfohlen**: Training auf CPU ist möglich aber langsamer

## Zukunftserweiterungen

- [ ] CLIP-basierte Auto-Annotation
- [ ] Semi-supervised Learning
- [ ] Active Learning für bessere Sample-Auswahl
- [ ] Modell-Ensembling für bessere Genauigkeit

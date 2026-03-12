import os
import gc
import time
import shutil
import json
import io
import base64
from pathlib import Path
from datetime import datetime
from collections import deque
import torch
import cv2
import mimetypes
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context, send_file
from ultralytics import YOLO

app = Flask(__name__)
app.config['MODEL_DIR'] = 'models'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_FOLDER_ABS'] = os.path.abspath(app.config['UPLOAD_FOLDER'])
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
MAX_VIDEO_MB = 150
MAX_VIDEO_DURATION_S = 60
torch.set_grad_enabled(False)  # Globale Inferenz ohne Gradienten

# Globaler Slot für das aktuell aktive Modell
active_model = {
    "name": None,
    "instance": None
}

# Results History für Verlauf und Vergleich
results_history = deque(maxlen=100)

# Verzeichnisse sicherstellen
os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def extract_results_data(results):
    """Extrahiert alle verfügbaren Daten aus YOLO Results-Objekten und liefert auch das letzte save_dir."""
    all_detections = []
    last_save_dir = None

    for r in results:
        if hasattr(r, "save_dir"):
            last_save_dir = Path(r.save_dir)
        detection = {
            "path": r.path,
            "speed": r.speed,
            "orig_shape": r.orig_shape,
            "names": r.names,
        }

        # Bounding Boxes
        if r.boxes is not None and r.boxes.xyxy is not None and len(r.boxes) > 0:
            detection["boxes"] = {
                "xyxy": r.boxes.xyxy.tolist(),
                "xywh": r.boxes.xywh.tolist() if r.boxes.xywh is not None else None,
                "conf": r.boxes.conf.tolist() if r.boxes.conf is not None else None,
                "cls": r.boxes.cls.tolist() if r.boxes.cls is not None else None,
            }
            # Detaillierte Detection-Liste
            boxes_list = []
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                conf = float(box.conf[0]) if box.conf is not None and len(box.conf) > 0 else 0
                bbox = [round(x, 2) for x in box.xyxy[0].tolist()] if box.xyxy is not None and len(box.xyxy) > 0 else []
                class_name = r.names.get(cls_id, "unknown") if cls_id >= 0 and cls_id in r.names else "unknown"
                boxes_list.append({
                    "class": class_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": bbox,
                    "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] if len(bbox) == 4 else [],
                })
            detection["detections"] = boxes_list
        elif r.boxes is not None:
            # Leere Boxes
            detection["boxes"] = {
                "xyxy": [],
                "xywh": [],
                "conf": [],
                "cls": [],
            }
            detection["detections"] = []

        # Keypoints (für Pose-Modelle)
        if r.keypoints is not None:
            kp_data = {}
            try:
                if r.keypoints.xy is not None and len(r.keypoints.xy) > 0:
                    kp_data["xy"] = r.keypoints.xy.tolist()
                if r.keypoints.xyn is not None and len(r.keypoints.xyn) > 0:
                    kp_data["xyn"] = r.keypoints.xyn.tolist()
                if r.keypoints.conf is not None and len(r.keypoints.conf) > 0:
                    kp_data["conf"] = r.keypoints.conf.tolist()
                if kp_data:
                    detection["keypoints"] = kp_data
            except (ValueError, RuntimeError):
                # Falls Keypoints leer oder ungültig sind
                pass

        # Masks (für Segmentierungs-Modelle)
        if r.masks is not None:
            detection["masks"] = {
                "xy": r.masks.xy.tolist() if r.masks.xy is not None else None,
                "xyn": r.masks.xyn.tolist() if r.masks.xyn is not None else None,
            }
        
        # Probs (für Klassifizierungs-Modelle)
        if r.probs is not None:
            detection["probs"] = {
                "top1": int(r.probs.top1),
                "top5": r.probs.top5.tolist(),
                "top1conf": float(r.probs.top1conf) if r.probs.top1conf is not None else None,
                "top5conf": r.probs.top5conf.tolist() if r.probs.top5conf is not None else None,
            }
        
        # OBB (für Oriented Bounding Boxes)
        if r.obb is not None:
            detection["obb"] = {
                "xyxyxyxy": r.obb.xyxyxyxy.tolist() if r.obb.xyxyxyxy is not None else None,
                "xywhr": r.obb.xywhr.tolist() if r.obb.xywhr is not None else None,
                "conf": r.obb.conf.tolist() if r.obb.conf is not None else None,
                "cls": r.obb.cls.tolist() if r.obb.cls is not None else None,
            }
        
        all_detections.append(detection)

    return all_detections, last_save_dir

def save_to_history(result_data, model_name, source_type, source_value, processing_time):
    """Speichert ein Ergebnis in der History."""
    history_entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": time.time(),
        "datetime": datetime.now().isoformat(),
        "model": model_name,
        "source_type": source_type,
        "source_value": source_value,
        "processing_time": processing_time,
        "detections_count": len(result_data.get("detections", [])),
        "detections": result_data.get("detections", []),
        "image_url": result_data.get("image_url"),
        "video_url": result_data.get("video_url")
    }
    results_history.append(history_entry)
    return history_entry

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history():
    """Gibt die Results-History zurück."""
    limit = request.args.get('limit', 50, type=int)
    source_type = request.args.get('source_type')
    
    history_list = list(results_history)
    
    # Filter nach source_type
    if source_type:
        history_list = [h for h in history_list if h['source_type'] == source_type]
    
    # Nach Timestamp sortieren (neueste zuerst)
    history_list.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Limit anwenden
    history_list = history_list[:limit]
    
    return jsonify({
        "count": len(history_list),
        "total_available": len(results_history),
        "history": history_list
    })

@app.route('/history/clear', methods=['POST'])
def clear_history():
    """Löscht die gesamte History."""
    results_history.clear()
    return jsonify({"status": "History gelöscht", "count": 0})

@app.route('/history/<entry_id>', methods=['GET'])
def get_history_entry(entry_id):
    """Gibt einen spezifischen History-Eintrag zurück."""
    for entry in results_history:
        if entry['id'] == entry_id:
            return jsonify(entry)
    return jsonify({"error": "Eintrag nicht gefunden"}), 404

@app.route('/exports/results/<export_format>', methods=['GET'])
def export_results(export_format):
    """Exportiert Ergebnisse in verschiedenen Formaten."""
    limit = request.args.get('limit', 100, type=int)
    history_list = list(results_history)[:limit]
    
    if export_format == 'json':
        return jsonify({
            "exported_at": datetime.now().isoformat(),
            "count": len(history_list),
            "results": history_list
        })
    
    elif export_format == 'csv':
        import csv
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Timestamp', 'Model', 'Source Type', 'Detections', 'Processing Time'])
        
        for entry in history_list:
            writer.writerow([
                entry['id'],
                entry['datetime'],
                entry['model'],
                entry['source_type'],
                entry['detections_count'],
                f"{entry['processing_time']:.2f}s"
            ])
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=results.csv'}
        )
    
    elif export_format == 'summary':
        # Zusammenfassung der Statistik
        total_detections = sum(e['detections_count'] for e in history_list)
        avg_time = sum(e['processing_time'] for e in history_list) / len(history_list) if history_list else 0
        
        # Detections nach Klasse gruppieren
        class_counts = {}
        for entry in history_list:
            for det in entry['detections']:
                cls = det.get('class', 'unknown')
                class_counts[cls] = class_counts.get(cls, 0) + 1
        
        return jsonify({
            "total_inferences": len(history_list),
            "total_detections": total_detections,
            "avg_processing_time": f"{avg_time:.2f}s",
            "detections_by_class": class_counts,
            "top_classes": sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        })
    
    return jsonify({"error": "Ungültiges Format. Unterstützt: json, csv, summary"}), 400

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """Stellt generierte Dateien (z.B. annotierte Ergebnisse) bereit."""
    full_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], filename)
    if not os.path.exists(full_path):
        return jsonify({"error": "Datei nicht gefunden"}), 404
    mime, _ = mimetypes.guess_type(full_path)
    resp = send_file(full_path, mimetype=mime, as_attachment=False, conditional=True, last_modified=None)
    resp.headers['Accept-Ranges'] = 'bytes'
    return resp

@app.route('/models', methods=['GET'])
def list_models():
    """Listet verfügbare Dateien und zeigt das aktuell aktive Modell an."""
    files = [f for f in os.listdir(app.config['MODEL_DIR']) if f.endswith('.pt')]
    return jsonify({
        "available_models": files,
        "active_model": active_model["name"]
    })

@app.route('/models/suggest', methods=['GET'])
def suggest_model():
    """Schlägt ein Modell basierend auf dem Einsatzzweck vor."""
    task = request.args.get('task', 'detect')  # detect, pose, seg, cls
    files = [f for f in os.listdir(app.config['MODEL_DIR']) if f.endswith('.pt')]
    
    # Filter models by task
    task_models = []
    for f in files:
        f_lower = f.lower()
        if task == 'pose' and 'pose' in f_lower:
            task_models.append(f)
        elif task == 'seg' and 'seg' in f_lower:
            task_models.append(f)
        elif task == 'cls' and 'cls' in f_lower:
            task_models.append(f)
        elif task == 'detect' and 'pose' not in f_lower and 'seg' not in f_lower and 'cls' not in f_lower:
            task_models.append(f)
    
    # Sort by size preference (n < s < m < l < x)
    size_order = {'n': 0, 's': 1, 'm': 2, 'l': 3, 'x': 4}
    def get_size_priority(name):
        name_lower = name.lower()
        for size, priority in size_order.items():
            if f'-{size}' in name_lower or f'{size}.' in name_lower:
                return priority
        return 5  # Unknown size
    
    task_models.sort(key=get_size_priority)
    
    # Suggest best match
    suggestion = task_models[0] if task_models else None
    
    return jsonify({
        "task": task,
        "available": task_models,
        "suggested": suggestion,
        "recommendation": {
            "fast": [m for m in task_models if '-n.' in m.lower() or '-s.' in m.lower()][:1],
            "balanced": [m for m in task_models if '-m.' in m.lower()][:1],
            "accurate": [m for m in task_models if '-l.' in m.lower() or '-x.' in m.lower()][:1]
        } if task_models else {}
    })

@app.route('/load/<model_name>', methods=['POST'])
def load_model(model_name):
    """Lädt ein Modell und entfernt das vorherige aus dem Speicher."""
    global active_model
    path = os.path.join(app.config['MODEL_DIR'], model_name)
    
    if not os.path.exists(path):
        return jsonify({"error": "Modell-Datei nicht gefunden"}), 404

    try:
        # 1. Altes Modell entfernen, falls vorhanden
        if active_model["instance"] is not None:
            print(f"Entlade Modell: {active_model['name']}")
            active_model["instance"] = None
            active_model["name"] = None
            
            # Speicherbereinigung forcieren
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # VRAM leeren falls GPU genutzt wird

        # 2. Neues Modell laden
        print(f"Lade neues Modell: {model_name}")
        active_model["instance"] = YOLO(path)
        active_model["name"] = model_name
        
        return jsonify({
            "status": f"Modell {model_name} aktiv. Altes Modell wurde entladen.",
            "active_model": model_name
        })
    except Exception as e:
        return jsonify(error_payload(e)), 500

@app.route('/test', methods=['POST'])
def test_model():
    """Nutzt das aktuell aktive Modell für eine Vorhersage mit erweiterten Optionen."""
    if active_model["instance"] is None:
        return jsonify({"error": "Kein Modell geladen. Bitte zuerst ein Modell aktivieren."}), 400

    # Unterstütze verschiedene Quellen
    source_type = request.form.get('source_type', 'image')
    source_value = request.form.get('source_value', '')
    
    # Inferenz-Parameter
    try:
        conf = float(request.form.get('conf', 0.25))
        iou = float(request.form.get('iou', 0.45))
        imgsz = int(request.form.get('imgsz', 640))
    except ValueError:
        return jsonify({"error": "Ungültige Parameter für conf/iou/imgsz"}), 400
    save = request.form.get('save', 'true').lower() == 'true'
    show_labels = request.form.get('show_labels', 'true').lower() == 'true'
    
    img_path = None
    
    try:
        # Quelle bestimmen
        if source_type == 'image' and 'image' in request.files:
            file = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            inference_source = img_path
        elif source_type == 'url' and source_value:
            inference_source = source_value
        elif source_type == 'webcam':
            try:
                cam_id = int(source_value) if source_value else 0
            except ValueError:
                return jsonify({"error": "Ungültige Webcam-ID. Bitte eine Zahl eingeben."}), 400
            inference_source = cam_id
        elif source_type == 'directory' and source_value:
            if not os.path.exists(source_value):
                return jsonify({"error": f"Verzeichnis nicht gefunden: {source_value}"}), 404
            inference_source = source_value
        elif source_type == 'glob' and source_value:
            inference_source = source_value
        elif source_type == 'screenshot':
            inference_source = 'screen'
        else:
            return jsonify({"error": f"Ungültige Quelle oder fehlende Daten für Typ: {source_type}"}), 400
        
        # Stream=True für alle Quellen die Videos/Streams sein könnten (verhindert RAM-Accumulation laut Ultralytics-Doku)
        # Besonders wichtig für YouTube-URLs, Videos, Webcams und RTSP-Streams
        stream_mode = source_type in {'video', 'url', 'webcam', 'directory', 'glob', 'stream'}
        
        with torch.inference_mode():
            results_gen = active_model["instance"](
                inference_source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                save=save,
                show=False,
                stream=True if stream_mode else False,
                verbose=False
            )

        # Ergebnisse extrahieren - Generator MUSS iteriert werden um "Waiting for stream" Warnungen zu vermeiden
        # Bei stream=True returned model einen Generator der konsumiert werden muss
        all_detections = []
        last_save_dir = None
        for r in results_gen:
            det, save_dir = extract_results_data([r])
            all_detections.extend(det)
            if save_dir:
                last_save_dir = save_dir

        # Annotiertes Bild speichern (nur wenn genau ein Frame zu erwarten ist und save aktiviert)
        annotated_filename = None
        if save and source_type == 'image' and all_detections:
            annotated_filename = f"result_{int(time.time() * 1000)}.jpg"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], annotated_filename)
            try:
                # Wir haben all_detections schon konsumiert; erneut laufen für save
                with torch.inference_mode():
                    res_single = active_model["instance"](inference_source, conf=conf, iou=iou, imgsz=imgsz, save=False, show=False, stream=False, verbose=False)
                if res_single and len(res_single) == 1:
                    res_single[0].save(filename=annotated_path)
            except Exception:
                annotated_filename = None
        
        response_payload = {
            "model_used": active_model["name"],
            "source_type": source_type,
            "source_value": str(inference_source),
            "parameters": {
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
            },
            "results": all_detections,
            "detections": all_detections[0].get("detections", []) if all_detections else [],
            "image_url": f"/uploads/{annotated_filename}" if annotated_filename else None,
            "speed": all_detections[0].get("speed", {}) if all_detections else {},
        }

        # Annotiertes Video/Stream Ergebnis für Video-Quellen anhängen
        if save and source_type in {'video', 'url', 'webcam', 'directory', 'glob', 'stream'} and last_save_dir:
            upload_root = Path(app.config['UPLOAD_FOLDER_ABS']).resolve()
            annotated_video = pick_annotated_video(last_save_dir, inference_source, upload_root)
            if annotated_video:
                response_payload["video_url"] = f"/uploads/{annotated_video.name}"

        # In History speichern
        save_to_history(
            response_payload, 
            active_model["name"], 
            source_type, 
            str(inference_source),
            all_detections[0].get("speed", {}).get("inference", 0) if all_detections else 0
        )

        return jsonify(response_payload)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Aufräumen
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass


@app.route('/test_video', methods=['POST'])
def test_video():
    """Inferiert ein einzelnes Video (synchron) und liefert einen Link zum annotierten Video."""
    if active_model["instance"] is None:
        return jsonify({"error": "Kein Modell geladen. Bitte zuerst ein Modell aktivieren."}), 400

    if 'video' not in request.files:
        return jsonify({"error": "Kein Video hochgeladen"}), 400

    file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], file.filename)
    file.save(video_path)

    err = validate_video(video_path, MAX_VIDEO_MB, MAX_VIDEO_DURATION_S)
    if err:
        return jsonify({"error": err}), 400

    try:
        # Einzigartiges Verzeichnis für dieses Video erstellen
        timestamp = int(time.time())
        save_dir = Path(app.config['UPLOAD_FOLDER_ABS']) / f"results_{timestamp}"
        save_dir.mkdir(exist_ok=True)

        # Stream=True verhindert RAM-Accumulation (siehe Ultralytics-Doku)
        with torch.inference_mode():
            results = active_model["instance"](
                video_path,
                stream=True,
                save=True,
                project=str(save_dir.parent),
                name=f"results_{timestamp}",
                exist_ok=True,
                verbose=False
            )

        # Iteration erzwingt Verarbeitung und Schreiben der annotierten Datei
        for r in results:
            result_save_dir = Path(r.save_dir).resolve()

        # Das annotierte Video liegt im results-Verzeichnis mit dem Originalnamen
        upload_root = Path(app.config['UPLOAD_FOLDER_ABS']).resolve()
        annotated = pick_annotated_video(save_dir, video_path, upload_root)

        return jsonify({
            "model_used": active_model["name"],
            "video_url": f"/uploads/{annotated.name}" if annotated else None,
            "save_dir": str(save_dir)
        })
    except Exception as e:
        print_exc(e)
        return jsonify(error_payload(e)), 500


@app.route('/test_video_stream', methods=['POST'])
def test_video_stream():
    """Streamt Fortschritt während der Video-Inferenz via SSE."""
    if active_model["instance"] is None:
        return jsonify({"error": "Kein Modell geladen. Bitte zuerst ein Modell aktivieren."}), 400
    if 'video' not in request.files:
        return jsonify({"error": "Kein Video hochgeladen"}), 400

    file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], file.filename)
    file.save(video_path)

    err = validate_video(video_path, MAX_VIDEO_MB, MAX_VIDEO_DURATION_S)
    if err:
        return jsonify({"error": err}), 400

    # Einzigartiges Verzeichnis für dieses Video erstellen
    timestamp = int(time.time())
    save_dir = Path(app.config['UPLOAD_FOLDER_ABS']) / f"results_{timestamp}"
    save_dir.mkdir(exist_ok=True)

    # versuche Frame-Anzahl zu bestimmen
    total_frames = None
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
    except Exception:
        total_frames = None

    def generate():
        processed = 0
        try:
            with torch.inference_mode():
                results = active_model["instance"](
                    video_path,
                    stream=True,
                    save=True,
                    project=str(save_dir.parent),
                    name=f"results_{timestamp}",
                    exist_ok=True,
                    verbose=False
                )
            result_save_dir = None
            for r in results:
                processed += 1
                # progress
                pct = int(processed / total_frames * 100) if total_frames else None
                payload = {
                    "frame": processed,
                    "total_frames": total_frames,
                    "progress": pct,
                    "message": f"Frame {processed}/{total_frames}" if total_frames else f"Frame {processed}"
                }
                yield f"data: {json.dumps(payload)}\n\n"
                result_save_dir = Path(r.save_dir).resolve()

            upload_root = Path(app.config['UPLOAD_FOLDER_ABS']).resolve()
            annotated = pick_annotated_video(result_save_dir, video_path, upload_root)

            payload_done = {
                "done": True,
                "model_used": active_model["name"],
                "video_url": f"/uploads/{annotated.name}" if annotated else None,
                "save_dir": str(result_save_dir) if result_save_dir else None
            }
            yield f"data: {json.dumps(payload_done)}\n\n"
        except Exception as e:
            print_exc(e)
            yield f"data: {json.dumps(error_payload(e))}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route('/test_url_stream', methods=['POST'])
def test_url_stream():
    """Streamt Fortschritt während der Inferenz einer URL (YouTube, RTSP, etc.) via SSE."""
    if active_model["instance"] is None:
        return jsonify({"error": "Kein Modell geladen. Bitte zuerst ein Modell aktivieren."}), 400

    source_value = request.form.get('source_value', '')
    if not source_value:
        return jsonify({"error": "URL erforderlich"}), 400

    conf = float(request.form.get('conf', 0.25))
    iou = float(request.form.get('iou', 0.45))
    imgsz = int(request.form.get('imgsz', 640))
    save = request.form.get('save', 'true').lower() == 'true'

    # Einzigartiges Verzeichnis für diese Inferenz erstellen
    timestamp = int(time.time())
    save_dir = Path(app.config['UPLOAD_FOLDER_ABS']) / f"results_url_{timestamp}"
    save_dir.mkdir(exist_ok=True)

    # Versuche Frame-Anzahl zu bestimmen (funktioniert bei YouTube oft nicht im Voraus)
    total_frames = None
    fps = None
    try:
        import cv2
        cap = cv2.VideoCapture(source_value)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            fps = cap.get(cv2.CAP_PROP_FPS) or None
            duration = total_frames / fps if total_frames and fps else None
            cap.release()
            print(f"Stream-Info: {total_frames} Frames, {fps} FPS, Dauer: {duration}s" if total_frames else "Stream-Info: Unbekannt")
    except Exception as e:
        print(f"Konnte Stream-Info nicht lesen: {e}")
        total_frames = None

    def generate():
        processed = 0
        result_save_dir = None
        try:
            with torch.inference_mode():
                results = active_model["instance"](
                    source_value,
                    stream=True,
                    save=save,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    project=str(save_dir.parent),
                    name=f"results_url_{timestamp}",
                    exist_ok=True,
                    verbose=False
                )
                for r in results:
                    processed += 1
                    # Fortschritt berechnen
                    pct = int(processed / total_frames * 100) if total_frames else None
                    payload = {
                        "frame": processed,
                        "total_frames": total_frames,
                        "progress": pct,
                        "message": f"Frame {processed}/{total_frames}" if total_frames else f"Frame {processed} verarbeitet"
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    result_save_dir = Path(r.save_dir).resolve()

            upload_root = Path(app.config['UPLOAD_FOLDER_ABS']).resolve()
            annotated = pick_annotated_video(result_save_dir, source_value, upload_root) if save else None

            payload_done = {
                "done": True,
                "model_used": active_model["name"],
                "video_url": f"/uploads/{annotated.name}" if annotated else None,
                "save_dir": str(result_save_dir) if result_save_dir else None,
                "total_frames_processed": processed
            }
            yield f"data: {json.dumps(payload_done)}\n\n"
        except Exception as e:
            print_exc(e)
            yield f"data: {json.dumps({'error': str(e), 'frame': processed})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


def is_subpath(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def pick_annotated_video(save_dir: Path, original_path: str, upload_root: Path) -> Path | None:
    """Wählt die annotierte Videodatei und kopiert sie ins Upload-Verzeichnis, falls nötig."""
    if not save_dir:
        return None
    save_dir = save_dir.resolve()
    orig = Path(original_path).resolve()

    # Ultralytics speichert annotierte Videos im save_dir mit dem Originalnamen
    # Das annotierte Video liegt direkt im save_dir
    annotated_candidates = []

    # 1. Suche im save_dir nach MP4-Dateien
    if save_dir.exists():
        for f in save_dir.glob("*.mp4"):
            if f.resolve() != orig:
                annotated_candidates.append(f.resolve())
        for f in save_dir.glob("*.avi"):
            if f.resolve() != orig:
                annotated_candidates.append(f.resolve())
        for f in save_dir.glob("*.mov"):
            if f.resolve() != orig:
                annotated_candidates.append(f.resolve())

    # 2. Suche in runs/ Verzeichnissen
    runs_dir = Path.home() / "runs"
    if runs_dir.exists():
        for f in runs_dir.rglob("*.mp4"):
            if f.resolve() != orig:
                annotated_candidates.append(f.resolve())

    # 3. Suche im upload_root
    if upload_root.exists():
        for f in upload_root.glob("*.mp4"):
            if f.resolve() != orig and "annotated" in f.name.lower():
                annotated_candidates.append(f.resolve())

    if not annotated_candidates:
        return None

    # Bevorzugen: Dateien im save_dir, dann nach jüngster Modifikationszeit
    def score(p: Path):
        in_save = save_dir in p.parents
        mtime = p.stat().st_mtime
        return (
            1 if in_save else 0,
            mtime,
        )

    candidate = sorted(annotated_candidates, key=score, reverse=True)[0]

    # Sauberen Dateinamen generieren (ohne ungültige Zeichen)
    # YouTube-URLs enthalten 'watch?v=' was ungültig ist
    import re
    orig_name = orig.name if orig.name else "video"
    # Entferne ungültige Zeichen für Dateinamen
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', orig_name)
    # Kürze lange Namen
    if len(safe_name) > 50:
        safe_name = safe_name[:50]
    
    # Immer nach uploads kopieren, um eine saubere URL zu haben
    dest = upload_root / f"annotated_{safe_name}_{int(time.time())}{candidate.suffix}"
    try:
        shutil.copy2(candidate, dest)
    except Exception as e:
        print(f"Fehler beim Kopieren: {e}")
        dest = candidate

    # Falls nicht MP4 oder MP4 mit inkompatiblem Codec, nach H.264 (yuv420p) konvertieren
    if dest.suffix.lower() != ".mp4":
        mp4_dest = upload_root / f"annotated_{safe_name}_{int(time.time())}.mp4"
        if convert_to_mp4(dest, mp4_dest):
            dest = mp4_dest
    else:
        # dennoch transkodieren, um Browserkompatibilität sicherzustellen
        mp4_dest = upload_root / f"annotated_{safe_name}_{int(time.time())}.mp4"
        if convert_to_mp4(dest, mp4_dest):
            dest = mp4_dest

    return dest if dest.exists() else None


def convert_to_mp4(src: Path, dest: Path) -> bool:
    """Konvertiert ein Video nach MP4/H.264 (yuv420p). Nutzt ffmpeg, fallback OpenCV."""
    try:
        # Versuch mit ffmpeg (falls installiert)
        import subprocess, shutil as _shutil
        if _shutil.which("ffmpeg"):
            cmd = [
                "ffmpeg", "-y",
                "-i", str(src),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                str(dest)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return dest.exists() and dest.stat().st_size > 0
    except Exception as e:
        print(f"FFmpeg Konvertierung fehlgeschlagen: {e}")

    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        cap.release()
        out.release()
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e:
        print(f"OpenCV Konvertierung fehlgeschlagen: {e}")
        return False


def validate_video(path: str, max_mb: int, max_duration_s: int) -> str | None:
    """Pragmatischer Schutz gegen zu große/lange Videos, um Hänger/RAM-Overflow zu vermeiden."""
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > max_mb:
            return f"Video zu groß ({size_mb:.1f} MB). Max {max_mb} MB."

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return "Video konnte nicht geöffnet werden."
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frames > 0:
            duration = frames / fps
            if duration > max_duration_s:
                return f"Video zu lang ({duration:.1f}s). Max {max_duration_s}s."
    except Exception:
        # Im Zweifel nicht blockieren, aber auch nicht hängen bleiben
        return None
    return None


def error_payload(e: Exception) -> dict:
    return {"error": str(e)}


def print_exc(e: Exception):
    try:
        import traceback
        traceback.print_exc()
    except Exception:
        print(e)


@app.route('/test_webcam', methods=['GET', 'POST'])
def test_webcam():
    """Führt Inferenz auf einer Webcam durch und gibt annotierten Stream zurück."""
    if active_model["instance"] is None:
        return jsonify({"error": "Kein Modell geladen. Bitte zuerst ein Modell aktivieren."}), 400
    
    cam_id = request.values.get('cam_id', '0')
    try:
        cam_id = int(cam_id)
    except ValueError:
        return jsonify({"error": "Ungültige Webcam-ID"}), 400
    
    conf = float(request.values.get('conf', 0.25))
    imgsz = int(request.values.get('imgsz', 640))
    
    # Teste ob Webcam verfügbar ist
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        return jsonify({"error": f"Webcam {cam_id} nicht verfügbar"}), 404
    cap.release()
    
    def generate():
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        max_frames = 300  # Max 300 Frames (~10 Sekunden bei 30fps)
        
        try:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Inferenz
                with torch.inference_mode():
                    results = active_model["instance"](frame, conf=conf, imgsz=imgsz, verbose=False)
                
                # Annotieren
                annotated = results[0].plot()
                
                # Als JPEG encodieren
                _, buffer = cv2.imencode('.jpg', annotated)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
        finally:
            cap.release()
    
    return Response(
        stream_with_context(generate()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route('/test_stream', methods=['POST'])
def test_stream():
    """Führt Inferenz auf einem Video-Stream (RTSP/RTMP) durch."""
    if active_model["instance"] is None:
        return jsonify({"error": "Kein Modell geladen. Bitte zuerst ein Modell aktivieren."}), 400
    
    stream_url = request.form.get('stream_url', '')
    if not stream_url:
        return jsonify({"error": "Stream-URL erforderlich"}), 400
    
    conf = float(request.form.get('conf', 0.25))
    imgsz = int(request.form.get('imgsz', 640))
    
    # Teste ob Stream verfügbar ist
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return jsonify({"error": f"Stream nicht verfügbar: {stream_url}"}), 404
    cap.release()
    
    def generate():
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        max_frames = 300
        
        try:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    # Versuch, Stream neu zu verbinden
                    time.sleep(1)
                    cap.open(stream_url)
                    continue
                
                # Inferenz
                results = active_model["instance"](frame, conf=conf, imgsz=imgsz, verbose=False)
                
                # Annotieren
                annotated = results[0].plot()
                
                # Als JPEG encodieren
                _, buffer = cv2.imencode('.jpg', annotated)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                frame_count += 1
        finally:
            cap.release()
    
    return Response(
        stream_with_context(generate()),
        mimetype='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


@app.route('/test_batch', methods=['POST'])
def test_batch():
    """Führt Batch-Inferenz auf mehreren Bildern durch."""
    if active_model["instance"] is None:
        return jsonify({"error": "Kein Modell geladen. Bitte zuerst ein Modell aktivieren."}), 400
    
    if 'images' not in request.files:
        return jsonify({"error": "Keine Bilder hochgeladen"}), 400
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "Keine Bilder ausgewählt"}), 400
    
    conf = float(request.form.get('conf', 0.25))
    iou = float(request.form.get('iou', 0.45))
    imgsz = int(request.form.get('imgsz', 640))
    
    img_paths = []
    unsupported = []
    for file in files:
        if not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
            safe_name = f"{int(time.time()*1000)}_{file.filename}"
            img_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], safe_name)
            file.save(img_path)
            img_paths.append(img_path)
        elif ext in {'.mp4', '.mov', '.avi', '.mkv'}:
            unsupported.append(file.filename)
        else:
            unsupported.append(file.filename)
    
    if unsupported and not img_paths:
        return jsonify({"error": f"Nicht unterstützte Dateitypen: {', '.join(unsupported)}"}), 400
    if not img_paths:
        return jsonify({"error": "Keine gültigen Bilddateien gefunden"}), 400
    
    try:
        # Batch-Inferenz
        with torch.inference_mode():
            results = active_model["instance"](
                img_paths,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
            )
        
        # Ergebnisse extrahieren
        all_detections, _ = extract_results_data(results)
        
        # Annotierte Bilder speichern
        annotated_files = []
        for i, r in enumerate(results):
            annotated_filename = f"result_batch_{i}_{int(time.time() * 1000)}.jpg"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], annotated_filename)
            try:
                r.save(filename=annotated_path)
                annotated_files.append(f"/uploads/{annotated_filename}")
            except Exception:
                pass
        
        return jsonify({
            "model_used": active_model["name"],
            "processed_count": len(results),
            "results": all_detections,
            "annotated_images": annotated_files,
            "parameters": {
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
            },
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Aufräumen
        for img_path in img_paths:
            if os.path.exists(img_path):
                try:
                    os.remove(img_path)
                except Exception:
                    pass

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint für Monitoring."""
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_memory = None
    if gpu_available:
        gpu_memory = {
            "total": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
            "allocated": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
            "reserved": round(torch.cuda.memory_reserved(0) / 1024**3, 2)
        }
    
    return jsonify({
        "status": "healthy",
        "model_loaded": active_model["name"],
        "gpu": {
            "available": gpu_available,
            "name": gpu_name,
            "memory": gpu_memory
        } if gpu_available else {"available": False},
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "timestamp": time.time()
    })

@app.route('/compare', methods=['POST'])
def compare_models():
    """Vergleicht mehrere Modelle mit demselben Bild."""
    if 'image' not in request.files:
        return jsonify({"error": "Kein Bild hochgeladen"}), 400
    
    models = request.form.getlist('models')
    if not models:
        return jsonify({"error": "Keine Modelle angegeben"}), 400
    
    conf = float(request.form.get('conf', 0.25))
    iou = float(request.form.get('iou', 0.45))
    imgsz = int(request.form.get('imgsz', 640))
    
    # Bild speichern
    file = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    
    results = []
    start_time = time.time()
    
    try:
        for model_name in models:
            model_path = os.path.join(app.config['MODEL_DIR'], model_name)
            if not os.path.exists(model_path):
                results.append({
                    "model": model_name,
                    "error": "Modell nicht gefunden"
                })
                continue
            
            try:
                # Modell laden
                model = YOLO(model_path)
                
                # Inferenz
                with torch.inference_mode():
                    res = model(img_path, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                
                # Ergebnisse extrahieren
                detections = []
                if res[0].boxes is not None and res[0].boxes.xyxy is not None:
                    for i, box in enumerate(res[0].boxes):
                        cls_id = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                        conf_val = float(box.conf[0]) if box.conf is not None and len(box.conf) > 0 else 0
                        bbox = [round(x, 2) for x in box.xyxy[0].tolist()]
                        class_name = res[0].names.get(cls_id, "unknown")
                        detections.append({
                            "class": class_name,
                            "class_id": cls_id,
                            "confidence": conf_val,
                            "bbox": bbox
                        })
                
                results.append({
                    "model": model_name,
                    "detections": detections,
                    "detections_count": len(detections),
                    "inference_time": res[0].speed.get('inference', 0) if res else 0,
                    "success": True
                })
                
                # Modell aus Speicher entfernen
                del model
                
            except Exception as e:
                results.append({
                    "model": model_name,
                    "error": str(e),
                    "success": False
                })
        
        total_time = time.time() - start_time
        
        # Vergleichs-Statistik
        comparison = {
            "total_time": round(total_time, 2),
            "models_compared": len(results),
            "results": results,
            "fastest_model": min((r for r in results if r.get('success')), key=lambda x: x.get('inference_time', float('inf')), default=None),
            "most_detections": max((r for r in results if r.get('success')), key=lambda x: x.get('detections_count', 0), default=None)
        }
        
        return jsonify(comparison)
    
    finally:
        # Aufräumen
        if os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

@app.route('/system/info', methods=['GET'])
def system_info():
    """Systeminformationen für Debugging."""
    import platform

    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "cpu_count": os.cpu_count()
    }
    
    # Optional: psutil für Memory-Info
    try:
        import psutil
        info["memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError:
        info["memory_gb"] = None
    
    return jsonify(info)

if __name__ == '__main__':
    # Threaded=True ist wichtig für SSE-Streams und parallele Requests
    # debug=False kann bei längeren Inferenzen stabiler sein
    app.run(debug=True, port=5000, threaded=True)

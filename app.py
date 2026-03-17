import os, gc, time, shutil, json, io, base64, re, subprocess, csv, numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import torch, cv2, mimetypes
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, Response, stream_with_context
from ultralytics import YOLO

app = Flask(__name__)
app.config['MODEL_DIR'] = 'models'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['UPLOAD_FOLDER_ABS'] = os.path.abspath(app.config['UPLOAD_FOLDER'])
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
MAX_VIDEO_MB = 150
MAX_VIDEO_DURATION_S = 60
torch.set_grad_enabled(False)

active_model = {"name": None, "instance": None}
results_history = deque(maxlen=100)

os.makedirs(app.config['MODEL_DIR'], exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def extract_results_data(results):
    all_detections, last_save_dir = [], None
    for r in results:
        if hasattr(r, "save_dir"): last_save_dir = Path(r.save_dir)
        detection = {"path": r.path, "speed": r.speed, "orig_shape": r.orig_shape, "names": r.names}
        if r.boxes is not None and r.boxes.xyxy is not None and len(r.boxes) > 0:
            detection["boxes"] = {"xyxy": r.boxes.xyxy.tolist(), "xywh": r.boxes.xywh.tolist() if r.boxes.xywh else None,
                                   "conf": r.boxes.conf.tolist() if r.boxes.conf else None, "cls": r.boxes.cls.tolist() if r.boxes.cls else None}
            boxes_list = []
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                conf_val = float(box.conf[0]) if box.conf is not None and len(box.conf) > 0 else 0
                bbox = [round(x, 2) for x in box.xyxy[0].tolist()] if box.xyxy is not None and len(box.xyxy) > 0 else []
                boxes_list.append({"class": r.names.get(cls_id, "unknown"), "class_id": cls_id, "confidence": conf_val, "bbox": bbox,
                                   "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] if len(bbox) == 4 else []})
            detection["detections"] = boxes_list
        else:
            detection["boxes"] = {"xyxy": [], "xywh": [], "conf": [], "cls": []}
            detection["detections"] = []
        if r.keypoints is not None:
            kp_data = {}
            try:
                if r.keypoints.xy is not None: kp_data["xy"] = r.keypoints.xy.tolist()
                if r.keypoints.xyn is not None: kp_data["xyn"] = r.keypoints.xyn.tolist()
                if r.keypoints.conf is not None: kp_data["conf"] = r.keypoints.conf.tolist()
                if kp_data: detection["keypoints"] = kp_data
            except (ValueError, RuntimeError): pass
        if r.masks is not None:
            detection["masks"] = {"xy": r.masks.xy.tolist() if r.masks.xy else None, "xyn": r.masks.xyn.tolist() if r.masks.xyn else None}
        if r.probs is not None:
            detection["probs"] = {"top1": int(r.probs.top1), "top5": r.probs.top5.tolist(), "top1conf": float(r.probs.top1conf) if r.probs.top1conf else None}
        if r.obb is not None:
            detection["obb"] = {"xyxyxyxy": r.obb.xyxyxyxy.tolist() if r.obb.xyxyxyxy else None, "xywhr": r.obb.xywhr.tolist() if r.obb.xywhr else None,
                                "conf": r.obb.conf.tolist() if r.obb.conf else None, "cls": r.obb.cls.tolist() if r.obb.cls else None}
        all_detections.append(detection)
    return all_detections, last_save_dir

def save_to_history(result_data, model_name, source_type, source_value, processing_time):
    entry = {"id": datetime.now().strftime("%Y%m%d%H%M%S%f"), "timestamp": time.time(), "datetime": datetime.now().isoformat(),
             "model": model_name, "source_type": source_type, "source_value": source_value, "processing_time": processing_time,
             "detections_count": len(result_data.get("detections", [])), "detections": result_data.get("detections", []),
             "image_url": result_data.get("image_url"), "video_url": result_data.get("video_url")}
    results_history.append(entry)
    return entry

@app.route('/')
def index(): return render_template('index.html')

@app.route('/history', methods=['GET'])
def get_history():
    limit = request.args.get('limit', 50, type=int)
    source_type = request.args.get('source_type')
    history_list = list(results_history)
    if source_type: history_list = [h for h in history_list if h['source_type'] == source_type]
    history_list.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({"count": len(history_list), "total_available": len(results_history), "history": history_list[:limit]})

@app.route('/history/clear', methods=['POST'])
def clear_history():
    results_history.clear()
    return jsonify({"status": "History gelöscht", "count": 0})

@app.route('/history/<entry_id>', methods=['GET'])
def get_history_entry(entry_id):
    for entry in results_history:
        if entry['id'] == entry_id: return jsonify(entry)
    return jsonify({"error": "Eintrag nicht gefunden"}), 404

@app.route('/exports/results/<export_format>', methods=['GET'])
def export_results(export_format):
    limit = request.args.get('limit', 100, type=int)
    history_list = list(results_history)[:limit]
    if export_format == 'json':
        return jsonify({"exported_at": datetime.now().isoformat(), "count": len(history_list), "results": history_list})
    elif export_format == 'csv':
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Timestamp', 'Model', 'Source Type', 'Detections', 'Processing Time'])
        for entry in history_list:
            writer.writerow([entry['id'], entry['datetime'], entry['model'], entry['source_type'], entry['detections_count'], f"{entry['processing_time']:.2f}s"])
        output.seek(0)
        return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=results.csv'})
    elif export_format == 'summary':
        total_detections = sum(e['detections_count'] for e in history_list)
        avg_time = sum(e['processing_time'] for e in history_list) / len(history_list) if history_list else 0
        class_counts = {}
        for entry in history_list:
            for det in entry['detections']:
                cls = det.get('class', 'unknown')
                class_counts[cls] = class_counts.get(cls, 0) + 1
        return jsonify({"total_inferences": len(history_list), "total_detections": total_detections,
                        "avg_processing_time": f"{avg_time:.2f}s", "detections_by_class": class_counts,
                        "top_classes": sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]})
    return jsonify({"error": "Ungültiges Format"}), 400

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    full_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], filename)
    if not os.path.exists(full_path): return jsonify({"error": "Datei nicht gefunden"}), 404
    mime, _ = mimetypes.guess_type(full_path)
    return send_file(full_path, mimetype=mime, as_attachment=False, conditional=True)

@app.route('/models', methods=['GET'])
def list_models():
    files = [f for f in os.listdir(app.config['MODEL_DIR']) if f.endswith('.pt')]
    return jsonify({"available_models": files, "active_model": active_model["name"]})

# ==================== YOLO MODEL DOWNLOAD ====================
YOLO_MODELS = [
    # YOLOv8 Models
    {"name": "yolov8n.pt", "type": "detect", "size": "3.2 MB", "speed": "80 FPS", "accuracy": "Low", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"},
    {"name": "yolov8s.pt", "type": "detect", "size": "11.2 MB", "speed": "40 FPS", "accuracy": "Medium", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"},
    {"name": "yolov8m.pt", "type": "detect", "size": "25.9 MB", "speed": "20 FPS", "accuracy": "High", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"},
    {"name": "yolov8l.pt", "type": "detect", "size": "43.7 MB", "speed": "10 FPS", "accuracy": "Very High", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"},
    {"name": "yolov8x.pt", "type": "detect", "size": "68.2 MB", "speed": "5 FPS", "accuracy": "Highest", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"},
    
    # YOLOv8 Pose
    {"name": "yolov8n-pose.pt", "type": "pose", "size": "3.4 MB", "speed": "75 FPS", "accuracy": "Low", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt"},
    {"name": "yolov8s-pose.pt", "type": "pose", "size": "11.8 MB", "speed": "38 FPS", "accuracy": "Medium", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt"},
    {"name": "yolov8m-pose.pt", "type": "pose", "size": "27.2 MB", "speed": "18 FPS", "accuracy": "High", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt"},
    
    # YOLOv8 Segmentation
    {"name": "yolov8n-seg.pt", "type": "seg", "size": "3.5 MB", "speed": "70 FPS", "accuracy": "Low", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"},
    {"name": "yolov8s-seg.pt", "type": "seg", "size": "12.4 MB", "speed": "35 FPS", "accuracy": "Medium", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt"},
    
    # YOLOv8 Classification
    {"name": "yolov8n-cls.pt", "type": "cls", "size": "2.8 MB", "speed": "85 FPS", "accuracy": "Low", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-cls.pt"},
    {"name": "yolov8s-cls.pt", "type": "cls", "size": "10.2 MB", "speed": "45 FPS", "accuracy": "Medium", "version": "v8", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-cls.pt"},
    
    # YOLOv5 Models
    {"name": "yolov5n.pt", "type": "detect", "size": "2.2 MB", "speed": "90 FPS", "accuracy": "Low", "version": "v5", "url": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5n.pt"},
    {"name": "yolov5s.pt", "type": "detect", "size": "7.2 MB", "speed": "50 FPS", "accuracy": "Medium", "version": "v5", "url": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5s.pt"},
    {"name": "yolov5m.pt", "type": "detect", "size": "21.2 MB", "speed": "25 FPS", "accuracy": "High", "version": "v5", "url": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5m.pt"},
    {"name": "yolov5l.pt", "type": "detect", "size": "46.5 MB", "speed": "12 FPS", "accuracy": "Very High", "version": "v5", "url": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5l.pt"},
    {"name": "yolov5x.pt", "type": "detect", "size": "86.7 MB", "speed": "6 FPS", "accuracy": "Highest", "version": "v5", "url": "https://github.com/ultralytics/yolov5/releases/download/v6.2/yolov5x.pt"},
    
    # YOLOv10 Models
    {"name": "yolov10n.pt", "type": "detect", "size": "2.8 MB", "speed": "85 FPS", "accuracy": "Low", "version": "v10", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov10n.pt"},
    {"name": "yolov10s.pt", "type": "detect", "size": "9.8 MB", "speed": "45 FPS", "accuracy": "Medium", "version": "v10", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov10s.pt"},
    {"name": "yolov10m.pt", "type": "detect", "size": "23.5 MB", "speed": "22 FPS", "accuracy": "High", "version": "v10", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov10m.pt"},
]

@app.route('/models/catalog', methods=['GET'])
def model_catalog():
    """Gibt Katalog der verfügbaren YOLO-Modelle zurück"""
    return jsonify({"models": YOLO_MODELS})

@app.route('/models/download/<model_name>', methods=['POST'])
def download_model(model_name):
    """Lädt ein YOLO-Modell herunter"""
    model_info = next((m for m in YOLO_MODELS if m['name'] == model_name), None)
    if not model_info:
        return jsonify({"error": "Modell nicht gefunden"}), 404
    
    model_path = os.path.join(app.config['MODEL_DIR'], model_name)
    if os.path.exists(model_path):
        return jsonify({"status": "Modell bereits vorhanden", "path": model_path})
    
    try:
        import requests
        from tqdm import tqdm
        
        # Download mit Fortschrittsanzeige
        response = requests.get(model_info['url'], stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            return jsonify({
                "status": "Download erfolgreich",
                "path": model_path,
                "size": os.path.getsize(model_path)
            })
        else:
            return jsonify({"error": "Download fehlgeschlagen"}), 500
            
    except Exception as e:
        if os.path.exists(model_path):
            os.remove(model_path)
        return jsonify({"error": str(e)}), 500

@app.route('/models/download_stream/<model_name>', methods=['POST'])
def download_model_stream(model_name):
    """Streamt den Download-Fortschritt"""
    model_info = next((m for m in YOLO_MODELS if m['name'] == model_name), None)
    if not model_info:
        return jsonify({"error": "Modell nicht gefunden"}), 404
    
    model_path = os.path.join(app.config['MODEL_DIR'], model_name)
    
    def generate():
        try:
            import requests
            
            response = requests.get(model_info['url'], stream=True)
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = int(downloaded / total_size * 100) if total_size > 0 else 0
                        yield f"data: {json.dumps({'progress': progress, 'downloaded': downloaded, 'total': total_size})}\n\n"
            
            yield f"data: {json.dumps({'done': True, 'path': model_path})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={"Cache-Control": "no-cache"})

@app.route('/models/suggest', methods=['GET'])
def suggest_model():
    task = request.args.get('task', 'detect')
    files = [f for f in os.listdir(app.config['MODEL_DIR']) if f.endswith('.pt')]
    task_models = []
    for f in files:
        fl = f.lower()
        if (task == 'pose' and 'pose' in fl) or (task == 'seg' and 'seg' in fl) or (task == 'cls' and 'cls' in fl) or \
           (task == 'detect' and 'pose' not in fl and 'seg' not in fl and 'cls' not in fl):
            task_models.append(f)
    size_order = {'n': 0, 's': 1, 'm': 2, 'l': 3, 'x': 4}
    task_models.sort(key=lambda n: next((p for s, p in size_order.items() if f'-{s}' in n.lower() or f'{s}.' in n.lower()), 5))
    return jsonify({"task": task, "available": task_models, "suggested": task_models[0] if task_models else None,
                    "recommendation": {"fast": [m for m in task_models if '-n.' in m.lower() or '-s.' in m.lower()][:1],
                                       "balanced": [m for m in task_models if '-m.' in m.lower()][:1],
                                       "accurate": [m for m in task_models if '-l.' in m.lower() or '-x.' in m.lower()][:1]} if task_models else {}})

@app.route('/load/<model_name>', methods=['POST'])
def load_model(model_name):
    global active_model
    path = os.path.join(app.config['MODEL_DIR'], model_name)
    if not os.path.exists(path): return jsonify({"error": "Modell nicht gefunden"}), 404
    try:
        if active_model["instance"]:
            print(f"Entlade: {active_model['name']}")
            active_model["instance"] = None
            active_model["name"] = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"Lade: {model_name}")
        active_model["instance"] = YOLO(path)
        active_model["name"] = model_name
        return jsonify({"status": f"Modell {model_name} aktiv", "active_model": model_name})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/test', methods=['POST'])
def test_model():
    if not active_model["instance"]: return jsonify({"error": "Kein Modell geladen"}), 400
    source_type = request.form.get('source_type', 'image')
    source_value = request.form.get('source_value', '')
    try:
        conf, iou, imgsz = float(request.form.get('conf', 0.25)), float(request.form.get('iou', 0.45)), int(request.form.get('imgsz', 640))
    except ValueError: return jsonify({"error": "Ungültige Parameter"}), 400
    save = request.form.get('save', 'true').lower() == 'true'
    img_path = None
    try:
        if source_type == 'image' and 'image' in request.files:
            file = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(img_path)
            inference_source = img_path
        elif source_type == 'url' and source_value: inference_source = source_value
        elif source_type == 'webcam':
            try: cam_id = int(source_value) if source_value else 0
            except ValueError: return jsonify({"error": "Ungültige Webcam-ID"}), 400
            inference_source = cam_id
        elif source_type == 'directory' and source_value:
            if not os.path.exists(source_value): return jsonify({"error": f"Verzeichnis nicht gefunden"}), 404
            inference_source = source_value
        elif source_type == 'glob' and source_value: inference_source = source_value
        elif source_type == 'screenshot': inference_source = 'screen'
        else: return jsonify({"error": f"Ungültige Quelle: {source_type}"}), 400
        stream_mode = source_type in {'video', 'url', 'webcam', 'directory', 'glob', 'stream'}
        with torch.inference_mode():
            results_gen = active_model["instance"](inference_source, conf=conf, iou=iou, imgsz=imgsz, save=save, show=False, stream=stream_mode, verbose=False)
        all_detections, last_save_dir = [], None
        for r in results_gen:
            det, save_dir = extract_results_data([r])
            all_detections.extend(det)
            if save_dir: last_save_dir = save_dir
        annotated_filename = None
        if save and source_type == 'image' and all_detections:
            annotated_filename = f"result_{int(time.time() * 1000)}.jpg"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], annotated_filename)
            try:
                with torch.inference_mode():
                    res_single = active_model["instance"](inference_source, conf=conf, iou=iou, imgsz=imgsz, save=False, show=False, stream=False, verbose=False)
                if res_single and len(res_single) == 1: res_single[0].save(filename=annotated_path)
            except Exception: annotated_filename = None
        response_payload = {"model_used": active_model["name"], "source_type": source_type, "source_value": str(inference_source),
                            "parameters": {"conf": conf, "iou": iou, "imgsz": imgsz}, "results": all_detections,
                            "detections": all_detections[0].get("detections", []) if all_detections else [],
                            "image_url": f"/uploads/{annotated_filename}" if annotated_filename else None,
                            "speed": all_detections[0].get("speed", {}) if all_detections else {}}
        if save and source_type in {'video', 'url', 'webcam', 'directory', 'glob', 'stream'} and last_save_dir:
            upload_root = Path(app.config['UPLOAD_FOLDER_ABS']).resolve()
            annotated_video = pick_annotated_video(last_save_dir, inference_source, upload_root)
            if annotated_video: response_payload["video_url"] = f"/uploads/{annotated_video.name}"
        save_to_history(response_payload, active_model["name"], source_type, str(inference_source),
                        all_detections[0].get("speed", {}).get("inference", 0) if all_detections else 0)
        return jsonify(response_payload)
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally:
        if img_path and os.path.exists(img_path):
            try: os.remove(img_path)
            except: pass

@app.route('/test_video', methods=['POST'])
def test_video():
    if not active_model["instance"]: return jsonify({"error": "Kein Modell geladen"}), 400
    if 'video' not in request.files: return jsonify({"error": "Kein Video"}), 400
    file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], file.filename)
    file.save(video_path)
    err = validate_video(video_path, MAX_VIDEO_MB, MAX_VIDEO_DURATION_S)
    if err: return jsonify({"error": err}), 400
    try:
        timestamp = int(time.time())
        save_dir = Path(app.config['UPLOAD_FOLDER_ABS']) / f"results_{timestamp}"
        save_dir.mkdir(exist_ok=True)
        with torch.inference_mode():
            results = active_model["instance"](video_path, stream=True, save=True, project=str(save_dir.parent),
                                                name=f"results_{timestamp}", exist_ok=True, verbose=False)
        for r in results: result_save_dir = Path(r.save_dir).resolve()
        upload_root = Path(app.config['UPLOAD_FOLDER_ABS']).resolve()
        annotated = pick_annotated_video(result_save_dir, video_path, upload_root)
        return jsonify({"model_used": active_model["name"], "video_url": f"/uploads/{annotated.name}" if annotated else None, "save_dir": str(save_dir)})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/test_video_stream', methods=['POST'])
def test_video_stream():
    if not active_model["instance"]: return jsonify({"error": "Kein Modell geladen"}), 400
    if 'video' not in request.files: return jsonify({"error": "Kein Video"}), 400
    file = request.files['video']
    video_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], file.filename)
    file.save(video_path)
    err = validate_video(video_path, MAX_VIDEO_MB, MAX_VIDEO_DURATION_S)
    if err: return jsonify({"error": err}), 400
    timestamp = int(time.time())
    save_dir = Path(app.config['UPLOAD_FOLDER_ABS']) / f"results_{timestamp}"
    save_dir.mkdir(exist_ok=True)
    total_frames = None
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        cap.release()
    except: pass
    def generate():
        processed, result_save_dir = 0, None
        try:
            with torch.inference_mode():
                results = active_model["instance"](video_path, stream=True, save=True, project=str(save_dir.parent),
                                                    name=f"results_{timestamp}", exist_ok=True, verbose=False)
            for r in results:
                processed += 1
                pct = int(processed / total_frames * 100) if total_frames else None
                yield f"data: {json.dumps({'frame': processed, 'total_frames': total_frames, 'progress': pct, 'message': f'Frame {processed}/{total_frames}' if total_frames else f'Frame {processed}'})}\n\n"
                result_save_dir = Path(r.save_dir).resolve()
            upload_root = Path(app.config['UPLOAD_FOLDER_ABS']).resolve()
            annotated = pick_annotated_video(result_save_dir, video_path, upload_root)
            yield f"data: {json.dumps({'done': True, 'model_used': active_model['name'], 'video_url': f'/uploads/{annotated.name}' if annotated else None, 'save_dir': str(result_save_dir) if result_save_dir else None})}\n\n"
        except Exception as e: yield f"data: {json.dumps({'error': str(e)})}\n\n"
    return Response(generate(), mimetype='text/event-stream', headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route('/test_url_stream', methods=['POST'])
def test_url_stream():
    if not active_model["instance"]: return jsonify({"error": "Kein Modell geladen"}), 400
    source_value = request.form.get('source_value', '')
    if not source_value: return jsonify({"error": "URL erforderlich"}), 400
    conf, iou, imgsz = float(request.form.get('conf', 0.25)), float(request.form.get('iou', 0.45)), int(request.form.get('imgsz', 640))
    save = request.form.get('save', 'true').lower() == 'true'
    timestamp = int(time.time())
    save_dir = Path(app.config['UPLOAD_FOLDER_ABS']) / f"results_url_{timestamp}"
    save_dir.mkdir(exist_ok=True)
    total_frames, fps = None, None
    try:
        cap = cv2.VideoCapture(source_value)
        if cap.isOpened():
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if total_frames and fps else None
            cap.release()
    except: pass
    def generate():
        processed, result_save_dir = 0, None
        try:
            with torch.inference_mode():
                results = active_model["instance"](source_value, stream=True, save=save, conf=conf, iou=iou, imgsz=imgsz,
                                                    project=str(save_dir.parent), name=f"results_url_{timestamp}", exist_ok=True, verbose=False)
            for r in results:
                processed += 1
                pct = int(processed / total_frames * 100) if total_frames else None
                yield f"data: {json.dumps({'frame': processed, 'total_frames': total_frames, 'progress': pct, 'message': f'Frame {processed}/{total_frames}' if total_frames else f'Frame {processed} verarbeitet'})}\n\n"
                result_save_dir = Path(r.save_dir).resolve()
            upload_root = Path(app.config['UPLOAD_FOLDER_ABS']).resolve()
            annotated = pick_annotated_video(result_save_dir, source_value, upload_root) if save else None
            yield f"data: {json.dumps({'done': True, 'model_used': active_model['name'], 'video_url': f'/uploads/{annotated.name}' if annotated else None, 'save_dir': str(result_save_dir) if result_save_dir else None, 'total_frames_processed': processed})}\n\n"
        except Exception as e: yield f"data: {json.dumps({'error': str(e), 'frame': processed})}\n\n"
    return Response(generate(), mimetype='text/event-stream', headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

def is_subpath(path: Path, root: Path) -> bool:
    try: path.resolve().relative_to(root.resolve()); return True
    except: return False

def pick_annotated_video(save_dir: Path, original_path: str, upload_root: Path) -> Path | None:
    if not save_dir: return None
    save_dir, orig = save_dir.resolve(), Path(original_path).resolve()
    annotated_candidates = []
    if save_dir.exists():
        for f in save_dir.glob("*.mp4"):
            if f.resolve() != orig: annotated_candidates.append(f.resolve())
        for f in save_dir.glob("*.avi"):
            if f.resolve() != orig: annotated_candidates.append(f.resolve())
        for f in save_dir.glob("*.mov"):
            if f.resolve() != orig: annotated_candidates.append(f.resolve())
    runs_dir = Path.home() / "runs"
    if runs_dir.exists():
        for f in runs_dir.rglob("*.mp4"):
            if f.resolve() != orig: annotated_candidates.append(f.resolve())
    if upload_root.exists():
        for f in upload_root.glob("*.mp4"):
            if f.resolve() != orig and "annotated" in f.name.lower(): annotated_candidates.append(f.resolve())
    if not annotated_candidates: return None
    def score(p: Path): return (1 if save_dir in p.parents else 0, p.stat().st_mtime)
    candidate = sorted(annotated_candidates, key=score, reverse=True)[0]
    orig_name = orig.name if orig.name else "video"
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', orig_name)
    if len(safe_name) > 50: safe_name = safe_name[:50]
    dest = upload_root / f"annotated_{safe_name}_{int(time.time())}{candidate.suffix}"
    try: shutil.copy2(candidate, dest)
    except Exception as e: print(f"Kopierfehler: {e}"); dest = candidate
    if dest.suffix.lower() != ".mp4":
        mp4_dest = upload_root / f"annotated_{safe_name}_{int(time.time())}.mp4"
        if convert_to_mp4(dest, mp4_dest): dest = mp4_dest
    else:
        mp4_dest = upload_root / f"annotated_{safe_name}_{int(time.time())}.mp4"
        if convert_to_mp4(dest, mp4_dest): dest = mp4_dest
    return dest if dest.exists() else None

def convert_to_mp4(src: Path, dest: Path) -> bool:
    try:
        import shutil as _shutil
        if _shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-an", str(dest)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return dest.exists() and dest.stat().st_size > 0
    except Exception as e: print(f"FFmpeg Fehler: {e}")
    try:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened(): return False
        fps, width, height = cap.get(cv2.CAP_PROP_FPS) or 25.0, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(str(dest), fourcc, fps, (width, height))
        while True:
            ret, frame = cap.read()
            if not ret: break
            out.write(frame)
        cap.release(); out.release()
        return dest.exists() and dest.stat().st_size > 0
    except Exception as e: print(f"OpenCV Fehler: {e}"); return False

def validate_video(path: str, max_mb: int, max_duration_s: int) -> str | None:
    try:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > max_mb: return f"Video zu groß ({size_mb:.1f} MB). Max {max_mb} MB."
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): return "Video nicht lesbar."
        fps, frames = cap.get(cv2.CAP_PROP_FPS) or 0, cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        cap.release()
        if fps > 0 and frames > 0:
            duration = frames / fps
            if duration > max_duration_s: return f"Video zu lang ({duration:.1f}s). Max {max_duration_s}s."
    except: return None
    return None

@app.route('/test_webcam', methods=['GET', 'POST'])
def test_webcam():
    if not active_model["instance"]: return jsonify({"error": "Kein Modell geladen"}), 400
    cam_id = request.values.get('cam_id', '0')
    try: cam_id = int(cam_id)
    except ValueError: return jsonify({"error": "Ungültige Webcam-ID"}), 400
    conf, imgsz = float(request.values.get('conf', 0.25)), int(request.values.get('imgsz', 640))
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened(): return jsonify({"error": f"Webcam {cam_id} nicht verfügbar"}), 404
    cap.release()
    def generate():
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count, max_frames = 0, 300
        try:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret: break
                with torch.inference_mode(): results = active_model["instance"](frame, conf=conf, imgsz=imgsz, verbose=False)
                annotated = results[0].plot()
                _, buffer = cv2.imencode('.jpg', annotated)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                frame_count += 1
        finally: cap.release()
    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame', headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route('/test_stream', methods=['POST'])
def test_stream():
    if not active_model["instance"]: return jsonify({"error": "Kein Modell geladen"}), 400
    stream_url = request.form.get('stream_url', '')
    if not stream_url: return jsonify({"error": "Stream-URL erforderlich"}), 400
    conf, imgsz = float(request.form.get('conf', 0.25)), int(request.form.get('imgsz', 640))
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened(): return jsonify({"error": f"Stream nicht verfügbar"}), 404
    cap.release()
    def generate():
        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count, max_frames = 0, 300
        try:
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret: time.sleep(1); cap.open(stream_url); continue
                results = active_model["instance"](frame, conf=conf, imgsz=imgsz, verbose=False)
                annotated = results[0].plot()
                _, buffer = cv2.imencode('.jpg', annotated)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                frame_count += 1
        finally: cap.release()
    return Response(stream_with_context(generate()), mimetype='multipart/x-mixed-replace; boundary=frame', headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route('/test_batch', methods=['POST'])
def test_batch():
    if not active_model["instance"]: return jsonify({"error": "Kein Modell geladen"}), 400
    if 'images' not in request.files: return jsonify({"error": "Keine Bilder"}), 400
    files = request.files.getlist('images')
    if not files: return jsonify({"error": "Keine Bilder ausgewählt"}), 400
    conf, iou, imgsz = float(request.form.get('conf', 0.25)), float(request.form.get('iou', 0.45)), int(request.form.get('imgsz', 640))
    img_paths, unsupported = [], []
    for file in files:
        if not file.filename: continue
        ext = Path(file.filename).suffix.lower()
        if ext in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
            safe_name = f"{int(time.time()*1000)}_{file.filename}"
            img_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], safe_name)
            file.save(img_path)
            img_paths.append(img_path)
        elif ext in {'.mp4', '.mov', '.avi', '.mkv'}: unsupported.append(file.filename)
        else: unsupported.append(file.filename)
    if unsupported and not img_paths: return jsonify({"error": f"Nicht unterstützt: {', '.join(unsupported)}"}), 400
    if not img_paths: return jsonify({"error": "Keine gültigen Bilder"}), 400
    try:
        with torch.inference_mode(): results = active_model["instance"](img_paths, conf=conf, iou=iou, imgsz=imgsz)
        all_detections, _ = extract_results_data(results)
        annotated_files = []
        for i, r in enumerate(results):
            annotated_filename = f"result_batch_{i}_{int(time.time() * 1000)}.jpg"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], annotated_filename)
            try: r.save(filename=annotated_path); annotated_files.append(f"/uploads/{annotated_filename}")
            except: pass
        return jsonify({"model_used": active_model["name"], "processed_count": len(results), "results": all_detections,
                        "annotated_images": annotated_files, "parameters": {"conf": conf, "iou": iou, "imgsz": imgsz}})
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally:
        for img_path in img_paths:
            if os.path.exists(img_path):
                try: os.remove(img_path)
                except: pass

@app.route('/health', methods=['GET'])
def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
    gpu_memory = {"total": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
                  "allocated": round(torch.cuda.memory_allocated(0) / 1024**3, 2),
                  "reserved": round(torch.cuda.memory_reserved(0) / 1024**3, 2)} if gpu_available else None
    return jsonify({"status": "healthy", "model_loaded": active_model["name"],
                    "gpu": {"available": gpu_available, "name": gpu_name, "memory": gpu_memory} if gpu_available else {"available": False},
                    "cpu_count": os.cpu_count(), "torch_version": torch.__version__, "timestamp": time.time()})

@app.route('/compare', methods=['POST'])
def compare_models():
    if 'image' not in request.files: return jsonify({"error": "Kein Bild"}), 400
    models = request.form.getlist('models')
    if not models: return jsonify({"error": "Keine Modelle"}), 400
    conf, iou, imgsz = float(request.form.get('conf', 0.25)), float(request.form.get('iou', 0.45)), int(request.form.get('imgsz', 640))
    file = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)
    results, start_time = [], time.time()
    try:
        for model_name in models:
            model_path = os.path.join(app.config['MODEL_DIR'], model_name)
            if not os.path.exists(model_path):
                results.append({"model": model_name, "error": "Modell nicht gefunden"}); continue
            try:
                model = YOLO(model_path)
                with torch.inference_mode(): res = model(img_path, conf=conf, iou=iou, imgsz=imgsz, verbose=False)
                detections = []
                if res and res[0].boxes is not None and res[0].boxes.xyxy is not None:
                    for i, box in enumerate(res[0].boxes):
                        cls_id = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                        conf_val = float(box.conf[0]) if box.conf is not None and len(box.conf) > 0 else 0
                        bbox = [round(x, 2) for x in box.xyxy[0].tolist()]
                        detections.append({"class": res[0].names.get(cls_id, "unknown"), "class_id": cls_id, "confidence": conf_val, "bbox": bbox})
                results.append({"model": model_name, "detections": detections, "detections_count": len(detections),
                                "inference_time": res[0].speed.get('inference', 0) if res else 0, "success": True})
                del model
            except Exception as e: results.append({"model": model_name, "error": str(e), "success": False})
        total_time = time.time() - start_time
        comparison = {"total_time": round(total_time, 2), "models_compared": len(results), "results": results,
                      "fastest_model": min((r for r in results if r.get('success')), key=lambda x: x.get('inference_time', float('inf')), default=None),
                      "most_detections": max((r for r in results if r.get('success')), key=lambda x: x.get('detections_count', 0), default=None)}
        return jsonify(comparison)
    finally:
        if os.path.exists(img_path):
            try: os.remove(img_path)
            except: pass
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

@app.route('/system/info', methods=['GET'])
def system_info():
    import platform
    info = {"platform": platform.system(), "platform_version": platform.version(), "python_version": platform.python_version(),
            "torch_version": torch.__version__, "torch_cuda": torch.version.cuda if hasattr(torch.version, 'cuda') else None,
            "cuda_available": torch.cuda.is_available(), "cpu_count": os.cpu_count()}
    try:
        import psutil
        info["memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except ImportError: info["memory_gb"] = None
    return jsonify(info)

# ==================== GROUNDING DINO / DATASET ====================
grounding_dino_model = {"instance": None, "loaded": False}
grounding_dino_available = False
few_shot_model = {"samples": [], "classes": {}, "loaded": False}

def check_grounding_dino():
    """Prüft ob Grounding DINO verfügbar ist"""
    global grounding_dino_available
    try:
        import groundingdino
        grounding_dino_available = True
        return True
    except ImportError:
        grounding_dino_available = False
        return False

# ==================== FEW-SHOT AUTO-DETECTION ====================
class FewShotDetector:
    """Few-Shot Objekterkennung mit Template-Matching und Feature-Vergleich"""
    
    def __init__(self):
        self.samples = []  # Gespeicherte Sample-Bilder mit Annotationen
        self.class_templates = {}  # Feature-Templates pro Klasse
        self.clip_model = None
        self.clip_preprocess = None
        
    def load_clip(self):
        """Lädt CLIP Modell für Feature-Extraction"""
        try:
            import clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
            self.clip_model.eval()
            return True
        except (ImportError, FileNotFoundError):
            return False
    
    def extract_features(self, image_path):
        """Extrahiert Features aus einem Bild"""
        try:
            from PIL import Image
            device = next(self.clip_model.parameters()).device if self.clip_model else "cpu"
            image = Image.open(image_path)
            image_input = self.clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = self.clip_model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()
        except Exception as e:
            return None
    
    def extract_patch_features(self, image_cv, bbox):
        """Extrahiert Features aus einem Bildausschnitt"""
        try:
            from PIL import Image
            import io
            x1, y1, x2, y2 = [int(c) for c in bbox]
            patch = image_cv[y1:y2, x1:x2]
            if patch.size == 0:
                return None
            patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            device = next(self.clip_model.parameters()).device if self.clip_model else "cpu"
            image_input = self.clip_preprocess(patch_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                features = self.clip_model.encode_image(image_input)
                features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy()
        except Exception as e:
            return None
    
    def add_sample(self, image_path, annotations):
        """Fügt ein Sample-Bild mit Annotationen hinzu"""
        sample = {
            "image_path": image_path,
            "annotations": annotations,
            "features": self.extract_features(image_path) if self.clip_model else None
        }
        self.samples.append(sample)
        
        # Templates pro Klasse erstellen
        image_cv = cv2.imread(image_path)
        for ann in annotations:
            cls = ann.get("class", "object")
            bbox = ann.get("bbox", [])
            if bbox and image_cv is not None:
                patch_features = self.extract_patch_features(image_cv, bbox)
                if patch_features is not None:
                    if cls not in self.class_templates:
                        self.class_templates[cls] = []
                    self.class_templates[cls].append(patch_features)
        
        return len(self.samples)
    
    def clear_samples(self):
        """Löscht alle Samples"""
        self.samples = []
        self.class_templates = {}
    
    def detect_similar(self, image_path, threshold=0.7):
        """Erkennt ähnliche Objekte in einem neuen Bild"""
        if not self.class_templates or not self.clip_model:
            return []
        
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            image_cv = cv2.imread(image_path)
            if image_cv is None:
                return []
            
            image_h, image_w = image_cv.shape[:2]
            detections = []
            
            # Erstelle Feature-Vergleich mit verschiedenen Patch-Größen
            # Verwende Sliding-Window-Ansatz für einfache Erkennung
            patch_sizes = [(64, 64), (96, 96), (128, 128), (160, 160), (200, 200)]
            strides = [32, 40, 48, 56, 64]
            
            for patch_size, stride in zip(patch_sizes, strides):
                ph, pw = patch_size
                for y in range(0, image_h - ph, stride):
                    for x in range(0, image_w - pw, stride):
                        patch = image_cv[y:y+ph, x:x+pw]
                        if patch.size == 0:
                            continue
                        
                        try:
                            patch_pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
                            device = next(self.clip_model.parameters()).device
                            image_input = self.clip_preprocess(patch_pil).unsqueeze(0).to(device)
                            with torch.no_grad():
                                patch_features = self.clip_model.encode_image(image_input)
                                patch_features = patch_features / patch_features.norm(dim=-1, keepdim=True)
                                patch_features = patch_features.cpu().numpy()
                            
                            # Vergleiche mit allen Klassen-Templates
                            best_class = None
                            best_score = 0
                            
                            for cls, templates in self.class_templates.items():
                                for template in templates:
                                    similarity = np.dot(patch_features, template.T)[0][0]
                                    if similarity > best_score:
                                        best_score = similarity
                                        best_class = cls
                            
                            if best_score >= threshold and best_class:
                                # Prüfe ob diese Detection nicht zu nah an einer bestehenden ist
                                is_duplicate = False
                                for existing in detections:
                                    ex, ey, ew, eh = existing["bbox"]
                                    if abs(x - ex) < pw/2 and abs(y - ey) < ph/2:
                                        if best_score > existing["confidence"]:
                                            detections.remove(existing)
                                        else:
                                            is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    detections.append({
                                        "class": best_class,
                                        "bbox": [x, y, x + pw, y + ph],
                                        "bbox_normalized": [x/image_w, y/image_h, (x+pw)/image_w, (y+ph)/image_h],
                                        "confidence": float(best_score)
                                    })
                        except Exception:
                            continue
            
            # Non-Maximum Suppression
            detections = self._nms(detections, iou_threshold=0.5)
            return detections
            
        except Exception as e:
            return []
    
    def _nms(self, detections, iou_threshold=0.5):
        """Non-Maximum Suppression für Overlapping Detections"""
        if not detections:
            return []
        
        # Sortiere nach Confidence
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Entferne überlappende Detections
            remaining = []
            for det in detections:
                iou = self._calculate_iou(best["bbox"], det["bbox"])
                if iou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Berechnet IoU zwischen zwei Bounding Boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def batch_detect(self, image_paths, threshold=0.7, progress_callback=None):
        """Führt Auto-Detection auf mehreren Bildern durch"""
        results = []
        total = len(image_paths)
        
        for i, img_path in enumerate(image_paths):
            detections = self.detect_similar(img_path, threshold)
            results.append({
                "filename": os.path.basename(img_path),
                "image_path": img_path,
                "annotations": detections,
                "annotation_count": len(detections)
            })
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results

# Initialisiere Few-Shot Detector
few_shot_detector = FewShotDetector()

def load_grounding_dino():
    """Lädt Grounding DINO Modell für zero-shot Objekterkennung"""
    try:
        from groundingdino.util.inference import load_model
        model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", 
                          "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        return {"model": model, "loaded": True, "error": None}
    except Exception as e:
        return {"model": None, "loaded": False, "error": str(e)}

def run_grounding_dino_inference(image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
    """Führt Grounding DINO Inference durch"""
    try:
        from groundingdino.util.inference import load_model, load_image, predict, annotate
        import torchvision.transforms as transforms
        from PIL import Image
        
        # Bild laden
        image_pil = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image_pil).unsqueeze(0)
        
        # Modell laden falls nicht geladen
        if not grounding_dino_model["loaded"] or grounding_dino_model["instance"] is None:
            grounding_dino_model["instance"] = load_model(
                "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            )
            grounding_dino_model["loaded"] = True
        
        model = grounding_dino_model["instance"]
        
        # Prediction
        with torch.inference_mode():
            outputs = model(image_tensor, captions=[text_prompt])
        
        # Ergebnisse extrahieren
        detection_results = []
        if hasattr(outputs, "pred_boxes") and outputs.pred_boxes is not None:
            boxes = outputs.pred_boxes[0]
            scores = outputs.pred_logits[0].sigmoid().max(dim=1)[0]
            logits = outputs.pred_logits[0].sigmoid()
            
            for idx, (box, score) in enumerate(zip(boxes, scores)):
                if score.item() > box_threshold:
                    # Bounding Box normalisieren
                    bbox_norm = box.tolist()
                    detection_results.append({
                        "bbox_normalized": bbox_norm,
                        "confidence": round(score.item(), 4),
                        "prompt": text_prompt,
                        "class_id": idx
                    })
        
        return {"success": True, "detections": detection_results, "image_size": image_pil.size}
    except Exception as e:
        return {"success": False, "error": str(e), "detections": []}

def run_yolo_zero_shot(image_path, text_prompt, box_threshold=0.25):
    """Fallback: Verwendet YOLO für Objekterkennung wenn Grounding DINO nicht verfügbar ist"""
    try:
        from ultralytics import YOLO
        
        # Versuche ein vorhandenes Modell zu laden oder lade ein Standardmodell
        model = None
        model_files = [f for f in os.listdir(app.config['MODEL_DIR']) if f.endswith('.pt')]
        
        if model_files:
            # Nimm das erste verfügbare Modell
            model_path = os.path.join(app.config['MODEL_DIR'], model_files[0])
            model = YOLO(model_path)
        else:
            # Lade YOLOv8n als Fallback (kleines schnelles Modell)
            model = YOLO('yolov8n.pt')
        
        # Inference auf CPU durchführen für bessere Kompatibilität
        device = 'cpu'  # Erzwinge CPU-Nutzung für Kompatibilität
        with torch.inference_mode():
            results = model(image_path, conf=box_threshold, verbose=False, device=device)
        
        detection_results = []
        if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
            image_cv = cv2.imread(image_path)
            image_h, image_w = image_cv.shape[:2]
            
            for i, box in enumerate(results[0].boxes):
                cls_id = int(box.cls[0]) if box.cls is not None and len(box.cls) > 0 else -1
                conf_val = float(box.conf[0]) if box.conf is not None and len(box.conf) > 0 else 0
                bbox = box.xyxy[0].tolist() if box.xyxy is not None else []
                
                if len(bbox) == 4:
                    # Normalisierte Bounding Box
                    bbox_norm = [bbox[0]/image_w, bbox[1]/image_h, bbox[2]/image_w, bbox[3]/image_h]
                    class_name = results[0].names.get(cls_id, "object") if results[0].names else "object"
                    
                    # Prüfen ob die Klasse zum Prompt passt (einfaches Matching)
                    prompt_words = text_prompt.lower().split()
                    if any(word in class_name.lower() for word in prompt_words) or not prompt_words:
                        detection_results.append({
                            "bbox_normalized": bbox_norm,
                            "confidence": round(conf_val, 4),
                            "prompt": text_prompt,
                            "class": class_name,
                            "class_id": cls_id
                        })
        
        return {"success": True, "detections": detection_results, "image_size": (image_w, image_h)}
    except Exception as e:
        return {"success": False, "error": str(e), "detections": []}

@app.route('/dataset/annotate', methods=['POST'])
def dataset_annotate():
    """Erstellt Annotationen mit Grounding DINO oder YOLO Fallback"""
    if 'image' not in request.files:
        return jsonify({"error": "Kein Bild"}), 400

    text_prompt = request.form.get('prompt', '')
    if not text_prompt:
        return jsonify({"error": "Text-Prompt erforderlich"}), 400

    try:
        box_threshold = float(request.form.get('box_threshold', 0.35))
        text_threshold = float(request.form.get('text_threshold', 0.25))
    except ValueError:
        return jsonify({"error": "Ungültige Threshold-Werte"}), 400

    file = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], f"gdino_{int(time.time()*1000)}_{file.filename}")
    file.save(img_path)

    try:
        # Versuche Grounding DINO, falle zurück auf YOLO
        result = run_grounding_dino_inference(img_path, text_prompt, box_threshold, text_threshold)
        
        # Wenn Grounding DINO fehlschlägt, verwende YOLO Fallback
        if not result.get("success") or not grounding_dino_available:
            result = run_yolo_zero_shot(img_path, text_prompt, box_threshold)

        if result.get("success"):
            # Annotiertes Bild erstellen
            image_cv = cv2.imread(img_path)
            image_h, image_w = image_cv.shape[:2]

            annotations = []
            for det in result["detections"]:
                bbox_norm = det["bbox_normalized"]
                x1 = int(bbox_norm[0] * image_w)
                y1 = int(bbox_norm[1] * image_h)
                x2 = int(bbox_norm[2] * image_w)
                y2 = int(bbox_norm[3] * image_h)

                # Bounding Box zeichnen
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_name = det.get("class", text_prompt)
                cv2.putText(image_cv, f"{class_name} {det['confidence']:.2f}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                annotations.append({
                    "class": class_name,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_normalized": bbox_norm,
                    "confidence": det["confidence"]
                })

            # Annotiertes Bild speichern
            annotated_filename = f"annotated_{int(time.time()*1000)}.jpg"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], annotated_filename)
            cv2.imwrite(annotated_path, image_cv)

            return jsonify({
                "success": True,
                "annotations": annotations,
                "annotation_count": len(annotations),
                "annotated_image": f"/uploads/{annotated_filename}",
                "prompt": text_prompt,
                "method": "grounding_dino" if grounding_dino_available else "yolo_fallback"
            })
        else:
            return jsonify({"success": False, "error": result.get("error", "Unbekannter Fehler")}), 500
    finally:
        if os.path.exists(img_path):
            try: os.remove(img_path)
            except: pass

@app.route('/dataset/annotate_batch', methods=['POST'])
def dataset_annotate_batch():
    """Erstellt Annotationen für mehrere Bilder mit Grounding DINO oder YOLO Fallback"""
    if 'images' not in request.files:
        return jsonify({"error": "Keine Bilder"}), 400

    text_prompt = request.form.get('prompt', '')
    if not text_prompt:
        return jsonify({"error": "Text-Prompt erforderlich"}), 400

    try:
        box_threshold = float(request.form.get('box_threshold', 0.35))
        text_threshold = float(request.form.get('text_threshold', 0.25))
    except ValueError:
        return jsonify({"error": "Ungültige Threshold-Werte"}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "Keine Bilder ausgewählt"}), 400

    results = []
    annotated_images = []

    for file in files:
        if not file.filename:
            continue

        img_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], f"gdino_{int(time.time()*1000)}_{file.filename}")
        file.save(img_path)

        try:
            # Versuche Grounding DINO, falle zurück auf YOLO
            result = run_grounding_dino_inference(img_path, text_prompt, box_threshold, text_threshold)
            
            # Wenn Grounding DINO fehlschlägt, verwende YOLO Fallback
            if not result.get("success") or not grounding_dino_available:
                result = run_yolo_zero_shot(img_path, text_prompt, box_threshold)

            if result.get("success"):
                image_cv = cv2.imread(img_path)
                image_h, image_w = image_cv.shape[:2]

                annotations = []
                for det in result["detections"]:
                    bbox_norm = det["bbox_normalized"]
                    x1 = int(bbox_norm[0] * image_w)
                    y1 = int(bbox_norm[1] * image_h)
                    x2 = int(bbox_norm[2] * image_w)
                    y2 = int(bbox_norm[3] * image_h)

                    class_name = det.get("class", text_prompt)
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    annotations.append({
                        "filename": file.filename,
                        "class": class_name,
                        "bbox": [x1, y1, x2, y2],
                        "bbox_normalized": bbox_norm,
                        "confidence": det["confidence"]
                    })

                annotated_filename = f"annotated_{int(time.time()*1000)}_{file.filename}"
                annotated_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], annotated_filename)
                cv2.imwrite(annotated_path, image_cv)
                annotated_images.append(f"/uploads/{annotated_filename}")

                results.append({
                    "filename": file.filename,
                    "annotations": annotations,
                    "annotation_count": len(annotations)
                })
        finally:
            if os.path.exists(img_path):
                try: os.remove(img_path)
                except: pass

    return jsonify({
        "success": True,
        "processed_count": len(results),
        "results": results,
        "annotated_images": annotated_images[:5],  # Nur erste 5 für Preview
        "prompt": text_prompt,
        "method": "grounding_dino" if grounding_dino_available else "yolo_fallback"
    })

@app.route('/dataset/export/yolo', methods=['POST'])
def dataset_export_yolo():
    """Exportiert Annotationen im YOLO-Format"""
    data = request.get_json()
    annotations = data.get('annotations', [])
    class_names = data.get('class_names', ['object'])
    image_width = data.get('image_width', 640)
    image_height = data.get('image_height', 640)
    
    yolo_lines = []
    for ann in annotations:
        bbox_norm = ann.get('bbox_normalized', ann.get('bbox', []))
        if len(bbox_norm) == 4:
            # YOLO Format: class x_center y_center width height (alle normalisiert 0-1)
            x1, y1, x2, y2 = bbox_norm
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            class_id = ann.get('class_id', 0)
            if class_id >= len(class_names):
                class_id = 0
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    yolo_content = '\n'.join(yolo_lines)
    
    # Als Datei speichern
    label_filename = f"label_{int(time.time()*1000)}.txt"
    label_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], label_filename)
    with open(label_path, 'w') as f:
        f.write(yolo_content)
    
    # classes.txt erstellen
    classes_content = '\n'.join(class_names)
    classes_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], f"classes_{int(time.time()*1000)}.txt")
    with open(classes_path, 'w') as f:
        f.write(classes_content)
    
    return jsonify({
        "success": True,
        "yolo_format": yolo_content,
        "label_file": f"/uploads/{label_filename}",
        "classes_file": f"/uploads/{classes_path}",
        "class_names": class_names
    })

@app.route('/dataset/export/coco', methods=['POST'])
def dataset_export_coco():
    """Exportiert Annotationen im COCO-Format"""
    data = request.get_json()
    annotations = data.get('annotations', [])
    class_names = data.get('class_names', ['object'])
    image_id = data.get('image_id', 1)
    image_width = data.get('image_width', 640)
    image_height = data.get('image_height', 640)
    
    coco_format = {
        "images": [{
            "id": image_id,
            "width": image_width,
            "height": image_height,
            "file_name": "image.jpg"
        }],
        "annotations": [],
        "categories": [{"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(class_names)]
    }
    
    for ann_id, ann in enumerate(annotations):
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            class_id = ann.get('class_id', 0)
            
            coco_format["annotations"].append({
                "id": ann_id + 1,
                "image_id": image_id,
                "category_id": class_id,
                "bbox": [x1, y1, width, height],
                "area": width * height,
                "iscrowd": 0,
                "confidence": ann.get('confidence', 1.0)
            })
    
    return jsonify({
        "success": True,
        "coco_format": coco_format,
        "download_url": f"/exports/coco_{int(time.time()*1000)}.json"
    })

@app.route('/dataset/templates', methods=['GET'])
def dataset_templates():
    """Gibt Vorlagen für häufige Use-Cases zurück"""
    templates = [
        {"id": "objects", "name": "Allgemeine Objekte", "prompt": "object item thing", "description": "Erkennt allgemeine Objekte"},
        {"id": "people", "name": "Personen", "prompt": "person human people", "description": "Erkennt Personen"},
        {"id": "vehicles", "name": "Fahrzeuge", "prompt": "car truck bus motorcycle bicycle vehicle", "description": "Erkennt verschiedene Fahrzeuge"},
        {"id": "animals", "name": "Tiere", "prompt": "dog cat bird animal", "description": "Erkennt Tiere"},
        {"id": "food", "name": "Essen", "prompt": "food fruit vegetable meal", "description": "Erkennt Lebensmittel"},
        {"id": "electronics", "name": "Elektronik", "prompt": "phone laptop computer electronics device", "description": "Erkennt Elektronikgeräte"},
        {"id": "furniture", "name": "Möbel", "prompt": "chair table sofa furniture", "description": "Erkennt Möbelstücke"},
        {"id": "safety", "name": "Sicherheit", "prompt": "helmet vest safety equipment person", "description": "Erkennt Sicherheitsausrüstung"}
    ]
    return jsonify({"templates": templates})

# ==================== YOLO TRAINING ====================
training_job = {"active": False, "process": None, "results": None, "logs": []}

@app.route('/training/status', methods=['GET'])
def training_status():
    """Gibt den aktuellen Training-Status zurück"""
    return jsonify({
        "active": training_job["active"],
        "results": training_job["results"],
        "logs": training_job["logs"][-50:]  # Letzte 50 Logs
    })

@app.route('/training/start', methods=['POST'])
def training_start():
    """Startet ein YOLO-Training"""
    global training_job
    
    if training_job["active"]:
        return jsonify({"error": "Training läuft bereits"}), 400
    
    # Parameter auslesen
    try:
        epochs = int(request.form.get('epochs', 100))
        batch = int(request.form.get('batch', 16))
        imgsz = int(request.form.get('imgsz', 640))
        model_name = request.form.get('model', 'yolov8n.pt')
    except ValueError:
        return jsonify({"error": "Ungültige Parameter"}), 400
    
    # Dataset prüfen
    if 'dataset' not in request.files:
        return jsonify({"error": "Kein Dataset"}), 400
    
    dataset_file = request.files['dataset']
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], f"dataset_{int(time.time())}.zip")
    dataset_file.save(dataset_path)
    
    # Training in separatem Thread starten
    def run_training():
        global training_job
        training_job["logs"] = []
        start_time = time.time()
        
        try:
            from ultralytics import YOLO
            
            # Dataset entpacken
            import zipfile
            extract_dir = os.path.join(app.config['UPLOAD_FOLDER_ABS'], f"dataset_{int(time.time())}")
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            # data.yaml finden
            data_yaml = None
            for root, dirs, files in os.walk(extract_dir):
                if 'data.yaml' in files:
                    data_yaml = os.path.join(root, 'data.yaml')
                    break
            
            if not data_yaml:
                training_job["logs"].append({"type": "error", "message": "data.yaml nicht gefunden"})
                training_job["active"] = False
                return
            
            training_job["logs"].append({"type": "info", "message": f"Lade Modell: {model_name}"})
            
            # Modell laden
            model_path = os.path.join(app.config['MODEL_DIR'], model_name)
            if not os.path.exists(model_path):
                # Modell herunterladen
                model_path = model_name  # Ultralytics lädt automatisch
            
            model = YOLO(model_path)
            
            training_job["logs"].append({"type": "info", "message": f"Starte Training: {epochs} Epochen, Batch={batch}, imgsz={imgsz}"})
            
            # Callback für Training-Fortschritt
            def on_epoch(epoch_info):
                epoch = epoch_info.get('epoch', 0)
                metrics = epoch_info.get('metrics', {})
                training_job["logs"].append({
                    "type": "info",
                    "message": f"Epoch {epoch}/{epochs}: mAP50={metrics.get('metrics/mAP50', 0):.4f}, Loss={metrics.get('train/box_loss', 0):.4f}"
                })
            
            # Training starten
            results = model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                project=app.config['UPLOAD_FOLDER_ABS'],
                name=f"training_{int(time.time())}",
                exist_ok=True,
                verbose=False,
                callbacks={"on_epoch": on_epoch}
            )
            
            # Ergebnisse speichern
            training_job["results"] = {
                "map50": float(results.results_dict.get('metrics/mAP50', 0)),
                "map95": float(results.results_dict.get('metrics/mAP50-95', 0)),
                "epochs": epochs,
                "best_epoch": int(results.results_dict.get('epoch', 0)),
                "training_time": time.time() - start_time,
                "model_path": str(results.save_dir / 'best.pt')
            }
            
            training_job["logs"].append({"type": "success", "message": f"Training abgeschlossen! mAP50: {training_job['results']['map50']:.4f}"})
            
        except Exception as e:
            training_job["logs"].append({"type": "error", "message": str(e)})
        finally:
            training_job["active"] = False
    
    # Thread starten
    import threading
    training_job["active"] = True
    training_job["process"] = threading.Thread(target=run_training)
    training_job["process"].start()
    
    return jsonify({
        "status": "Training gestartet",
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "model": model_name
    })

@app.route('/training/stop', methods=['POST'])
def training_stop():
    """Stoppt das aktuelle Training"""
    global training_job
    
    if not training_job["active"]:
        return jsonify({"error": "Kein Training aktiv"}), 400
    
    # Training kann nicht sauber gestoppt werden (Ultralytics Limitation)
    training_job["active"] = False
    training_job["logs"].append({"type": "warning", "message": "Training gestoppt"})
    
    return jsonify({"status": "Training gestoppt"})

@app.route('/training/download', methods=['GET'])
def training_download():
    """Lädt das trainierte Modell herunter"""
    if not training_job["results"]:
        return jsonify({"error": "Kein Training abgeschlossen"}), 404
    
    model_path = training_job["results"].get("model_path")
    if not model_path or not os.path.exists(model_path):
        return jsonify({"error": "Modell nicht gefunden"}), 404
    
    return send_file(model_path, as_attachment=True, download_name='trained_model.pt')

# ==================== FEW-SHOT AUTO-DETECTION ENDPOINTS ====================

@app.route('/dataset/fewshot/status', methods=['GET'])
def fewshot_status():
    """Gibt den Status des Few-Shot Detectors zurück"""
    return jsonify({
        "samples_count": len(few_shot_detector.samples),
        "classes": list(few_shot_detector.class_templates.keys()),
        "clip_loaded": few_shot_detector.clip_model is not None,
        "ready": len(few_shot_detector.class_templates) > 0 and few_shot_detector.clip_model is not None
    })

@app.route('/dataset/fewshot/add_sample', methods=['POST'])
def fewshot_add_sample():
    """Fügt ein Sample-Bild mit Annotationen zum Few-Shot Detector hinzu"""
    if 'image' not in request.files:
        return jsonify({"error": "Kein Bild"}), 400
    
    annotations_json = request.form.get('annotations', '[]')
    try:
        annotations = json.loads(annotations_json)
    except json.JSONDecodeError:
        return jsonify({"error": "Ungültige Annotationen"}), 400
    
    if not annotations:
        return jsonify({"error": "Mindestens eine Annotation erforderlich"}), 400
    
    file = request.files['image']
    img_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], f"sample_{int(time.time()*1000)}_{file.filename}")
    file.save(img_path)
    
    try:
        # Versuche CLIP zu laden falls noch nicht geladen
        if few_shot_detector.clip_model is None:
            clip_loaded = few_shot_detector.load_clip()
            if not clip_loaded:
                # Fallback: Verwende einfache Template-Matching Methode
                pass
        
        sample_count = few_shot_detector.add_sample(img_path, annotations)
        
        return jsonify({
            "success": True,
            "sample_count": sample_count,
            "classes": list(few_shot_detector.class_templates.keys()),
            "message": f"Sample {sample_count} hinzugefügt ({len(annotations)} Annotationen)"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Behalte Sample-Bilder für spätere Verwendung
        pass

@app.route('/dataset/fewshot/clear', methods=['POST'])
def fewshot_clear():
    """Löscht alle Few-Shot Samples"""
    few_shot_detector.clear_samples()
    return jsonify({
        "success": True,
        "message": "Alle Samples gelöscht"
    })

@app.route('/dataset/fewshot/auto_detect', methods=['POST'])
def fewshot_auto_detect():
    """Führt Auto-Detection auf mehreren Bildern durch"""
    if 'images' not in request.files:
        return jsonify({"error": "Keine Bilder"}), 400
    
    if not few_shot_detector.class_templates:
        return jsonify({"error": "Keine Samples geladen. Bitte zuerst 2-3 Beispielbilder annotieren."}), 400
    
    try:
        threshold = float(request.form.get('threshold', 0.7))
    except ValueError:
        threshold = 0.7
    
    files = request.files.getlist('images')
    if not files:
        return jsonify({"error": "Keine Bilder ausgewählt"}), 400
    
    # Speichere temporär die Bilder
    temp_paths = []
    for file in files:
        if file.filename:
            img_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], f"autodetect_{int(time.time()*1000)}_{file.filename}")
            file.save(img_path)
            temp_paths.append(img_path)
    
    try:
        # Führe Auto-Detection durch
        results = []
        for i, img_path in enumerate(temp_paths):
            detections = few_shot_detector.detect_similar(img_path, threshold)
            
            # Annotiertes Bild erstellen
            image_cv = cv2.imread(img_path)
            image_h, image_w = image_cv.shape[:2]
            
            for det in detections:
                bbox_norm = det["bbox_normalized"]
                x1 = int(bbox_norm[0] * image_w)
                y1 = int(bbox_norm[1] * image_h)
                x2 = int(bbox_norm[2] * image_w)
                y2 = int(bbox_norm[3] * image_h)
                
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_cv, f"{det['class']} {det['confidence']:.2f}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            annotated_filename = f"autodetect_annotated_{int(time.time()*1000)}_{os.path.basename(img_path)}"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER_ABS'], annotated_filename)
            cv2.imwrite(annotated_path, image_cv)
            
            results.append({
                "filename": os.path.basename(img_path),
                "annotations": detections,
                "annotation_count": len(detections),
                "annotated_image": f"/uploads/{annotated_filename}"
            })
        
        total_annotations = sum(r["annotation_count"] for r in results)
        
        return jsonify({
            "success": True,
            "processed_count": len(results),
            "results": results,
            "total_annotations": total_annotations,
            "classes_detected": list(set(det["class"] for r in results for det in r["annotations"]))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Aufräumen der temporären Bilder
        for img_path in temp_paths:
            if os.path.exists(img_path):
                try: os.remove(img_path)
                except: pass

@app.route('/dataset/fewshot/load_clip', methods=['POST'])
def fewshot_load_clip():
    """Lädt das CLIP Modell manuell"""
    try:
        loaded = few_shot_detector.load_clip()
        if loaded:
            return jsonify({
                "success": True,
                "message": "CLIP Modell erfolgreich geladen",
                "device": "cuda" if torch.cuda.is_available() else "cpu"
            })
        else:
            return jsonify({
                "success": False,
                "message": "CLIP Modell nicht verfügbar. Installiere: pip install clip-by-openai"
            }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True, use_reloader=False)

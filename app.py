import os, gc, time, shutil, json, io, base64, re, subprocess
from pathlib import Path
from datetime import datetime
from collections import deque
import torch, cv2, mimetypes
from flask import Flask, render_template, request, jsonify, send_from_directory, Response, stream_with_context
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

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)

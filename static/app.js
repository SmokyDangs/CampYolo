const navButtons = Array.from(document.querySelectorAll('nav button'));
const sections = Array.from(document.querySelectorAll('.tab-section'));
const dropzone = document.getElementById('dropzone');
const batchDropzone = document.getElementById('batchDropzone');
const imageInput = document.getElementById('imageInput');
const videoInput = document.getElementById('videoInput');
const batchInput = document.getElementById('batchInput');
const previewBox = document.getElementById('previewBox');
const previewName = document.getElementById('previewName');
const resultLink = document.getElementById('resultLink');
const testStatusPill = document.getElementById('testStatusPill');
const testStatusLabel = document.getElementById('testStatusLabel');
const elapsedLabel = document.createElement('span');
elapsedLabel.style.marginLeft = '6px';
elapsedLabel.className = 'muted';
testStatusPill?.appendChild(elapsedLabel);
const runBtn = document.getElementById('runBtn');
const modeButtons = {
    image: document.getElementById('modeImage'),
    video: document.getElementById('modeVideo'),
    url: document.getElementById('modeUrl'),
    webcam: document.getElementById('modeWebcam'),
    batch: document.getElementById('modeBatch')
};
let currentMode = 'image';
let statusTimer = null;
let statusStart = null;
const logEntries = document.getElementById('logEntries');
const progressWrap = document.getElementById('progressWrap');
const progressInner = document.getElementById('progressInner');
const progressText = document.getElementById('progressText');
const progressEta = document.getElementById('progressEta');
let progressTimer = null;
let progressStart = null;
let filePickerOpen = false;

// Input-Gruppen
const inputGroups = {
    image: document.getElementById('imageInputGroup'),
    video: document.getElementById('imageInputGroup'),
    url: document.getElementById('urlInputGroup'),
    webcam: document.getElementById('webcamInputGroup'),
    batch: document.getElementById('batchInputGroup')
};

navButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        switchSection(btn.dataset.target);
    });
});

function switchSection(targetId) {
    sections.forEach(sec => sec.classList.toggle('active', sec.id === targetId));
    navButtons.forEach(b => b.classList.toggle('active', b.dataset.target === targetId));
    document.getElementById(targetId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function updateModelList() {
    const res = await fetch('/models');
    const data = await res.json();
    const list = document.getElementById('modelList');
    const models = data.available_models;
    const active = data.active_model;

    if (active) {
        document.getElementById('selectedModel').value = active;
        document.getElementById('activeModelLabel').textContent = active;
        document.getElementById('activeModelPill').style.display = 'inline-flex';
    } else {
        document.getElementById('activeModelPill').style.display = 'none';
    }

    if (!models.length) {
        list.innerHTML = `<li class="empty-state">Keine Modelle gefunden. Lege .pt-Dateien im Ordner "models" ab.</li>`;
        return;
    }

    list.innerHTML = models.map((m, idx) =>
        `<li class="model-card">
            <h4>${m}</h4>
            <small>Datei #${idx + 1}</small>
            ${active === m ? '<span class="chip">Aktiv</span>' : ''}
            <div class="inline-actions">
                <button class="action" onclick="loadModel('${m}')">${active === m ? 'Neu laden' : 'Laden'}</button>
                <button onclick="prefillModel('${m}')">Auswählen</button>
            </div>
        </li>`
    ).join('');
}

async function loadModel(name) {
    const res = await fetch(`/load/${name}`, { method: 'POST' });
    const data = await res.json();
    showToast(data.status || data.error, res.ok ? 'success' : 'error');
    document.getElementById('selectedModel').value = name;
    await updateModelList();
}

function prefillModel(name) {
    document.getElementById('selectedModel').value = name;
    switchSection('testing');
}

async function runInference() {
    const modelName = document.getElementById('selectedModel').value;
    
    if (!modelName) {
        showToast('Bitte ein Modell auswählen oder eintippen.', 'error');
        return;
    }

    // Parameter sammeln
    const conf = document.getElementById('confThreshold').value;
    const iou = document.getElementById('iouThreshold').value;
    const imgsz = document.getElementById('imgSize').value;
    const save = document.getElementById('saveResults').checked;
    const showLabels = document.getElementById('showLabels').checked;

    setStatus('Läuft …', true);
    runBtn.disabled = true;
    startProgress();
    logMsg(`Starte ${currentMode}-Inferenz mit ${modelName}`, 'info');

    try {
        if (currentMode === 'image') {
            await runImageInference(modelName, conf, iou, imgsz, save, showLabels);
        } else if (currentMode === 'video') {
            await runVideoInference(modelName);
        } else if (currentMode === 'url') {
            await runUrlInference(modelName, conf, iou, imgsz, save, showLabels);
        } else if (currentMode === 'webcam') {
            await runWebcamInference(modelName, conf, imgsz);
        } else if (currentMode === 'batch') {
            await runBatchInference(modelName, conf, iou, imgsz);
        }
    } catch (err) {
        stopProgress(false);
        setStatus('Fehler');
        logMsg(`Fehler: ${err.message}`, 'error');
        showToast(err.message, 'error');
        runBtn.disabled = false;
    }
}

async function runImageInference(modelName, conf, iou, imgsz, save, showLabels) {
    if (!imageInput.files.length) {
        throw new Error('Bitte ein Bild auswählen.');
    }

    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('source_type', 'image');
    formData.append('image', imageInput.files[0]);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    formData.append('save', save ? 'true' : 'false');
    formData.append('show_labels', showLabels ? 'true' : 'false');

    const res = await fetch('/test', { method: 'POST', body: formData });
    const data = await res.json();
    
    if (!res.ok) {
        throw new Error(data.error || 'Inferenz fehlgeschlagen');
    }
    
    handleResult(res.ok, data);
}

async function runVideoInference(modelName) {
    if (!videoInput.files.length) {
        throw new Error('Bitte ein Video auswählen.');
    }

    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('video', videoInput.files[0]);

    previewBox.innerHTML = '<span class="muted">Video wird verarbeitet …</span>';
    previewName.textContent = 'Verarbeitung läuft';

    await streamVideoInference(formData);
}

async function runUrlInference(modelName, conf, iou, imgsz, save, showLabels) {
    const urlValue = document.getElementById('urlInput').value;
    if (!urlValue) {
        throw new Error('Bitte eine URL eingeben.');
    }

    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('source_type', 'url');
    formData.append('source_value', urlValue);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    formData.append('save', save ? 'true' : 'false');
    formData.append('show_labels', showLabels ? 'true' : 'false');

    const res = await fetch('/test', { method: 'POST', body: formData });
    const data = await res.json();
    
    if (!res.ok) {
        throw new Error(data.error || 'Inferenz fehlgeschlagen');
    }
    
    handleResult(res.ok, data);
}

async function runWebcamInference(modelName, conf, imgsz) {
    const camId = document.getElementById('webcamIdInput').value;
    
    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('cam_id', camId);
    formData.append('conf', conf);
    formData.append('imgsz', imgsz);

    // Zeige Webcam-Stream im Preview
    previewBox.innerHTML = '<div style="display:grid;place-items:center;height:100%;"><span class="muted">Webcam wird initialisiert...</span></div>';
    previewName.textContent = `Webcam ${camId}`;
    
    // Stream als Bild anzeigen
    const streamUrl = `/test_webcam?${new URLSearchParams({ cam_id: camId, conf, imgsz })}`;
    previewBox.innerHTML = `<img src="/test_webcam" style="width:100%;height:100%;object-fit:contain;" alt="Webcam Stream">`;
    
    logMsg(`Webcam-Stream gestartet (ID: ${camId})`, 'success');
    stopProgress(true);
    setStatus('Webcam aktiv');
    runBtn.disabled = false;
}

async function runBatchInference(modelName, conf, iou, imgsz) {
    if (!batchInput.files.length) {
        throw new Error('Bitte mindestens ein Bild auswählen.');
    }

    const formData = new FormData();
    formData.append('model_name', modelName);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    
    for (let i = 0; i < batchInput.files.length; i++) {
        formData.append('images', batchInput.files[i]);
    }

    const res = await fetch('/test_batch', { method: 'POST', body: formData });
    const data = await res.json();
    
    if (!res.ok) {
        throw new Error(data.error || 'Batch-Inferenz fehlgeschlagen');
    }
    
    handleBatchResult(res.ok, data);
}

// Initialisieren
updateModelList();
switchSection('hub');

// --- Drag & Drop / Preview ---
if (dropzone && imageInput) {
    dropzone.addEventListener('click', e => {
        if (e.target.closest('button')) return;
        triggerFile(e);
    });

    ['dragenter', 'dragover'].forEach(evt =>
        dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.add('dragover'); })
    );
    ['dragleave', 'drop'].forEach(evt =>
        dropzone.addEventListener(evt, e => { e.preventDefault(); dropzone.classList.remove('dragover'); })
    );
    dropzone.addEventListener('drop', e => {
        if (e.dataTransfer.files.length) {
            const dt = new DataTransfer();
            Array.from(e.dataTransfer.files).forEach(f => dt.items.add(f));
            const targetInput = currentMode === 'image' ? imageInput : videoInput;
            targetInput.files = dt.files;
            previewFile(targetInput.files[0]);
        }
    });

    imageInput.addEventListener('change', e => {
        if (e.target.files.length) previewFile(e.target.files[0]);
    });
    videoInput.addEventListener('change', e => {
        if (e.target.files.length) previewFile(e.target.files[0]);
    });
}

// Batch Dropzone
if (batchDropzone && batchInput) {
    batchDropzone.addEventListener('click', e => {
        if (e.target.closest('button')) return;
        batchInput.click();
    });

    ['dragenter', 'dragover'].forEach(evt =>
        batchDropzone.addEventListener(evt, e => { e.preventDefault(); batchDropzone.classList.add('dragover'); })
    );
    ['dragleave', 'drop'].forEach(evt =>
        batchDropzone.addEventListener(evt, e => { e.preventDefault(); batchDropzone.classList.remove('dragover'); })
    );
    batchDropzone.addEventListener('drop', e => {
        if (e.dataTransfer.files.length) {
            const dt = new DataTransfer();
            Array.from(e.dataTransfer.files).forEach(f => dt.items.add(f));
            batchInput.files = dt.files;
            const count = batchInput.files.length;
            previewName.textContent = `${count} Datei(en)`;
            previewBox.innerHTML = `<div style="display:grid;place-items:center;height:100%;"><span class="muted">${count} Bilder ausgewählt</span></div>`;
            logMsg(`${count} Bilder für Batch-Inferenz ausgewählt`, 'info');
        }
    });

    batchInput.addEventListener('change', e => {
        if (e.target.files.length) {
            const count = e.target.files.length;
            previewName.textContent = `${count} Datei(en)`;
            previewBox.innerHTML = `<div style="display:grid;place-items:center;height:100%;"><span class="muted">${count} Bilder ausgewählt</span></div>`;
            logMsg(`${count} Bilder für Batch-Inferenz ausgewählt`, 'info');
        }
    });
}

function previewFile(file) {
    previewName.textContent = file.name;
    const url = URL.createObjectURL(file);
    if (file.type.startsWith('video')) {
        previewBox.innerHTML = `<video controls src="${url}" style="width:100%; height:100%; object-fit:contain;"></video>`;
        if (resultLink) resultLink.style.display = 'none';
    } else {
        previewBox.innerHTML = `<img src="${url}" alt="preview" style="width:100%;height:100%;object-fit:contain;">`;
        if (resultLink) resultLink.style.display = 'none';
    }
}

function resetTest() {
    imageInput.value = '';
    videoInput.value = '';
    batchInput.value = '';
    document.getElementById('urlInput').value = '';
    document.getElementById('webcamIdInput').value = '0';
    previewName.textContent = 'Keine Datei';
    previewBox.innerHTML = '<span class="muted">Vorschau erscheint hier</span>';
    if (resultLink) resultLink.style.display = 'none';
    document.getElementById('output').textContent = '';
    setStatus();
    logMsg('Zurückgesetzt', 'info');
    stopProgress(false);
}

function setStatus(text = null, startTimer = false) {
    if (!text) {
        testStatusPill.style.display = 'none';
        stopStatusTimer();
        return;
    }
    testStatusPill.style.display = 'inline-flex';
    testStatusLabel.textContent = text;
    if (startTimer) startStatusTimer();
}

function setMode(mode) {
    currentMode = mode;
    Object.entries(modeButtons).forEach(([key, btn]) => {
        btn.classList.toggle('active', key === mode);
    });
    
    // Input-Gruppen umschalten
    Object.entries(inputGroups).forEach(([key, el]) => {
        if (el) el.style.display = (key === mode || (mode === 'video' && key === 'image')) ? 'block' : 'none';
    });
    
    // Dropzone-Prompt aktualisieren
    const prompt = mode === 'image' ? 'Bild hier ablegen oder klicken zum Auswählen' :
                   mode === 'video' ? 'Video hier ablegen oder klicken zum Auswählen' :
                   'Datei auswählen';
    document.getElementById('dropPrompt').textContent = prompt;
    
    // Preview zurücksetzen
    previewBox.innerHTML = '<span class="muted">Vorschau erscheint hier</span>';
    previewName.textContent = 'Keine Datei';
}

function triggerFile(e) {
    if (e) e.stopPropagation();
    if (filePickerOpen) return;
    const input = (currentMode === 'image' ? imageInput : videoInput);
    if (!input) return;
    filePickerOpen = true;
    const resetFlag = () => { filePickerOpen = false; input.removeEventListener('change', resetFlag); };
    input.addEventListener('change', resetFlag);
    // Fallback falls kein change/event (Abbrechen) kommt
    setTimeout(() => { filePickerOpen = false; }, 2000);
    input.click();
}

// --- Toasts ---
const toastStack = document.getElementById('toastStack');

function showToast(message, type = 'success') {
    if (!toastStack) return alert(message);
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    toastStack.appendChild(el);
    setTimeout(() => el.remove(), 4000);
}

// --- Status Timer & Logs ---
function startStatusTimer() {
    stopStatusTimer();
    statusStart = Date.now();
    elapsedLabel.textContent = '(0s)';
    statusTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - statusStart) / 1000);
        elapsedLabel.textContent = `(${elapsed}s)`;
    }, 1000);
}

function stopStatusTimer() {
    if (statusTimer) {
        clearInterval(statusTimer);
        statusTimer = null;
    }
    const elapsed = statusStart ? Math.floor((Date.now() - statusStart) / 1000) : 0;
    statusStart = null;
    elapsedLabel.textContent = '';
    return elapsed;
}

function logMsg(msg, type = 'info') {
    if (!logEntries) return;
    const line = document.createElement('div');
    const ts = new Date().toLocaleTimeString();
    line.className = type;
    line.textContent = `[${ts}] ${msg}`;
    logEntries.appendChild(line);
    logEntries.parentElement.scrollTop = logEntries.parentElement.scrollHeight;
}

// --- Streaming Video Inference ---
async function streamVideoInference(formData) {
    try {
        const res = await fetch('/test_video_stream', { method: 'POST', body: formData });
        if (!res.ok || !res.body) {
            const data = await res.json().catch(() => ({}));
            throw new Error(data.error || 'Stream konnte nicht gestartet werden');
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            let boundary;
            while ((boundary = buffer.indexOf('\n\n')) >= 0) {
                const raw = buffer.slice(0, boundary).trim();
                buffer = buffer.slice(boundary + 2);
                if (!raw.startsWith('data:')) continue;
                const jsonStr = raw.replace(/^data:\s*/, '');
                let payload;
                try { payload = JSON.parse(jsonStr); } catch { continue; }
                handleStreamPayload(payload);
            }
        }
    } catch (err) {
        const elapsed = stopStatusTimer();
        stopProgress(false);
        setStatus('Fehler');
        logMsg(`Fehler nach ${elapsed}s: ${err.message}`, 'error');
        showToast(err.message, 'error');
        runBtn.disabled = false;
    }
}

function handleStreamPayload(p) {
    if (p.error) {
        const elapsed = stopStatusTimer();
        stopProgress(false);
        setStatus('Fehler');
        logMsg(`Fehler nach ${elapsed}s: ${p.error}`, 'error');
        showToast(p.error, 'error');
        runBtn.disabled = false;
        return;
    }
    if (p.progress !== undefined) {
        const eta = computeEta(p);
        setProgress(p.progress ?? 0, eta);
        logMsg(p.message || `Frame ${p.frame}`, 'info');
        return;
    }
    if (p.done) {
        const elapsed = stopStatusTimer();
        stopProgress(true);
        setStatus(`Fertig (${elapsed}s)`);
        logMsg(`Video fertig in ${elapsed}s`, 'success');
        document.getElementById('output').textContent = JSON.stringify(p, null, 2);
        if (p.video_url) setAnnotatedVideo(p.video_url);
        runBtn.disabled = false;
    }
}

function handleResult(ok, data) {
    if (ok) {
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        if (data.image_url) {
            previewBox.innerHTML = `<img src="${data.image_url}?t=${Date.now()}" alt="annotated result" style="width:100%;height:100%;object-fit:contain;">`;
            previewName.textContent = 'Ergebnis (annotiert)';
        }
        if (data.video_url) setAnnotatedVideo(data.video_url);
        const elapsed = stopStatusTimer();
        stopProgress(true);
        setStatus(`Fertig (${elapsed}s)`);
        logMsg(`Fertig in ${elapsed}s`, 'success');
        
        // Detaillierte Ergebnisse loggen
        if (data.detections && data.detections.length > 0) {
            logMsg(`${data.detections.length} Objekt(e) erkannt`, 'success');
            data.detections.forEach(d => {
                logMsg(`  • ${d.class} (${(d.confidence * 100).toFixed(1)}%)`, 'info');
            });
        }
        if (data.results && data.results[0]?.keypoints) {
            logMsg('Keypoints erkannt (Pose-Modell)', 'success');
        }
    } else {
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        const elapsed = stopStatusTimer();
        stopProgress(false);
        setStatus('Fehler');
        logMsg(`Fehler nach ${elapsed}s: ${data.error || 'Unbekannt'}`, 'error');
    }
    runBtn.disabled = false;
}

function handleBatchResult(ok, data) {
    if (ok) {
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        const elapsed = stopStatusTimer();
        stopProgress(true);
        setStatus(`Fertig (${elapsed}s)`);
        logMsg(`Batch-Inferenz: ${data.processed_count} Bilder verarbeitet`, 'success');
        
        if (data.annotated_images && data.annotated_images.length > 0) {
            const firstImg = data.annotated_images[0];
            previewBox.innerHTML = `<img src="${firstImg}?t=${Date.now()}" alt="batch result" style="width:100%;height:100%;object-fit:contain;">`;
            previewName.textContent = `Batch-Ergebnis (${data.processed_count} Bilder)`;
        }
        runBtn.disabled = false;
    } else {
        document.getElementById('output').textContent = JSON.stringify(data, null, 2);
        const elapsed = stopStatusTimer();
        stopProgress(false);
        setStatus('Fehler');
        logMsg(`Fehler nach ${elapsed}s: ${data.error || 'Unbekannt'}`, 'error');
        runBtn.disabled = false;
    }
}

function setAnnotatedVideo(url) {
    const videoUrl = `${url}?t=${Date.now()}`;
    previewBox.innerHTML = `
        <div style="display:grid;place-items:center;height:100%;gap:12px;">
            <video controls autoplay style="width:100%; height:100%; object-fit:contain;" 
                   onerror="this.parentElement.innerHTML='<span class=\\'error\\'>Video konnte nicht geladen werden</span>'">
                <source src="${videoUrl}" type="video/mp4">
                Dein Browser unterstützt das Video-Tag nicht.
            </video>
        </div>
    `;
    previewName.textContent = 'Ergebnis-Video (annotiert)';
    if (resultLink) {
        resultLink.href = videoUrl;
        resultLink.textContent = 'Ergebnis öffnen';
        resultLink.style.display = 'inline';
    }
    logMsg(`Annotiertes Video: ${url}`, 'success');
    
    // Video-Element finden und laden
    const video = previewBox.querySelector('video');
    if (video) {
        video.load();
        video.onloadeddata = () => {
            logMsg('Video erfolgreich geladen', 'success');
        };
        video.onerror = () => {
            logMsg('Fehler beim Laden des Videos', 'error');
        };
    }
}

// --- Progress bar ---
function startProgress() {
    if (!progressWrap) return;
    stopProgressTimer();
    progressWrap.style.display = 'grid';
    progressStart = Date.now();
    setProgress(5, '—');
    progressTimer = setInterval(() => {
        const elapsed = (Date.now() - progressStart) / 1000;
        const pct = Math.min(90, 5 + (elapsed / 30) * 85);
        setProgress(Math.round(pct), `${Math.max(1, Math.round(30 - elapsed))}s`);
    }, 700);
}

function stopProgress(success) {
    stopProgressTimer();
    if (!progressWrap) return;
    if (success) {
        setProgress(100, '0s');
        setTimeout(() => progressWrap.style.display = 'none', 800);
    } else {
        progressWrap.style.display = 'none';
    }
    progressStart = null;
}

function stopProgressTimer() {
    if (progressTimer) clearInterval(progressTimer);
    progressTimer = null;
}

function setProgress(pct, etaText) {
    if (!progressInner) return;
    progressInner.style.width = `${pct}%`;
    if (progressText) progressText.textContent = `${pct}%`;
    if (progressEta) progressEta.textContent = etaText;
}

function computeEta(p) {
    if (!p.total_frames || !p.frame || !progressStart) return '…';
    const elapsed = (Date.now() - progressStart) / 1000;
    const fps = p.frame / Math.max(elapsed, 0.001);
    const remaining = Math.max(0, (p.total_frames - p.frame) / fps);
    return `${Math.round(remaining)}s`;
}

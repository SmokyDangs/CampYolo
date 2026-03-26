// YOLO Vision - Kompakte Client-Implementation
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

const state = {
    mode: 'image',
    activeModel: null,
    processing: false,
    stats: { totalInferences: 0, totalDetections: 0, avgTime: 0 },
    settings: {
        conf: 0.25,
        iou: 0.45,
        imgsz: 640,
        saveResults: true,
        showLabels: true,
        theme: 'dark'
    }
};

const els = {};

document.addEventListener('DOMContentLoaded', () => {
    Object.assign(els, {
        navBtns: $$('nav button'),
        sections: $$('.tab-section'),
        dropzone: $('dropzone'),
        batchDropzone: $('batchDropzone'),
        imageInput: $('imageInput'),
        videoInput: $('videoInput'),
        batchInput: $('batchInput'),
        previewBox: $('previewBox'),
        previewName: $('previewName'),
        resultLink: $('resultLink'),
        testStatusPill: $('testStatusPill'),
        testStatusLabel: $('testStatusLabel'),
        runBtn: $('runBtn'),
        logEntries: $('logEntries'),
        progressWrap: $('progressWrap'),
        progressInner: $('progressInner'),
        progressText: $('progressText'),
        progressEta: $('progressEta'),
        output: $('output'),
        selectedModel: $('selectedModel'),
        urlInput: $('urlInput'),
        streamUrlInput: $('streamUrlInput'),
        webcamIdInput: $('webcamIdInput'),
        directoryInput: $('directoryInput'),
        directoryDropPrompt: $('directoryDropPrompt'),
        localPathInput: $('localPathInput'),
        confThreshold: $('confThreshold'),
        iouThreshold: $('iouThreshold'),
        imgSize: $('imgSize'),
        saveResults: $('saveResults'),
        showLabels: $('showLabels'),
        detectionResults: $('detectionResults'),
        statsTotal: $('statsTotal'),
        statsDetections: $('statsDetections'),
        statsTime: $('statsTime')
    });

    setupNav();
    setupModeTabs();
    setupDropzones();
    setupInputs();
    setupKeyboard();
    updateModelList();
    loadStats();
    setupParamLabels();
    setupDatasetTab();
    setupFewShotDetection();
    loadSettings();
    setupSettingsAutoSave();
});

function setupNav() {
    els.navBtns.forEach(btn => {
        btn.addEventListener('click', () => switchSection(btn.dataset.target));
    });
}

function setupModeTabs() {
    $$('.source-tab').forEach(btn => {
        btn.addEventListener('click', function() { setMode(this.dataset.mode); });
    });
}

function setupInputs() {
    els.imageInput?.addEventListener('change', handleFileSelect);
    els.videoInput?.addEventListener('change', handleFileSelect);
    els.batchInput?.addEventListener('change', handleBatchSelect);
    els.directoryInput?.addEventListener('change', handleDirectorySelect);
    
    ['confThreshold', 'iouThreshold', 'imgSize'].forEach(id => {
        $(id)?.addEventListener('input', setupParamLabels);
    });
}

function setupParamLabels() {
    $('confValue').textContent = els.confThreshold?.value || '0.25';
    $('iouValue').textContent = els.iouThreshold?.value || '0.45';
    $('sizeValue').textContent = els.imgSize?.value || '640';
    $('compareConfValue').textContent = $('compareConf')?.value || '0.25';
    $('compareIouValue').textContent = $('compareIou')?.value || '0.45';
    $('compareSizeValue').textContent = $('compareSize')?.value || '640';
}

function setupDropzones() {
    setupDropzone(els.dropzone, () => state.mode === 'video' ? els.videoInput : els.imageInput);
    setupDropzone(els.batchDropzone, () => els.batchInput, true);
    setupDropzone($('directoryDropzone'), () => els.directoryInput);
    setupLocalPathDropzone();
}

function setupLocalPathDropzone() {
    // Drag & Drop für lokale Pfade aus Dateiexplorer
    const localPathGroup = $('localPathInputGroup');
    if (!localPathGroup) return;
    
    // Globales Drag & Drop für alle Dateien
    document.addEventListener('dragover', (e) => {
        e.preventDefault();
        // Visuelles Feedback bei Drag über lokalen Pfad-Bereich
        if (state.mode === 'local_path') {
            localPathGroup.style.borderColor = 'var(--accent-primary)';
            localPathGroup.style.background = 'var(--accent-gradient-subtle)';
        }
    });
    
    document.addEventListener('dragleave', (e) => {
        if (e.target === localPathGroup || !localPathGroup.contains(e.target)) {
            localPathGroup.style.borderColor = '';
            localPathGroup.style.background = '';
        }
    });
    
    document.addEventListener('drop', (e) => {
        e.preventDefault();
        const input = els.localPathInput;
        if (!input) return;
        
        // Dateien aus Explorer verarbeiten
        if (e.dataTransfer?.files?.length > 0) {
            const file = e.dataTransfer.files[0];
            // Hinweis: Browser können aus Sicherheitsgründen keinen vollständigen Pfad lesen
            // User muss den Pfad manuell eingeben oder Copy-Paste verwenden
            showToast('Datei erkannt. Bitte Pfad manuell eingeben oder kopieren.', 'info');
        }
        
        // Text-Pfad aus Clipboard (Strg+V im Dokument)
        const text = e.dataTransfer?.getData('text');
        if (text && isValidPath(text)) {
            input.value = text.trim();
            showToast(`Pfad eingefügt: ${text.trim()}`, 'success');
        }
    });
}

function isValidPath(text) {
    // Prüfen ob Text ein gültiger Pfad sein könnte
    const trimmed = text.trim();
    // Windows: C:\..., Linux/Mac: /...
    return /^[A-Z]:\\/.test(trimmed) || /^\//.test(trimmed) || /^\\\\/.test(trimmed);
}

function setupDropzone(dz, getInput, isBatch = false) {
    if (!dz) return;
    
    // Dropzone-Klick triggert das versteckte Input
    dz.addEventListener('click', (e) => {
        // Wenn der Klick bereits vom Input kam (Bubbling), nichts tun
        if (e.target.tagName === 'INPUT') return;
        
        const input = getInput();
        if (input) {
            input.click();
        }
    });

    // Bubbling vom Input unterbinden, um Rekursion zu vermeiden
    const inputs = dz.querySelectorAll('input[type="file"]');
    inputs.forEach(input => {
        input.addEventListener('click', (e) => e.stopPropagation());
    });
    
    ['dragenter', 'dragover'].forEach(evt => {
        dz.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.add('dragover');
        });
    });
    
    ['dragleave', 'drop'].forEach(evt => {
        dz.addEventListener(evt, e => {
            e.preventDefault();
            e.stopPropagation();
            dz.classList.remove('dragover');
        });
    });
    
    dz.addEventListener('drop', e => {
        e.preventDefault();
        e.stopPropagation();
        const files = Array.from(e.dataTransfer.files);
        if (!files.length) return;
        
        const input = getInput();
        if (!input) return;
        
        const dt = new DataTransfer();
        files.forEach(f => dt.items.add(f));
        input.files = dt.files;
        
        if (isBatch) handleBatchSelect({ target: input });
        else handleFileSelect({ target: input });
    });
}

function switchSection(targetId) {
    els.sections.forEach(sec => sec.classList.toggle('active', sec.id === targetId));
    els.navBtns.forEach(btn => btn.classList.toggle('active', btn.dataset.target === targetId));
    $(targetId)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    if (targetId === 'compare') initCompare();
}

function setMode(mode) {
    state.mode = mode;
    
    $$('.source-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });
    
    $$('.input-group').forEach(el => el.style.display = 'none');
    
    const modeMap = {
        image: 'imageInputGroup',
        video: 'imageInputGroup',
        url: 'urlInputGroup',
        webcam: 'webcamInputGroup',
        batch: 'batchInputGroup',
        directory: 'directoryInputGroup',
        local_path: 'localPathInputGroup',
        stream: 'streamInputGroup'
    };
    
    const group = $(modeMap[mode]);
    if (group) group.style.display = 'block';
    
    const prompt = $('dropPrompt');
    const prompts = {
        image: 'Bild ablegen oder klicken',
        video: 'Video ablegen oder klicken',
        batch: 'Mehrere Bilder ablegen'
    };
    if (prompt) prompt.textContent = prompts[mode] || 'Auswählen';
    
    if (els.imageInput) {
        els.imageInput.accept = mode === 'video' ? 'video/*' : 'image/*';
    }
    
    resetPreview();
    log(`Modus: ${mode}`, 'info');
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    els.previewName.textContent = file.name;
    const url = URL.createObjectURL(file);
    const isVideo = file.type.startsWith('video/') || file.name.match(/\.(mp4|avi|mov|mkv|webm)$/i);
    
    els.previewBox.innerHTML = isVideo
        ? `<video controls src="${url}" class="preview-media"></video>`
        : `<img src="${url}" alt="preview" class="preview-media">`;
    
    els.resultLink.style.display = 'none';
    log(`${isVideo ? 'Video' : 'Bild'}: ${file.name}`, 'success');
}

function handleBatchSelect(e) {
    const count = e.target.files.length;
    els.previewName.textContent = `${count} Datei(en)`;
    els.previewBox.innerHTML = `
        <div class="preview-folder">
            <div>
                <div class="preview-folder-icon">📁</div>
                <div class="preview-folder-count">${count} Bilder</div>
            </div>
        </div>`;
    log(`${count} Bilder für Batch`, 'info');
}

function handleDirectorySelect(e) {
    const files = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
    const count = files.length;
    els.directoryDropPrompt.textContent = `${count} Bild(er) in Ordner`;
    els.previewName.textContent = `Ordner: ${count} Bilder`;
    els.previewBox.innerHTML = `
        <div class="preview-folder">
            <div>
                <div class="preview-folder-icon">📁</div>
                <div class="preview-folder-count">${count} Bilder</div>
            </div>
        </div>`;
    log(`${count} Bilder aus Verzeichnis gewählt`, 'info');
}

function resetPreview() {
    els.previewName.textContent = 'Keine Datei';
    els.previewBox.innerHTML = `
        <div class="preview-placeholder">
            <svg class="dropzone-icon preview-placeholder-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
            </svg>
            <div>Vorschau hier</div>
        </div>`;
    els.resultLink.style.display = 'none';
    if (els.detectionResults) els.detectionResults.innerHTML = '';
}

async function updateModelList() {
    try {
        const res = await fetch('/models');
        const data = await res.json();
        const list = $('modelList');
        const models = data.available_models || [];
        const active = data.active_model;
        state.activeModel = active;
        
        const activePill = $('activeModelPill');
        const activeLabel = $('activeModelLabel');
        
        if (active) {
            els.selectedModel.value = active;
            if (activePill) {
                activePill.classList.add('active');
                activeLabel.textContent = active;
            }
        } else if (activePill) {
            activePill.classList.remove('active');
        }
        
        if (!models.length) {
            list.innerHTML = `<div class="empty-state">Keine Modelle gefunden<br><small>.pt-Dateien in "models/" ablegen</small></div>`;
            return;
        }
        
        list.innerHTML = models.map((m, i) => {
            const type = detectModelType(m);
            const isActive = active === m;
            return `
                <li class="model-card">
                    <h4>${m}</h4>
                    <div class="model-info">
                        <span class="model-tag">${type}</span>
                        <span class="model-tag">#${i + 1}</span>
                    </div>
                    ${isActive ? '<span class="chip">✓ Aktiv</span>' : ''}
                    <div class="inline-actions">
                        <button class="action" onclick="loadModel('${m}')">${isActive ? '↻ Neu laden' : '▶ Laden'}</button>
                        <button onclick="prefillModel('${m}')" class="ghost">Auswählen</button>
                    </div>
                </li>`;
        }).join('');
        
        log(`${models.length} Modelle verfügbar`, 'success');
    } catch (err) {
        showToast('Fehler: ' + err.message, 'error');
    }
}

function detectModelType(fn) {
    const l = fn.toLowerCase();
    if (l.includes('pose')) return 'Pose';
    if (l.includes('seg')) return 'Seg';
    if (l.includes('obb')) return 'OBB';
    if (l.includes('cls')) return 'Cls';
    if (l.includes('detect') || l.includes('yolo')) return 'Detect';
    return 'Custom';
}

async function loadModel(name) {
    try {
        const res = await fetch(`/load/${name}`, { method: 'POST' });
        const data = await res.json();
        showToast(data.status || data.error, res.ok ? 'success' : 'error');
        els.selectedModel.value = name;
        await updateModelList();
        log(`Modell: ${name}`, res.ok ? 'success' : 'error');
    } catch (err) {
        showToast('Netzwerkfehler', 'error');
    }
}

function prefillModel(name) {
    els.selectedModel.value = name;
    switchSection('testing');
    showToast(`"${name}" ausgewählt`, 'success');
}

async function runInference() {
    const modelName = els.selectedModel.value;
    
    if (!modelName) { showToast('Bitte Modell wählen', 'error'); return; }
    if (state.activeModel !== modelName) {
        await loadModel(modelName);
        if (state.activeModel !== modelName) { showToast('Modell-Laden fehlgeschlagen', 'error'); return; }
    }
    
    const conf = parseFloat(els.confThreshold?.value) || 0.25;
    const iou = parseFloat(els.iouThreshold?.value) || 0.45;
    const imgsz = parseInt(els.imgSize?.value) || 640;
    const save = els.saveResults?.checked !== false;
    
    state.processing = true;
    els.runBtn.disabled = true;
    startProgress();
    setStatus(`Verarbeite...`, true);
    log(`Starte ${state.mode}-Inferenz mit ${modelName}`, 'info');
    
    try {
        const start = Date.now();
        let result;
        
        switch (state.mode) {
            case 'image': result = await runImageInference(modelName, conf, iou, imgsz, save); break;
            case 'video': result = await runVideoStreaming(modelName, save); break;
            case 'url': result = await runUrlStreaming(modelName, conf, iou, imgsz, save); break;
            case 'webcam': result = await runWebcamInference(modelName, conf, imgsz); break;
            case 'batch': result = await runBatchInference(modelName, conf, iou, imgsz); break;
            case 'directory': result = await runDirectoryInference(modelName, conf, iou, imgsz); break;
            case 'local_path': result = await runLocalPathInference(modelName, conf, iou, imgsz); break;
            case 'stream': result = await runStreamInference(modelName, conf, iou, imgsz, save); break;
            default: throw new Error('Unbekannter Modus');
        }
        
        const time = ((Date.now() - start) / 1000).toFixed(1);
        updateStats(1, result?.detections?.length || 0, time);
        
        stopProgress(true);
        setStatus(`Fertig (${time}s)`, false);
        els.runBtn.disabled = false;
        state.processing = false;
    } catch (err) {
        stopProgress(false);
        setStatus('Fehler', false);
        els.runBtn.disabled = false;
        state.processing = false;
        log(`Fehler: ${err.message}`, 'error');
        showToast(err.message, 'error');
    }
}

async function runImageInference(modelName, conf, iou, imgsz, save) {
    if (state.mode === 'video') return await runVideoStreaming(modelName, save);
    
    if (!els.imageInput?.files.length) throw new Error('Bitte Datei wählen');
    
    const formData = new FormData();
    formData.append('source_type', 'image');
    formData.append('image', els.imageInput.files[0]);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    formData.append('save', save ? 'true' : 'false');
    
    const res = await fetch('/test', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Inferenz fehlgeschlagen');
    
    handleImageResult(data);
    return data;
}

async function runVideoStreaming(modelName, save) {
    if (!els.videoInput?.files.length) throw new Error('Bitte Video wählen');
    
    const file = els.videoInput.files[0];
    const formData = new FormData();
    formData.append('video', file);

    els.previewBox.innerHTML = `
        <div class="preview-video-processing">
            <div class="preview-video-processing-content">
                <div class="spinner preview-video-processing-spinner"></div>
                <div class="preview-video-processing-label">Video wird verarbeitet...</div>
            </div>
        </div>`;
    els.previewName.textContent = file.name;
    
    return await streamInferenceFetch('/test_video_stream', formData);
}

async function runUrlStreaming(modelName, conf, iou, imgsz, save) {
    const urlValue = els.urlInput?.value || els.streamUrlInput?.value;
    if (!urlValue) throw new Error('Bitte URL eingeben');
    
    const formData = new FormData();
    formData.append('source_value', urlValue);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    formData.append('save', save ? 'true' : 'false');
    
    return await streamInference('/test_url_stream', formData);
}

async function streamInference(endpoint, formData) {
    return new Promise((resolve, reject) => {
        const es = new EventSourcePolyfill(endpoint, { body: formData, headers: { 'Accept': 'text/event-stream' } });
        let lastPayload = null;
        const start = Date.now();
        
        es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            lastPayload = data;
            
            if (data.error) { es.close(); reject(new Error(data.error)); return; }
            
            if (data.progress !== undefined) {
                const eta = computeEta(data);
                setProgress(data.progress || 0, eta);
                log(data.message || `Frame ${data.frame}`, 'info');
                if (data.total_frames) els.previewName.textContent = `Frame ${data.frame}/${data.total_frames}`;
                return;
            }
            
            if (data.done) {
                es.close();
                const time = ((Date.now() - start) / 1000).toFixed(1);
                if (data.video_url) setVideoResult(data.video_url);
                log(`Fertig in ${time}s`, 'success');
                els.output.textContent = JSON.stringify(data, null, 2);
                resolve(data);
            }
        };
        
        es.onerror = () => { es.close(); reject(new Error('Verbindung unterbrochen')); };
    });
}

async function streamInferenceFetch(endpoint, formData) {
    const res = await fetch(endpoint, { method: 'POST', body: formData });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || 'Stream fehlgeschlagen');
    }
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let lastPayload = null;
    
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        let boundary;
        while ((boundary = buffer.indexOf('\n\n')) >= 0) {
            const raw = buffer.slice(0, boundary).trim();
            buffer = buffer.slice(boundary + 2);
            
            if (!raw.startsWith('data:')) continue;
            
            try {
                const payload = JSON.parse(raw.replace(/^data:\s*/, ''));
                lastPayload = payload;
                handleStreamPayload(payload);
            } catch (e) { continue; }
        }
    }
    
    return lastPayload;
}

function handleStreamPayload(p) {
    if (p.error) { stopProgress(false); setStatus('Fehler', false); log(p.error, 'error'); showToast(p.error, 'error'); return; }
    if (p.progress !== undefined) { setProgress(p.progress || 0, computeEta(p)); log(p.message || `Frame ${p.frame}`, 'info'); return; }
    if (p.done) {
        stopProgress(true);
        setStatus('Fertig', false);
        if (p.video_url) setVideoResult(p.video_url);
        els.output.textContent = JSON.stringify(p, null, 2);
        log(`${p.total_frames_processed || 0} Frames`, 'success');
    }
}

async function runWebcamInference(modelName, conf, imgsz) {
    const camId = els.webcamIdInput?.value || '0';
    els.previewBox.innerHTML = `<img src="/test_webcam?cam_id=${camId}&conf=${conf}&imgsz=${imgsz}" class="preview-media" alt="Webcam">`;
    els.previewName.textContent = `Webcam ${camId}`;
    log(`Webcam-Stream gestartet (ID: ${camId})`, 'success');
    stopProgress(true);
    setStatus('Webcam aktiv', false);
    els.runBtn.disabled = false;
    return { success: true };
}

async function runBatchInference(modelName, conf, iou, imgsz) {
    if (!els.batchInput?.files.length) throw new Error('Bitte Bilder wählen');
    
    const formData = new FormData();
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    
    for (let i = 0; i < els.batchInput.files.length; i++) {
        formData.append('images', els.batchInput.files[i]);
    }
    
    const res = await fetch('/test_batch', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Batch fehlgeschlagen');
    
    handleBatchResult(data);
    return data;
}

async function runDirectoryInference(modelName, conf, iou, imgsz) {
    const files = Array.from(els.directoryInput?.files || []).filter(f => f.type.startsWith('image/'));
    if (!files.length) throw new Error('Bitte Ordner mit Bildern wählen');
    
    const formData = new FormData();
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    
    for (let i = 0; i < files.length; i++) {
        formData.append('images', files[i]);
    }
    
    const res = await fetch('/test_batch', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Verzeichnis-Upload fehlgeschlagen');
    
    handleBatchResult(data);
    return data;
}

async function runLocalPathInference(modelName, conf, iou, imgsz) {
    const pathValue = els.localPathInput?.value?.trim();
    if (!pathValue) throw new Error('Bitte System-Pfad eingeben');

    const formData = new FormData();
    formData.append('path', pathValue);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);

    const res = await fetch('/test_local_path', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'Pfad-Inferenz fehlgeschlagen');

    // Ergebnis basierend auf Pfad-Typ verarbeiten
    if (data.path_type === 'directory') {
        handleBatchResult(data);
    } else {
        handleImageResult(data);
    }
    
    log(`${data.files_count || 1} Datei(en) verarbeitet`, 'success');
    return data;
}

async function runStreamInference(modelName, conf, iou, imgsz, save) {
    return await runUrlStreaming(modelName, conf, iou, imgsz, save);
}

function handleImageResult(data) {
    els.output.textContent = JSON.stringify(data, null, 2);
    
    if (data.image_url) {
        els.previewBox.innerHTML = `<img src="${data.image_url}?t=${Date.now()}" alt="result" class="preview-media">`;
        els.previewName.textContent = 'Ergebnis (annotiert)';
    }
    
    if (data.detections?.length > 0) {
        showDetections(data.detections);
        log(`${data.detections.length} Objekt(e) erkannt`, 'success');
    }
    
    if (data.video_url) setVideoResult(data.video_url);
}

function showDetections(detections) {
    if (!els.detectionResults) return;
    els.detectionResults.innerHTML = detections.map(d => `
        <div class="detection-item">
            <span class="detection-class">${d.class}</span>
            <span class="detection-conf">${(d.confidence * 100).toFixed(1)}%</span>
        </div>`).join('');
}

function handleBatchResult(data) {
    els.output.textContent = JSON.stringify(data, null, 2);
    
    if (data.annotated_images?.length > 0) {
        els.previewBox.innerHTML = `<img src="${data.annotated_images[0]}?t=${Date.now()}" alt="batch" class="preview-media">`;
        els.previewName.textContent = `Batch: ${data.processed_count} Bilder`;
    }
    
    log(`${data.processed_count} Bilder verarbeitet`, 'success');
}

function setVideoResult(url) {
    const videoUrl = `${url}?t=${Date.now()}`;
    els.previewBox.innerHTML = `
        <video controls autoplay playsinline class="preview-media">
            <source src="${videoUrl}" type="video/mp4">
        </video>`;
    els.previewName.textContent = 'Ergebnis-Video';
    els.resultLink.href = videoUrl;
    els.resultLink.style.display = 'inline';
    els.resultLink.textContent = '📥 Download';
    els.resultLink.target = '_blank';
    log(`Video: ${url}`, 'success');
    
    const video = els.previewBox.querySelector('video');
    if (video) {
        video.load();
        video.onloadeddata = () => log('Video geladen', 'success');
        video.onerror = () => log('Video-Fehler', 'error');
    }
}

function setStatus(text, isLoading = false) {
    if (!els.testStatusPill) return;
    if (!text) { els.testStatusPill.style.display = 'none'; return; }
    
    els.testStatusPill.style.display = 'inline-flex';
    els.testStatusPill.className = `pill ${isLoading ? 'loading' : 'active'}`;
    els.testStatusLabel.textContent = text;
}

let progressTimer = null;
let progressStart = null;

function startProgress() {
    if (!els.progressWrap) return;
    stopProgress(false);
    els.progressWrap.style.display = 'grid';
    progressStart = Date.now();
    setProgress(5, '—');
    
    progressTimer = setInterval(() => {
        const elapsed = (Date.now() - progressStart) / 1000;
        const pct = Math.min(90, 5 + (elapsed / 30) * 85);
        setProgress(Math.round(pct), `${Math.max(1, Math.round(30 - elapsed))}s`);
    }, 700);
}

function stopProgress(success) {
    if (progressTimer) clearInterval(progressTimer);
    progressTimer = null;
    if (!els.progressWrap) return;
    
    if (success) {
        setProgress(100, '0s');
        setTimeout(() => els.progressWrap.style.display = 'none', 800);
    } else {
        els.progressWrap.style.display = 'none';
    }
    progressStart = null;
}

function setProgress(pct, eta) {
    if (els.progressInner) els.progressInner.style.width = `${pct}%`;
    if (els.progressText) els.progressText.textContent = `${pct}%`;
    if (els.progressEta) els.progressEta.textContent = eta || '—';
}

function computeEta(p) {
    if (!p.total_frames || !p.frame || !progressStart) return '…';
    const elapsed = (Date.now() - progressStart) / 1000;
    const fps = p.frame / Math.max(elapsed, 0.001);
    const remaining = Math.max(0, (p.total_frames - p.frame) / fps);
    return `${Math.round(remaining)}s`;
}

function updateStats(inferences = 1, detections = 0, time = 0) {
    state.stats.totalInferences += inferences;
    state.stats.totalDetections += detections;
    const total = state.stats.totalInferences;
    state.stats.avgTime = ((state.stats.avgTime * (total - 1)) + parseFloat(time)) / total;
    saveStats();
    renderStats();
}

function renderStats() {
    if (els.statsTotal) els.statsTotal.textContent = state.stats.totalInferences;
    if (els.statsDetections) els.statsDetections.textContent = state.stats.totalDetections;
    if (els.statsTime) els.statsTime.textContent = `${state.stats.avgTime.toFixed(1)}s`;
}

function saveStats() {
    try { localStorage.setItem('yolo_stats', JSON.stringify(state.stats)); } catch (e) {}
}

function loadStats() {
    try {
        const saved = localStorage.getItem('yolo_stats');
        if (saved) { state.stats = JSON.parse(saved); renderStats(); }
    } catch (e) {}
}

function log(msg, type = 'info') {
    if (!els.logEntries) return;
    const line = document.createElement('div');
    line.className = type;
    line.textContent = `[${new Date().toLocaleTimeString('de-DE')}] ${msg}`;
    els.logEntries.appendChild(line);
    const container = els.logEntries.parentElement;
    if (container) container.scrollTop = container.scrollHeight;
}

function showToast(msg, type = 'success') {
    const stack = $('toastStack');
    if (!stack) { console.log(`[${type}] ${msg}`); return; }

    const icons = {
        success: '✓',
        error: '✕',
        warning: '⚠',
        info: 'ℹ'
    };
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <span class="toast-icon">${icons[type] || icons.success}</span>
        <span>${msg}</span>
    `;
    stack.appendChild(toast);
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function resetTest() {
    [els.imageInput, els.videoInput, els.batchInput].forEach(i => { if (i) i.value = ''; });
    [els.urlInput, els.streamUrlInput].forEach(i => { if (i) i.value = ''; });
    if (els.webcamIdInput) els.webcamIdInput.value = '0';
    
    resetPreview();
    if (els.output) els.output.textContent = '';
    setStatus();
    stopProgress(false);
    log('Zurückgesetzt', 'info');
}

function setupKeyboard() {
    document.addEventListener('keydown', e => {
        // Don't trigger when typing in inputs/textareas
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
            // Allow Ctrl+V for path input in local_path mode
            if ((e.ctrlKey || e.metaKey) && e.key === 'v' && state.mode === 'local_path') {
                setTimeout(() => {
                    const input = els.localPathInput;
                    const text = input?.value?.trim();
                    if (text && isValidPath(text)) {
                        showToast(`Pfad eingefügt: ${text}`, 'success');
                    }
                }, 100);
            }
            return;
        }

        // Ctrl+Enter: Start inference
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            if (!state.processing) runInference();
        }
        // Escape: Reset
        if (e.key === 'Escape') {
            e.preventDefault();
            resetTest();
        }
    });
}

class EventSourcePolyfill {
    constructor(url, options = {}) {
        this.url = url;
        this.options = options;
        this.onmessage = null;
        this.onerror = null;
        this.connect();
    }
    
    async connect() {
        try {
            const res = await fetch(this.url, {
                method: 'POST',
                body: this.options.body,
                headers: { 'Accept': 'text/event-stream', ...this.options.headers }
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        const data = line.slice(5).trim();
                        if (this.onmessage) this.onmessage({ data });
                    }
                }
            }
        } catch (err) { if (this.onerror) this.onerror(err); }
    }
    
    close() {}
}

// ==================== COMPARE ====================
let compareModels = [];
let compareImage = null;

async function initCompare() {
    await updateCompareModelList();
    setupCompareDropzone();
}

async function updateCompareModelList() {
    const res = await fetch('/models');
    const data = await res.json();
    const list = $('compareModelList');
    const models = data.available_models || [];
    
    if (!models.length) { list.innerHTML = '<div class="empty-state">Keine Modelle</div>'; return; }

    list.innerHTML = models.map(m => `
        <label class="model-compare-card">
            <input type="checkbox" value="${m}" onchange="toggleCompareModel('${m}')" class="model-compare-checkbox">
            <div class="model-compare-info">
                <div class="model-compare-name">${m}</div>
                <div class="model-compare-type">${detectModelType(m)}</div>
            </div>
        </label>`).join('');
}

function toggleCompareModel(name) {
    compareModels = compareModels.includes(name)
        ? compareModels.filter(m => m !== name)
        : [...compareModels, name];
}

function setupCompareDropzone() {
    const dz = $('compareDropzone');
    const input = $('compareImage');
    if (!dz || !input) return;
    
    dz.addEventListener('click', e => { e.preventDefault(); input.click(); });
    
    ['dragenter', 'dragover'].forEach(evt => {
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover'); });
    });
    
    dz.addEventListener('drop', e => {
        e.preventDefault();
        if (e.dataTransfer.files.length) {
            compareImage = e.dataTransfer.files[0];
            $('compareDropPrompt').textContent = compareImage.name;
            log(`Vergleich: ${compareImage.name}`, 'success');
        }
    });
    
    input.addEventListener('change', e => {
        if (e.target.files.length) {
            compareImage = e.target.files[0];
            $('compareDropPrompt').textContent = compareImage.name;
            log(`Vergleich: ${compareImage.name}`, 'success');
        }
    });
}

async function runCompare() {
    if (compareModels.length < 2) { showToast('Mind. 2 Modelle wählen', 'error'); return; }
    if (!compareImage) { showToast('Bild wählen', 'error'); return; }
    
    const btn = $('compareBtn');
    btn.disabled = true;
    btn.textContent = '⏳...';
    
    const formData = new FormData();
    formData.append('image', compareImage);
    compareModels.forEach(m => formData.append('models', m));
    formData.append('conf', $('compareConf').value);
    formData.append('iou', $('compareIou').value);
    formData.append('imgsz', $('compareSize').value);
    
    try {
        const res = await fetch('/compare', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        displayCompareResults(data);
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '⚖ Vergleich starten';
    }
}

function displayCompareResults(data) {
    $('compareResults').style.display = 'block';

    let html = `
        <div class="stats-grid-wrapper">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">${data.models_compared}</div>
                    <div class="stat-label">Modelle</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${data.total_time}s</div>
                    <div class="stat-label">Gesamtzeit</div>
                </div>
            </div>
        </div>
        <div class="stats-content-grid">`;

    data.results.forEach(r => {
        html += r.success
            ? `<div class="model-card">
                <h4 class="stats-model-title">${r.model}</h4>
                <div class="stats-detection-grid">
                    <div class="stat-detection-card">
                        <div class="stat-detection-value">${r.detections_count}</div>
                        <div class="stat-detection-label">Treffer</div>
                    </div>
                    <div class="stat-detection-card">
                        <div class="stat-time-value">${r.inference_time}ms</div>
                        <div class="stat-detection-label">Inferenz</div>
                    </div>
                </div>
            </div>`
            : `<div class="model-card model-error"><div class="model-error-message">❌ ${r.model}: ${r.error}</div></div>`;
    });

    html += '</div>';
    $('compareResultsContent').innerHTML = html;
}

// ==================== HISTORY ====================
async function loadHistory(search = '', source = '') {
    try {
        const limit = $('historyLimit')?.value || '50';
        let url = `/history?limit=${limit}`;
        if (source) url += `&source_type=${source}`;
        
        const res = await fetch(url);
        const data = await res.json();
        const list = $('historyList');
        const empty = $('historyEmpty');

        let history = data.history || [];
        
        // Client-side search filter
        if (search) {
            history = history.filter(h => 
                h.model?.toLowerCase().includes(search) ||
                h.source_value?.toLowerCase().includes(search) ||
                h.source_type?.toLowerCase().includes(search)
            );
        }

        if (!history.length) {
            list.style.display = 'none';
            empty.style.display = 'block';
            return;
        }

        list.style.display = 'grid';
        empty.style.display = 'none';

        list.innerHTML = history.map(entry => {
            const date = new Date(entry.timestamp * 1000).toLocaleString('de-DE');
            return `
                <div class="model-card history-card animate-fade-in">
                    <div class="history-header">
                        <div>
                            <div class="history-model">${entry.model}</div>
                            <div class="history-meta">${entry.source_type} • ${entry.detections_count} Treffer</div>
                            <div class="history-date">${date}</div>
                        </div>
                        <div class="history-actions">
                            ${entry.image_url ? `<a href="${entry.image_url}" target="_blank" class="ghost" title="Bild anzeigen">🖼️</a>` : ''}
                            ${entry.video_url ? `<a href="${entry.video_url}" target="_blank" class="ghost" title="Video anzeigen">🎬</a>` : ''}
                            ${entry.json_url ? `<a href="${entry.json_url}" target="_blank" class="ghost" title="JSON anzeigen">📄</a>` : ''}
                        </div>
                    </div>
                </div>`;
        }).join('');
    } catch (err) { showToast('Fehler: ' + err.message, 'error'); }
}

async function clearHistory() {
    if (!confirm('Verlauf löschen?')) return;
    try {
        await fetch('/history/clear', { method: 'POST' });
        loadHistory();
        showToast('Verlauf gelöscht', 'success');
    } catch (err) { showToast(err.message, 'error'); }
}

function exportHistory(format) {
    window.open(`/exports/results/${format}`, '_blank');
    showToast('Export gestartet', 'success');
}

// ==================== DATASET / GROUNDING DINO ====================
let datasetImages = [];
let datasetAnnotations = [];
let datasetTemplates = [];

// Dataset Tab Initialisierung
function setupDatasetTab() {
    setupDatasetDropzone();
    setupDatasetParamLabels();
}

function setupDatasetDropzone() {
    const dz = $('datasetDropzone');
    const input = $('datasetImage');
    if (!dz || !input) return;

    dz.addEventListener('click', e => { e.preventDefault(); input.click(); });

    ['dragenter', 'dragover'].forEach(evt => {
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover'); });
    });

    dz.addEventListener('drop', e => {
        e.preventDefault();
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        handleDatasetFiles(files);
    });

    input.addEventListener('change', e => {
        const files = Array.from(e.target.files);
        handleDatasetFiles(files);
    });
}

function handleDatasetFiles(files) {
    if (!files.length) return;
    datasetImages = files;
    $('datasetDropPrompt').textContent = `${files.length} Bild(er) ausgewählt`;
    log(`${files.length} Bilder für Dataset`, 'info');
}

function setupDatasetParamLabels() {
    const boxThresh = $('boxThreshold');
    const textThresh = $('textThreshold');
    if (boxThresh) $('boxThresholdValue').textContent = boxThresh.value;
    if (textThresh) $('textThresholdValue').textContent = textThresh.value;

    boxThresh?.addEventListener('input', () => {
        $('boxThresholdValue').textContent = boxThresh.value;
    });
    textThresh?.addEventListener('input', () => {
        $('textThresholdValue').textContent = textThresh.value;
    });
}

async function loadTemplates() {
    try {
        const res = await fetch('/dataset/templates');
        const data = await res.json();
        datasetTemplates = data.templates || [];

        const prompt = $('datasetPrompt');
        if (prompt && datasetTemplates.length) {
            const templateText = datasetTemplates.map(t => `${t.name}: "${t.prompt}"`).join('\n');
            showToast(`${datasetTemplates.length} Vorlagen geladen`, 'success');
            log(`Vorlagen: ${templateText}`, 'info');
        }
    } catch (err) {
        showToast('Fehler: ' + err.message, 'error');
    }
}

function applyTemplate(templateId) {
    const template = datasetTemplates.find(t => t.id === templateId);
    if (template) {
        $('datasetPrompt').value = template.prompt;
        showToast(`Vorlage "${template.name}" angewendet`, 'success');
    }
}

async function annotateImages() {
    const prompt = $('datasetPrompt')?.value?.trim();
    if (!prompt) { showToast('Bitte Text-Prompt eingeben', 'error'); return; }
    if (!datasetImages.length) { showToast('Bitte Bilder auswählen', 'error'); return; }

    const btn = $('annotateBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Verarbeite...';

    const boxThreshold = $('boxThreshold')?.value || 0.35;
    const textThreshold = $('textThreshold')?.value || 0.25;

    setDatasetStatus('Verarbeite...', true);
    startDatasetProgress();

    try {
        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('box_threshold', boxThreshold);
        formData.append('text_threshold', textThreshold);

        datasetImages.forEach((file, i) => {
            formData.append('images', file);
        });

        const res = await fetch('/dataset/annotate_batch', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) throw new Error(data.error || 'Annotation fehlgeschlagen');

        stopDatasetProgress(true);
        setDatasetStatus('Fertig', false);
        displayDatasetResults(data);

        log(`${data.processed_count} Bilder annotiert`, 'success');
        showToast(`${data.processed_count} Bilder annotiert`, 'success');
    } catch (err) {
        stopDatasetProgress(false);
        setDatasetStatus('Fehler', false);
        log(`Fehler: ${err.message}`, 'error');
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🚀 Annotation starten';
    }
}

function displayDatasetResults(data) {
    datasetAnnotations = data.results || [];
    const content = $('datasetResultsContent');
    const preview = $('datasetPreview');
    const previewContent = $('datasetPreviewContent');

    if (!content) return;

    // Ergebnisse anzeigen
    let html = `<div class="stats-grid-wrapper">
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">${data.processed_count}</div><div class="stat-label">Bilder</div></div>`;
    const totalAnnotations = data.results?.reduce((sum, r) => sum + (r.annotation_count || 0), 0) || 0;
    html += `<div class="stat-card"><div class="stat-value">${totalAnnotations}</div><div class="stat-label">Annotationen</div></div>
        </div>
    </div>`;

    // Annotationen pro Bild
    if (data.results?.length) {
        html += `<div class="stats-content-grid">`;
        data.results.forEach((r, i) => {
            html += `<div class="model-card">
                <div class="annotation-file-header">
                    <div>
                        <div class="annotation-file-name">${r.filename}</div>
                        <div class="annotation-count">${r.annotation_count} Objekte erkannt</div>
                    </div>
                    <button class="ghost" onclick="showAnnotationDetails(${i})">📋 Details</button>
                </div>
                ${r.annotations?.length ? `<div class="annotation-tags">
                    ${r.annotations.map(a => `<span class="chip">${a.class} (${(a.confidence * 100).toFixed(0)}%)</span>`).join('')}
                </div>` : ''}
            </div>`;
        });
        html += `</div>`;
    }

    content.innerHTML = html;
    $('datasetResults').style.display = 'block';

    // Preview der annotierten Bilder
    if (data.annotated_images?.length && previewContent) {
        previewContent.innerHTML = data.annotated_images.map(url =>
            `<div class="preview-item"><img src="${url}" alt="annotated"></div>`
        ).join('');
        preview.style.display = 'block';
    }
}

function showAnnotationDetails(index) {
    const result = datasetAnnotations[index];
    if (!result) return;

    const details = JSON.stringify(result, null, 2);
    log(`Details: ${result.filename} - ${result.annotation_count} Annotationen`, 'info');
    
    // JSON Export für dieses Bild
    const blob = new Blob([details], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${result.filename.replace(/\.[^.]+$/, '')}_annotations.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function exportDataset(format) {
    if (!datasetAnnotations.length) { showToast('Keine Annotationen zum Exportieren', 'error'); return; }

    const allAnnotations = datasetAnnotations.flatMap(r => r.annotations || []);
    if (!allAnnotations.length) { showToast('Keine Annotationen gefunden', 'error'); return; }

    const classNames = [...new Set(allAnnotations.map(a => a.class))];

    if (format === 'json') {
        const blob = new Blob([JSON.stringify({ annotations: datasetAnnotations, classes: classNames }, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dataset_annotations_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('JSON Export gestartet', 'success');
    } else if (format === 'yolo') {
        exportToYOLO(allAnnotations, classNames);
    } else if (format === 'coco') {
        exportToCOCO(allAnnotations, classNames);
    }
}

async function exportToYOLO(annotations, classNames) {
    try {
        const res = await fetch('/dataset/export/yolo', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ annotations, class_names: classNames })
        });
        const data = await res.json();
        if (data.success) {
            showToast('YOLO Export erstellt', 'success');
            log(`YOLO Format: ${classNames.length} Klassen`, 'success');
        }
    } catch (err) {
        showToast('Export Fehler: ' + err.message, 'error');
    }
}

async function exportToCOCO(annotations, classNames) {
    try {
        const res = await fetch('/dataset/export/coco', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ annotations, class_names: classNames })
        });
        const data = await res.json();
        if (data.success) {
            const blob = new Blob([JSON.stringify(data.coco_format, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `coco_annotations_${Date.now()}.json`;
            a.click();
            URL.revokeObjectURL(url);
            showToast('COCO Export erstellt', 'success');
        }
    } catch (err) {
        showToast('Export Fehler: ' + err.message, 'error');
    }
}

function clearDataset() {
    datasetImages = [];
    datasetAnnotations = [];
    $('datasetImage').value = '';
    $('datasetPrompt').value = '';
    $('datasetDropPrompt').textContent = 'Bilder ablegen oder klicken';
    $('datasetResults')?.style.setProperty('display', 'none');
    $('datasetPreview')?.style.setProperty('display', 'none');
    setDatasetStatus();
    stopDatasetProgress(false);
    log('Dataset zurückgesetzt', 'info');
}

function setDatasetStatus(text, isLoading = false) {
    const pill = $('datasetStatusPill');
    const label = $('datasetStatusLabel');
    if (!pill) return;
    if (!text) { pill.style.display = 'none'; return; }
    pill.style.display = 'inline-flex';
    pill.className = `pill ${isLoading ? 'loading' : 'active'}`;
    label.textContent = text;
}

let datasetProgressTimer = null;
let datasetProgressStart = null;

function startDatasetProgress() {
    const wrap = $('datasetProgressWrap');
    if (!wrap) return;
    stopDatasetProgress(false);
    wrap.style.display = 'grid';
    datasetProgressStart = Date.now();
    setDatasetProgress(5, '—');

    datasetProgressTimer = setInterval(() => {
        const elapsed = (Date.now() - datasetProgressStart) / 1000;
        const pct = Math.min(90, 5 + (elapsed / 30) * 85);
        setDatasetProgress(Math.round(pct), `${Math.max(1, Math.round(30 - elapsed))}s`);
    }, 700);
}

function stopDatasetProgress(success) {
    if (datasetProgressTimer) clearInterval(datasetProgressTimer);
    datasetProgressTimer = null;
    const wrap = $('datasetProgressWrap');
    if (!wrap) return;
    if (success) {
        setDatasetProgress(100, '0s');
        setTimeout(() => wrap.style.display = 'none', 800);
    } else {
        wrap.style.display = 'none';
    }
    datasetProgressStart = null;
}

function setDatasetProgress(pct, eta) {
    const inner = $('datasetProgressInner');
    const text = $('datasetProgressText');
    const etaEl = $('datasetProgressEta');
    if (inner) inner.style.width = `${pct}%`;
    if (text) text.textContent = `${pct}%`;
    if (etaEl) etaEl.textContent = eta || '—';
}

// Global exports
window.runInference = runInference;
window.resetTest = resetTest;
window.switchSection = switchSection;
window.setMode = setMode;
window.loadModel = loadModel;
window.prefillModel = prefillModel;
window.updateModelList = updateModelList;
window.runCompare = runCompare;
window.initCompare = initCompare;
window.loadHistory = loadHistory;
window.clearHistory = clearHistory;
window.exportHistory = exportHistory;
// Dataset exports
window.loadTemplates = loadTemplates;
window.applyTemplate = applyTemplate;
window.annotateImages = annotateImages;
window.exportDataset = exportDataset;
window.clearDataset = clearDataset;
window.showAnnotationDetails = showAnnotationDetails;
let catalogModels = [];

async function toggleModelCatalog() {
    const catalog = $('modelCatalog');
    if (catalog.style.display === 'none') {
        catalog.style.display = 'block';
        await loadModelCatalog();
    } else {
        catalog.style.display = 'none';
    }
}

async function loadModelCatalog() {
    try {
        const res = await fetch('/models/catalog');
        const data = await res.json();
        catalogModels = data.models || [];
        filterModels('all');
    } catch (err) {
        showToast('Katalog-Fehler: ' + err.message, 'error');
    }
}

function filterModels(version) {
    const list = $('modelCatalogList');
    if (!list) return;

    // Filter logic
    const filtered = version === 'all' 
        ? catalogModels 
        : catalogModels.filter(m => m.version === version);

    // Update filter buttons
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.version === version);
    });

    if (filtered.length === 0) {
        list.innerHTML = '<div class="empty-state">Keine Modelle für diese Version</div>';
        return;
    }

    list.innerHTML = filtered.map(m => `
        <div class="model-card catalog-card">
            <div style="display:flex; justify-content:space-between; align-items:start;">
                <h4 style="margin:0;">${m.name}</h4>
                <span class="chip">${m.version.toUpperCase()}</span>
            </div>
            <div class="model-info" style="margin:10px 0;">
                <span class="model-tag">${m.type}</span>
                <span class="model-tag">${m.size}</span>
            </div>
            <div style="font-size:11px; color:var(--text-secondary); margin-bottom:12px;">
                <div>Accuracy: ${m.accuracy}</div>
                <div>Speed: ${m.speed}</div>
            </div>
            <button class="action btn-block" onclick="downloadModel('${m.name}')" id="dl-btn-${m.name.replace('.','-')}">
                📥 Herunterladen
            </button>
        </div>
    `).join('');
}

async function downloadModel(name) {
    const btn = document.getElementById(`dl-btn-${name.replace('.','-')}`);
    const originalText = btn.innerHTML;
    
    btn.disabled = true;
    btn.innerHTML = '⏳ Lädt...';
    
    try {
        const res = await fetch(`/models/download/${name}`, { method: 'POST' });
        const data = await res.json();
        
        if (res.ok) {
            showToast(`"${name}" erfolgreich geladen`, 'success');
            btn.innerHTML = '✅ Fertig';
            // Hauptliste aktualisieren
            await updateModelList();
        } else {
            throw new Error(data.error || 'Download fehlgeschlagen');
        }
    } catch (err) {
        showToast(err.message, 'error');
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

// ==================== FEW-SHOT AUTO-DETECTION ====================
let fewShotSamples = [];
let currentSampleAnnotations = null;
let currentSampleImage = null;

function handleSampleFile(file) {
    currentSampleImage = file;
    $('sampleDropPrompt').textContent = file.name;
    log(`Sample: ${file.name}`, 'info');
    
    // Sample direkt hinzufügen
    addSampleToFewShot(file);
}

// ==================== AUTO ANNOTATION ====================
async function annotateSample() {
    const prompt = $('samplePrompt')?.value?.trim();
    if (!prompt) { showToast('Bitte Prompt eingeben', 'error'); return; }
    if (!currentSampleImage) { showToast('Bitte Sample-Bild auswählen', 'error'); return; }

    const btn = $('annotateSampleBtn');
    btn.disabled = true;
    btn.textContent = '⏳...';

    try {
        const formData = new FormData();
        formData.append('prompt', prompt);
        formData.append('box_threshold', 0.25);
        formData.append('image', currentSampleImage);

        const res = await fetch('/dataset/annotate', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) throw new Error(data.error);

        currentSampleAnnotations = data.annotations || [];

        // Vorschau der Annotationen
        showSampleAnnotations(currentSampleAnnotations);

        // Hinzufügen-Button aktivieren
        $('addSampleBtn').disabled = currentSampleAnnotations.length === 0;

        showToast(`${currentSampleAnnotations.length} Objekte annotiert`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🏷️ Sample annotieren';
    }
}

function setupFewShotDetection() {
    setupSampleDropzone();
    setupAutoDetectDropzone();
    setupAutoDetectThreshold();
    updateFewShotStatus();
}

function setupSampleDropzone() {
    const dz = $('sampleDropzone');
    const input = $('sampleImage');
    if (!dz || !input) return;

    dz.addEventListener('click', e => { e.preventDefault(); input.click(); });

    ['dragenter', 'dragover'].forEach(evt => {
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover'); });
    });

    dz.addEventListener('drop', e => {
        e.preventDefault();
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length) handleSampleFile(files[0]);
    });

    input.addEventListener('change', e => {
        if (e.target.files.length) handleSampleFile(e.target.files[0]);
    });
}

function setupAutoDetectDropzone() {
    const dz = $('autoDetectDropzone');
    const input = $('autoDetectImages');
    if (!dz || !input) return;

    dz.addEventListener('click', e => { e.preventDefault(); input.click(); });

    ['dragenter', 'dragover'].forEach(evt => {
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(evt => {
        dz.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); dz.classList.remove('dragover'); });
    });

    dz.addEventListener('drop', e => {
        e.preventDefault();
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        handleAutoDetectFiles(files);
    });

    input.addEventListener('change', e => {
        const files = Array.from(e.target.files);
        handleAutoDetectFiles(files);
    });
}

function handleAutoDetectFiles(files) {
    if (!files.length) return;
    $('autoDetectPrompt').textContent = `${files.length} Bilder ausgewählt`;
    $('autoDetectBtn').disabled = fewShotSamples.length === 0;
    log(`${files.length} Bilder für Auto-Detection`, 'info');
    window._autoDetectFiles = files;
}

function setupAutoDetectThreshold() {
    const thresh = $('autoDetectThreshold');
    if (thresh) {
        $('autoDetectThresholdValue').textContent = thresh.value;
        thresh.addEventListener('input', () => {
            $('autoDetectThresholdValue').textContent = thresh.value;
        });
    }
}

function showSampleAnnotations(annotations) {
    const list = $('samplesList');
    if (!list) return;

    list.innerHTML = `<div style="font-size:11px; color:var(--text-secondary); margin-bottom:4px;">Annotationen:</div>` +
        annotations.map(a => `
            <div class="detection-item" style="padding:4px 8px; font-size:11px;">
                <span class="detection-class">${a.class}</span>
                <span class="detection-conf">${(a.confidence * 100).toFixed(0)}%</span>
            </div>
        `).join('');
}

async function addSampleToFewShot() {
    if (!currentSampleImage || !currentSampleAnnotations?.length) {
        showToast('Keine Annotationen vorhanden', 'error');
        return;
    }

    const btn = $('addSampleBtn');
    btn.disabled = true;
    btn.textContent = '⏳...';

    try {
        const formData = new FormData();
        formData.append('image', currentSampleImage);
        formData.append('annotations', JSON.stringify(currentSampleAnnotations));

        const res = await fetch('/dataset/fewshot/add_sample', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) throw new Error(data.error);

        fewShotSamples.push({
            image: currentSampleImage,
            annotations: currentSampleAnnotations
        });

        updateFewShotStatus();
        updateSamplesList();
        
        // Reset für nächstes Sample
        currentSampleImage = null;
        currentSampleAnnotations = null;
        $('sampleDropPrompt').textContent = 'Beispielbild ablegen';
        $('samplePrompt').value = '';
        $('samplesList').innerHTML = '';
        $('addSampleBtn').disabled = true;

        showToast(data.message, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '➕ Hinzufügen';
    }
}

function updateFewShotStatus() {
    const pill = $('fewshotStatusPill');
    const text = $('fewshotStatusText');
    const autoDetectBtn = $('autoDetectBtn');
    
    if (fewShotSamples.length === 0) {
        pill.style.background = 'rgba(245,158,11,0.15)';
        pill.style.color = 'var(--warning)';
        pill.style.borderColor = 'rgba(245,158,11,0.3)';
        text.textContent = 'Keine Samples';
        if (autoDetectBtn) autoDetectBtn.disabled = true;
    } else if (fewShotSamples.length < 2) {
        pill.style.background = 'rgba(245,158,11,0.15)';
        pill.style.color = 'var(--warning)';
        pill.style.borderColor = 'rgba(245,158,11,0.3)';
        text.textContent = `${fewShotSamples.length} Sample (mind. 2 empfohlen)`;
        if (autoDetectBtn) autoDetectBtn.disabled = true;
    } else {
        pill.style.background = 'rgba(16,185,129,0.15)';
        pill.style.color = 'var(--success)';
        pill.style.borderColor = 'rgba(16,185,129,0.3)';
        const classes = [...new Set(fewShotSamples.flatMap(s => s.annotations.map(a => a.class)))];
        text.textContent = `${fewShotSamples.length} Samples • ${classes.length} Klassen`;
        if (autoDetectBtn) autoDetectBtn.disabled = false;
    }
}

function updateSamplesList() {
    const list = $('samplesList');
    if (!list || fewShotSamples.length === 0) return;

    let html = '<div style="font-size:11px; color:var(--text-secondary); margin:8px 0 4px;">Gespeicherte Samples:</div>';
    fewShotSamples.forEach((s, i) => {
        const classes = [...new Set(s.annotations.map(a => a.class))].join(', ');
        html += `<div class="model-card" style="padding:8px; margin-bottom:4px;">
            <div style="font-size:11px; font-weight:600;">Sample ${i + 1}</div>
            <div style="font-size:10px; color:var(--text-secondary);">${s.annotations.length} Objekte: ${classes}</div>
        </div>`;
    });
    list.innerHTML = html;
}

async function runAutoDetection() {
    if (!fewShotSamples.length) {
        showToast('Bitte zuerst Samples hinzufügen', 'error');
        return;
    }
    if (!window._autoDetectFiles?.length) {
        showToast('Bitte Bilder zum Labeln auswählen', 'error');
        return;
    }

    const btn = $('autoDetectBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Verarbeite...';

    const threshold = $('autoDetectThreshold')?.value || 0.7;

    try {
        const formData = new FormData();
        formData.append('threshold', threshold);
        window._autoDetectFiles.forEach((file, i) => {
            formData.append('images', file);
        });

        const res = await fetch('/dataset/fewshot/auto_detect', { method: 'POST', body: formData });
        const data = await res.json();

        if (!res.ok) throw new Error(data.error);

        displayAutoDetectResults(data);
        showToast(`${data.total_annotations} Objekte automatisch erkannt`, 'success');
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '🎯 Auto-Detect';
    }
}

function displayAutoDetectResults(data) {
    const container = $('autoDetectResults');
    const content = $('autoDetectResultsContent');
    if (!container || !content) return;

    container.style.display = 'block';

    let html = `<div class="stats-grid" style="margin-bottom:16px;">`;
    html += `<div class="stat-card"><div class="stat-value">${data.processed_count}</div><div class="stat-label">Bilder</div></div>`;
    html += `<div class="stat-card"><div class="stat-value">${data.total_annotations}</div><div class="stat-label">Objekte</div></div>`;
    html += `</div>`;

    if (data.results?.length) {
        html += `<div style="display:grid; gap:12px;">`;
        data.results.forEach((r, i) => {
            html += `<div class="model-card">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                    <div>
                        <div style="font-weight:600;">${r.filename}</div>
                        <div style="font-size:12px; color:var(--text-secondary);">${r.annotation_count} Objekte</div>
                    </div>
                    ${r.annotated_image ? `<a href="${r.annotated_image}" target="_blank" class="ghost" style="padding:4px 8px; font-size:11px;">🖼️ Annotiert</a>` : ''}
                </div>
                ${r.annotations?.length ? `<div style="margin-top:8px; display:flex; flex-wrap:wrap; gap:4px;">
                    ${r.annotations.map(a => `<span class="chip">${a.class} (${(a.confidence * 100).toFixed(0)}%)</span>`).join('')}
                </div>` : ''}
            </div>`;
        });
        html += `</div>`;
    }

    content.innerHTML = html;
}

function clearFewShot() {
    if (!confirm('Alle Few-Shot Samples löschen?')) return;
    
    fetch('/dataset/fewshot/clear', { method: 'POST' })
        .then(() => {
            fewShotSamples = [];
            currentSampleAnnotations = null;
            currentSampleImage = null;
            $('samplesList').innerHTML = '';
            $('autoDetectResults').style.display = 'none';
            $('autoDetectPrompt').textContent = 'Bilder zum automatischen Labeln';
            window._autoDetectFiles = null;
            updateFewShotStatus();
            showToast('Samples gelöscht', 'success');
        })
        .catch(err => showToast(err.message, 'error'));
}

// Few-Shot Detection zu global exports hinzufügen
window.setupFewShotDetection = setupFewShotDetection;
window.annotateSample = annotateSample;
window.addSampleToFewShot = addSampleToFewShot;
window.runAutoDetection = runAutoDetection;
window.clearFewShot = clearFewShot;

// ==================== MODEL DOWNLOAD ====================
let modelCatalog = [];
let currentFilter = 'all';
let downloadingModels = {};

async function loadModelCatalog() {
    try {
        const res = await fetch('/models/catalog');
        const data = await res.json();
        modelCatalog = data.models || [];
        renderModelCatalog(modelCatalog);
    } catch (err) {
        showToast('Fehler: ' + err.message, 'error');
    }
}

function toggleModelCatalog() {
    const catalog = document.getElementById('modelCatalog');
    if (catalog.style.display === 'none') {
        catalog.style.display = 'block';
        if (modelCatalog.length === 0) loadModelCatalog();
    } else {
        catalog.style.display = 'none';
    }
}

function filterModels(version) {
    currentFilter = version;
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.version === version);
    });
    
    if (version === 'all') {
        renderModelCatalog(modelCatalog);
    } else {
        renderModelCatalog(modelCatalog.filter(m => m.version === version));
    }
}

function renderModelCatalog(models) {
    const list = document.getElementById('modelCatalogList');
    if (!list) return;
    
    if (!models.length) {
        list.innerHTML = '<div class="empty-state">Keine Modelle gefunden</div>';
        return;
    }
    
    list.innerHTML = models.map(model => {
        const isDownloading = downloadingModels[model.name];
        const isInstalled = false; // Will be checked
        
        return `
            <div class="model-catalog-item" id="model-${model.name.replace(/\./g, '-')}">
                <div class="model-catalog-header">
                    <span class="model-catalog-name">${model.name}</span>
                    <span class="model-catalog-version">${model.version}</span>
                </div>
                <div class="model-catalog-type">
                    <span class="tag ${model.type}">${model.type.toUpperCase()}</span>
                </div>
                <div class="model-catalog-stats">
                    <div class="model-catalog-stat">
                        <div class="value">${model.size}</div>
                        <div class="label">Größe</div>
                    </div>
                    <div class="model-catalog-stat">
                        <div class="value">${model.speed}</div>
                        <div class="label">Speed</div>
                    </div>
                    <div class="model-catalog-stat">
                        <div class="value">${model.accuracy}</div>
                        <div class="label">Genauigkeit</div>
                    </div>
                </div>
                <div class="model-catalog-actions">
                    ${isDownloading 
                        ? `<button class="btn btn-primary" disabled>⏳ Lädt... ${isDownloading.progress}%</button>`
                        : `<button class="btn btn-primary" onclick="downloadModel('${model.name}')">📥 Download</button>`
                    }
                </div>
                ${isDownloading ? `
                    <div class="download-progress">
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${isDownloading.progress}%"></div>
                        </div>
                        <div class="progress-text">
                            <span>${formatBytes(isDownloading.downloaded)}</span>
                            <span>${formatBytes(isDownloading.total)}</span>
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }).join('');
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

async function downloadModel(modelName) {
    const modelEl = document.getElementById(`model-${modelName.replace(/\./g, '-')}`);
    if (!modelEl) return;
    
    // Check if already downloading
    if (downloadingModels[modelName]) return;
    
    // Initialize download state
    downloadingModels[modelName] = { progress: 0, downloaded: 0, total: 0 };
    modelEl.classList.add('downloading');
    
    try {
        const res = await fetch(`/models/download_stream/${modelName}`, { method: 'POST' });
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            
            for (const line of lines) {
                if (line.startsWith('data:')) {
                    try {
                        const data = JSON.parse(line.slice(5).trim());
                        
                        if (data.error) {
                            throw new Error(data.error);
                        }
                        
                        if (data.progress !== undefined) {
                            downloadingModels[modelName] = data;
                            renderModelCatalog(getFilteredModels());
                        }
                        
                        if (data.done) {
                            downloadingModels[modelName] = null;
                            modelEl.classList.remove('downloading');
                            modelEl.classList.add('installed');
                            showToast(`Modell ${modelName} erfolgreich heruntergeladen!`, 'success');
                            updateModelList();
                            return;
                        }
                    } catch (e) {
                        console.error(e);
                    }
                }
            }
        }
    } catch (err) {
        downloadingModels[modelName] = null;
        modelEl.classList.remove('downloading');
        showToast('Download fehlgeschlagen: ' + err.message, 'error');
        renderModelCatalog(getFilteredModels());
    }
}

function getFilteredModels() {
    if (currentFilter === 'all') return modelCatalog;
    return modelCatalog.filter(m => m.version === currentFilter);
}

// ==================== SETTINGS AUTO-SAVE ====================
function loadSettings() {
    try {
        const saved = localStorage.getItem('yolo_settings');
        if (saved) {
            const settings = JSON.parse(saved);
            Object.assign(state.settings, settings);
            
            // Apply settings
            if (settings.conf && els.confThreshold) {
                els.confThreshold.value = settings.conf;
            }
            if (settings.iou && els.iouThreshold) {
                els.iouThreshold.value = settings.iou;
            }
            if (settings.imgsz && els.imgSize) {
                els.imgSize.value = settings.imgsz;
            }
            if (settings.saveResults !== undefined && els.saveResults) {
                els.saveResults.checked = settings.saveResults;
            }
            if (settings.showLabels !== undefined && els.showLabels) {
                els.showLabels.checked = settings.showLabels;
            }
            
            setupParamLabels();
            showToast('Einstellungen geladen', 'info');
        }
    } catch (err) {
        console.error('Failed to load settings:', err);
    }
}

function setupSettingsAutoSave() {
    // Auto-save on parameter changes
    [els.confThreshold, els.iouThreshold, els.imgSize].forEach(el => {
        el?.addEventListener('change', saveSettings);
    });
    
    [els.saveResults, els.showLabels].forEach(el => {
        el?.addEventListener('change', saveSettings);
    });
}

function saveSettings() {
    try {
        state.settings.conf = els.confThreshold?.value || 0.25;
        state.settings.iou = els.iouThreshold?.value || 0.45;
        state.settings.imgsz = els.imgSize?.value || 640;
        state.settings.saveResults = els.saveResults?.checked ?? true;
        state.settings.showLabels = els.showLabels?.checked ?? true;
        
        localStorage.setItem('yolo_settings', JSON.stringify(state.settings));
        
        // Show save notification (debounced)
        clearTimeout(window.settingsSaveTimeout);
        window.settingsSaveTimeout = setTimeout(() => {
            showToast('Einstellungen gespeichert', 'success');
        }, 500);
    } catch (err) {
        console.error('Failed to save settings:', err);
    }
}

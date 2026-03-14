// YOLO Vision - Kompakte Client-Implementation
const $ = id => document.getElementById(id);
const $$ = sel => document.querySelectorAll(sel);

const state = {
    mode: 'image',
    activeModel: null,
    processing: false,
    stats: { totalInferences: 0, totalDetections: 0, avgTime: 0 }
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
}

function setupDropzone(dz, getInput, isBatch = false) {
    if (!dz) return;
    
    const clickHandler = (e) => {
        e.preventDefault();
        getInput()?.click();
    };
    
    dz.addEventListener('click', clickHandler);
    
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
        ? `<video controls src="${url}" style="width:100%;height:100%;object-fit:contain;"></video>`
        : `<img src="${url}" alt="preview" style="width:100%;height:100%;object-fit:contain;">`;
    
    els.resultLink.style.display = 'none';
    log(`${isVideo ? 'Video' : 'Bild'}: ${file.name}`, 'success');
}

function handleBatchSelect(e) {
    const count = e.target.files.length;
    els.previewName.textContent = `${count} Datei(en)`;
    els.previewBox.innerHTML = `
        <div style="display:grid;place-items:center;height:100%;text-align:center;">
            <div>
                <div style="font-size:48px;margin-bottom:8px;">📁</div>
                <div style="color:var(--text-secondary);">${count} Bilder</div>
            </div>
        </div>`;
    log(`${count} Bilder für Batch`, 'info');
}

function resetPreview() {
    els.previewName.textContent = 'Keine Datei';
    els.previewBox.innerHTML = `
        <div style="text-align:center;color:var(--text-muted);font-size:12px;">
            <svg class="dropzone-icon" style="margin:0 auto 8px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                activePill.style.display = 'inline-flex';
                activeLabel.textContent = active;
            }
        } else if (activePill) {
            activePill.style.display = 'none';
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
        <div style="display:grid;place-items:center;height:100%;">
            <div style="text-align:center;">
                <div class="spinner" style="margin:0 auto 16px;"></div>
                <div style="color:var(--text-secondary);">Video wird verarbeitet...</div>
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
    els.previewBox.innerHTML = `<img src="/test_webcam?cam_id=${camId}&conf=${conf}&imgsz=${imgsz}" style="width:100%;height:100%;object-fit:contain;" alt="Webcam">`;
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

async function runStreamInference(modelName, conf, iou, imgsz, save) {
    return await runUrlStreaming(modelName, conf, iou, imgsz, save);
}

function handleImageResult(data) {
    els.output.textContent = JSON.stringify(data, null, 2);
    
    if (data.image_url) {
        els.previewBox.innerHTML = `<img src="${data.image_url}?t=${Date.now()}" alt="result" style="width:100%;height:100%;object-fit:contain;">`;
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
        els.previewBox.innerHTML = `<img src="${data.annotated_images[0]}?t=${Date.now()}" alt="batch" style="width:100%;height:100%;object-fit:contain;">`;
        els.previewName.textContent = `Batch: ${data.processed_count} Bilder`;
    }
    
    log(`${data.processed_count} Bilder verarbeitet`, 'success');
}

function setVideoResult(url) {
    const videoUrl = `${url}?t=${Date.now()}`;
    els.previewBox.innerHTML = `
        <video controls autoplay playsinline style="width:100%;height:100%;object-fit:contain;">
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
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : '⚠';
    toast.innerHTML = `<span style="font-size:18px;">${icon}</span><span>${msg}</span>`;
    stack.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
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
        if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); if (!state.processing) runInference(); }
        if (e.key === 'Escape') resetTest();
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
        <label class="model-card" style="cursor:pointer;display:flex;align-items:center;gap:10px;padding:12px;">
            <input type="checkbox" value="${m}" onchange="toggleCompareModel('${m}')" style="width:18px;height:18px;">
            <div style="flex:1;">
                <div style="font-weight:600;font-size:13px;">${m}</div>
                <div style="font-size:11px;color:var(--text-secondary);">${detectModelType(m)}</div>
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
        <div class="stats-grid" style="margin-bottom:20px;">
            <div class="stat-card">
                <div class="stat-value">${data.models_compared}</div>
                <div class="stat-label">Modelle</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.total_time}s</div>
                <div class="stat-label">Gesamtzeit</div>
            </div>
        </div>
        <div style="display:grid;gap:12px;">`;
    
    data.results.forEach(r => {
        html += r.success
            ? `<div class="model-card">
                <h4 style="margin:0 0 12px;">${r.model}</h4>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:12px;">
                    <div style="text-align:center;padding:10px;background:var(--bg-secondary);border-radius:10px;">
                        <div style="font-size:20px;font-weight:700;color:var(--accent-primary);">${r.detections_count}</div>
                        <div style="font-size:11px;color:var(--text-secondary);">Treffer</div>
                    </div>
                    <div style="text-align:center;padding:10px;background:var(--bg-secondary);border-radius:10px;">
                        <div style="font-size:20px;font-weight:700;color:var(--success);">${r.inference_time}ms</div>
                        <div style="font-size:11px;color:var(--text-secondary);">Inferenz</div>
                    </div>
                </div>
            </div>`
            : `<div class="model-card" style="border-color:var(--error);"><div style="color:var(--error);">❌ ${r.model}: ${r.error}</div></div>`;
    });
    
    html += '</div>';
    $('compareResultsContent').innerHTML = html;
}

// ==================== HISTORY ====================
async function loadHistory() {
    try {
        const res = await fetch('/history?limit=50');
        const data = await res.json();
        const list = $('historyList');
        const empty = $('historyEmpty');
        
        if (!data.history?.length) {
            list.style.display = 'none';
            empty.style.display = 'block';
            return;
        }
        
        list.style.display = 'grid';
        empty.style.display = 'none';
        
        list.innerHTML = data.history.map(entry => {
            const date = new Date(entry.timestamp * 1000).toLocaleString('de-DE');
            return `
                <div class="model-card" style="padding:16px;">
                    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;">
                        <div>
                            <div style="font-weight:600;font-size:14px;">${entry.model}</div>
                            <div style="font-size:12px;color:var(--text-secondary);">${entry.source_type} • ${entry.detections_count} Treffer</div>
                            <div style="font-size:11px;color:var(--text-muted);margin-top:4px;">${date}</div>
                        </div>
                        <div style="display:flex;gap:8px;">
                            ${entry.image_url ? `<a href="${entry.image_url}" target="_blank" class="ghost" style="padding:8px 12px;">🖼️</a>` : ''}
                            ${entry.video_url ? `<a href="${entry.video_url}" target="_blank" class="ghost" style="padding:8px 12px;">🎬</a>` : ''}
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

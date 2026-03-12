// YOLO Prediction App - Enhanced JavaScript
// Modern, feature-rich client for YOLO model inference

// State Management
const state = {
    currentMode: 'image',
    selectedModel: null,
    activeModel: null,
    isProcessing: false,
    eventSource: null,
    stats: {
        totalInferences: 0,
        totalDetections: 0,
        avgProcessingTime: 0
    }
};

// DOM Elements Cache
const elements = {};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    cacheElements();
    setupEventListeners();
    updateModelList();
    loadStats();
    setupKeyboardShortcuts();
});

// Cache DOM elements for performance
function cacheElements() {
    elements.navButtons = document.querySelectorAll('nav button');
    elements.sections = document.querySelectorAll('.tab-section');
    elements.dropzone = document.getElementById('dropzone');
    elements.batchDropzone = document.getElementById('batchDropzone');
    elements.imageInput = document.getElementById('imageInput');
    elements.videoInput = document.getElementById('videoInput');
    elements.batchInput = document.getElementById('batchInput');
    elements.previewBox = document.getElementById('previewBox');
    elements.previewName = document.getElementById('previewName');
    elements.resultLink = document.getElementById('resultLink');
    elements.testStatusPill = document.getElementById('testStatusPill');
    elements.testStatusLabel = document.getElementById('testStatusLabel');
    elements.runBtn = document.getElementById('runBtn');
    elements.logEntries = document.getElementById('logEntries');
    elements.progressWrap = document.getElementById('progressWrap');
    elements.progressInner = document.getElementById('progressInner');
    elements.progressText = document.getElementById('progressText');
    elements.progressEta = document.getElementById('progressEta');
    elements.output = document.getElementById('output');
    elements.selectedModel = document.getElementById('selectedModel');
    elements.urlInput = document.getElementById('urlInput');
    elements.streamUrlInput = document.getElementById('streamUrlInput');
    elements.webcamIdInput = document.getElementById('webcamIdInput');
    elements.confThreshold = document.getElementById('confThreshold');
    elements.iouThreshold = document.getElementById('iouThreshold');
    elements.imgSize = document.getElementById('imgSize');
    elements.saveResults = document.getElementById('saveResults');
    elements.showLabels = document.getElementById('showLabels');
    elements.detectionResults = document.getElementById('detectionResults');
    elements.statsTotal = document.getElementById('statsTotal');
    elements.statsDetections = document.getElementById('statsDetections');
    elements.statsTime = document.getElementById('statsTime');
}

// Setup Event Listeners
function setupEventListeners() {
    // Navigation
    elements.navButtons.forEach(btn => {
        btn.addEventListener('click', () => switchSection(btn.dataset.target));
    });

    // Mode buttons - Fix: Proper event delegation
    document.querySelectorAll('.source-tab').forEach(btn => {
        btn.addEventListener('click', function() {
            const mode = this.dataset.mode;
            if (mode) {
                setMode(mode);
            }
        });
    });

    // Dropzones
    setupDropzone(elements.dropzone, elements.imageInput);
    setupDropzone(elements.batchDropzone, elements.batchInput, true);

    // File inputs
    elements.imageInput?.addEventListener('change', handleFileSelect);
    elements.videoInput?.addEventListener('change', handleFileSelect);
    elements.batchInput?.addEventListener('change', handleBatchSelect);

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
}

// Setup Dropzone with Button Handler
function setupDropzone(dropzone, input, isBatch = false) {
    if (!dropzone || !input) return;

    // Add click handler to ALL buttons inside dropzone
    const buttons = dropzone.querySelectorAll('button');
    buttons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            // Use the correct file input based on mode
            const targetInput = isBatch 
                ? input 
                : (state.currentMode === 'video' ? elements.videoInput : elements.imageInput);
            targetInput?.click();
        });
    });

    // Click on dropzone (not on button)
    dropzone.addEventListener('click', (e) => {
        if (e.target.closest('button')) return;
        const targetInput = isBatch
            ? input
            : (state.currentMode === 'video' ? elements.videoInput : elements.imageInput);
        targetInput?.click();
    });

    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const files = Array.from(e.dataTransfer.files);
        if (files.length === 0) return;
        
        if (isBatch) {
            const dt = new DataTransfer();
            files.forEach(f => dt.items.add(f));
            input.files = dt.files;
            handleBatchSelect({ target: input });
        } else {
            const targetInput = state.currentMode === 'video' ? elements.videoInput : elements.imageInput;
            if (targetInput) {
                const dt = new DataTransfer();
                files.forEach(f => dt.items.add(f));
                targetInput.files = dt.files;
                handleFileSelect({ target: targetInput });
            }
        }
    });
}

// Switch Section
function switchSection(targetId) {
    elements.sections.forEach(sec => {
        sec.classList.toggle('active', sec.id === targetId);
    });
    
    elements.navButtons.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.target === targetId);
    });

    // Scroll to section
    document.getElementById(targetId)?.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'start' 
    });
}

// Set Mode (image, video, url, webcam, batch, stream)
function setMode(mode) {
    state.currentMode = mode;
    
    // Update tab buttons
    document.querySelectorAll('.source-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Hide all input groups first
    const allInputGroups = document.querySelectorAll('.input-group');
    allInputGroups.forEach(el => {
        el.style.display = 'none';
    });

    // Show selected input group based on mode
    const modeGroupMap = {
        'image': 'imageInputGroup',
        'video': 'imageInputGroup',  // Video uses same group but different file input
        'url': 'urlInputGroup',
        'webcam': 'webcamInputGroup',
        'batch': 'batchInputGroup',
        'stream': 'streamInputGroup'
    };
    
    const groupId = modeGroupMap[mode];
    if (groupId) {
        const group = document.getElementById(groupId);
        if (group) {
            group.style.display = 'block';
        }
    }

    // Update dropzone prompt based on mode
    const prompt = document.getElementById('dropPrompt');
    const prompts = {
        image: 'Bild hier ablegen oder klicken',
        video: 'Video hier ablegen oder klicken',
        batch: 'Mehrere Bilder hier ablegen',
        stream: 'YouTube oder RTSP URL eingeben'
    };
    if (prompt) {
        prompt.textContent = prompts[mode] || 'Datei auswählen';
    }

    // Update file input accept attribute based on mode
    if (elements.imageInput) {
        if (mode === 'video') {
            elements.imageInput.accept = 'video/*';
        } else if (mode === 'image') {
            elements.imageInput.accept = 'image/*';
        } else {
            elements.imageInput.accept = 'image/*';
        }
    }

    // Reset preview
    resetPreview();
    logMsg(`Modus gewechselt: ${mode}`, 'info');
}

// Handle File Selection
function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;

    elements.previewName.textContent = file.name;
    const url = URL.createObjectURL(file);

    if (file.type.startsWith('video/') || file.name.toLowerCase().match(/\.(mp4|avi|mov|mkv|webm)$/)) {
        elements.previewBox.innerHTML = `
            <video controls src="${url}" style="width:100%; height:100%; object-fit:contain;"></video>
        `;
        logMsg(`Video geladen: ${file.name}`, 'success');
    } else {
        elements.previewBox.innerHTML = `
            <img src="${url}" alt="preview" style="width:100%; height:100%; object-fit:contain;">
        `;
        logMsg(`Bild geladen: ${file.name}`, 'success');
    }

    elements.resultLink.style.display = 'none';
}

// Handle Batch Selection
function handleBatchSelect(e) {
    const files = e.target.files;
    const count = files.length;
    
    elements.previewName.textContent = `${count} Datei(en)`;
    elements.previewBox.innerHTML = `
        <div style="display:grid; place-items:center; height:100%; text-align:center;">
            <div>
                <div style="font-size:48px; margin-bottom:8px;">📁</div>
                <div style="color:var(--text-secondary);">${count} Bilder ausgewählt</div>
            </div>
        </div>
    `;
    
    logMsg(`${count} Bilder für Batch-Inferenz ausgewählt`, 'info');
}

// Reset Preview
function resetPreview() {
    elements.previewName.textContent = 'Keine Datei';
    elements.previewBox.innerHTML = `
        <div style="text-align:center; color:var(--text-muted);">
            <svg class="dropzone-icon" style="margin:0 auto 12px;" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path>
            </svg>
            <div>Vorschau erscheint hier</div>
        </div>
    `;
    elements.resultLink.style.display = 'none';
    if (elements.detectionResults) {
        elements.detectionResults.innerHTML = '';
    }
}

// Update Model List
async function updateModelList() {
    try {
        const res = await fetch('/models');
        const data = await res.json();
        const list = document.getElementById('modelList');
        const models = data.available_models || [];
        const active = data.active_model;
        state.activeModel = active;

        // Update active model indicator
        const activePill = document.getElementById('activeModelPill');
        const activeLabel = document.getElementById('activeModelLabel');
        
        if (active) {
            elements.selectedModel.value = active;
            state.selectedModel = active;
            if (activePill) {
                activePill.style.display = 'inline-flex';
                activeLabel.textContent = active;
            }
        } else if (activePill) {
            activePill.style.display = 'none';
        }

        // Render model cards
        if (!models.length) {
            list.innerHTML = `
                <div class="empty-state">
                    <svg class="empty-state-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4"></path>
                    </svg>
                    <div>Keine Modelle gefunden</div>
                    <small>Lege .pt-Dateien im Ordner "models" ab</small>
                </div>
            `;
            return;
        }

        list.innerHTML = models.map((m, idx) => {
            const modelType = detectModelType(m);
            const isActive = active === m;
            
            return `
                <li class="model-card">
                    <h4>${m}</h4>
                    <div class="model-info">
                        <span class="model-tag">${modelType}</span>
                        <span class="model-tag">#${idx + 1}</span>
                    </div>
                    ${isActive ? '<span class="chip">✓ Aktiv</span>' : ''}
                    <div class="inline-actions">
                        <button class="action" onclick="loadModel('${m}')">
                            ${isActive ? '↻ Neu laden' : '▶ Laden'}
                        </button>
                        <button onclick="prefillModel('${m}')" class="ghost">Auswählen</button>
                    </div>
                </li>
            `;
        }).join('');

        logMsg(`${models.length} Modelle verfügbar`, 'success');
    } catch (err) {
        showToast('Fehler beim Laden der Modelle: ' + err.message, 'error');
    }
}

// Detect model type from filename
function detectModelType(filename) {
    const lower = filename.toLowerCase();
    if (lower.includes('pose')) return 'Pose';
    if (lower.includes('seg')) return 'Segmentation';
    if (lower.includes('obb')) return 'OBB';
    if (lower.includes('cls')) return 'Classification';
    if (lower.includes('detect') || lower.includes('yolo')) return 'Detection';
    return 'Custom';
}

// Load Model
async function loadModel(name) {
    try {
        const res = await fetch(`/load/${name}`, { method: 'POST' });
        const data = await res.json();
        
        showToast(data.status || data.error, res.ok ? 'success' : 'error');
        elements.selectedModel.value = name;
        state.selectedModel = name;
        
        await updateModelList();
        logMsg(`Modell geladen: ${name}`, res.ok ? 'success' : 'error');
    } catch (err) {
        showToast('Netzwerkfehler: ' + err.message, 'error');
    }
}

// Prefill Model
function prefillModel(name) {
    elements.selectedModel.value = name;
    state.selectedModel = name;
    switchSection('testing');
    showToast(`Modell "${name}" ausgewählt`, 'success');
}

// Run Inference
async function runInference() {
    const modelName = elements.selectedModel.value;

    if (!modelName) {
        showToast('Bitte ein Modell auswählen', 'error');
        return;
    }

    // Ensure the selected model is actually loaded on the backend
    if (state.activeModel !== modelName) {
        await loadModel(modelName);
        if (state.activeModel !== modelName) {
            showToast('Modell konnte nicht geladen werden', 'error');
            return;
        }
    }

    // Get parameters
    const conf = parseFloat(elements.confThreshold?.value) || 0.25;
    const iou = parseFloat(elements.iouThreshold?.value) || 0.45;
    const imgsz = parseInt(elements.imgSize?.value) || 640;
    const save = elements.saveResults?.checked !== false;

    state.isProcessing = true;
    elements.runBtn.disabled = true;
    startProgress();
    setStatus(`Verarbeite...`, true);
    logMsg(`Starte ${state.currentMode}-Inferenz mit ${modelName}`, 'info');

    try {
        const startTime = Date.now();
        let result;

        switch (state.currentMode) {
            case 'image':
                result = await runImageInference(modelName, conf, iou, imgsz, save);
                break;
            case 'video':
                result = await runVideoStreaming(modelName, save);
                break;
            case 'url':
                result = await runUrlStreaming(modelName, conf, iou, imgsz, save);
                break;
            case 'webcam':
                result = await runWebcamInference(modelName, conf, imgsz);
                break;
            case 'batch':
                result = await runBatchInference(modelName, conf, iou, imgsz);
                break;
            case 'stream':
                result = await runStreamInference(modelName, conf, iou, imgsz, save);
                break;
            default:
                throw new Error('Unbekannter Modus');
        }

        const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);
        updateStats(1, result?.detections?.length || 0, processingTime);
        
        stopProgress(true);
        setStatus(`Fertig (${processingTime}s)`, false);
        elements.runBtn.disabled = false;
        state.isProcessing = false;

    } catch (err) {
        stopProgress(false);
        setStatus('Fehler', false);
        elements.runBtn.disabled = false;
        state.isProcessing = false;
        logMsg(`Fehler: ${err.message}`, 'error');
        showToast(err.message, 'error');
    }
}

// Image Inference
async function runImageInference(modelName, conf, iou, imgsz, save) {
    // Check if we're in video mode but using image input group
    if (state.currentMode === 'video') {
        return await runVideoStreaming(modelName, save);
    }
    
    if (!elements.imageInput?.files.length) {
        throw new Error('Bitte ein Bild auswählen');
    }

    const formData = new FormData();
    formData.append('source_type', 'image');
    formData.append('image', elements.imageInput.files[0]);
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

// Video Streaming Inference
async function runVideoStreaming(modelName, save) {
    if (!elements.videoInput?.files.length) {
        throw new Error('Bitte ein Video auswählen');
    }

    const file = elements.videoInput.files[0];
    const formData = new FormData();
    formData.append('video', file);

    elements.previewBox.innerHTML = `
        <div style="display:grid; place-items:center; height:100%;">
            <div style="text-align:center;">
                <div class="spinner" style="margin:0 auto 16px;"></div>
                <div style="color:var(--text-secondary);">Video wird verarbeitet...</div>
            </div>
        </div>
    `;
    elements.previewName.textContent = file.name;

    // Use fetch-based streaming instead of EventSource
    return await streamInferenceFetch('/test_video_stream', formData);
}

// URL Streaming Inference
async function runUrlStreaming(modelName, conf, iou, imgsz, save) {
    // Support both urlInput and streamUrlInput
    const urlValue = elements.urlInput?.value || elements.streamUrlInput?.value;
    if (!urlValue) {
        throw new Error('Bitte eine URL eingeben');
    }

    const formData = new FormData();
    formData.append('source_value', urlValue);
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);
    formData.append('save', save ? 'true' : 'false');

    return await streamInference('/test_url_stream', formData);
}

// Generic SSE Stream Handler
async function streamInference(endpoint, formData) {
    return new Promise((resolve, reject) => {
        const eventSource = new EventSourcePolyfill(endpoint, {
            body: formData,
            headers: {
                'Accept': 'text/event-stream'
            }
        });

        let lastPayload = null;
        const startTime = Date.now();

        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            lastPayload = data;

            if (data.error) {
                eventSource.close();
                reject(new Error(data.error));
                return;
            }

            if (data.progress !== undefined) {
                const eta = computeEta(data);
                setProgress(data.progress || 0, eta);
                logMsg(data.message || `Frame ${data.frame}`, 'info');
                
                if (data.total_frames) {
                    elements.previewName.textContent = `Frame ${data.frame}/${data.total_frames}`;
                }
                return;
            }

            if (data.done) {
                eventSource.close();
                const processingTime = ((Date.now() - startTime) / 1000).toFixed(1);
                
                if (data.video_url) {
                    setVideoResult(data.video_url);
                }
                
                logMsg(`Fertig in ${processingTime}s`, 'success');
                elements.output.textContent = JSON.stringify(data, null, 2);
                resolve(data);
            }
        };

        eventSource.onerror = (err) => {
            eventSource.close();
            reject(new Error('Verbindung unterbrochen'));
        };
    });
}

// Simple fetch-based streaming for browsers without EventSourcePolyfill
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
            
            const jsonStr = raw.replace(/^data:\s*/, '');
            try {
                const payload = JSON.parse(jsonStr);
                lastPayload = payload;
                handleStreamPayload(payload);
            } catch (e) {
                continue;
            }
        }
    }

    return lastPayload;
}

function handleStreamPayload(p) {
    if (p.error) {
        stopProgress(false);
        setStatus('Fehler', false);
        logMsg(p.error, 'error');
        showToast(p.error, 'error');
        return;
    }

    if (p.progress !== undefined) {
        const eta = computeEta(p);
        setProgress(p.progress || 0, eta);
        logMsg(p.message || `Frame ${p.frame}`, 'info');
        return;
    }

    if (p.done) {
        stopProgress(true);
        setStatus('Fertig', false);

        if (p.video_url) {
            setVideoResult(p.video_url);
        }

        elements.output.textContent = JSON.stringify(p, null, 2);
        logMsg(`Verarbeitet: ${p.total_frames_processed || 0} Frames`, 'success');
        // Don't release button here - runInference will do it
    }
}

// Webcam Inference
async function runWebcamInference(modelName, conf, imgsz) {
    const camId = elements.webcamIdInput?.value || '0';
    
    elements.previewBox.innerHTML = `
        <img src="/test_webcam?cam_id=${camId}&conf=${conf}&imgsz=${imgsz}" 
             style="width:100%; height:100%; object-fit:contain;" 
             alt="Webcam Stream">
    `;
    
    elements.previewName.textContent = `Webcam ${camId}`;
    logMsg(`Webcam-Stream gestartet (ID: ${camId})`, 'success');
    
    stopProgress(true);
    setStatus('Webcam aktiv', false);
    elements.runBtn.disabled = false;
    
    return { success: true };
}

// Batch Inference
async function runBatchInference(modelName, conf, iou, imgsz) {
    if (!elements.batchInput?.files.length) {
        throw new Error('Bitte mindestens ein Bild auswählen');
    }

    const formData = new FormData();
    formData.append('conf', conf);
    formData.append('iou', iou);
    formData.append('imgsz', imgsz);

    for (let i = 0; i < elements.batchInput.files.length; i++) {
        formData.append('images', elements.batchInput.files[i]);
    }

    const res = await fetch('/test_batch', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Batch-Inferenz fehlgeschlagen');

    handleBatchResult(data);
    return data;
}

// Stream Inference (YouTube, RTSP)
async function runStreamInference(modelName, conf, iou, imgsz, save) {
    return await runUrlStreaming(modelName, conf, iou, imgsz, save);
}

// Handle Image Result
function handleImageResult(data) {
    elements.output.textContent = JSON.stringify(data, null, 2);
    
    if (data.image_url) {
        elements.previewBox.innerHTML = `
            <img src="${data.image_url}?t=${Date.now()}" 
                 alt="annotated result" 
                 style="width:100%; height:100%; object-fit:contain;">
        `;
        elements.previewName.textContent = 'Ergebnis (annotiert)';
    }

    // Show detections
    if (data.detections && data.detections.length > 0) {
        showDetections(data.detections);
        logMsg(`${data.detections.length} Objekt(e) erkannt`, 'success');
    }

    if (data.video_url) {
        setVideoResult(data.video_url);
    }
}

// Show Detections
function showDetections(detections) {
    if (!elements.detectionResults) return;
    
    elements.detectionResults.innerHTML = detections.map(d => `
        <div class="detection-item">
            <span class="detection-class">${d.class}</span>
            <span class="detection-conf">${(d.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
}

// Handle Batch Result
function handleBatchResult(data) {
    elements.output.textContent = JSON.stringify(data, null, 2);
    
    if (data.annotated_images?.length > 0) {
        elements.previewBox.innerHTML = `
            <img src="${data.annotated_images[0]}?t=${Date.now()}" 
                 alt="batch result" 
                 style="width:100%; height:100%; object-fit:contain;">
        `;
        elements.previewName.textContent = `Batch: ${data.processed_count} Bilder`;
    }
    
    logMsg(`Batch: ${data.processed_count} Bilder verarbeitet`, 'success');
}

// Set Video Result
function setVideoResult(url) {
    const videoUrl = `${url}?t=${Date.now()}`;

    elements.previewBox.innerHTML = `
        <video controls autoplay playsinline style="width:100%; height:100%; object-fit:contain;">
            <source src="${videoUrl}" type="video/mp4">
            Dein Browser unterstützt das Video-Tag nicht.
        </video>
    `;

    elements.previewName.textContent = 'Ergebnis-Video (annotiert)';
    elements.resultLink.href = videoUrl;
    elements.resultLink.style.display = 'inline';
    elements.resultLink.textContent = '📥 Download';
    elements.resultLink.target = '_blank';

    logMsg(`Video: ${url}`, 'success');

    // Force video load
    const video = elements.previewBox.querySelector('video');
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

// Status Management
function setStatus(text, isLoading = false) {
    if (!elements.testStatusPill || !elements.testStatusLabel) return;
    
    if (!text) {
        elements.testStatusPill.style.display = 'none';
        return;
    }
    
    elements.testStatusPill.style.display = 'inline-flex';
    elements.testStatusPill.className = `pill ${isLoading ? 'loading' : 'active'}`;
    elements.testStatusLabel.textContent = text;
}

// Progress Management
let progressTimer = null;
let progressStart = null;

function startProgress() {
    if (!elements.progressWrap) return;
    
    stopProgress(false);
    elements.progressWrap.style.display = 'grid';
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
    
    if (!elements.progressWrap) return;
    
    if (success) {
        setProgress(100, '0s');
        setTimeout(() => elements.progressWrap.style.display = 'none', 800);
    } else {
        elements.progressWrap.style.display = 'none';
    }
    
    progressStart = null;
}

function setProgress(pct, etaText) {
    if (elements.progressInner) elements.progressInner.style.width = `${pct}%`;
    if (elements.progressText) elements.progressText.textContent = `${pct}%`;
    if (elements.progressEta) elements.progressEta.textContent = etaText || '—';
}

function computeEta(p) {
    if (!p.total_frames || !p.frame || !progressStart) return '…';
    
    const elapsed = (Date.now() - progressStart) / 1000;
    const fps = p.frame / Math.max(elapsed, 0.001);
    const remaining = Math.max(0, (p.total_frames - p.frame) / fps);
    
    return `${Math.round(remaining)}s`;
}

// Stats Management
function updateStats(inferences = 1, detections = 0, processingTime = 0) {
    state.stats.totalInferences += inferences;
    state.stats.totalDetections += detections;
    
    // Update average
    const total = state.stats.totalInferences;
    state.stats.avgProcessingTime = 
        ((state.stats.avgProcessingTime * (total - 1)) + parseFloat(processingTime)) / total;
    
    saveStats();
    renderStats();
}

function renderStats() {
    if (elements.statsTotal) {
        elements.statsTotal.textContent = state.stats.totalInferences;
    }
    if (elements.statsDetections) {
        elements.statsDetections.textContent = state.stats.totalDetections;
    }
    if (elements.statsTime) {
        elements.statsTime.textContent = `${state.stats.avgProcessingTime.toFixed(1)}s`;
    }
}

function saveStats() {
    try {
        localStorage.setItem('yolo_stats', JSON.stringify(state.stats));
    } catch (e) {
        // Ignore storage errors
    }
}

function loadStats() {
    try {
        const saved = localStorage.getItem('yolo_stats');
        if (saved) {
            state.stats = JSON.parse(saved);
            renderStats();
        }
    } catch (e) {
        // Ignore storage errors
    }
}

// Logging
function logMsg(msg, type = 'info') {
    if (!elements.logEntries) return;
    
    const line = document.createElement('div');
    line.className = type;
    const ts = new Date().toLocaleTimeString('de-DE');
    line.textContent = `[${ts}] ${msg}`;
    elements.logEntries.appendChild(line);
    
    // Auto-scroll
    const container = elements.logEntries.parentElement;
    if (container) {
        container.scrollTop = container.scrollHeight;
    }
}

// Toast Notifications
function showToast(message, type = 'success') {
    const toastStack = document.getElementById('toastStack');
    if (!toastStack) {
        console.log(`[${type.toUpperCase()}] ${message}`);
        return;
    }
    
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const icon = type === 'success' ? '✓' : type === 'error' ? '✕' : '⚠';
    toast.innerHTML = `
        <span style="font-size:18px;">${icon}</span>
        <span>${message}</span>
    `;
    
    toastStack.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// Reset Test
function resetTest() {
    if (elements.imageInput) elements.imageInput.value = '';
    if (elements.videoInput) elements.videoInput.value = '';
    if (elements.batchInput) elements.batchInput.value = '';
    if (elements.urlInput) elements.urlInput.value = '';
    if (elements.streamUrlInput) elements.streamUrlInput.value = '';
    if (elements.webcamIdInput) elements.webcamIdInput.value = '0';
    
    resetPreview();
    if (elements.output) elements.output.textContent = '';
    setStatus();
    stopProgress(false);
    logMsg('Zurückgesetzt', 'info');
}

// Keyboard Shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', (e) => {
        // Ctrl+R = Run inference
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            if (!state.isProcessing) runInference();
        }
        // Escape = Reset
        if (e.key === 'Escape') {
            resetTest();
        }
    });
}

function handleKeyboard(e) {
    // Handled in setupKeyboardShortcuts
}

// EventSource Polyfill for POST requests
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
            const response = await fetch(this.url, {
                method: 'POST',
                body: this.options.body,
                headers: {
                    'Accept': 'text/event-stream',
                    ...this.options.headers
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const reader = response.body.getReader();
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
                        if (this.onmessage) {
                            this.onmessage({ data });
                        }
                    }
                }
            }
        } catch (err) {
            if (this.onerror) {
                this.onerror(err);
            }
        }
    }
    
    close() {
        // Connection closes automatically when fetch completes
    }
}

// ==================== COMPARE FUNCTIONS ====================

let compareSelectedModels = [];
let compareImageFile = null;

// Initialize compare section
async function initCompare() {
    await updateCompareModelList();
    setupCompareDropzone();
}

async function updateCompareModelList() {
    const res = await fetch('/models');
    const data = await res.json();
    const list = document.getElementById('compareModelList');
    const models = data.available_models || [];
    
    if (!models.length) {
        list.innerHTML = '<div class="empty-state">Keine Modelle verfügbar</div>';
        return;
    }
    
    list.innerHTML = models.map(m => `
        <label class="model-card" style="cursor:pointer; display:flex; align-items:center; gap:10px; padding:12px;">
            <input type="checkbox" value="${m}" onchange="toggleCompareModel('${m}')" style="width:18px; height:18px;">
            <div style="flex:1;">
                <div style="font-weight:600; font-size:13px;">${m}</div>
                <div style="font-size:11px; color:var(--text-secondary);">${detectModelType(m)}</div>
            </div>
        </label>
    `).join('');
}

function toggleCompareModel(name) {
    if (compareSelectedModels.includes(name)) {
        compareSelectedModels = compareSelectedModels.filter(m => m !== name);
    } else {
        compareSelectedModels.push(name);
    }
}

function setupCompareDropzone() {
    const dropzone = document.getElementById('compareDropzone');
    const input = document.getElementById('compareImage');

    if (!dropzone || !input) return;

    // Add click handler to dropzone (including buttons)
    dropzone.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        input.click();
    });

    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove('dragover');
        });
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.dataTransfer.files.length) {
            compareImageFile = e.dataTransfer.files[0];
            document.getElementById('compareDropPrompt').textContent = compareImageFile.name;
            logMsg(`Bild für Vergleich: ${compareImageFile.name}`, 'success');
        }
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length) {
            compareImageFile = e.target.files[0];
            document.getElementById('compareDropPrompt').textContent = compareImageFile.name;
            logMsg(`Bild für Vergleich: ${compareImageFile.name}`, 'success');
        }
    });
}

async function runCompare() {
    if (compareSelectedModels.length < 2) {
        showToast('Bitte mindestens 2 Modelle auswählen', 'error');
        return;
    }
    
    if (!compareImageFile) {
        showToast('Bitte ein Bild auswählen', 'error');
        return;
    }
    
    const btn = document.getElementById('compareBtn');
    btn.disabled = true;
    btn.textContent = '⏳ Vergleiche...';
    
    const formData = new FormData();
    formData.append('image', compareImageFile);
    compareSelectedModels.forEach(m => formData.append('models', m));
    formData.append('conf', document.getElementById('compareConf').value);
    formData.append('iou', document.getElementById('compareIou').value);
    formData.append('imgsz', document.getElementById('compareSize').value);
    
    try {
        const res = await fetch('/compare', { method: 'POST', body: formData });
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.error);
        
        displayCompareResults(data);
    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = '⚖️ Vergleich starten';
    }
}

function displayCompareResults(data) {
    const container = document.getElementById('compareResultsContent');
    const resultsDiv = document.getElementById('compareResults');
    
    resultsDiv.style.display = 'block';
    
    let html = `
        <div class="stats-grid" style="margin-bottom:20px;">
            <div class="stat-card">
                <div class="stat-value">${data.models_compared}</div>
                <div class="stat-label">Modelle verglichen</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${data.total_time}s</div>
                <div class="stat-label">Gesamtzeit</div>
            </div>
        </div>
        <div style="display:grid; gap:12px;">
    `;
    
    data.results.forEach(r => {
        if (r.success) {
            html += `
                <div class="model-card">
                    <h4 style="margin:0 0 12px;">${r.model}</h4>
                    <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-bottom:12px;">
                        <div style="text-align:center; padding:10px; background:var(--bg-secondary); border-radius:10px;">
                            <div style="font-size:20px; font-weight:700; color:var(--accent-primary);">${r.detections_count}</div>
                            <div style="font-size:11px; color:var(--text-secondary);">Treffer</div>
                        </div>
                        <div style="text-align:center; padding:10px; background:var(--bg-secondary); border-radius:10px;">
                            <div style="font-size:20px; font-weight:700; color:var(--success);">${r.inference_time}ms</div>
                            <div style="font-size:11px; color:var(--text-secondary);">Inferenz</div>
                        </div>
                    </div>
                </div>
            `;
        } else {
            html += `<div class="model-card" style="border-color:var(--error);"><div style="color:var(--error);">❌ ${r.model}: ${r.error}</div></div>`;
        }
    });
    
    html += '</div>';
    container.innerHTML = html;
}

// ==================== HISTORY FUNCTIONS ====================

async function loadHistory() {
    try {
        const res = await fetch('/history?limit=50');
        const data = await res.json();
        
        const list = document.getElementById('historyList');
        const empty = document.getElementById('historyEmpty');
        
        if (!data.history || data.history.length === 0) {
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
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
                        <div>
                            <div style="font-weight:600; font-size:14px;">${entry.model}</div>
                            <div style="font-size:12px; color:var(--text-secondary);">
                                ${entry.source_type} • ${entry.detections_count} Treffer
                            </div>
                            <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">${date}</div>
                        </div>
                        <div style="display:flex; gap:8px;">
                            ${entry.image_url ? `<a href="${entry.image_url}" target="_blank" class="ghost" style="padding:8px 12px;">🖼️</a>` : ''}
                            ${entry.video_url ? `<a href="${entry.video_url}" target="_blank" class="ghost" style="padding:8px 12px;">🎬</a>` : ''}
                        </div>
                    </div>
                </div>
            `;
        }).join('');
    } catch (err) {
        showToast('Fehler: ' + err.message, 'error');
    }
}

async function clearHistory() {
    if (!confirm('Verlauf wirklich löschen?')) return;
    try {
        await fetch('/history/clear', { method: 'POST' });
        loadHistory();
        showToast('Verlauf gelöscht', 'success');
    } catch (err) {
        showToast(err.message, 'error');
    }
}

function exportHistory(format) {
    window.open(`/exports/results/${format}`, '_blank');
    showToast('Export gestartet', 'success');
}

// Export for global access
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

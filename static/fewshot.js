/**
 * Few-Shot Incremental Training UI Functions
 */

let fewshotSamples = [];
let fewshotClasses = new Set();
let fewshotTrainingStartTime = null;
let fewshotModelTrained = false;

document.addEventListener('DOMContentLoaded', () => {
    setupFewshotUI();
});

function setupFewshotUI() {
    // Parameter sliders
    setupParamSlider('fewshotEpochs', 'fewshotEpochsValue');
    setupParamSlider('fewshotBatch', 'fewshotBatchValue');
    setupParamSlider('fewshotImgsz', 'fewshotImgszValue');

    // Dropzone setup
    setupFewshotDropzone();
    setupPredictDropzone();

    // Status polling
    setInterval(checkFewshotStatus, 3000);
}

function setupParamSlider(sliderId, valueId) {
    const slider = document.getElementById(sliderId);
    const value = document.getElementById(valueId);
    if (slider && value) {
        slider.addEventListener('input', () => {
            value.textContent = slider.value;
        });
    }
}

function setupFewshotDropzone() {
    const dz = document.getElementById('fewshotDropzone');
    const input = document.getElementById('fewshotImageInput');
    if (!dz || !input) return;

    dz.addEventListener('click', () => input.click());

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
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length) handleFewshotImages(files);
    });

    input.addEventListener('change', e => {
        if (e.target.files.length) handleFewshotImages(Array.from(e.target.files));
    });
}

function setupPredictDropzone() {
    const dz = document.getElementById('fewshotPredictDropzone');
    const input = document.getElementById('fewshotPredictImage');
    if (!dz || !input) return;

    dz.addEventListener('click', () => input.click());

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
        const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
        if (files.length) runFewshotPrediction(files[0]);
    });

    input.addEventListener('change', e => {
        if (e.target.files.length) runFewshotPrediction(e.target.files[0]);
    });
}

function handleFewshotImages(files) {
    files.forEach(file => {
        const reader = new FileReader();
        reader.onload = e => {
            const img = new Image();
            img.src = e.target.result;
            img.onload = () => {
                // Öffne Annotation Editor
                openAnnotationEditor(file, img, e.target.result);
            };
        };
        reader.readAsDataURL(file);
    });
}

function openAnnotationEditor(file, img, dataUrl) {
    // Einfacher Annotation Editor Modal
    const modal = document.createElement('div');
    modal.className = 'annotation-modal';
    modal.innerHTML = `
        <div class="annotation-modal-content">
            <div class="annotation-modal-header">
                <h4>📝 Bild annotieren: ${file.name}</h4>
                <button class="close-btn" onclick="this.closest('.annotation-modal').remove()">✕</button>
            </div>
            <div class="annotation-modal-body">
                <div class="annotation-canvas-wrapper">
                    <canvas id="annotationCanvas"></canvas>
                </div>
                <div class="annotation-controls">
                    <div class="class-input">
                        <label>Klassen Name:</label>
                        <input type="text" id="newClassName" placeholder="z.B. person, car, dog..." list="classSuggestions">
                        <datalist id="classSuggestions">
                            ${Array.from(fewshotClasses).map(c => `<option value="${c}">`).join('')}
                        </datalist>
                    </div>
                    <div class="annotation-actions">
                        <button class="btn btn-primary" onclick="addAnnotation()">➕ Box hinzufügen</button>
                        <button class="btn btn-success" onclick="saveAnnotations()">💾 Speichern</button>
                        <button class="ghost" onclick="this.closest('.annotation-modal').remove()">Abbrechen</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    document.body.appendChild(modal);

    // Canvas setup
    const canvas = document.getElementById('annotationCanvas');
    const ctx = canvas.getContext('2d');
    const maxSize = 600;
    const scale = Math.min(maxSize / img.width, maxSize / img.height);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;

    // Bild zeichnen
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    // Annotationen speichern
    modal.currentAnnotations = [];
    modal.currentImage = { file, dataUrl, width: img.width, height: img.height, scale };

    // Canvas Click für Box Drawing
    let isDrawing = false;
    let startX, startY;

    canvas.addEventListener('mousedown', e => {
        isDrawing = true;
        const rect = canvas.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;
    });

    canvas.addEventListener('mousemove', e => {
        if (!isDrawing) return;
        const rect = canvas.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;

        // Canvas neu zeichnen
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        // Aktuelle Box zeichnen
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, currentX - startX, currentY - startY);
    });

    canvas.addEventListener('mouseup', e => {
        if (!isDrawing) return;
        isDrawing = false;
        const rect = canvas.getBoundingClientRect();
        const endX = e.clientX - rect.left;
        const endY = e.clientY - rect.top;

        // Box speichern (zurückskalieren auf Originalgröße)
        const x1 = Math.min(startX, endX) / scale;
        const y1 = Math.min(startY, endY) / scale;
        const x2 = Math.max(startX, endX) / scale;
        const y2 = Math.max(startY, endY) / scale;

        modal.currentAnnotations.push({
            bbox: [x1, y1, x2, y2],
            class: document.getElementById('newClassName').value || 'object'
        });

        // Box auf Canvas zeichnen
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, startY, endX - startX, endY - startY);
    });

    window.addAnnotation = function() {
        const className = document.getElementById('newClassName').value;
        if (!className) {
            showToast('Bitte Klassenname eingeben', 'error');
            return;
        }
        fewshotClasses.add(className);
        modal.currentAnnotations.forEach(ann => ann.class = className);
        showToast('Klasse gesetzt', 'success');
    };

    window.saveAnnotations = function() {
        if (modal.currentAnnotations.length === 0) {
            showToast('Mindestens eine Box zeichnen', 'error');
            return;
        }

        const className = document.getElementById('newClassName').value;
        if (!className) {
            showToast('Bitte Klassenname eingeben', 'error');
            return;
        }

        // Annotationen speichern
        modal.currentAnnotations.forEach(ann => {
            ann.class = className;
            fewshotClasses.add(className);
        });

        fewshotSamples.push({
            image: modal.currentImage,
            annotations: modal.currentAnnotations
        });

        updateFewshotSamplesUI();
        modal.remove();
        showToast(`${modal.currentAnnotations.length} Annotationen gespeichert`, 'success');
    };
}

function updateFewshotSamplesUI() {
    const listEl = document.getElementById('fewshotSamplesList');
    const gridEl = document.getElementById('fewshotSamplesGrid');
    const countEl = document.getElementById('fewshotSampleCount');
    const classesEl = document.getElementById('fewshotClassesList');

    if (!listEl || !gridEl || !countEl) return;

    countEl.textContent = fewshotSamples.length;

    if (fewshotSamples.length > 0) {
        listEl.style.display = 'block';
        gridEl.innerHTML = fewshotSamples.map((sample, i) => `
            <div class="sample-card">
                <img src="${sample.image.dataUrl}" alt="Sample ${i + 1}">
                <div class="sample-info">
                    <span class="sample-badges">
                        ${sample.annotations.map(a => `<span class="badge">${a.class}</span>`).join('')}
                    </span>
                    <span class="sample-count">${sample.annotations.length} Boxen</span>
                </div>
                <button class="sample-remove" onclick="removeFewshotSample(${i})">✕</button>
            </div>
        `).join('');
    } else {
        listEl.style.display = 'none';
    }

    // Klassen anzeigen
    if (classesEl) {
        if (fewshotClasses.size > 0) {
            classesEl.innerHTML = Array.from(fewshotClasses).map(c => 
                `<span class="class-chip">${c}</span>`
            ).join('');
        } else {
            classesEl.innerHTML = '<p style="font-size:13px; color:var(--text-muted);">Keine Klassen definiert.</p>';
        }
    }

    // Buttons aktivieren
    const startBtn = document.getElementById('startFewshotBtn');
    const incrementalBtn = document.getElementById('incrementalFewshotBtn');
    
    if (startBtn) startBtn.disabled = fewshotSamples.length < 3;
    if (incrementalBtn) incrementalBtn.disabled = !fewshotModelTrained;

    updateFewshotStatusPill();
}

function removeFewshotSample(index) {
    fewshotSamples.splice(index, 1);
    updateFewshotSamplesUI();
}

function clearFewshotSamples() {
    if (!confirm('Alle Samples löschen?')) return;
    fewshotSamples = [];
    fewshotClasses = new Set();
    updateFewshotSamplesUI();
    showToast('Alle Samples gelöscht', 'success');
}

function updateFewshotStatusPill() {
    const pill = document.getElementById('fewshotStatusPill');
    const label = document.getElementById('fewshotStatusLabel');
    
    if (!pill || !label) return;

    pill.style.display = 'inline-flex';
    
    if (fewshotSamples.length === 0) {
        pill.className = 'pill';
        label.textContent = 'Keine Samples';
    } else if (fewshotSamples.length < 3) {
        pill.className = 'pill';
        label.textContent = `${fewshotSamples.length}/3 Samples`;
    } else {
        pill.className = 'pill active';
        label.textContent = `${fewshotSamples.length} Samples, ${fewshotClasses.size} Klassen`;
    }
}

function startFewshotTraining() {
    if (fewshotSamples.length < 3) {
        showToast('Mindestens 3 Samples erforderlich', 'error');
        return;
    }

    const btn = document.getElementById('startFewshotBtn');
    const epochs = document.getElementById('fewshotEpochs').value;
    const batch = document.getElementById('fewshotBatch').value;
    const imgsz = document.getElementById('fewshotImgsz').value;
    const model = document.getElementById('fewshotBaseModel').value;

    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">Starte...</span>';

    // Samples zum Server senden
    const uploadPromises = fewshotSamples.map(sample => {
        const formData = new FormData();
        formData.append('image', sample.image.file);
        formData.append('annotations', JSON.stringify(sample.annotations));
        return fetch('/fewshot/train/add_sample', { method: 'POST', body: formData })
            .then(res => res.json());
    });

    Promise.all(uploadPromises).then(results => {
        // Training starten
        const trainFormData = new FormData();
        trainFormData.append('epochs', epochs);
        trainFormData.append('batch', batch);
        trainFormData.append('imgsz', imgsz);
        trainFormData.append('model', model);

        return fetch('/fewshot/train/start', { method: 'POST', body: trainFormData });
    }).then(res => res.json()).then(data => {
        if (data.error) {
            showToast(data.error, 'error');
            btn.disabled = false;
        } else {
            showToast('Few-Shot Training gestartet!', 'success');
            showFewshotProgress();
            fewshotTrainingStartTime = Date.now();
        }
    }).catch(err => {
        showToast('Fehler: ' + err.message, 'error');
        btn.disabled = false;
    });
}

function startIncrementalTraining() {
    if (!fewshotModelTrained) {
        showToast('Erst normales Training durchführen', 'error');
        return;
    }

    const btn = document.getElementById('incrementalFewshotBtn');
    const epochs = document.getElementById('fewshotEpochs').value;
    const batch = document.getElementById('fewshotBatch').value;
    const imgsz = document.getElementById('fewshotImgsz').value;

    btn.disabled = true;

    // Neue Samples hochladen
    const uploadPromises = fewshotSamples.map(sample => {
        const formData = new FormData();
        formData.append('image', sample.image.file);
        formData.append('annotations', JSON.stringify(sample.annotations));
        return fetch('/fewshot/train/add_sample', { method: 'POST', body: formData })
            .then(res => res.json());
    });

    Promise.all(uploadPromises).then(() => {
        const trainFormData = new FormData();
        trainFormData.append('epochs', epochs);
        trainFormData.append('batch', batch);
        trainFormData.append('imgsz', imgsz);

        return fetch('/fewshot/train/incremental', { method: 'POST', body: trainFormData });
    }).then(res => res.json()).then(data => {
        if (data.error) {
            showToast(data.error, 'error');
            btn.disabled = false;
        } else {
            showToast('Inkrementelles Training gestartet!', 'success');
            showFewshotProgress();
            fewshotTrainingStartTime = Date.now();
        }
    }).catch(err => {
        showToast('Fehler: ' + err.message, 'error');
        btn.disabled = false;
    });
}

function showFewshotProgress() {
    document.getElementById('fewshotPlaceholder').style.display = 'none';
    document.getElementById('fewshotProgress').style.display = 'block';
    document.getElementById('fewshotResults').style.display = 'none';
}

function checkFewshotStatus() {
    fetch('/training/status')
        .then(res => res.json())
        .then(data => {
            updateFewshotProgressUI(data);
        })
        .catch(err => console.error(err));
}

function updateFewshotProgressUI(data) {
    const statusPill = document.getElementById('fewshotStatusPill');
    const statusLabel = document.getElementById('fewshotStatusLabel');
    const btn = document.getElementById('startFewshotBtn');
    const incrementalBtn = document.getElementById('incrementalFewshotBtn');

    if (data.active) {
        if (statusPill) {
            statusPill.style.display = 'inline-flex';
            statusPill.className = 'pill loading';
            statusLabel.textContent = 'Training läuft...';
        }
        if (btn) btn.disabled = true;
        if (incrementalBtn) incrementalBtn.disabled = true;

        // Zeit aktualisieren
        if (fewshotTrainingStartTime) {
            const elapsed = Math.floor((Date.now() - fewshotTrainingStartTime) / 1000);
            document.getElementById('fewshotTime').textContent = formatTime(elapsed);
        }

        // Fortschritt aktualisieren
        const logs = data.logs || [];
        const epochLogs = logs.filter(l => l.message.includes('Epoch'));
        if (epochLogs.length > 0) {
            const lastEpoch = epochLogs[epochLogs.length - 1];
            const match = lastEpoch.message.match(/Epoch (\d+)\/(\d+)/);
            if (match) {
                const current = parseInt(match[1]);
                const total = parseInt(match[2]);
                const progress = (current / total) * 100;
                document.getElementById('fewshotProgressBar').style.width = progress + '%';
                document.getElementById('fewshotCurrentEpoch').textContent = `${current}/${total}`;

                const mapMatch = lastEpoch.message.match(/mAP50=([0-9.]+)/);
                if (mapMatch) {
                    document.getElementById('fewshotMap50').textContent = mapMatch[1];
                }
                const lossMatch = lastEpoch.message.match(/Loss=([0-9.]+)/);
                if (lossMatch) {
                    document.getElementById('fewshotLoss').textContent = lossMatch[1];
                }
            }
        }
    } else {
        if (statusPill) {
            statusPill.style.display = 'inline-flex';
            statusPill.className = 'pill active';
            statusLabel.textContent = fewshotModelTrained ? 'Modell trainiert' : 'Bereit';
        }
        if (btn) {
            btn.disabled = fewshotSamples.length < 3;
            btn.innerHTML = '<span class="btn-icon">🔥</span><span class="btn-text">Training starten</span>';
        }
        if (incrementalBtn) {
            incrementalBtn.disabled = !fewshotModelTrained;
        }

        if (data.results && data.results.model_name && data.results.model_name.startsWith('fewshot_')) {
            showFewshotResults(data.results);
            fewshotModelTrained = true;
        }
    }

    // Logs aktualisieren
    if (data.logs && data.logs.length > 0) {
        updateFewshotLogs(data.logs);
    }
}

function showFewshotResults(results) {
    document.getElementById('fewshotResults').style.display = 'block';
    document.getElementById('downloadFewshotBtn').disabled = false;

    document.getElementById('fewshotResultMap50').textContent = results.map50?.toFixed(4) || '-';
    document.getElementById('fewshotResultMap95').textContent = results.map95?.toFixed(4) || '-';
    document.getElementById('fewshotResultTime').textContent = formatTime(Math.floor(results.training_time || 0));
    document.getElementById('fewshotResultSamples').textContent = results.samples_count || '-';

    showToast('Training abgeschlossen!', 'success');
}

function updateFewshotLogs(logs) {
    const container = document.getElementById('fewshotLogs');
    if (!container) return;

    container.innerHTML = logs.map(log =>
        `<div class="log-entry ${log.type}">[${new Date().toLocaleTimeString()}] ${log.message}</div>`
    ).join('');

    container.scrollTop = container.scrollHeight;
}

function downloadFewshotModel() {
    window.open('/fewshot/train/download', '_blank');
    showToast('Download gestartet', 'success');
}

function runFewshotPrediction(file) {
    const formData = new FormData();
    formData.append('image', file);

    fetch('/fewshot/predict', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
                return;
            }

            const resultDiv = document.getElementById('fewshotPredictResult');
            const img = document.getElementById('fewshotPredictImage');
            const detectionsDiv = document.getElementById('fewshotPredictDetections');

            resultDiv.style.display = 'block';
            img.src = data.annotated_image;
            
            if (data.detections.length > 0) {
                detectionsDiv.innerHTML = `<strong>${data.detections.length} Objekte erkannt:</strong> ` +
                    data.detections.map(d => `${d.class} (${(d.confidence * 100).toFixed(0)}%)`).join(', ');
            } else {
                detectionsDiv.textContent = 'Keine Objekte erkannt';
            }

            showToast(`${data.detection_count} Objekte erkannt`, 'success');
        })
        .catch(err => {
            showToast('Fehler: ' + err.message, 'error');
        });
}

function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function showToast(message, type = 'info') {
    // Bestehende Toast-Funktion verwenden oder einfache Implementierung
    const existing = document.querySelector('.toast-notification');
    if (existing) existing.remove();

    const toast = document.createElement('div');
    toast.className = `toast-notification toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 24px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#6366f1'};
        color: white;
        border-radius: 8px;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// Globale exports
window.startFewshotTraining = startFewshotTraining;
window.startIncrementalTraining = startIncrementalTraining;
window.downloadFewshotModel = downloadFewshotModel;
window.clearFewshotSamples = clearFewshotSamples;

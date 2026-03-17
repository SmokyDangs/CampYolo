/**
 * YOLO Training UI Functions
 */

let trainingInterval = null;
let trainingStartTime = null;

// Initialize training UI
document.addEventListener('DOMContentLoaded', () => {
    setupTrainingUI();
});

function setupTrainingUI() {
    // Parameter sliders
    setupParamSlider('trainingEpochs', 'epochsValue');
    setupParamSlider('trainingBatch', 'batchValue');
    setupParamSlider('trainingImgsz', 'imgszValue');
    
    // Dataset dropzone
    setupTrainingDropzone();
    
    // Start polling for training status
    setInterval(checkTrainingStatus, 2000);
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

function setupTrainingDropzone() {
    const dz = document.getElementById('datasetDropzone');
    const input = document.getElementById('trainingDataset');
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
        const files = Array.from(e.dataTransfer.files).filter(f => f.name.endsWith('.zip'));
        if (files.length) handleTrainingDataset(files[0]);
    });
    
    input.addEventListener('change', e => {
        if (e.target.files.length) handleTrainingDataset(e.target.files[0]);
    });
}

function handleTrainingDataset(file) {
    const info = document.getElementById('datasetInfo');
    const name = document.getElementById('datasetName');
    const classes = document.getElementById('datasetClasses');
    const images = document.getElementById('datasetImages');
    const btn = document.getElementById('startTrainingBtn');
    
    if (info && name && images) {
        info.style.display = 'block';
        name.textContent = file.name;
        classes.textContent = 'YOLO-Format (data.yaml erforderlich)';
        images.textContent = 'ZIP hochgeladen';
        btn.disabled = false;
        
        addTrainingLog(`Dataset geladen: ${file.name}`, 'info');
    }
}

function startTraining() {
    const btn = document.getElementById('startTrainingBtn');
    const model = document.getElementById('trainingModel').value;
    const epochs = document.getElementById('trainingEpochs').value;
    const batch = document.getElementById('trainingBatch').value;
    const imgsz = document.getElementById('trainingImgsz').value;
    const datasetInput = document.getElementById('trainingDataset');
    
    if (!datasetInput.files.length) {
        showToast('Bitte Dataset auswählen', 'error');
        return;
    }
    
    btn.disabled = true;
    btn.innerHTML = '<span class="btn-icon">⏳</span><span class="btn-text">Starte...</span>';
    
    const formData = new FormData();
    formData.append('dataset', datasetInput.files[0]);
    formData.append('model', model);
    formData.append('epochs', epochs);
    formData.append('batch', batch);
    formData.append('imgsz', imgsz);
    
    fetch('/training/start', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            if (data.error) {
                showToast(data.error, 'error');
                btn.disabled = false;
            } else {
                showToast('Training gestartet!', 'success');
                showTrainingProgress();
                trainingStartTime = Date.now();
            }
        })
        .catch(err => {
            showToast('Fehler: ' + err.message, 'error');
            btn.disabled = false;
        });
}

function showTrainingProgress() {
    document.getElementById('trainingPlaceholder').style.display = 'none';
    document.getElementById('trainingProgress').style.display = 'block';
    document.getElementById('trainingResults').style.display = 'none';
}

function checkTrainingStatus() {
    fetch('/training/status')
        .then(res => res.json())
        .then(data => {
            updateTrainingUI(data);
        })
        .catch(err => console.error(err));
}

function updateTrainingUI(data) {
    const statusPill = document.getElementById('trainingStatusPill');
    const statusLabel = document.getElementById('trainingStatusLabel');
    const btn = document.getElementById('startTrainingBtn');
    
    if (data.active) {
        if (statusPill) {
            statusPill.style.display = 'inline-flex';
            statusPill.className = 'pill loading';
            statusLabel.textContent = 'Training läuft...';
        }
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="btn-icon">🔥</span><span class="btn-text">Training läuft...</span>';
        }
        
        // Update time
        if (trainingStartTime) {
            const elapsed = Math.floor((Date.now() - trainingStartTime) / 1000);
            document.getElementById('trainingTime').textContent = formatTime(elapsed);
        }
        
        // Update progress bar (simulated)
        const logs = data.logs || [];
        const epochLogs = logs.filter(l => l.message.includes('Epoch'));
        if (epochLogs.length > 0) {
            const lastEpoch = epochLogs[epochLogs.length - 1];
            const match = lastEpoch.message.match(/Epoch (\d+)\/(\d+)/);
            if (match) {
                const current = parseInt(match[1]);
                const total = parseInt(match[2]);
                const progress = (current / total) * 100;
                document.getElementById('trainingProgressBar').style.width = progress + '%';
                document.getElementById('currentEpoch').textContent = `${current}/${total}`;
                
                // Extract metrics
                const mapMatch = lastEpoch.message.match(/mAP50=([0-9.]+)/);
                if (mapMatch) {
                    document.getElementById('map50').textContent = mapMatch[1];
                }
                const lossMatch = lastEpoch.message.match(/Loss=([0-9.]+)/);
                if (lossMatch) {
                    document.getElementById('trainingLoss').textContent = lossMatch[1];
                }
            }
        }
    } else {
        if (statusPill) {
            statusPill.style.display = 'inline-flex';
            statusPill.className = 'pill active';
            statusLabel.textContent = 'Bereit';
        }
        if (btn) {
            btn.disabled = false;
            btn.innerHTML = '<span class="btn-icon">🔥</span><span class="btn-text">Training starten</span>';
        }
        
        // Show results if available
        if (data.results) {
            showTrainingResults(data.results);
        }
    }
    
    // Update logs
    if (data.logs && data.logs.length > 0) {
        updateTrainingLogs(data.logs);
    }
}

function showTrainingResults(results) {
    document.getElementById('trainingResults').style.display = 'block';
    document.getElementById('downloadResultBtn').disabled = false;
    
    document.getElementById('resultMap50').textContent = results.map50?.toFixed(4) || '-';
    document.getElementById('resultMap95').textContent = results.map95?.toFixed(4) || '-';
    document.getElementById('resultTime').textContent = formatTime(Math.floor(results.training_time || 0));
    document.getElementById('resultBest').textContent = results.best_epoch || '-';
    
    showToast('Training abgeschlossen!', 'success');
}

function updateTrainingLogs(logs) {
    const container = document.getElementById('trainingLogs');
    if (!container) return;
    
    container.innerHTML = logs.map(log => 
        `<div class="log-entry ${log.type}">[${new Date().toLocaleTimeString()}] ${log.message}</div>`
    ).join('');
    
    container.scrollTop = container.scrollHeight;
}

function addTrainingLog(message, type = 'info') {
    const container = document.getElementById('trainingLogs');
    if (!container) return;
    
    container.innerHTML += `<div class="log-entry ${type}">[${new Date().toLocaleTimeString()}] ${message}</div>`;
    container.scrollTop = container.scrollHeight;
}

function downloadModel() {
    window.open('/training/download', '_blank');
    showToast('Download gestartet', 'success');
}

function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

// Global exports
window.startTraining = startTraining;
window.downloadModel = downloadModel;

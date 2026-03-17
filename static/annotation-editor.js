/**
 * Simplified Annotation Editor - Stable Version
 */

const AnnEditor = {
    canvas: null,
    ctx: null,
    image: null,
    annotations: [],
    selected: [],
    mode: 'draw',
    currentClass: 'object',
    zoom: 1,
    isDrawing: false,
    dragStart: null,
    currentBox: null,
    history: [],
    historyIdx: -1,

    init(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return false;
        this.ctx = this.canvas.getContext('2d');
        this.bindEvents();
        return true;
    },

    loadImage(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                this.image = new Image();
                this.image.onload = () => {
                    this.canvas.width = Math.min(this.image.width, 800);
                    this.canvas.height = this.image.height * (this.canvas.width / this.image.width);
                    this.annotations = [];
                    this.selected = [];
                    this.history = [];
                    this.historyIdx = -1;
                    this.saveState();
                    this.render();
                    this.updateUI();
                    resolve(this.image);
                };
                this.image.src = e.target.result;
            };
            reader.onerror = reject;
            reader.readAsDataURL(file);
        });
    },

    bindEvents() {
        if (!this.canvas) return;
        
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('mouseleave', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            this.zoom = Math.max(0.5, Math.min(2, this.zoom + (e.deltaY > 0 ? -0.1 : 0.1)));
            this.render();
        });

        document.addEventListener('keydown', (e) => {
            if (document.getElementById('manualAnnotationEditor')?.style?.display === 'none') return;
            
            if (e.key === 'd' || e.key === 'D') this.setMode('draw');
            if (e.key === 'v' || e.key === 'V') this.setMode('select');
            if (e.key === 'Delete' || e.key === 'Backspace') this.deleteSelected();
            if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); this.undo(); }
            if ((e.ctrlKey || e.metaKey) && e.key === 'y') { e.preventDefault(); this.redo(); }
            if (e.key >= '1' && e.key <= '6') {
                const classes = ['person', 'car', 'bicycle', 'dog', 'cat', 'object'];
                this.setClass(classes[e.key - 1]);
            }
        });
    },

    getPos(e) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
            y: (e.clientY - rect.top) * (this.canvas.height / rect.height)
        };
    },

    onMouseDown(e) {
        const pos = this.getPos(e);
        
        if (this.mode === 'draw') {
            this.isDrawing = true;
            this.dragStart = pos;
            this.currentBox = { class: this.currentClass, x: pos.x, y: pos.y, w: 0, h: 0 };
        } else if (this.mode === 'select') {
            const idx = this.getBoxAt(pos);
            if (idx !== -1) {
                this.selected = e.shiftKey ? [...this.selected, idx] : [idx];
            } else if (!e.shiftKey) {
                this.selected = [];
            }
            this.render();
            this.updateLayerPanel();
        }
        this.updateUI();
    },

    onMouseMove(e) {
        if (!this.isDrawing || !this.currentBox) return;
        const pos = this.getPos(e);
        this.currentBox.w = pos.x - this.dragStart.x;
        this.currentBox.h = pos.y - this.dragStart.y;
        this.render();
    },

    onMouseUp(e) {
        if (!this.isDrawing || !this.currentBox) {
            this.isDrawing = false;
            return;
        }

        // Normalize
        if (this.currentBox.w < 0) {
            this.currentBox.x += this.currentBox.w;
            this.currentBox.w = Math.abs(this.currentBox.w);
        }
        if (this.currentBox.h < 0) {
            this.currentBox.y += this.currentBox.h;
            this.currentBox.h = Math.abs(this.currentBox.h);
        }

        // Add if valid
        if (this.currentBox.w > 10 && this.currentBox.h > 10) {
            this.annotations.push({
                class: this.currentBox.class,
                x: this.currentBox.x,
                y: this.currentBox.y,
                w: this.currentBox.w,
                h: this.currentBox.h
            });
            this.saveState();
        }

        this.currentBox = null;
        this.isDrawing = false;
        this.render();
        this.updateLayerPanel();
        this.updateUI();
    },

    getBoxAt(pos) {
        for (let i = this.annotations.length - 1; i >= 0; i--) {
            const b = this.annotations[i];
            if (pos.x >= b.x && pos.x <= b.x + b.w && pos.y >= b.y && pos.y <= b.y + b.h) {
                return i;
            }
        }
        return -1;
    },

    render() {
        if (!this.ctx || !this.image) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);

        // Draw annotations
        this.annotations.forEach((box, i) => {
            const isSelected = this.selected.includes(i);
            this.ctx.strokeStyle = isSelected ? '#00ffff' : '#00ff00';
            this.ctx.lineWidth = isSelected ? 3 : 2;
            this.ctx.strokeRect(box.x, box.y, box.w, box.h);

            // Label
            this.ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
            this.ctx.fillRect(box.x, box.y - 16, Math.max(50, this.ctx.measureText(box.class).width + 10), 16);
            this.ctx.fillStyle = '#000';
            this.ctx.font = '11px sans-serif';
            this.ctx.fillText(box.class, box.x + 4, box.y - 4);

            // Index
            this.ctx.fillStyle = isSelected ? '#00ffff' : '#00ff00';
            this.ctx.fillRect(box.x - 2, box.y - 2, 14, 14);
            this.ctx.fillStyle = '#000';
            this.ctx.fillText(i.toString(), box.x + 3, box.y + 8);
        });

        // Draw current box
        if (this.currentBox) {
            this.ctx.strokeStyle = '#ffff00';
            this.ctx.lineWidth = 2;
            this.ctx.setLineDash([5, 3]);
            this.ctx.strokeRect(this.currentBox.x, this.currentBox.y, this.currentBox.w, this.currentBox.h);
            this.ctx.setLineDash([]);
        }
    },

    setMode(mode) {
        this.mode = mode;
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
    },

    setClass(name) {
        this.currentClass = name;
        document.getElementById('classNameInput').value = name;
        document.getElementById('currentClassDisplay').textContent = name;
        document.querySelectorAll('.quick-class-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.class === name);
        });
    },

    saveState() {
        this.history = this.history.slice(0, this.historyIdx + 1);
        this.history.push(JSON.stringify(this.annotations));
        if (this.history.length > 30) this.history.shift();
        else this.historyIdx++;
    },

    undo() {
        if (this.historyIdx > 0) {
            this.historyIdx--;
            this.annotations = JSON.parse(this.history[this.historyIdx]);
            this.render();
            this.updateLayerPanel();
            this.updateUI();
        }
    },

    redo() {
        if (this.historyIdx < this.history.length - 1) {
            this.historyIdx++;
            this.annotations = JSON.parse(this.history[this.historyIdx]);
            this.render();
            this.updateLayerPanel();
            this.updateUI();
        }
    },

    deleteSelected() {
        if (this.selected.length === 0) return;
        this.selected.sort((a, b) => b - a).forEach(i => this.annotations.splice(i, 1));
        this.selected = [];
        this.saveState();
        this.render();
        this.updateLayerPanel();
        this.updateUI();
    },

    clearAll() {
        if (!confirm('Alle Annotationen löschen?')) return;
        this.annotations = [];
        this.selected = [];
        this.saveState();
        this.render();
        this.updateLayerPanel();
        this.updateUI();
    },

    updateUI() {
        document.getElementById('annotationCount').textContent = this.annotations.length;
        document.getElementById('selectedCount').textContent = this.selected.length;
        document.getElementById('saveAnnotationsBtn').disabled = this.annotations.length === 0;
    },

    updateLayerPanel() {
        const list = document.getElementById('annotationList');
        if (!list) return;

        if (this.annotations.length === 0) {
            list.innerHTML = '<div class="annotation-help">Keine Annotationen</div>';
            return;
        }

        list.innerHTML = this.annotations.map((box, i) => `
            <div class="layer-item ${this.selected.includes(i) ? 'selected' : ''}" 
                 onclick="AnnEditor.selectLayer(${i}, event)">
                <span class="layer-class">${box.class}</span>
                <span class="layer-coords">[${Math.round(box.x)},${Math.round(box.y)} ${Math.round(box.w)}×${Math.round(box.h)}]</span>
                <button class="layer-action-btn delete" onclick="event.stopPropagation(); AnnEditor.deleteLayer(${i})">✕</button>
            </div>
        `).join('');
    },

    selectLayer(i, e) {
        this.selected = e.shiftKey ? [...this.selected, i] : [i];
        this.render();
        this.updateLayerPanel();
        this.updateUI();
    },

    deleteLayer(i) {
        this.annotations.splice(i, 1);
        if (this.selected.includes(i)) this.selected = this.selected.filter(x => x !== i);
        this.saveState();
        this.render();
        this.updateLayerPanel();
        this.updateUI();
    },

    getAnnotations() {
        return this.annotations.map(box => ({
            class: box.class,
            bbox: [box.x, box.y, box.x + box.w, box.y + box.h],
            bbox_normalized: [
                box.x / this.canvas.width,
                box.y / this.canvas.height,
                (box.x + box.w) / this.canvas.width,
                (box.y + box.h) / this.canvas.height
            ],
            confidence: 1.0
        }));
    }
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    AnnEditor.init('annotationCanvas');
});

// Global functions
function setEditorMode(mode) { AnnEditor.setMode(mode); }
function undo() { AnnEditor.undo(); }
function redo() { AnnEditor.redo(); }
function deleteSelected() { AnnEditor.deleteSelected(); }
function clearAllAnnotations() { AnnEditor.clearAll(); }
function toggleAllLayers() {}
function expandAllLayers() {}

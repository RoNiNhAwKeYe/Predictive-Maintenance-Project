/**
 * Predictive Maintenance Dashboard - Frontend
 * Real-time acoustic sensor monitoring with CNN-LSTM hybrid predictions
 */

// ============ CONFIGURATION ============
const API_BASE_URL = 'http://localhost:5000';
const CHARTS = {};
let recordedAudioChunks = [];
let mediaRecorder = null;
let audioContext = null;
let predictionHistory = [];
let sensorStreamActive = false;
let predictionInProgress = false;
const MAX_HISTORY = 20;

// ============ INITIALIZATION ============
document.addEventListener('DOMContentLoaded', async () => {
    console.log('Dashboard initializing...');
    
    // Initialize audio context
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize charts
    initializeCharts();
    
    // Check system status
    await checkSystemStatus();
    
    // Start automatic sensor input capture
    await initSensorStream();
    
    // Load initial data
    updateTimestamp();
    setInterval(updateTimestamp, 1000);
    
    // Log system ready
    addLog('Dashboard initialized successfully', 'success');
});

// ============ EVENT LISTENERS ============
function setupEventListeners() {
    // Test buttons
    document.getElementById('btn-test-healthy')?.addEventListener('click', () => testPrediction('healthy'));
    document.getElementById('btn-test-faulty')?.addEventListener('click', () => testPrediction('faulty'));
    
    // File upload
    const fileInput = document.getElementById('audio-file');
    fileInput?.addEventListener('change', (e) => {
        document.getElementById('btn-predict-file').disabled = !e.target.files.length;
    });
    document.getElementById('btn-predict-file')?.addEventListener('click', predictFile);
    
    // Logs
    document.getElementById('btn-clear-logs')?.addEventListener('click', clearLogs);
    
    // Navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', (e) => {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
        });
    });
}

// ============ API FUNCTIONS ============
async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/status`);
        const data = await response.json();
        
        if (data.model_loaded) {
            updateSystemIndicator(true);
            document.getElementById('model-status').textContent = '✓ Loaded';
            addLog('Model successfully loaded', 'success');
        } else {
            updateSystemIndicator(false);
            document.getElementById('model-status').textContent = '⚠ Initializing...';
            addLog('Model initializing...', 'warning');
        }
    } catch (error) {
        console.error('System status check failed:', error);
        updateSystemIndicator(false);
        document.getElementById('model-status').textContent = '✗ Error';
        addLog('Failed to connect to backend', 'error');
    }
}

async function testPrediction(faultType) {
    const button = event.target;
    button.disabled = true;
    
    try {
        addLog(`Testing ${faultType} bearing scenario...`, 'warning');
        
        const response = await fetch(`${API_BASE_URL}/api/predict/synthetic`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                fault_type: faultType,
                duration: 0.1,
                sampling_rate: 20000
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            addLog(`Error: ${result.error}`, 'error');
        } else {
            updatePredictionDisplay(result);
            addLog(`Prediction: ${result.status} (${(result.confidence * 100).toFixed(1)}%)`, 'success');
        }
    } catch (error) {
        console.error('Prediction failed:', error);
        addLog(`Request failed: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
    }
}

// ============ AUDIO RECORDING ============
async function initSensorStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        addLog('Browser does not support microphone access', 'error');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        sensorStreamActive = true;
        recordedAudioChunks = [];

        mediaRecorder.ondataavailable = async (event) => {
            if (event.data && event.data.size > 0) {
                await sendSensorChunk(event.data);
            }
        };

        mediaRecorder.onstart = () => {
            document.getElementById('recording-status').textContent = '🔴 Live sensor stream active';
            document.getElementById('recording-status').className = 'recording-status recording';
            addLog('Automatic sensor stream started', 'success');
        };

        mediaRecorder.onstop = () => {
            document.getElementById('recording-status').textContent = '✓ Sensor stream paused';
            document.getElementById('recording-status').className = 'recording-status ready';
            addLog('Automatic sensor stream stopped', 'warning');
        };

        mediaRecorder.start(3000); // emit data every 3 seconds

        document.getElementById('btn-start-recording').disabled = true;
        document.getElementById('btn-stop-recording').disabled = false;
        document.getElementById('btn-predict-recording').disabled = true;
    } catch (error) {
        console.error('Sensor stream error:', error);
        addLog(`Automatic sensor capture failed: ${error.message}`, 'error');
    }
}

async function sendSensorChunk(audioBlob) {
    if (predictionInProgress) {
        return;
    }

    predictionInProgress = true;
    try {
        const arrayBuffer = await audioBlob.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        const base64 = btoa(binary);

        addLog('Auto-sending sensor data for prediction...', 'info');

        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                audio_base64: base64,
                sampling_rate: 20000
            })
        });

        const result = await response.json();
        if (result.error) {
            addLog(`Auto prediction error: ${result.error}`, 'error');
        } else {
            updatePredictionDisplay(result);
            addLog(`Auto prediction: ${result.status} (${(result.confidence * 100).toFixed(1)}%)`, 'success');
        }
    } catch (error) {
        console.error('Sensor prediction failed:', error);
        addLog(`Sensor prediction failed: ${error.message}`, 'error');
    } finally {
        predictionInProgress = false;
    }
}

async function startRecording() {
    addLog('Automatic sensor stream is active - manual recording disabled', 'info');
}

async function stopRecording() {
    addLog('Automatic sensor stream is active - manual recording disabled', 'info');
}

async function predictRecording() {
    addLog('Automatic sensor stream is active - manual prediction disabled', 'info');
}

// ============ FILE UPLOAD PREDICTION ============
async function predictFile() {
    const fileInput = document.getElementById('audio-file');
    if (!fileInput.files.length) {
        addLog('No file selected', 'error');
        return;
    }
    
    const button = event.target;
    button.disabled = true;
    
    try {
        const file = fileInput.files[0];
        addLog(`Processing file: ${file.name}...`, 'warning');
        
        const arrayBuffer = await file.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        const base64 = btoa(binary);
        
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                audio_base64: base64,
                sampling_rate: 20000
            })
        });
        
        const result = await response.json();
        
        if (result.error) {
            addLog(`Error: ${result.error}`, 'error');
        } else {
            updatePredictionDisplay(result);
            addLog(`File prediction: ${result.status} (${(result.confidence * 100).toFixed(1)}%)`, 'success');
        }
    } catch (error) {
        console.error('File prediction failed:', error);
        addLog(`File processing failed: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
    }
}

// ============ DISPLAY UPDATES ============
function updatePredictionDisplay(result) {
    const isFaulty = result.fault_probability >= 0.5;
    const statusText = isFaulty ? 'FAULTY' : 'HEALTHY';
    const statusColor = isFaulty ? 'red-text' : 'green-text';
    
    // Update prediction
    const predElement = document.getElementById('prediction-result');
    predElement.textContent = statusText;
    predElement.className = `big-value ${statusColor}`;
    
    // Update confidence
    const confidence = Math.round(result.fault_probability * 100);
    document.getElementById('confidence-percent').textContent = `${confidence}%`;
    document.getElementById('confidence-bar').style.width = `${confidence}%`;
    
    // Update RUL if available
    if (result.rul_hours !== undefined) {
        updateRULDisplay(result.rul_hours, result.rul_status);
    }
    
    // Update machine status
    const machineStatus = document.getElementById('machine-status');
    machineStatus.innerHTML = `
        <div class="big-value ${statusColor}">${statusText}</div>
        <p class="status-description">${isFaulty ? 'Fault detected - requires maintenance' : 'Operating normally'}</p>
    `;
    
    // Update badges and alerts
    updateBadgesAndAlerts(isFaulty, confidence);
    
    // Update charts
    addToPredictionHistory(result);
    updateCharts();
    
    // Generate synthetic waveform visualization
    generateWaveformChart(isFaulty);
}

function updateRULDisplay(rulHours, rulStatus) {
    // Update RUL display on dashboard
    // Args:
    //   rulHours: Remaining useful life in hours
    //   rulStatus: Status string (CRITICAL, WARNING, GOOD)
    
    const rulElement = document.getElementById('rul-hours');
    const rulBadge = document.getElementById('rul-badge');
    const rulBar = document.getElementById('rul-bar');
    const rulPercentage = document.getElementById('rul-percentage');
    
    // Display RUL hours
    rulElement.textContent = rulHours.toFixed(1);
    
    // Determine color and percentage based on RUL hours
    let percentFull = 0;
    let badgeClass = 'success';
    let badgeText = 'GOOD';
    
    if (rulStatus === 'CRITICAL') {
        percentFull = (rulHours / 50) * 100;  // Max 50 hours for critical
        badgeClass = 'danger';
        badgeText = 'CRITICAL';
        rulBar.style.background = 'linear-gradient(90deg, var(--color-danger) 0%, var(--color-danger) 100%)';
    } else if (rulStatus === 'WARNING') {
        percentFull = (rulHours / 50) * 100;
        badgeClass = 'warning';
        badgeText = 'WARNING';
        rulBar.style.background = 'linear-gradient(90deg, var(--color-warning) 0%, var(--color-warning) 100%)';
    } else {
        // GOOD
        percentFull = Math.min((rulHours / 100) * 100, 100);
        badgeClass = 'success';
        badgeText = 'GOOD';
        rulBar.style.background = 'linear-gradient(90deg, var(--color-success) 0%, var(--color-warning) 100%)';
    }
    
    // Update badge
    rulBadge.textContent = badgeText;
    rulBadge.className = `card-badge ${badgeClass}`;
    
    // Update progress bar
    rulBar.style.width = `${Math.min(percentFull, 100)}%`;
    
    // Update percentage text
    const displayPercent = Math.min((rulHours / 100) * 100, 100);
    rulPercentage.textContent = `${displayPercent.toFixed(0)}% remaining`;
}

function updateBadgesAndAlerts(isFaulty, confidence) {
    const statusBadge = document.getElementById('status-badge');
    const predictionBadge = document.getElementById('prediction-badge');
    const alertSection = document.getElementById('alerts-section');
    const alertMessage = document.getElementById('alert-message');
    
    if (isFaulty) {
        statusBadge.textContent = 'ALERT';
        statusBadge.className = 'card-badge danger';
        
        predictionBadge.textContent = 'FAULTY';
        predictionBadge.className = 'card-badge danger';
        
        alertSection.style.borderLeft = '4px solid #ef4444';
        alertMessage.textContent = `⚠️ BEARING FAULT DETECTED with ${confidence}% confidence - Immediate inspection required!`;
        alertMessage.style.color = '#dc2626';
        
        updateSystemIndicator(false);
    } else {
        statusBadge.textContent = 'NORMAL';
        statusBadge.className = 'card-badge success';
        
        predictionBadge.textContent = 'HEALTHY';
        predictionBadge.className = 'card-badge success';
        
        alertSection.style.borderLeft = '4px solid #10b981';
        alertMessage.textContent = `✓ System healthy - No faults detected`;
        alertMessage.style.color = '#059669';
        
        updateSystemIndicator(true);
    }
}

function updateSystemIndicator(healthy) {
    const indicator = document.getElementById('system-status-indicator');
    const statusText = document.getElementById('system-status-text');
    
    if (healthy) {
        indicator.classList.add('healthy');
        statusText.textContent = 'System Healthy';
    } else {
        indicator.classList.remove('healthy');
        statusText.textContent = 'Alert Active';
    }
}

function addToPredictionHistory(result) {
    predictionHistory.push({
        timestamp: new Date(),
        confidence: result.fault_probability,
        status: result.status,
        rul_hours: result.rul_hours || null
    });
    
    if (predictionHistory.length > MAX_HISTORY) {
        predictionHistory.shift();
    }
}

function updateTimestamp() {
    const now = new Date().toLocaleTimeString();
    document.getElementById('last-update').textContent = now;
}

// ============ CHARTS ============
function initializeCharts() {
    // Waveform Chart
    const waveCtx = document.getElementById('waveform-chart');
    if (waveCtx) {
        CHARTS.waveform = new Chart(waveCtx, {
            type: 'line',
            data: {
                labels: Array.from({length: 100}, (_, i) => i),
                datasets: [{
                    label: 'Acoustic Amplitude',
                    data: Array(100).fill(0),
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.3,
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        min: -1,
                        max: 1
                    }
                }
            }
        });
    }
    
    // History Chart
    const historyCtx = document.getElementById('history-chart');
    if (historyCtx) {
        CHARTS.history = new Chart(historyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Fault Confidence (%)',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.3,
                    borderWidth: 2,
                    fill: true,
                    pointRadius: 4,
                    pointBackgroundColor: '#f59e0b'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        min: 0,
                        max: 100
                    }
                }
            }
        });
    }
}

function generateWaveformChart(isFaulty) {
    if (!CHARTS.waveform) return;
    
    // Generate synthetic waveform data
    const samples = 100;
    const data = [];
    
    if (isFaulty) {
        // Faulty: multiple frequencies and impulses
        for (let i = 0; i < samples; i++) {
            const t = i / samples;
            const base = 0.3 * Math.sin(2 * Math.PI * 3 * t);
            const harmonic = 0.2 * Math.sin(2 * Math.PI * 6 * t);
            const impulse = (i % 15 === 0) ? 0.4 : 0;
            data.push(base + harmonic + impulse + Math.random() * 0.1);
        }
    } else {
        // Healthy: clean sinusoid with low noise
        for (let i = 0; i < samples; i++) {
            const t = i / samples;
            const signal = 0.3 * Math.sin(2 * Math.PI * 2 * t);
            data.push(signal + Math.random() * 0.05);
        }
    }
    
    CHARTS.waveform.data.datasets[0].data = data;
    CHARTS.waveform.update();
}

function updateCharts() {
    if (!CHARTS.history) return;
    
    CHARTS.history.data.labels = predictionHistory.map((_, i) => i + 1);
    CHARTS.history.data.datasets[0].data = predictionHistory.map(
        p => Math.round(p.confidence * 100)
    );
    CHARTS.history.update();
}

// ============ LOGGING ============
function addLog(message, type = 'info') {
    const container = document.getElementById('logs-container');
    const entry = document.createElement('p');
    
    const timestamp = new Date().toLocaleTimeString();
    entry.textContent = `[${timestamp}] ${message}`;
    entry.className = `log-entry ${type}`;
    
    container.insertBefore(entry, container.firstChild);
    
    // Keep only last 50 logs
    while (container.children.length > 50) {
        container.removeChild(container.lastChild);
    }
    
    console.log(`[${type.toUpperCase()}] ${message}`);
}

function clearLogs() {
    const container = document.getElementById('logs-container');
    container.innerHTML = '<p class="log-entry">Logs cleared</p>';
    addLog('Logs cleared by user', 'info');
}
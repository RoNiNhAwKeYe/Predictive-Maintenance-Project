# 🚀 Quick Reference Card

## System Status Check

### Backend Running?
```bash
# Test API is up
curl http://localhost:5000/api/status

# Expected response:
# {"status": "OK", "model_loaded": true}
```

### Model Loaded?
```bash
# Check if model is available
curl http://localhost:5000/api/health

# Expected response:
# {"status": "healthy", "model_available": true, "model_type": "CNN-LSTM"}
```

---

## API Endpoints

### 1. Predict from Audio (POST)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/wav" \
  --data-binary @your_audio_file.wav

# Response:
{
  "status": "FAULTY",
  "confidence": 0.85,
  "rul_hours": 5.2,
  "rul_status": "CRITICAL",
  "fault_probability": 0.85,
  "health_probability": 0.15
}
```

### 2. Predict from Synthetic Data (Quick Test)
```bash
curl -X POST http://localhost:5000/api/predict/synthetic \
  -H "Content-Type: application/json" \
  -d '{
    "fault_type": "faulty",
    "duration": 0.1,
    "sampling_rate": 20000
  }'

# Response includes RUL_HOURS and RUL_STATUS
```

### 3. Check System Status
```bash
curl http://localhost:5000/api/status

# Response: {"status": "OK", "model_loaded": true, "timestamp": "..."}
```

### 4. Get System Health
```bash
curl http://localhost:5000/api/health

# Response: {"status": "healthy", "model_available": true}
```

---

## Dashboard Features

### Buttons & Controls

| Button | Does What | Result |
|--------|-----------|--------|
| **Test Healthy** | Generates synthetic healthy bearing audio | Shows HEALTHY + 50-100h RUL |
| **Test Faulty** | Generates synthetic faulty bearing audio | Shows FAULTY + 1-10h RUL |
| **Start Recording** | Activates microphone input | Records audio for prediction |
| **Stop** | Stops recording | Prepares audio for prediction |
| **Predict** | Sends recorded audio to backend | Shows fault % + RUL hours |
| **Upload File** | File picker for audio samples | Sends file to backend |

### Display Cards

#### Machine Status
- Shows overall health
- Green (✓) = HEALTHY
- Red (✗) = FAULTY

#### CNN Prediction
- **Confidence**: Probability of fault (0-100%)
- **Status**: HEALTHY or FAULTY
- **Alert**: Visual indicator

#### Remaining Useful Life (NEW)
- **Hours**: 0-200+ hours remaining
- **Status**: GOOD/WARNING/CRITICAL
- **Progress Bar**: Visual representation
- **Percentage**: % of life remaining

#### Model Info
- Model type loaded
- Input specifications
- Status indicator

#### Event Logs
- Timestamped entries
- Color coded (green=success, red=error, yellow=info)
- Sortable & clearable

---

## Desktop Dashboard Layout

```
┌──────────────────────────────────────────────────┐
│ 🎙️  Predictive Maintenance | Server: Online    │
├──────────────────────────────────────────────────┤
│                                                  │
│  STATUS: ✓ HEALTHY                             │
│  ─────────────────────────────────────────── │
│                                                  │
│  ┌──────────────────┐ ┌──────────────────┐     │
│  │ MACHINE STATUS   │ │ CNN PREDICTION   │     │
│  │ ✓ HEALTHY        │ │ ✓ HEALTHY 95%   │     │
│  └──────────────────┘ └──────────────────┘     │
│                                                  │
│  ┌──────────────────┐ ┌──────────────────┐     │
│  │ REMAINING LIFE   │ │ MODEL STATUS     │     │
│  │ 78.5 hours       │ │ ✓ Loaded         │     │
│  │ ✓ GOOD           │ │ CNN-LSTM Hybrid  │     │
│  └──────────────────┘ └──────────────────┘     │
│                                                  │
│  CONTROLS:                                      │
│  [Test Healthy] [Test Faulty] [Upload] [Pred] │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│  [Start Recording] [Stop Recording]            │
│                                                  │
│  ┌──────────────────┐ ┌──────────────────┐     │
│  │ Waveform         │ │ Prediction       │     │
│  │ [Chart]          │ │ Trend [Chart]    │     │
│  └──────────────────┘ └──────────────────┘     │
│                                                  │
│  📋 RECENT EVENTS                              │
│  ✓ 14:35:22 Prediction: HEALTHY, RUL: 78.5h  │
│  ✓ 14:35:18 File uploaded successfully        │
│  ✓ 14:35:10 Backend connected successfully    │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## File Structure

```
Predictive-Maintenance-Project/
│
├── run.bat                          ← START HERE (Windows)
├── run.sh                           ← START HERE (Mac/Linux)
│
├── 📋 Documentation
│   ├── README.md                    ← Full system docs
│   ├── SETUP_GUIDE.md              ← Quick setup steps
│   ├── CNN_LSTM_RUL_MODEL.md       ← Model architecture
│   ├── TRAINING_GUIDE.md           ← How to train custom model
│   ├── RUL_ENHANCEMENT_SUMMARY.md  ← What's new in v1.1
│   └── QUICK_REFERENCE.md          ← This file!
│
├── 🔧 Backend
│   ├── backend/
│   │   ├── app.py                  ← Flask REST API server
│   │   ├── requirements.txt         ← Python dependencies
│   │   ├── best_model.keras        ← Trained CNN-LSTM model (auto-loaded)
│   │   ├── audio_sensor_simulator.py ← Generate test data
│   │   ├── CNN (1).ipynb           ← Training notebook
│   │   └── CNN_(1).ipynb           ← Alternative training notebook
│
└── 🎨 Frontend
    └── frontend/
        ├── index.html              ← Dashboard UI
        ├── script.js               ← API integration & interactions
        ├── style.css               ← Styling & responsive design
        └── README.md               ← Frontend documentation
```

---

## RUL Status Meanings

### GOOD 🟢 (RUL > 20 hours)
- Bearing is healthy
- Continue normal operation
- Routine monitoring sufficient
- **Action**: None required

### WARNING 🟡 (RUL 5-20 hours)
- Bearing showing early degradation
- Increased monitoring recommended
- Plan maintenance within days
- **Action**: Schedule maintenance appointment

### CRITICAL 🔴 (RUL ≤ 5 hours)
- Bearing severely degraded
- Failure imminent
- **Action**: Stop operation, replace immediately

---

## Video Tutorials

### Basic Usage (2 min)
1. Start server with run.bat/run.sh
2. Open http://localhost:5000
3. Click "Test Healthy Bearing"
4. View results: Status + RUL hours + Alert

### Live Recording (2 min)
1. Click "Start Recording"
2. Record 1-2 seconds of sound
3. Click "Stop Recording"
4. Click "Predict"
5. See fault probability + RUL estimate

### File Upload (2 min)
1. Click "Upload File"
2. Select an audio file (WAV, MP3, OGG, FLAC)
3. Click "Predict"
4. Results shown instantly

### Training Custom Model (30 min)
1. Open `backend/CNN (1).ipynb` in Jupyter
2. Add your training data
3. Modify hyperparameters if needed
4. Run all cells
5. Model saves as `best_model.keras`
6. Restart Flask server
7. New model now in use!

---

## Troubleshooting Flowchart

```
System Not Working?
│
├─ Server won't start
│  └─ ✓ Check Python 3.8+ installed
│  └─ ✓ Check dependencies: pip install -r backend/requirements.txt
│  └─ ✓ Check if port 5000 is free
│
├─ Dashboard shows "Server Offline"
│  └─ ✓ Is Flask running? Check terminal
│  └─ ✓ Try restarting: Stop + start again
│  └─ ✓ Check firewall settings
│
├─ Predictions are all HEALTHY
│  └─ ✓ Check if model is loaded (see logs)
│  └─ ✓ Try "Test Faulty Bearing" button
│  └─ ✓ Check model file exists: backend/best_model.keras
│
├─ RUL shows as None or 0
│  └─ ✓ Model needs 2 outputs (check backend logs)
│  └─ ✓ Train new model with RUL labels
│  └─ ✓ Restart Flask server
│
├─ Microphone not recording
│  └─ ✓ Grant browser microphone permission
│  └─ ✓ Use HTTPS or localhost (not IP address)
│  └─ ✓ Check browser console for errors (F12)
│
└─ Predictions are slow
   └─ ✓ First prediction always slower (warmup)
   └─ ✓ LSTM is slower than simple CNN (~200ms vs ~50ms)
   └─ ✓ Close other applications to free RAM
```

---

## Common Commands

### Start Server
```bash
# Windows
run.bat

# Mac/Linux
./run.sh
```

### View Logs
```bash
# The run.bat/run.sh scripts show logs automatically
# Or check the terminal where Flask is running
```

### Generate Test Data
```bash
cd backend
python audio_sensor_simulator.py
# Creates test_audio/ folder with synthetic samples
```

### Test API Endpoint
```bash
# Windows
curl http://localhost:5000/api/status

# Mac/Linux
curl http://localhost:5000/api/status
```

### Stop Server
```bash
# Press Ctrl+C in the terminal where Flask is running
```

---

## Performance Specs

| Metric | Value |
|--------|-------|
| **API Response Time** | 100-200ms (CNN-LSTM) |
| **First Prediction** | ~1-2 seconds (model warmup) |
| **Browser Compatibility** | Chrome, Firefox, Safari, Edge |
| **Mobile Support** | Yes (responsive design) |
| **Max File Size** | ~10MB audio |
| **Supported Formats** | WAV, MP3, OGG, FLAC |
| **Microphone Latency** | 50-100ms |

---

## Configuration Files

### backend/app.py
Key variables to customize:
```python
MODEL_PATH = 'best_model.keras'      # Where model is loaded from
SAMPLING_RATE = 20000                # Audio sample rate (Hz)
SPECTROGRAM_SIZE = (64, 64)          # Spectrogram dimensions
SEQUENCE_LENGTH = 10                 # LSTM sequence length
FAULT_THRESHOLD = 0.5                # Decision threshold
```

### frontend/script.js
Key variables to customize:
```javascript
const API_URL = 'http://localhost:5000'  // Backend address
const CHART_MAX_POINTS = 20              // History chart size
const RECORDING_SAMPLE_RATE = 20000      // Microphone sample rate
const RUL_BASE_HOURS = 100               // Max RUL for progress bar
```

### frontend/style.css
Key variables to customize:
```css
--color-primary: #2196F3;       /* Primary blue */
--color-success: #4CAF50;       /* Success green */
--color-warning: #FF9800;       /* Warning orange */
--color-danger: #F44336;        /* Danger red */
--font-family: 'Arial', sans-serif;
--border-radius: 12px;
```

---

## Next Steps

### ✓ System is ready! Choose what to do next:

📚 **Learn More**
- Read [README.md](README.md) for complete documentation
- Read [CNN_LSTM_RUL_MODEL.md](CNN_LSTM_RUL_MODEL.md) for model details
- Read [TRAINING_GUIDE.md](TRAINING_GUIDE.md) to train custom model

🎓 **Train Custom Model**
- Gather your bearing audio data + RUL labels
- Follow [TRAINING_GUIDE.md](TRAINING_GUIDE.md)
- Place trained model in `backend/best_model.keras`
- Restart server

🚀 **Deploy to Production**
- Use Docker: `docker build -t bearing-fault . && docker run -p 5000:5000 bearing-fault`
- Deploy to cloud: AWS ECS, Google Cloud Run, Azure Container Instances
- Set up HTTPS/SSL certificate

🔌 **Connect Real Sensors**
- Modify `/api/predict` endpoint to accept sensor data
- Stream audio from embedded device
- Log predictions to database

---

**Version**: 1.1 (with CNN-LSTM & RUL)  
**Last Updated**: April 2024  
**Support**: Check README.md for troubleshooting

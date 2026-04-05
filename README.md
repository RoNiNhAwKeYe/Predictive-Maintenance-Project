# 🔧 Predictive Maintenance - Acoustic Bearing Fault Detection

A comprehensive system for **real-time acoustic sensor monitoring** using a CNN-LSTM hybrid model to detect bearing faults and estimate remaining useful life before failure.

## 📋 Project Overview

**Predictive-Maintenance-Project** combines deep learning (CNN-LSTM) with a modern web dashboard to analyze acoustic signals from industrial bearings:

- **Backend**: Flask REST API with TensorFlow/Keras CNN-LSTM hybrid model
- **Model**: TimeDistributed CNN + LSTM network for fault detection and RUL prediction
- **Input**: 20 kHz acoustic sensor data (raw waveforms or spectrograms)
- **Output**: Fault probability and RUL estimate
- **Frontend**: Modern, responsive dashboard with real-time predictions
- **Features**: Live microphone input, file upload, synthetic testing, prediction history

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** (tested on Python 3.11)
- **pip** or **conda** package manager
- **Git** (optional)
- **Modern browser** (Chrome, Firefox, Edge, Safari)

### Installation

#### 1. **Clone/Navigate to Project**

```bash
cd "d:\डेस्कटॉप\Projects\Deep Learning Project\Predictive-Maintenance-Project"
```

#### 2. **Create Python Virtual Environment**

**Windows (PowerShell/CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. **Install Dependencies**

```bash
pip install -r backend/requirements.txt
```

> ⚠️ **Note**: TensorFlow installation can take 5-10 minutes

---

## 📂 Project Structure

```
Predictive-Maintenance-Project/
├── backend/
│   ├── app.py                         # Flask app with API endpoints
│   ├── CNN (1).ipynb                  # CNN model training notebook
│   ├── CNN_(1).ipynb                  # CNN model training notebook
│   └── requirements.txt                # Python dependencies
│
├── frontend/
│   ├── index.html                     # Dashboard UI
│   ├── style.css                      # Modern styling (dark/responsive)
│   ├── script.js                      # API integration & interactions
│   └── README.md                      # Frontend documentation
│
├── README.md                          # This file
└── CNN_MODEL_SUMMARY.md              # Detailed model documentation

```

---

## ▶️ Running the System

### Step 1: Start Backend Server

```bash
cd backend
python app.py
```

**Expected Output:**
```
============================================================
Predictive Maintenance Backend
============================================================
Starting server on http://localhost:5000
API Endpoints:
  GET  /api/status - System status
  GET  /api/health - Health check
  POST /api/predict - Make prediction on audio
  POST /api/predict/synthetic - Test with synthetic data
============================================================
```

### Step 2: Open Frontend in Browser

Open your browser and navigate to:
```
http://localhost:5000
```

You should see the **Predictive Maintenance Dashboard** with:
- ✓ System status indicator
- ✓ Real-time prediction display
- ✓ Control panel for testing
- ✓ Live recording capability
- ✓ Prediction history charts
- ✓ Event logs

---

## 🎯 Usage Guide

### 1. **Test with Synthetic Data** (Recommended First)

Useful for testing without real sensor data:

1. Click **"Test Healthy Bearing"** - System should show HEALTHY status
2. Click **"Test Faulty Bearing"** - System should show FAULTY status with alerts

### 2. **Live Microphone Recording**

Capture real acoustic data from your microphone:

1. Click **"Start Recording"** 
2. Hold microphone near your audio source (bearing, machinery, etc.)
3. Click **"Stop Recording"** when done
4. Click **"Predict"** to get instant classification

### 3. **Upload Audio File**

Analyze pre-recorded audio:

1. Click **"Choose File"** and select a `.wav` or `.mp3` file
2. Click **"Predict from File"**
3. System processes and displays results

### 4. **Monitor Prediction History**

- Real-time confidence trend chart shows all predictions
- Event logs show timestamps and details
- System alerts on fault detection

---

## 🧠 CNN-LSTM Hybrid Model Details

### Architecture
```
Input: Sequence of Spectrograms (Multi-timestep)
    ↓
TimeDistributed Conv2D(32) → BatchNorm → MaxPool(2×2)
    ↓
TimeDistributed Conv2D(64) → BatchNorm → MaxPool(2×2)
    ↓
TimeDistributed Conv2D(128) → BatchNorm → MaxPool(2×2)
    ↓
TimeDistributed Flatten → Dense(128, ReLU)
    ↓
LSTM(128, return_sequences=True)
    ↓
LSTM(64)
    ↓
├─ Fault Detection Branch → Dense(64) → Sigmoid(1)
└─ RUL Prediction Branch → Dense(64) → ReLU(1)
    ↓
Output: 
  - Fault Probability [0.0, 1.0]
  - Remaining Useful Life (hours)
```

### Input Specification
| Property | Value |
|----------|-------|
| **Audio Sample Rate** | 20 kHz |
| **Filter Range** | 500 - 9,000 Hz |
| **Spectrogram Size** | 64 × 64 pixels |
| **Type** | Grayscale (1 channel) |

### Output Specification
| Property | Value |
|----------|-------|
| **Fault Output** | Probability [0.0, 1.0] |
| **Fault Interpretation** | P(Fault) |
| **Fault Threshold** | 0.5 |
| **RUL Output** | Hours (float) |
| **RUL Interpretation** | Remaining hours until failure |
| **RUL Range** | 0.5 - 200+ hours |
| **RUL Status** | CRITICAL (≤5h), WARNING (5-20h), GOOD (>20h) |
| **Labels** | HEALTHY / FAULTY |

### Preprocessing Pipeline
1. **Handle NaN/Inf** - Linear interpolation
2. **Bandpass Filter** - 500-9,000 Hz (Butterworth, order 4)
3. **Z-Score Normalization** - Mean=0, Std=1
4. **Outlier Clipping** - Values beyond ±5σ clipped
5. **STFT** - FFT window 256 samples, 50% overlap
6. **Log Scale** - log(1 + power) compression
7. **Resize** - Bilinear interpolation to 64×64

---

## 🔌 API Endpoints

### 1. System Status
```
GET /api/status
Response: { "system": "...", "model_loaded": true, "timestamp": "..." }
```

### 2. Health Check
```
GET /api/health
Response: { "status": "healthy", "model_loaded": true }
```

### 3. Make Prediction (Audio)
```
POST /api/predict
Content-Type: application/json

Request:
{
  "audio_base64": "base64_encoded_wav_data",
  "sampling_rate": 20000
}

Response:
{
  "status": "HEALTHY|FAULTY",
  "confidence": 0.234,
  "alert_level": "NORMAL|CRITICAL",
  "fault_probability": 0.234,
  "health_probability": 0.766,
  "timestamp": "2024-...",
  "threshold": 0.5
}
```

### 4. Synthetic Test Prediction
```
POST /api/predict/synthetic
Content-Type: application/json

Request:
{
  "fault_type": "healthy|faulty",
  "duration": 0.1,
  "sampling_rate": 20000
}

Response: (Same as /api/predict)
```

---

## 🛠️ Configuration

### Backend Configuration
Edit `backend/app.py` to modify:

```python
# Model threshold
threshold = 0.5  # Change to adjust sensitivity

# Audio preprocessing
sr = 20000  # Sampling rate
duration = 0.1  # Duration in seconds
```

### Frontend Configuration
Edit `frontend/script.js` to modify:

```javascript
// Backend API URL
const API_BASE_URL = 'http://localhost:5000';

// Prediction history size
const MAX_HISTORY = 20;

// Audio recording settings
sampling_rate: 20000
```

---

## 📊 Training the Model (Optional)

To train your own CNN-LSTM hybrid model:

1. Open `backend/CNN (1).ipynb` in Jupyter
2. Follow the notebook to preprocess bearing data
3. Train the CNN-LSTM hybrid model on your dataset
4. Export trained weights: `model.save('best_model.keras')`
5. Place `best_model.keras` in the `backend/` directory
6. Restart `app.py` - it will auto-load your model

```python
# The app will look for:
model_path = 'best_model.keras'
```

---

## 🐛 Troubleshooting

### Problem: "Failed to connect to backend"

**Solution:**
1. Make sure Flask server is running: `python app.py` in `backend/` folder
2. Check if port 5000 is available: `netstat -ano | find "5000"` (Windows)
3. Try accessing `http://localhost:5000/api/status` directly

### Problem: "Model not loaded"

**Solution:**
1. Check `backend/app.py` logs for loading errors
2. Ensure TensorFlow is installed: `pip install tensorflow`
3. If using custom model, place `best_model.keras` in `backend/` directory
4. Restart Flask server

### Problem: "Recording not working"

**Solution:**
1. Grant microphone permission to your browser
2. Use HTTPS or localhost (browsers restrict microphone access)
3. Check browser console (F12) for JavaScript errors

### Problem: "CORS errors"

**Solution:**
Flask-CORS is configured. If issues persist:
```python
# In app.py, verify:
CORS(app)  # Should allow all origins for localhost
```

---

## 📈 Performance Metrics

Typical performance on test data:

| Metric | Value |
|--------|-------|
| **Accuracy** | ~95% |
| **Precision** | ~94% |
| **Recall** | ~96% |
| **AUC-ROC** | ~0.99 |
| **Inference Time** | ~50ms per sample |
| **Model Size** | ~12 MB |

---

## 🔐 Security Notes

- **Local Network Only**: Current setup is for localhost only
- **For Production**: 
  - Enable HTTPS
  - Add authentication
  - Deploy behind reverse proxy (nginx/Apache)
  - Validate all inputs
  - Rate limiting on API endpoints

---

## 📚 Documentation

Additional documentation:

- **[CNN_MODEL_SUMMARY.md](CNN_MODEL_SUMMARY.md)** - Detailed model architecture
- **[frontend/README.md](frontend/README.md)** - Frontend architecture
- **Jupyter Notebooks** - Model training and experimentation

---

## 🤝 Contributing

To extend this project:

1. **Improve Model**: Train on more bearing fault data
2. **Add Features**: Integration with IoT sensors, cloud deployment
3. **Dashboard**: Add more visualizations (spectrograms, FFT plots)
4. **Automation**: Schedule predictions, email alerts

---

## 📝 License

This project is created for educational and demonstration purposes.

---

## 🎓 Technical Stack

- **Backend**: Python, Flask, TensorFlow/Keras, SciPy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla), Chart.js
- **Data**: NumPy, SciPy (signal processing)
- **Model**: CNN-LSTM hybrid for fault detection and RUL prediction

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Jupyter notebooks for training details
3. Check browser console (F12) for errors
4. Verify all dependencies are installed

---

**Last Updated**: April 2024  
**Version**: 1.0.0

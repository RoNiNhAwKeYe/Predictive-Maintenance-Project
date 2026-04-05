# 🚀 Quick Setup Guide

## One-Minute Quick Start

### Windows Users:
1. Navigate to the project folder
2. Double-click **`run.bat`**
3. Wait for dependencies to install (first run only, 5-10 minutes)
4. Open browser to **http://localhost:5000**
5. Click "Test Healthy Bearing" or "Test Faulty Bearing"

### macOS/Linux Users:
```bash
cd Predictive-Maintenance-Project
chmod +x run.sh
./run.sh
```
Then open **http://localhost:5000**

---

## Dashboard Overview
```
┌─────────────────────────────────────────────────────┐
│  🎙️  Predictive Maintenance | Status: Healthy     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ⚠️  ALERTS: No issues detected                   │
│                                                     │
│  ┌──────────────────┐  ┌──────────────────┐       │
│  │  Machine Status  │  │  CNN Prediction  │       │
│  │   HEALTHY ✓      │  │   HEALTHY ✓      │       │
│  └──────────────────┘  └──────────────────┘       │
│                                                     │
│  ┌──────────────────┐  ┌──────────────────┐       │
│  │ Remaining Life   │  │  Model Status    │       │
│  │  78.5 hours      │  │  ✓ Loaded        │       │
│  │  GOOD ✓          │  │  CNN-LSTM Hybrid │       │
│  └──────────────────┘  └──────────────────┘       │
│                                                     │
│  ┌──────────────────────────────────────────┐     │
│  │  Control Panel                           │     │
│  │  [Test Healthy] [Test Faulty]            │     │
│  │  [Start Recording] [Stop] [Predict]      │     │
│  │  [Upload File] [Predict]                 │     │
│  └──────────────────────────────────────────┘     │
│                                                     │
│  ┌──────────────────┐  ┌──────────────────┐       │
│  │  Waveform Chart  │  │ Prediction Trend │       │
│  │  [Graph]         │  │  [Graph]         │       │
│  └──────────────────┘  └──────────────────┘       │
│                                                     │
│  📋 LOGS                                            │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.9 | 3.11+ |
| **RAM** | 4 GB | 8 GB+ |
| **Storage** | 500 MB | 2 GB (with dependencies) |
| **Browser** | Any modern | Chrome/Firefox |
| **GPU** | Not required | NVIDIA CUDA (faster inference) |

---

## Step-by-Step Setup

### Step 1: Install Python
Download from https://www.python.org/downloads/

**Important**: Check "Add Python to PATH" during installation

### Step 2: Navigate to Project
```bash
cd "d:\डेस्कटॉप\Projects\Deep Learning Project\Predictive-Maintenance-Project"
```

### Step 3: Run Setup Script

**Windows**:
```bash
run.bat
```

**macOS/Linux**:
```bash
./run.sh
```

This will:
1. ✓ Create Python virtual environment
2. ✓ Install all dependencies
3. ✓ Start Flask server
4. ✓ Display API endpoints

### Step 4: Open Dashboard
Open browser to: **http://localhost:5000**

---

## Testing the System

### Test 1: Healthy Bearing (Expected: Green, HEALTHY)
1. Click **"Test Healthy Bearing"** button
2. Wait for response (usually < 1 second)
3. See green "HEALTHY" status
4. Confidence bar shows low fault probability

### Test 2: Faulty Bearing (Expected: Red, FAULTY)
1. Click **"Test Faulty Bearing"** button
2. See red "FAULTY" status with alert
3. Confidence bar shows high fault probability
4. Check event logs at bottom

### Test 3: Live Recording
1. Click **"Start Recording"**
2. Speak into microphone or play bearing sound
3. Click **"Stop Recording"**
4. Click **"Predict"** button
5. Results appear in real-time

### Test 4: Upload File
1. Click **"Choose File"**
2. Select any audio file (WAV, MP3, etc.)
3. Click **"Predict from File"**
4. Results display with analysis

---

## Troubleshooting

### Issue: "Failed to connect to backend"

**Cause**: Flask server isn't running  
**Solution**:
1. Check that you ran `run.bat` (Windows) or `./run.sh` (Mac/Linux)
2. Look for error messages in the terminal
3. Ensure port 5000 is not in use:
   - Windows: `netstat -ano | find ":5000"`
   - Mac/Linux: `lsof -i :5000`

### Issue: "Module not found" errors

**Cause**: Dependencies didn't install  
**Solution**:
```bash
# Activate virtual environment first
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

# Then install manually
pip install -r backend/requirements.txt
```

### Issue: TensorFlow installation takes forever

**Normal!** TensorFlow is large. On first install it may take 10-15 minutes.  
Leave it running, don't interrupt.

### Issue: Microphone not working

**Cause**: Browser permissions or no microphone  
**Solution**:
1. Check browser microphone permissions (address bar)
2. Grant access when prompted
3. Use localhost/127.0.0.1 (not IP address)
4. HTTPS recommended (localhost works without it)

### Issue: File upload not working

**Solution**:
1. Ensure file is actually selected
2. Check browser console (F12) for errors
3. Try a smaller file first
4. Supported formats: WAV, MP3, OGG, FLAC

---

## File Structure

```
Predictive-Maintenance-Project/
├── run.bat                 ← START HERE (Windows)
├── run.sh                  ← START HERE (Mac/Linux)
├── README.md               ← Full documentation
├── SETUP_GUIDE.md          ← This file
│
├── backend/
│   ├── app.py              ← Flask REST API
│   ├── requirements.txt     ← Python dependencies
│   ├── audio_sensor_simulator.py  ← Generate test audio
│   ├── CNN (1).ipynb       ← Model training notebook
│   └── CNN_(1).ipynb       ← Model training notebook
│
├── frontend/
│   ├── index.html          ← Dashboard UI
│   ├── style.css           ← Styling & layout
│   ├── script.js           ← Frontend logic
│   └── README.md           ← Frontend docs
│
└── venv/                   ← Created by setup script
```

---

## Common Workflows

### Workflow 1: Quick Test
```
1. Run setup script
2. Wait for "Starting server..."
3. Open http://localhost:5000
4. Click "Test Healthy Bearing"
5. Click "Test Faulty Bearing"
6. Done! ✓
```

### Workflow 2: Test with Own Audio
```
1. Run setup script
2. Open dashboard
3. Click "Choose File"
4. Select your audio file
5. Click "Predict from File"
6. View results
```

### Workflow 3: Live Recording Test
```
1. Run setup script
2. Open dashboard
3. Click "Start Recording"
4. Record for 1-2 seconds
5. Click "Stop Recording"
6. Click "Predict"
7. View instant results
```

### Workflow 4: Generate Test Data
```
1. Open terminal/command prompt
2. cd backend
3. python audio_sensor_simulator.py
4. Check "test_audio" folder
5. Upload generated files to dashboard
```

---

## Key Features Explained

### 🎙️ Real-time Recording
- Captures audio from your microphone
- Directly processes through CNN
- Shows predictions instantly
- Records events in log

### 📊 Prediction History
- Tracks last 20 predictions
- Shows confidence trend over time
- Visual chart updates in real-time
- Helps identify patterns

### 🎨 Responsive Design
- Auto-adjusts to screen size
- Mobile-friendly interface
- Touch-friendly buttons
- Fast even on slow connections

### 📋 Event Logging
- Color-coded entries (green/yellow/red)
- Timestamped for reference
- Shows successes and errors
- Clearable log

---

## Performance Tips

1. **First Run Slower**: Initial model load takes 5-30 seconds
2. **Use Chrome**: Best browser performance
3. **Close Extra Tabs**: Save RAM for TensorFlow
4. **Predictions Fast**: Usually < 1 second after first load
5. **Large Files**: Audio files under 10MB work best

---

## Next Steps

### After Setup Works:

1. **Explore Features**
   - Try different test scenarios
   - Upload your own audio files
   - Test live recording

2. **Customize Dashboard**
   - Edit `frontend/style.css` for colors
   - Modify `frontend/script.js` for behavior
   - Add your own charts

3. **Train Custom Model**
   - Open `backend/CNN (1).ipynb`
   - Add your own training data
   - Save trained model as `best_model.keras`
   - Restart Flask server

4. **Deploy to Production**
   - Use Docker for deployment
   - Set up HTTPS/SSL
   - Use production WSGI server
   - Add authentication/security

---

## API Quick Reference

### Test API Endpoints

**Health Check**:
```bash
curl http://localhost:5000/api/health
```

**System Status**:
```bash
curl http://localhost:5000/api/status
```

**Test Prediction** (with synthetic data):
```bash
curl -X POST http://localhost:5000/api/predict/synthetic \
  -H "Content-Type: application/json" \
  -d '{"fault_type":"healthy","duration":0.1,"sampling_rate":20000}'
```

---

## Support Resources

| Topic | Location |
|-------|----------|
| Full Documentation | [README.md](README.md) |
| Backend Details | [backend/README.md] (create if needed) |
| Frontend Details | [frontend/README.md](frontend/README.md) |
| Model Details | [CNN_MODEL_SUMMARY.md](CNN_MODEL_SUMMARY.md) |
| Code | Look at .ipynb files |

---

## Common Questions

**Q: Can I use my own model?**  
A: Yes! Train it in the notebooks, save as `best_model.keras`, restart server.

**Q: How accurate is the model?**  
A: ~95% accuracy on test data. Accuracy depends on training data quality.

**Q: Can I connect real sensors?**  
A: Yes! Modify `backend/app.py` to accept sensor input format.

**Q: How do I deploy to production?**  
A: Use Docker, add authentication, deploy to cloud (AWS/GCP/Azure).

**Q: Is my data sent to cloud servers?**  
A: No! Everything runs locally on your machine.

---

## Getting Help

1. **Check Logs**: Look at browser console (F12) and terminal
2. **Read Documentation**: Check README.md and frontend/README.md
3. **Restart Server**: Sometimes fixes temporary issues
4. **Reinstall Deps**: `pip install -r backend/requirements.txt --upgrade`

---

**You're all set!** 🎉  
Start with the setup script and explore the dashboard.

Questions? Check README.md for detailed documentation.

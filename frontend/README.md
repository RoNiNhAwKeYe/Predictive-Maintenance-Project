# 🎨 Frontend Documentation

## Overview

The Predictive Maintenance Dashboard is a modern, responsive web interface for real-time acoustic bearing fault detection. It provides an intuitive way to visualize CNN-LSTM hybrid predictions and interact with the backend API.

---

## Architecture

### Technology Stack
- **HTML5** - Semantic markup
- **CSS3** - Modern CSS with CSS variables, Grid, Flexbox
- **JavaScript (Vanilla)** - No dependencies except Chart.js
- **Chart.js** - Interactive data visualization

### Browser Support
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## Features

### 1. **Real-time Dashboard**
- System health indicator with pulsing animation
- Current machine status display
- CNN-LSTM hybrid prediction with confidence meter
- Model information panel

### 2. **Live Recording**
- Microphone input capture
- Record/Stop controls
- Direct prediction on recorded audio
- User-friendly status indicators

### 3. **File Upload**
- Upload pre-recorded audio files
- Supported formats: WAV, MP3, and other HTML5 audio formats
- Automatic processing and prediction

### 3. **CNN-LSTM Hybrid Predictions**
- **Fault Detection**: Probability [0-100%] with confidence meter
- **Remaining Useful Life (RUL)**: Predicted hours until failure
- RUL Status: CRITICAL (≤5h), WARNING (5-20h), GOOD (>20h)
- Visual progress bars for both metrics

### 5. **Visualization**
- **Waveform Chart**: Acoustic signal amplitude over time
- **Prediction History**: Trend of fault confidence over time
- **Event Logs**: Timestamped log of all system events

### 6. **Responsive Grid Layout**
- Auto-adjusting 2-column layout
- Mobile-friendly single-column on small screens
- Accessible card-based design

---

## UI Components

### Navbar
```html
<nav class="navbar">
  - Brand title with icon
  - Navigation menu (Dashboard, Predictions, Controls, Logs)
  - System status indicator
</nav>
```

### Cards
Reusable card components:
- `card` - Standard card with header and body
- `card.alert` - Full-width alert section
- `card.control` - Interactive control panel
- `card.logs` - Event log display

### Key Sections

#### 1. Alerts Section
**Full-width, prominent alert display**
- Green for healthy status
- Red for faulty status
- Confidence percentage clearly displayed

#### 2. System Status
- Large "HEALTHY" / "FAULTY" indicator
- Badge showing system state
- Descriptive text

#### 3. CNN-LSTM Hybrid Prediction
- Prediction result with color coding
- Confidence meter with progress bar
- Percentage display

#### 4. Control Panel
- **Test Buttons**: Test healthy/faulty scenarios
- **Recording Controls**: Start/Stop recording
- **File Upload**: Select and predict from files
- **Status Display**: Recording progress

#### 5. Charts
Two interactive Chart.js visualizations:
- **Waveform Chart**: Shows acoustic signal
- **History Chart**: Shows prediction confidence trend

#### 6. Event Logs
- Timestamped entries
- Color-coded severity (success/warning/error)
- Auto-scrolling
- Clear logs button

---

## CSS Styling

### Design System

**Colors**:
```css
--color-primary: #1f2937      /* Dark blue-gray */
--color-accent: #3b82f6       /* Bright blue */
--color-success: #10b981      /* Green */
--color-danger: #ef4444       /* Red */
--color-warning: #f59e0b      /* Amber */
--color-info: #0ea5e9         /* Cyan */
```

**Spacing**:
```css
- Padding: 0.75rem, 1rem, 1.5rem, 2rem
- Gap: 0.75rem, 1.5rem
```

**Shadows**:
```css
--shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05)
--shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1)
--shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1)
```

**Border Radius**: 8px

### Responsive Design

**Breakpoints**:
- **Mobile**: < 768px (single column)
- **Tablet**: 768px - 1024px (adaptive grid)
- **Desktop**: > 1024px (full grid)

**Grid**:
```css
.grid-layout {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 1.5rem;
}
```

---

## JavaScript API

### Configuration
```javascript
const API_BASE_URL = 'http://localhost:5000';
const MAX_HISTORY = 20;
```

### Main Functions

#### System Status
```javascript
checkSystemStatus()
// Check if backend is running and model is loaded
```

#### Predictions
```javascript
testPrediction(faultType)
// testPrediction('healthy') or testPrediction('faulty')

predictRecording()
// Predict from recorded audio

predictFile()
// Predict from uploaded file
```

#### Recording
```javascript
startRecording()
stopRecording()
// Microphone recording control
```

#### Display Updates
```javascript
updatePredictionDisplay(result)
// Update all UI elements with prediction

updateBadgesAndAlerts(isFaulty, confidence)
// Update status badges and alert messages

updateSystemIndicator(healthy)
// Update system health indicator animation
```

#### Charts
```javascript
initializeCharts()
// Initialize Chart.js charts

generateWaveformChart(isFaulty)
// Generate simulated waveform based on fault status

updateCharts()
// Update chart data after prediction
```

#### Logging
```javascript
addLog(message, type)
// addLog('Message', 'success|warning|error|info')

clearLogs()
// Clear event log
```

---

## API Integration

### Backend Communication

**Base URL**: `http://localhost:5000`

**Endpoints Used**:
1. `GET /api/status` - Check system status
2. `POST /api/predict` - Send audio for prediction
3. `POST /api/predict/synthetic` - Test with synthetic data

### Request/Response Format

**Prediction Request**:
```javascript
{
  audio_base64: "base64_encoded_wav",
  sampling_rate: 20000
}
```

**Prediction Response**:
```javascript
{
  status: "HEALTHY|FAULTY",
  confidence: 0.234,
  fault_probability: 0.234,
  health_probability: 0.766,
  alert_level: "NORMAL|CRITICAL",
  timestamp: "ISO8601"
}
```

---

## State Management

### Global Variables
```javascript
CHARTS {}              // Chart.js instances
recordedAudioChunks [] // Audio blob chunks
mediaRecorder         // MediaRecorder instance
predictionHistory []  // Array of predictions
```

### Prediction History
Maintains last 20 predictions with:
- Timestamp
- Confidence score
- Status (HEALTHY/FAULTY)

---

## Event Logging

Color-coded log entries:
- **Success** (Green): Successful operations
- **Warning** (Yellow): Attention-required operations
- **Error** (Red): Failed operations
- **Info** (Gray): Informational messages

Each entry includes:
- Timestamp (HH:MM:SS)
- Message text
- Severity color

---

## User Interactions

### Workflow: Test Data
1. User clicks "Test Healthy Bearing"
2. Frontend sends POST to `/api/predict/synthetic`
3. Backend generates synthetic audio and predicts
4. Results displayed with green indicators
5. Log entry added

### Workflow: Live Recording
1. User clicks "Start Recording"
2. Browser requests microphone permission
3. Audio captured via MediaRecorder API
4. User clicks "Stop Recording"
5. User clicks "Predict"
6. Audio sent to `/api/predict`
7. Results displayed with animation
8. Chart updated with confidence trend

### Workflow: File Upload
1. User selects audio file
2. "Predict" button becomes enabled
3. File converted to base64
4. POST request to `/api/predict`
5. Results displayed
6. File input cleared

---

## Accessibility Features

- Semantic HTML markup
- ARIA labels on interactive elements
- Keyboard navigation support
- High contrast color scheme
- Readable font sizes (minimum 14px)
- Focus indicators on buttons

---

## Performance Optimizations

- **Lazy Chart Initialization**: Charts only created if DOM elements exist
- **Debounced Logging**: Log entries limited to 50 entries
- **Event Delegation**: Single event listener for navigation
- **CSS Variables**: Easy theme switching
- **Hardware Acceleration**: CSS transforms for smooth animations

---

## Customization

### Change Color Scheme
Edit CSS variables in `style.css`:
```css
:root {
    --color-primary: #new_color;
    --color-accent: #new_color;
    /* ... */
}
```

### Add Custom Chart
```javascript
CHARTS.myChart = new Chart(ctx, {
    type: 'chart_type',
    data: { /* ... */ },
    options: { /* ... */ }
});
```

### Modify Grid Layout
```css
.grid-layout {
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    /* Adjust minmax value */
}
```

---

## Browser DevTools Tips

1. **Check Console**: `F12` → Console for JS errors
2. **Network Tab**: Monitor API requests/responses
3. **Elements Tab**: Inspect HTML/CSS
4. **Application Tab**: Check localStorage, sessionStorage

---

## Future Enhancements

- Dark/Light theme toggle
- Real-time WebSocket updates
- Export prediction logs as CSV
- Spectogram visualization
- FFT frequency domain charts
- Anomaly detection visualization
- Multi-language support

---

## File Structure
```
frontend/
├── index.html          # Main dashboard HTML
├── style.css           # Complete CSS styling
├── script.js           # All JavaScript functionality
└── README.md           # This file
```

---

**Last Updated**: April 2024

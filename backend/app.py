"""
Predictive Maintenance Backend
Real-time acoustic sensor monitoring with CNN model
"""
import os
import json
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from datetime import datetime
import logging
from scipy.io import wavfile
from scipy import signal
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='../frontend', static_folder='../frontend')
CORS(app)

# Global model cache
model = None
model_loaded = False

# ============ MODEL HANDLER ============
def load_model():
    """Load or initialize the CNN model"""
    global model, model_loaded
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        # Try to load existing trained model
        model_path = 'best_model.keras'
        if os.path.exists(model_path):
            logger.info("Loading trained model from disk...")
            model = keras.models.load_model(model_path)
            model_loaded = True
            logger.info("✓ Model loaded successfully")
        else:
            logger.warning("No trained model found. Building fresh model...")
            model = build_cnn_model()
            model_loaded = True
            logger.info("✓ Fresh CNN model created")
        return True
    except ImportError:
        logger.error("TensorFlow not installed. Install with: pip install tensorflow scipy librosa")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def build_cnn_lstm_model():
    """Build the CNN-LSTM hybrid model for fault detection + RUL prediction"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    # Input: sequence of spectrograms (time series)
    # Shape: (batch_size, sequence_length, 64, 64, 1)
    inputs = keras.Input(shape=(None, 64, 64, 1))
    
    # TimeDistributed CNN for feature extraction from each spectrogram
    x = layers.TimeDistributed(
        keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
        ])
    )(inputs)
    
    # LSTM to capture temporal dependencies
    lstm_out = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
    lstm_out = layers.LSTM(64, dropout=0.2)(lstm_out)
    
    # Fault Detection Branch
    fault_branch = layers.Dense(64, activation='relu')(lstm_out)
    fault_branch = layers.Dropout(0.3)(fault_branch)
    fault_output = layers.Dense(1, activation='sigmoid', name='fault_detection')(fault_branch)
    
    # RUL Prediction Branch
    rul_branch = layers.Dense(64, activation='relu')(lstm_out)
    rul_branch = layers.Dropout(0.3)(rul_branch)
    rul_output = layers.Dense(1, activation='relu', name='rul_prediction')(rul_branch)
    
    model = keras.Model(inputs=inputs, outputs=[fault_output, rul_output])
    
    model.compile(
        optimizer='adam',
        loss={
            'fault_detection': 'binary_crossentropy',
            'rul_prediction': 'mse'
        },
        loss_weights={
            'fault_detection': 0.5,
            'rul_prediction': 0.5
        },
        metrics={
            'fault_detection': ['accuracy'],
            'rul_prediction': ['mae']
        }
    )
    return model

def build_cnn_model():
    """Legacy: Build simple CNN model for backward compatibility"""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=(64, 64, 1)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def preprocess_audio(audio_data, sr=20000):
    """
    Preprocess raw audio to 64x64 spectrogram
    
    Args:
        audio_data: numpy array of audio samples
        sr: sampling rate (default 20kHz)
    
    Returns:
        spectrogram: 64x64 normalized spectrogram
    """
    try:
        # Handle NaN/Inf
        audio_data = np.nan_to_num(audio_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Bandpass filter: 500-9000 Hz
        nyquist = sr / 2
        low = 500 / nyquist
        high = 9000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        audio_filtered = signal.filtfilt(b, a, audio_data)
        
        # Z-score normalization
        mean = np.mean(audio_filtered)
        std = np.std(audio_filtered)
        if std > 0:
            audio_filtered = (audio_filtered - mean) / std
        
        # Clip outliers (±5σ)
        audio_filtered = np.clip(audio_filtered, -5, 5)
        
        # Compute STFT
        f, t, Sxx = signal.spectrogram(
            audio_filtered,
            fs=sr,
            window='hann',
            nperseg=256,
            noverlap=128,
            nfft=256
        )
        
        # Log magnitude
        spectrogram = np.log1p(np.abs(Sxx))
        
        # Normalize to [0, 1]
        spec_min = np.min(spectrogram)
        spec_max = np.max(spectrogram)
        if spec_max > spec_min:
            spectrogram = (spectrogram - spec_min) / (spec_max - spec_min)
        
        # Resize to 64x64 using bilinear interpolation
        from scipy.ndimage import zoom
        current_shape = spectrogram.shape
        zoom_factors = (64 / current_shape[0], 64 / current_shape[1])
        spectrogram_64 = zoom(spectrogram, zoom_factors, order=1)
        
        # Ensure exactly 64x64
        spectrogram_64 = spectrogram_64[:64, :64]
        if spectrogram_64.shape != (64, 64):
            spectrogram_padded = np.zeros((64, 64))
            h, w = spectrogram_64.shape
            spectrogram_padded[:h, :w] = spectrogram_64
            spectrogram_64 = spectrogram_padded
        
        return spectrogram_64.astype(np.float32)
    
    except Exception as e:
        logger.error(f"Error preprocessing audio: {e}")
        return None

def predict_fault(audio_data, sr=20000):
    """
    Make prediction on audio data
    Returns both fault detection and RUL estimation
    
    Args:
        audio_data: numpy array of audio samples
        sr: sampling rate
    
    Returns:
        dict with prediction results
    """
    global model, model_loaded
    
    if not model_loaded:
        return {'error': 'Model not loaded'}
    
    try:
        # Preprocess audio
        spectrogram = preprocess_audio(audio_data, sr)
        if spectrogram is None:
            return {'error': 'Audio preprocessing failed'}
        
        # Add batch and channel dimensions
        spec_input = np.expand_dims(np.expand_dims(spectrogram, 0), -1)
        
        # Check if model is CNN-LSTM (multi-output) or simple CNN
        model_outputs = len(model.outputs) if hasattr(model, 'outputs') else 1
        
        if model_outputs > 1:
            # CNN-LSTM hybrid model with RUL prediction
            # Input: (batch, sequence_length, 64, 64, 1)
            # For single spectrogram, wrap in sequence dimension
            spec_input = np.expand_dims(spec_input, 1)  # (1, 1, 64, 64, 1)
            
            fault_pred, rul_pred = model.predict(spec_input, verbose=0)
            fault_probability = float(fault_pred[0][0])
            rul_estimate = max(0.5, float(rul_pred[0][0]))  # Ensure positive RUL
        else:
            # Simple CNN model (backward compatibility)
            fault_probability = float(model.predict(spec_input, verbose=0)[0][0])
            rul_estimate = None
        
        confidence = fault_probability
        threshold = 0.5
        is_fault = confidence >= threshold
        
        status = 'FAULTY' if is_fault else 'HEALTHY'
        alert_level = 'CRITICAL' if is_fault else 'NORMAL'
        
        result = {
            'status': status,
            'confidence': confidence,
            'alert_level': alert_level,
            'threshold': threshold,
            'timestamp': datetime.now().isoformat(),
            'spectrogram_shape': str(spectrogram.shape),
            'fault_probability': confidence,
            'health_probability': 1.0 - confidence,
        }
        
        # Add RUL if available
        if rul_estimate is not None:
            result['rul_hours'] = float(rul_estimate)
            result['rul_status'] = get_rul_status(rul_estimate)
        
        return result
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {'error': str(e)}

def get_rul_status(rul_hours):
    """
    Determine RUL status for visual indication
    
    Args:
        rul_hours: Remaining Useful Life in hours
        
    Returns:
        Status string: CRITICAL, WARNING, or GOOD
    """
    if rul_hours <= 5:
        return 'CRITICAL'
    elif rul_hours <= 20:
        return 'WARNING'
    else:
        return 'GOOD'

def generate_synthetic_audio(duration=0.1, sr=20000, fault_type='healthy'):
    """
    Generate synthetic bearing sensor data for testing
    
    Args:
        duration: duration in seconds
        sr: sampling rate
        fault_type: 'healthy' or 'faulty'
    
    Returns:
        tuple: (numpy array of audio samples, estimated RUL in hours)
    """
    samples = int(duration * sr)
    t = np.linspace(0, duration, samples, False)
    
    if fault_type == 'healthy':
        # Clean bearing: low amplitude, single dominant frequency
        signal_clean = 0.1 * np.sin(2 * np.pi * 2000 * t)  # 2 kHz dominant
        noise = 0.02 * np.random.randn(samples)
        audio = signal_clean + noise
        rul_estimate = np.random.uniform(50, 100)  # 50-100 hours RUL
    else:
        # Faulty bearing: multiple frequencies, impulses
        base = 0.2 * np.sin(2 * np.pi * 1500 * t)  # 1.5 kHz
        harmonic = 0.15 * np.sin(2 * np.pi * 3000 * t)  # 3 kHz harmonic
        
        # Add impulses (characteristic of bearing wear)
        impulse_freq = 100  # impulses per second
        impulse_times = np.arange(0, duration, 1/impulse_freq)
        impulses = np.zeros(samples)
        for imp_time in impulse_times:
            imp_idx = int(imp_time * sr)
            if imp_idx < samples:
                impulse_width = int(0.001 * sr)  # 1ms impulse
                impulses[imp_idx:min(imp_idx+impulse_width, samples)] = 0.5
        
        noise = 0.03 * np.random.randn(samples)
        audio = base + harmonic + impulses + noise
        rul_estimate = np.random.uniform(1, 10)  # 1-10 hours RUL (critical)
    
    # Clip to valid range
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32), rul_estimate

# ============ API ENDPOINTS ============

@app.route('/')
def index():
    """Serve the main dashboard"""
    return app.send_static_file('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'system': 'Predictive Maintenance System',
        'version': '1.0',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests
    
    Expected JSON:
    {
        'audio': [array of samples or base64 encoded wav],
        'sampling_rate': 20000,
        'duration': 0.1
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get audio data
        if 'audio' in data and isinstance(data['audio'], list):
            audio_data = np.array(data['audio'], dtype=np.float32)
        elif 'audio_base64' in data:
            # Decode base64 audio
            audio_bytes = base64.b64decode(data['audio_base64'])
            sr, audio_data = wavfile.read(io.BytesIO(audio_bytes))
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize INT16
        else:
            return jsonify({'error': 'No audio data provided'}), 400
        
        sr = data.get('sampling_rate', 20000)
        
        # Make prediction
        result = predict_fault(audio_data, sr)
        
        if 'error' in result:
            return jsonify(result), 500
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/synthetic', methods=['POST'])
def predict_synthetic():
    """
    Generate synthetic data and make prediction
    
    Expected JSON:
    {
        'fault_type': 'healthy' or 'faulty',
        'duration': 0.1,
        'sampling_rate': 20000
    }
    """
    try:
        data = request.get_json() or {}
        
        fault_type = data.get('fault_type', 'healthy')
        duration = float(data.get('duration', 0.1))
        sr = int(data.get('sampling_rate', 20000))
        
        # Generate synthetic audio with RUL estimate
        audio_data, rul_estimate = generate_synthetic_audio(duration, sr, fault_type)
        
        # Make prediction
        result = predict_fault(audio_data, sr)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Add RUL from synthetic generation if not from model
        if 'rul_hours' not in result:
            result['rul_hours'] = rul_estimate
            result['rul_status'] = get_rul_status(rul_estimate)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Synthetic prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

# ============ ERROR HANDLERS ============

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    logger.error(f"Server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

# ============ STARTUP ============

@app.before_request
def initialize():
    """Initialize on first request"""
    global model_loaded
    if not model_loaded:
        load_model()

if __name__ == '__main__':
    # Load model on startup
    load_model()
    
    # Run Flask app
    print("\n" + "="*60)
    print("Predictive Maintenance Backend")
    print("="*60)
    print("Starting server on http://localhost:5000")
    print("API Endpoints:")
    print("  GET  /api/status - System status")
    print("  GET  /api/health - Health check")
    print("  POST /api/predict - Make prediction on audio")
    print("  POST /api/predict/synthetic - Test with synthetic data")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

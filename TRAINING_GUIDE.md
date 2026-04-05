## 🔬 Training Guide: CNN-LSTM for RUL Prediction

This guide covers training your own CNN-LSTM model for bearing fault detection and remaining useful life prediction.

---

## 📋 Prerequisites

### Python Packages
```bash
pip install tensorflow keras numpy scipy matplotlib scikit-learn pandas
```

### Data Requirements
- Audio samples from bearing runs: healthy & faulty bearings
- RUL labels: hours remaining for each sample
- Recommended: 100+ healthy runs, 100+ failure runs
- Sample duration: 2-10 seconds each

---

## 📊 Dataset Preparation

### Step 1: Load and Explore Data
```python
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal

# Example: Load NASA Prognostics bearing dataset
data = loadmat('bearing_run_1.mat')
audio = data['bearing'][0, 0, 2]  # Vertical acceleration

# Basic stats
print(f"Shape: {audio.shape}")
print(f"Duration: {len(audio) / 20000:.2f}s")  # 20 kHz sampling rate
print(f"Min: {audio.min():.4f}, Max: {audio.max():.4f}")
```

### Step 2: Create Label DataFrame
```python
# Create labels for each sample
labels = {
    'file': ['run_1.mat', 'run_2.mat', ...],
    'fault': [0, 0, ..., 1, 1],  # 0=healthy, 1=faulty
    'rul_hours': [150.5, 148.2, ..., 8.3, 2.1]  # Remaining useful life
}
df_labels = pd.DataFrame(labels)
print(df_labels.head())
```

### Step 3: Create Training/Validation Split
```python
from sklearn.model_selection import train_test_split

train_files, val_files = train_test_split(
    df_labels, test_size=0.2, random_state=42
)
print(f"Training: {len(train_files)}, Validation: {len(val_files)}")
```

---

## 🎯 Signal Processing Pipeline

### Audio Preprocessing Function
```python
def preprocess_audio(audio, sr=20000):
    """
    Converts raw audio to normalized spectrogram
    
    Args:
        audio: numpy array of audio samples
        sr: sampling rate (Hz)
    
    Returns:
        spec: 64x64 normalized spectrogram [0, 1]
    """
    # 1. Bandpass filter (500-9000 Hz)
    nyquist = sr / 2
    low = 500 / nyquist
    high = 9000 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, audio)
    
    # 2. Normalize (Z-score)
    filtered = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    # 3. Clip outliers (±5 sigma)
    filtered = np.clip(filtered, -5, 5)
    
    # 4. STFT to spectrogram
    f, t, Sxx = signal.spectrogram(
        filtered, sr,
        window='hann',
        nperseg=256,
        noverlap=128,
        nfft=256
    )
    
    # 5. Log magnitude
    Sxx_log = np.log(1 + np.abs(Sxx))
    
    # 6. Normalize to [0, 1]
    Sxx_norm = (Sxx_log - Sxx_log.min()) / (Sxx_log.max() - Sxx_log.min() + 1e-8)
    
    # 7. Resize to 64x64
    spec = cv2.resize(Sxx_norm, (64, 64))
    
    return spec
```

### Batch Processing
```python
import cv2

def create_spectrograms(audio_list, label_list):
    """Create batches of spectrograms with labels"""
    specs = []
    labels_fault = []
    labels_rul = []
    
    for audio, (fault, rul) in zip(audio_list, label_list):
        spec = preprocess_audio(audio)
        specs.append(spec)
        labels_fault.append(fault)
        labels_rul.append(rul)
    
    return np.array(specs), np.array(labels_fault), np.array(labels_rul)
```

---

## 🏗️ Build CNN-LSTM Model

### Model Architecture Code
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM,
    Dense, Dropout, Input, Concatenate, BatchNormalization
)
from tensorflow.keras.models import Model

def build_cnn_lstm_model(sequence_length=10):
    """
    Builds CNN-LSTM hybrid model for:
    - Fault detection (binary: healthy/faulty)
    - RUL prediction (regression: hours remaining)
    
    Args:
        sequence_length: number of spectrograms per sequence
    
    Returns:
        model: Keras Model with 2 outputs
    """
    
    # Input: sequence of spectrograms
    inputs = Input(shape=(sequence_length, 64, 64, 1))
    
    # TimeDistributed CNN for feature extraction
    x = TimeDistributed(Conv2D(32, (3, 3), activation='relu'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    x = TimeDistributed(Conv2D(128, (3, 3), activation='relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D((2, 2)))(x)
    
    # Flatten and dense for each timestep
    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(128, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    
    # LSTM layers for temporal dependency
    x = LSTM(128, return_sequences=True, dropout=0.2)(x)
    x = LSTM(64, dropout=0.2)(x)
    
    # Output Branch 1: Fault Detection
    fault_branch = Dense(64, activation='relu')(x)
    fault_branch = Dropout(0.3)(fault_branch)
    fault_output = Dense(1, activation='sigmoid', name='fault_detection')(fault_branch)
    
    # Output Branch 2: RUL Prediction
    rul_branch = Dense(64, activation='relu')(x)
    rul_branch = Dropout(0.3)(rul_branch)
    rul_output = Dense(1, activation='relu', name='rul_prediction')(rul_branch)
    
    # Build model
    model = Model(inputs=inputs, outputs=[fault_output, rul_output])
    
    return model

# Build model
model = build_cnn_lstm_model(sequence_length=10)
model.summary()
```

---

## 🎓 Training the Model

### Compile Model
```python
# Compile with multi-task loss
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
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
```

### Create Training Sequences
```python
def create_sequences(specs, fault_labels, rul_labels, seq_length=10):
    """Create sliding window sequences from spectrograms"""
    X = []
    y_fault = []
    y_rul = []
    
    for i in range(len(specs) - seq_length):
        X.append(specs[i:i+seq_length])  # 10 consecutive spectrograms
        y_fault.append(fault_labels[i+seq_length])  # Label from last spectrogram
        y_rul.append(rul_labels[i+seq_length])
    
    return np.array(X), np.array(y_fault), np.array(y_rul)

# Create train sequences
X_train, y_fault_train, y_rul_train = create_sequences(
    specs_train, fault_labels_train, rul_labels_train,
    seq_length=10
)

# Create validation sequences
X_val, y_fault_val, y_rul_val = create_sequences(
    specs_val, fault_labels_val, rul_labels_val,
    seq_length=10
)

print(f"Train X shape: {X_train.shape}")
print(f"Train fault y shape: {y_fault_train.shape}")
print(f"Train RUL y shape: {y_rul_train.shape}")
```

### Training with Callbacks
```python
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    ),
    ModelCheckpoint(
        'best_model.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]

# Train model
history = model.fit(
    X_train,
    {'fault_detection': y_fault_train, 'rul_prediction': y_rul_train},
    batch_size=32,
    epochs=50,
    validation_data=(
        X_val,
        {'fault_detection': y_fault_val, 'rul_prediction': y_rul_val}
    ),
    callbacks=callbacks,
    verbose=1
)
```

---

## 📈 Evaluate Model

### Performance Metrics
```python
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report

# Get predictions
fault_preds, rul_preds = model.predict(X_val)

# Fault Detection Metrics
fault_binary = (fault_preds > 0.5).astype(int).flatten()
print("Fault Detection Report:")
print(classification_report(y_fault_val, fault_binary))

cm = confusion_matrix(y_fault_val, fault_binary)
print(f"\nConfusion Matrix:\n{cm}")

auc = roc_auc_score(y_fault_val, fault_preds)
print(f"AUC-ROC: {auc:.4f}")

# RUL Prediction Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_rul_val, rul_preds)
rmse = np.sqrt(mean_squared_error(y_rul_val, rul_preds))
mape = np.mean(np.abs((y_rul_val - rul_preds.flatten()) / y_rul_val)) * 100

print(f"\nRUL Prediction:")
print(f"MAE: {mae:.2f} hours")
print(f"RMSE: {rmse:.2f} hours")
print(f"MAPE: {mape:.2f}%")
```

### Visualize Training History
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Fault loss
axes[0, 0].plot(history.history['fault_detection_loss'])
axes[0, 0].plot(history.history['val_fault_detection_loss'])
axes[0, 0].set_title('Fault Detection Loss')
axes[0, 0].legend(['Train', 'Val'])

# Fault accuracy
axes[0, 1].plot(history.history['fault_detection_accuracy'])
axes[0, 1].plot(history.history['val_fault_detection_accuracy'])
axes[0, 1].set_title('Fault Detection Accuracy')
axes[0, 1].legend(['Train', 'Val'])

# RUL loss (MSE)
axes[1, 0].plot(history.history['rul_prediction_loss'])
axes[1, 0].plot(history.history['val_rul_prediction_loss'])
axes[1, 0].set_title('RUL Prediction Loss (MSE)')
axes[1, 0].legend(['Train', 'Val'])

# RUL MAE
axes[1, 1].plot(history.history['rul_prediction_mae'])
axes[1, 1].plot(history.history['val_rul_prediction_mae'])
axes[1, 1].set_title('RUL Prediction MAE')
axes[1, 1].legend(['Train', 'Val'])

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()
```

---

## 💾 Save & Deploy Model

### Save Trained Model
```python
# Save best model
model.save('best_model.keras')
print("✓ Model saved as best_model.keras")

# Also save weights separately
model.save_weights('model_weights.h5')
print("✓ Weights saved as model_weights.h5")
```

### Load for Inference
```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('best_model.keras')

# Test inference
test_spec = preprocess_audio(test_audio)
test_spec_seq = np.expand_dims(np.expand_dims(test_spec, 0), -1)  # (1, 1, 64, 64, 1)

fault_pred, rul_pred = model.predict(test_spec_seq, verbose=0)
print(f"Fault Probability: {fault_pred[0][0]:.4f}")
print(f"RUL Hours: {rul_pred[0][0]:.2f}")
```

### Copy to Backend
```bash
# Copy trained model to Flask backend
cp best_model.keras /path/to/backend/

# Restart Flask server
# The app will automatically load and use your trained model
```

---

## 🔧 Hyperparameter Tuning

### Try Different Configurations
```python
from tensorflow.keras.optimizers import Adam
import itertools

# Grid search parameters
params_grid = {
    'lstm_units': [64, 128, 256],
    'dropout': [0.2, 0.3, 0.4],
    'learning_rate': [1e-3, 1e-4, 1e-5],
    'batch_size': [16, 32, 64]
}

best_val_loss = float('inf')
best_params = None

for lstm_units, dropout, lr, bs in itertools.product(*params_grid.values()):
    print(f"\nTesting: LSTM={lstm_units}, dropout={dropout}, lr={lr}, bs={bs}")
    
    # Build model
    model = build_cnn_lstm_model(lstm_units=lstm_units, dropout=dropout)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss={...},
        metrics={...}
    )
    
    # Train briefly to evaluate
    history = model.fit(
        X_train, {...},
        batch_size=bs,
        epochs=10,
        validation_data=(X_val, {...}),
        verbose=0
    )
    
    val_loss = history.history['val_loss'][-1]
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {
            'lstm_units': lstm_units,
            'dropout': dropout,
            'learning_rate': lr,
            'batch_size': bs
        }
        print(f"✓ New best! Val loss: {val_loss:.4f}")

print(f"\n🏆 Best parameters: {best_params}")
```

---

## 📊 Working with Different Data Formats

### From WAV Files
```python
import librosa

def load_wav_and_process(filepath):
    """Load WAV and create spectrogram"""
    audio, sr = librosa.load(filepath, sr=20000)
    spec = preprocess_audio(audio, sr=sr)
    return spec
```

### From Sensor CSV
```python
def load_csv_and_process(filepath):
    """Load CSV sensor data and create spectrogram"""
    df = pd.read_csv(filepath)
    audio = df['acceleration'].values
    spec = preprocess_audio(audio)
    return spec
```

### From MATLAB Files
```python
from scipy.io import loadmat

def load_mat_and_process(filepath, key='bearing'):
    """Load MATLAB .mat file and create spectrogram"""
    data = loadmat(filepath)
    audio = data[key].flatten()
    spec = preprocess_audio(audio)
    return spec
```

---

## ⚠️ Common Pitfalls & Solutions

| Problem | Solution |
|---------|----------|
| **Model not learning** | Check data normalization, increase learning rate, reduce regularization |
| **High train loss, good val loss** | Normal! Your model is not memorizing |
| **Overfitting** | Increase dropout, reduce model size, use L1/L2 regularization |
| **Exploding loss** | Reduce learning rate, check for NaN in data, normalize inputs |
| **RUL predictions unrealistic** | Check RUL label range, verify ReLU activation, inspect training targets |
| **Fault/RUL disagreement** | Normal! Different tasks optimize differently. Adjust loss_weights |

---

## 📝 Training Checklist

- [ ] Dataset prepared with fault & RUL labels
- [ ] Audio preprocessing pipeline tested
- [ ] Training/validation split created (80/20)
- [ ] Sequences generated with sliding window
- [ ] Model architecture defined
- [ ] Callbacks configured (early stopping, checkpoint)
- [ ] Model trained (50+ epochs typically)
- [ ] Validation metrics evaluated
- [ ] Model saved as `best_model.keras`
- [ ] Model copied to `backend/` directory
- [ ] Flask server restarted
- [ ] Test predictions from dashboard

---

**Training Guide Version**: 1.0  
**Last Updated**: April 2024  
**Framework**: TensorFlow 2.14+

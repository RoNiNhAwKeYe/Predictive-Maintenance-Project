# CNN Model — Quick Reference & Code Snippets

## Quick Facts

```
Dataset:      NASA IMS Bearing (run-to-failure)
Task:         Binary classification (Healthy=0, Faulty=1)
Input:        64×64 grayscale spectrogram images
Output:       Fault probability [0.0, 1.0]
Threshold:    0.5 (default)
Model Size:   ~286k parameters
Training:     50 epochs, batch size 32, Adam optimizer
```

---

## Preprocessing Configuration

```python
FS             = 20_000   # Sampling frequency (Hz)
APPLY_FILTER   = True     # Butterworth bandpass
LOWCUT         = 500      # Hz
HIGHCUT        = 9_000    # Hz
OUTLIER_CLIP   = 5.0      # ±N std devs

# Windowing
WINDOW_SIZE = 2048        # samples (~0.1 s at 20 kHz)
HOP_SIZE    = 1024        # 50% overlap

# STFT → Spectrogram
NPERSEG      = 256        # FFT window length
NOVERLAP     = 128        # FFT overlap (50%)
SPEC_H       = 64         # Output height (freq bins)
SPEC_W       = 64         # Output width (time frames)
```

---

## CNN Architecture at a Glance

```
Input  (64, 64, 1)
   ↓
 Block 1: Conv2D(32, 3×3) → BatchNorm → MaxPool(2×2)
   ↓
 Block 2: Conv2D(64, 3×3) → BatchNorm → MaxPool(2×2)
   ↓
 Block 3: Conv2D(128, 3×3) → BatchNorm → MaxPool(2×2)
   ↓
 Flatten (8,192 units)
   ↓
 Dense(256, ReLU) → Dropout(0.5)
   ↓
 Dense(1, Sigmoid) → Output probability
```

**Total Parameters**: ~286,000  
**Activations**: ReLU (hidden) | Sigmoid (output)  
**Regularization**: BatchNorm + Dropout(0.5)

---

## Complete Preprocessing Function

```python
def preprocess_signal(sig, fs=FS, apply_filter=APPLY_FILTER,
                      lowcut=LOWCUT, highcut=HIGHCUT, clip_std=OUTLIER_CLIP):
    """
    Full preprocessing: handle NaN → filter → normalize → clip
    """
    s = sig.copy().astype(np.float32)
    
    # Step 1: Handle missing values
    bad_mask = ~np.isfinite(s)
    if bad_mask.any():
        idx = np.arange(len(s))
        good_idx = idx[~bad_mask]
        s[bad_mask] = np.interp(idx[bad_mask], good_idx, s[good_idx])
    
    # Step 2: Bandpass filter
    if apply_filter and len(s) > 100:
        try:
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = signal.butter(4, [low, high], btype='band')
            s = signal.filtfilt(b, a, s)
        except:
            pass
    
    # Step 3: Z-score normalization
    std = s.std()
    if std > 1e-8:
        s = (s - s.mean()) / std
    
    # Step 4: Clip outliers
    s = np.clip(s, -clip_std, clip_std)
    
    return s
```

---

## Convert Signal Window to Spectrogram

```python
def window_to_spectrogram(win, fs=FS, nperseg=NPERSEG,
                          noverlap=NOVERLAP, h=SPEC_H, w=SPEC_W):
    """
    Window (1D) → Spectrogram image (2D, normalized)
    """
    from scipy.signal import stft as scipy_stft
    from skimage.transform import resize as sk_resize
    
    # STFT
    _, _, Zxx = scipy_stft(win, fs=fs, nperseg=nperseg,
                            noverlap=noverlap, window='hann')
    
    # Power & log scale
    spec = np.abs(Zxx) ** 2
    spec = np.log1p(spec)
    
    # Min-max normalize
    s_min, s_max = spec.min(), spec.max()
    if s_max - s_min > 1e-8:
        spec = (spec - s_min) / (s_max - s_min)
    else:
        spec = np.zeros_like(spec)
    
    # Resize to target dims
    spec = sk_resize(spec, (h, w), anti_aliasing=True).astype(np.float32)
    return spec
```

---

## Segmentation & Spectrogram Batch

```python
def segment_signal(sig, window_size=WINDOW_SIZE, hop_size=HOP_SIZE):
    """Split 1-D signal into overlapping windows"""
    segments = []
    start = 0
    while start + window_size <= len(sig):
        segments.append(sig[start : start + window_size])
        start += hop_size
    return np.array(segments, dtype=np.float32)

# Example usage:
clean_signal = preprocess_signal(raw_signal)
windows = segment_signal(clean_signal)  # (N, 2048)
spectrograms = np.array(
    [window_to_spectrogram(w) for w in windows]
)  # (N, 64, 64)
spectrograms = spectrograms[..., np.newaxis]  # (N, 64, 64, 1) for CNN
```

---

## Training Configuration

```python
# Model compilation
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32

# Callbacks
cb_early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=7,
    restore_best_weights=True,
    verbose=1
)

cb_reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

cb_checkpoint = callbacks.ModelCheckpoint(
    'best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=0
)

# Fit
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[cb_early_stop, cb_reduce_lr, cb_checkpoint],
    verbose=1
)
```

---

## Prediction on New Data

### Single Spectrogram (Already Preprocessed)
```python
def predict_single_spectrogram(spec_2d, threshold=0.5):
    """
    Input:  spec_2d of shape (64, 64) — normalized [0, 1]
    Output: (label_str, probability)
    """
    x = spec_2d[np.newaxis, :, :, np.newaxis]  # (1, 64, 64, 1)
    prob = float(model.predict(x, verbose=0)[0, 0])
    label = 'Faulty' if prob >= threshold else 'Healthy'
    return label, prob

# Example
spec = spectrograms[0]  # (64, 64)
label, prob = predict_single_spectrogram(spec)
print(f"Prediction: {label} (probability={prob:.3f})")
```

### Full Pipeline: Raw Signal → Prediction
```python
def predict_from_raw_signal(raw_signal, fs=FS, threshold=0.5):
    """
    End-to-end: raw 1D signal → preprocessing → windowing → 
    spectrogram → CNN → majority vote
    """
    # Preprocess
    clean = preprocess_signal(raw_signal, fs=fs)
    
    # Segment
    wins = segment_signal(clean)
    if len(wins) == 0:
        return 'Error: Signal too short', 0.0
    
    # STFT → spectrogram
    specs = np.array([window_to_spectrogram(w, fs=fs) for w in wins])
    specs = specs[:, :, :, np.newaxis]  # (N, 64, 64, 1)
    
    # Batch predict
    probs = model.predict(specs, verbose=0).ravel()
    
    # Majority vote
    mean_prob = float(probs.mean())
    label = 'Faulty' if mean_prob >= threshold else 'Healthy'
    
    return label, mean_prob

# Example
raw_audio = np.loadtxt('bearing_data.txt')[:, 0]  # Channel 0
label, prob = predict_from_raw_signal(raw_audio)
print(f"Overall Prediction: {label} (avg probability={prob:.3f})")
```

### Batch Prediction
```python
# X_batch shape: (N, 64, 64, 1)
probs = model.predict(X_batch, verbose=0).ravel()  # (N,)
predictions = (probs >= 0.5).astype(int)            # (N,) → {0, 1}

# For confidence
confidence = np.abs(probs - 0.5) * 2  # Distance from decision boundary
```

---

## Evaluation Metrics

```python
# Test set evaluation
test_metrics = model.evaluate(X_test, y_test, verbose=0, return_dict=True)
print(f"Test Accuracy : {test_metrics['accuracy']:.4f}")
print(f"Test Precision: {test_metrics['precision']:.4f}")
print(f"Test Recall   : {test_metrics['recall']:.4f}")
print(f"Test AUC      : {test_metrics['auc']:.4f}")

# Confusion matrix
y_pred = (model.predict(X_test, verbose=0) >= 0.5).astype(int).ravel()
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred, 
                           target_names=['Healthy', 'Faulty']))
```

---

## Data Labeling Strategy

```python
HEALTHY_FRAC = 0.35  # First 35% → Healthy (0)
FAULTY_FRAC  = 0.35  # Last 35%  → Faulty (1)
# Middle 30% excluded (transitional zone)

n_files = len(all_files)  # Assume chronologically sorted
healthy_cutoff = int(n_files * HEALTHY_FRAC)
faulty_start = int(n_files * (1 - FAULTY_FRAC))

labels = np.full(n_files, -1, dtype=int)
labels[:healthy_cutoff] = 0      # Healthy
labels[faulty_start:] = 1         # Faulty
# labels in [healthy_cutoff, faulty_start) = -1 (excluded)
```

---

## Model Import & Export

```python
# Save trained model
model.save('bearing_cnn.keras')

# Load model
from tensorflow.keras.models import load_model
model = load_model('bearing_cnn.keras')

# Save only weights
model.save_weights('model_weights.h5')

# Load weights into new model
model = build_cnn()
model.load_weights('model_weights.h5')
```

---

## Hyperparameter Tuning Notes

| Parameter | Default | Range to Try | Impact |
|-----------|---------|---|---|
| **Learning Rate** | 1e-3 | 1e-4 to 1e-2 | Too high: diverge; too low: slow |
| **Batch Size** | 32 | 16, 32, 64 | Larger: faster but less stable |
| **Dropout** | 0.5 | 0.3-0.7 | Prevents overfitting |
| **Conv Filters** | [32,64,128] | Vary depths | More filters = more capacity |
| **Window Size** | 2048 | 1024-4096 | Longer: more context, fewer samples |
| **Threshold** | 0.5 | 0.3-0.7 | Balance precision vs recall |

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| **Poor accuracy** | Insufficient preprocessing | Check filter cutoffs, z-norm range |
| **Overfitting** | Too many parameters | Increase dropout, reduce filters |
| **Underfitting** | Model too simple | Add conv blocks, increase filters |
| **NaN loss** | Exploding gradients | Reduce learning rate, add batch norm |
| **Memory error** | Batch too large | Reduce batch size, window size |
| **Slow training** | Data not GPU-optimized | Ensure float32, use generators |


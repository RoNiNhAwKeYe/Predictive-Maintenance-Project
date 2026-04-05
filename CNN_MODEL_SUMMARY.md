# CNN Model Summary — Acoustic-Based Bearing Fault Detection

## 1. Overview

**Project**: Predictive Maintenance using CNN-LSTM  
**Dataset**: NASA IMS Bearing Dataset (Run-to-Failure)  
**Task**: Binary classification — Healthy (0) vs Faulty (1)  
**Input**: Acoustic sensor signals converted to spectrogram images  
**Output**: Fault probability (0.0-1.0) with binary classification at threshold 0.5

---

## 2. Input Specifications

### Raw Acoustic Data
- **Source**: NASA IMS bearing test data (.txt files, multi-channel)
- **Sampling Rate**: 20,000 Hz (20 kHz)
- **Number of Channels**: Multi-bearing setup (4 bearings per test), channel 0 selected
- **Duration per File**: Variable (minutes to hours of continuous monitoring)
- **Data Format**: Time-series 1D signals, 1 sample per channel per time step

### Signal Windowing
- **Window Size**: 2,048 samples ≈ 0.1 seconds at 20 kHz
- **Hop Size**: 1,024 samples (50% overlap)
- **Purpose**: Creates multiple training samples from each long file, ensures fixed input size

### Spectrogram (CNN Input Image)

The time-domain signal is converted to a 2D spectrogram image:

| Feature | Details |
|---------|---------|
| **Image Height** | 64 frequency bins |
| **Image Width** | 64 time frames |
| **Channels** | 1 (grayscale) |
| **Format** | Log-magnitude power spectrogram |
| **Value Range** | [0, 1] (normalized) |
| **STFT Parameters** | `nperseg=256`, `noverlap=128` (Hann window) |

**CNN Input Shape**: `(batch_size, 64, 64, 1)`

---

## 3. Preprocessing Pipeline

All raw signals undergo the following 4-step preprocessing:

### Step 1: Handle Missing/Infinite Values
- Replace NaN and Inf values via linear interpolation
- Ensures continuity of signal for filter operations

### Step 2: Bandpass Filtering (Optional)
- **Type**: Butterworth bandpass filter (order=4)
- **Cutoff Frequencies**: 500 Hz (low) — 9,000 Hz (high)
- **Purpose**: Isolates bearing fault frequencies, attenuates noise
- **Method**: Zero-phase filtering (`scipy.signal.filtfilt`)

### Step 3: Z-Score Normalization
- Formula: $$\text{signal}_{\text{norm}} = \frac{\text{signal} - \mu}{\sigma}$$
- Achieves zero mean and unit variance
- Ensures all samples have consistent amplitude range

### Step 4: Outlier Clipping
- Clip values beyond ±5 standard deviations
- Removes extreme noise spikes without signal loss
- **Range after clipping**: [-5, +5]

### STFT → Spectrogram Generation

1. **STFT Computation**:
   - FFT window: 256 samples, 50% overlap
   - Output: Complex frequency-time matrix

2. **Power Spectrogram**:
   - $$S = |\text{STFT}|^2$$

3. **Log-Scale Compression**:
   - $$S_{\text{log}} = \log(1 + S)$$
   - Compresses dynamic range, emphasizes subtle patterns

4. **Min-Max Normalization**:
   - $$S_{\text{norm}} = \frac{S - S_{\min}}{S_{\max} - S_{\min}}$$
   - Scales all values to [0, 1]

5. **Resize to Fixed Dimensions**:
   - Bilinear interpolation to 64×64 pixels
   - Ensures consistent input for CNN

---

## 4. Output Specifications

### Classification Output

| Property | Details |
|----------|---------|
| **Output Type** | Binary probability |
| **Output Range** | [0.0, 1.0] |
| **Interpretation** | P(Faulty) — probability the bearing is faulty |
| **Decision Threshold** | 0.5 |
| **Class 0 (Healthy)** | Output < 0.5 |
| **Class 1 (Faulty)** | Output ≥ 0.5 |

### Training Labels
- **Healthy**: First 35% of sorted files (chronologically early, low degradation)
- **Excluded**: Middle 30% (transitional zone, unclear status)
- **Faulty**: Last 35% of sorted files (chronologically late, high degradation)

---

## 5. CNN Model Architecture

### Full Model Summary

```
Input Layer
    ↓
├─ Conv2D(32 filters, 3×3 kernel, ReLU, padding='same')
├─ BatchNormalization()
├─ MaxPooling2D(2×2)                        [64×64 → 32×32]
    ↓
├─ Conv2D(64 filters, 3×3 kernel, ReLU, padding='same')
├─ BatchNormalization()
├─ MaxPooling2D(2×2)                        [32×32 → 16×16]
    ↓
├─ Conv2D(128 filters, 3×3 kernel, ReLU, padding='same')
├─ BatchNormalization()
├─ MaxPooling2D(2×2)                        [16×16 → 8×8]
    ↓
├─ Flatten()                                [128×8×8 = 8,192 units]
├─ Dense(256, ReLU)
├─ Dropout(0.5)                             [50% dropout for regularization]
├─ Dense(1, Sigmoid)                        [Binary output]
    ↓
Output: Faulty probability
```

### Model Details

| Component | Value |
|-----------|-------|
| **Total Parameters** | ~286k trainable parameters |
| **Activation Functions** | ReLU (hidden), Sigmoid (output) |
| **Pooling** | Max pooling (2×2) after each conv block |
| **Regularization** | BatchNorm + Dropout(0.5) |

### Layer Breakdown

**Block 1**: 32 filters, 3×3 kernels
- Learns low-level features (edges, streaks in spectrograms)
- Output spatial dims: 64×64 → 32×32

**Block 2**: 64 filters, 3×3 kernels
- Learns mid-level features (texture patterns, frequency regions)
- Output spatial dims: 32×32 → 16×16

**Block 3**: 128 filters, 3×3 kernels
- Learns high-level features (degradation signatures)
- Output spatial dims: 16×16 → 8×8

**Classification Head**:
- Flattens all feature maps (8,192 units)
- Dense(256) + Dropout(0.5) for robust feature fusion
- Dense(1, Sigmoid) outputs final probability

---

## 6. Training Configuration

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam (learning rate = 1e-3) |
| **Loss Function** | Binary Crossentropy |
| **Batch Size** | 32 samples |
| **Epochs** | Up to 50 (with early stopping) |
| **Validation Split** | 15% of training data |
| **Test Split** | 15% of total data |
| **Train/Val/Test Ratio** | ~70% / 15% / 15% |

### Callbacks
- **EarlyStopping**: Monitor `val_loss`, patience=7 epochs
- **ReduceLROnPlateau**: Reduce learning rate by 0.5× if no improvement (patience=3), min_lr=1e-6
- **ModelCheckpoint**: Save best weights based on `val_accuracy` to `/content/best_model.keras`

### Metrics Tracked
- **Accuracy**: Fraction of correct predictions
- **Precision**: TP / (TP + FP) — of predicted faults, how many correct
- **Recall**: TP / (TP + FN) — of actual faults, how many detected
- **AUC**: Area under ROC curve — threshold-independent performance

---

## 7. Data Flow Example

### From Raw Signal to Prediction

```
Raw 1-D Audio Signal (20 kHz)
    ↓ [Load .txt file, select channel 0]
1-D Array (variable length, ~1M+ samples)
    ↓ [Preprocess: filter, normalize, clip]
Cleaned 1-D Signal
    ↓ [Window into 2048-sample segments, 50% overlap]
Multiple 2048-Sample Windows
    ↓ [STFT + log-magnitude + resize]
64×64 Spectrograms (grayscale, normalized)
    ↓ [Add channel: (64, 64) → (64, 64, 1)]
CNN Input Batch: shape (batch_size, 64, 64, 1)
    ↓ [Forward pass through CNN]
Output logits → Sigmoid activation
    ↓
Probability ∈ [0, 1]
    ↓ [Threshold at 0.5]
Class Label: "Healthy" or "Faulty"
```

---

## 8. Model Files & Checkpoints

### Training Artifacts
- **Best Model Weights**: Saved during training to `/content/best_model.keras`
  - Restored before test evaluation
  - Format: TensorFlow `.keras` (compatible with tf.keras.models)

### Local Checkpoints
- **No persistent weights in backend folder** — weights saved to Google Drive during Colab execution
- To use offline: Must re-train or export weights from trained session

### Dataset Location
- **Source**: Google Drive `/MyDrive/IMS_Bearing/1st_test/1st_test`
- **Format**: Multi-channel text files (channels separated by columns)
- **Channel Selection**: Channel 0 by default (adjustable in code)

---

## 9. Key Code Snippets

### Inference on New Signal
```python
def predict_from_raw_signal(raw_signal, channel=0, threshold=0.5):
    # Preprocess
    clean = preprocess_signal(raw_signal)
    # Segment
    wins = segment_signal(clean)
    # Spectrogram per window
    specs = np.array([window_to_spectrogram(w) for w in wins])
    specs = specs[:, :, :, np.newaxis]
    # Batch predict
    probs = model.predict(specs, verbose=0).ravel()
    mean_prob = float(probs.mean())
    label = 'Faulty' if mean_prob >= threshold else 'Healthy'
    return label, mean_prob
```

### Single Spectrogram Prediction
```python
def predict_single_spectrogram(spec_2d, threshold=0.5):
    x = spec_2d[np.newaxis, :, :, np.newaxis]  # (1, 64, 64, 1)
    prob = float(model.predict(x, verbose=0)[0, 0])
    label = 'Faulty' if prob >= threshold else 'Healthy'
    return label, prob
```

---

## 10. Important Notes

### Assumptions
1. **Run-to-Failure Data**: Files are chronologically sorted; later files have higher failure probability
2. **Stationarity**: Statistics change slowly over time (suitable for RUL estimation)
3. **Class Balance**: First/last 35% split creates balanced dataset (excludes transitional zone)
4. **Channel 0**: Default bearing channel; should match your data layout

### Potential Improvements
- **Real RUL Labels**: Current model uses binary labels; extend to continuous RUL regression
- **CNN-LSTM Hybrid**: Sequence 10 spectrograms for temporal degradation patterns (code included but not fully trained)
- **Data Augmentation**: Rotate, shift, or add noise to spectrograms
- **Ensemble Models**: Combine multiple CNN initializations
- **Threshold Tuning**: Adjust decision boundary based on cost matrix (e.g., false alarms vs missed faults)

### Deployment Considerations
- Model expects **single 64×64 grayscale image** or **batch of such images**
- Input preprocessing is **essential** — skipping filter/norm will degrade performance
- Monitor prediction confidence; retraining recommended if performance drifts
- Consider online learning for long-term deployment

---

## Summary Table

| Aspect | Specification |
|--------|---|
| **Input** | 2048-sample audio window @ 20 kHz → 64×64 log-magnitude spectrogram |
| **Output** | Binary probability [0, 1] for fault classification |
| **Architecture** | 3× Conv blocks (32/64/128 filters) + Dense classifier |
| **Parameters** | ~286k trainable parameters |
| **Training Data** | NASA IMS bearing dataset, run-to-failure, binary labeled |
| **Preprocessing** | Filter (500-9000 Hz) + Z-norm + clip + STFT |
| **Loss** | Binary crossentropy |
| **Optimizer** | Adam (lr=1e-3) |
| **Metrics** | Accuracy, Precision, Recall, AUC |
| **Format** | TensorFlow `.keras` (tf.keras.Sequential) |


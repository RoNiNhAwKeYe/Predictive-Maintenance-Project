# CNN-LSTM Hybrid Model for Bearing Fault Detection & RUL Prediction

## 🎯 Model Overview

This is a **hybrid CNN-LSTM architecture** that performs **dual-task learning**:
1. **Fault Detection**: Binary classification (Healthy/Faulty)
2. **RUL Estimation**: Remaining Useful Life prediction in hours

The model processes **sequential acoustic spectrograms** to capture both spatial features (from CNN) and temporal dependencies (from LSTM).

---

## 🏗️ Architecture Details

### Input Specification
| Property | Value |
|----------|-------|
| **Format** | Time series of 64×64 spectrograms |
| **Shape** | (batch_size, sequence_length, 64, 64, 1) |
| **Sequence Length** | Variable (typically 10-100 time steps) |
| **Sample Rate** | 20 kHz |
| **Time Window** | ~32ms per spectrogram (varies by sequence) |
| **Data Type** | Float32, range [0, 1] normalized |

### Feature Extraction (CNN)
```
TimeDistributed CNN Block:
  Conv2D(32, 3×3) → ReLU → BatchNorm → MaxPool(2×2)     [64×64 → 32×32]
    ↓
  Conv2D(64, 3×3) → ReLU → BatchNorm → MaxPool(2×2)     [32×32 → 16×16]
    ↓
  Conv2D(128, 3×3) → ReLU → BatchNorm → MaxPool(2×2)    [16×16 → 8×8]
    ↓
  Flatten + Dense(128, ReLU) + Dropout(0.3)              [→ 128 features]
```
Applied to each spectrogram in the sequence independently.

### Temporal Modeling (LSTM)
```
LSTM Layer 1:
  Input: (batch, sequence_length, 128)
  Units: 128, return_sequences=True, dropout=0.2
  Output: (batch, sequence_length, 128)
    ↓
LSTM Layer 2:
  Input: (batch, sequence_length, 128)
  Units: 64, dropout=0.2
  Output: (batch, 64)  ← Captures full sequence context
```

### Dual Output Heads (Multi-task Learning)

#### Fault Detection Branch
```
Dense(64, ReLU) + Dropout(0.3)
    ↓
Dense(1, Sigmoid)  → Output: [0.0, 1.0]
```
- **Output Range**: 0.0 (healthy) → 1.0 (faulty)
- **Threshold**: 0.5 for classification
- **Loss**: Binary Crossentropy
- **Weight**: 0.5 in total loss

#### RUL Prediction Branch
```
Dense(64, ReLU) + Dropout(0.3)
    ↓
Dense(1, ReLU)  → Output: [0, ∞)
```
- **Output Range**: Hours (0.5 - 200+)
- **Activation**: ReLU ensures non-negative RUL
- **Loss**: Mean Squared Error (MSE)
- **Weight**: 0.5 in total loss

---

## 📊 Input Preprocessing

### Audio to Spectrogram Pipeline
```
Raw Audio Signal (20 kHz)
    ↓ [Bandpass Filter: 500-9000 Hz]
Filtered Signal
    ↓ [Z-score: μ=0, σ=1]
Normalized Signal
    ↓ [Clip Outliers: ±5σ]
Clipped Signal
    ↓ [STFT: window=256, hop=128]
Complex Spectrogram
    ↓ [Log Magnitude: log(1 + |X|)]
Log Spectrogram
    ↓ [Resize to 64×64]
Final Spectrogram [0, 1] normalized
```

### Sequence Formation
- Raw audio: ~2-10 seconds per sample
- Split into overlapping windows
- Each window → spectrogram
- Stack spectrograms in temporal order
- Sequences: 10-100 timesteps (varies by data)

---

## 🎓 Training Configuration

### Hyperparameters (Typical)
| Parameter | Value |
|-----------|-------|
| **Optimizer** | Adam (lr=1e-3) |
| **Batch Size** | 32 |
| **Epochs** | 50 (with early stopping) |
| **Validation Split** | 20% |
| **Early Stopping** | Patience=7 epochs |
| **Learning Rate Schedule** | ReduceLROnPlateau (patience=3) |

### Loss Functions
```
Total Loss = 0.5 × BCE(fault_pred, fault_true) + 0.5 × MSE(rul_pred, rul_true)
```
- **Balanced**: Equal weight on both tasks
- **Can be tuned**: Increase fault weight if more important

### Metrics
| Task | Metrics |
|------|---------|
| **Fault Detection** | Accuracy, Precision, Recall, AUC-ROC |
| **RUL Prediction** | MAE (Mean Absolute Error), RMSE |

---

## 📈 Expected Performance

### Fault Detection
- **Accuracy**: ~90-95%
- **Precision**: ~88-94%
- **Recall**: ~90-96%
- **AUC-ROC**: ~0.95-0.99

### RUL Prediction
- **Mean Absolute Error (MAE)**: ±2-5 hours
- **Root Mean Squared Error (RMSE)**: ±3-7 hours
- **Mean Absolute Percentage Error (MAPE)**: ~10-20%

*Performance varies based on training data quality and diversity.*

---

## 🔌 Deployment & Inference

### Model File
```
best_model.keras
├─ Weights: CNN filters + LSTM gates + Dense layers
├─ Architecture: Functional model with 2 outputs
└─ Input Shape: (None, None, 64, 64, 1)  # Dynamic sequence length
```

### Single Prediction (Inference)
```python
# Load model
model = load_model('best_model.keras')

# Prepare input
audio_signal = load_audio('bearing.wav')  # (...) audio samples
spectrograms = audio_to_spectrograms(audio_signal)  # (seq_len, 64, 64)
input_batch = np.expand_dims(spectrograms, (0, -1))  # (1, seq_len, 64, 64, 1)

# Predict
fault_prob, rul_hours = model.predict(input_batch)
# fault_prob: 0.75 (75% probability of fault)
# rul_hours: 8.2 (8.2 hours remaining)
```

### Batch Prediction
```python
# For multiple samples
fault_probs, rul_estimates = model.predict(input_batch, batch_size=32)
# Faster inference with batching
```

---

## 🛠️ Advanced Usage

### Fine-tuning on Custom Data
```python
# Load pre-trained model
model = load_model('best_model.keras')

# Freeze CNN layers, fine-tune LSTM only
for layer in model.layers[:-4]:
    layer.trainable = False

# Compile with lower learning rate
model.compile(
    optimizer='adam',  # New lr=1e-5
    loss={'fault_detection': 'binary_crossentropy', 
          'rul_prediction': 'mse'},
    loss_weights={'fault_detection': 0.5, 'rul_prediction': 0.5}
)

# Train on custom data
model.fit(custom_data, ...)
```

### Ensemble Predictions
Combine multiple model predictions for robustness:
```python
fault_probs = []
rul_estimates = []

for model in models:
    f, r = model.predict(input_data)
    fault_probs.append(f)
    rul_estimates.append(r)

# Average predictions
avg_fault = np.mean(fault_probs)  # 0.72 (72% fault)
avg_rul = np.mean(rul_estimates)  # 7.8 hours
```

### Transfer Learning
Use pre-trained weights as initialization:
```python
# Load weights into different architecture
model_custom = build_custom_architecture()
pre_trained_model = load_model('best_model.keras')

# Copy compatible weights
model_custom.get_layer('lstm_1').set_weights(
    pre_trained_model.get_layer('lstm').get_weights()
)
```

---

## 📊 API Response Format

### Single Prediction Response
```json
{
  "status": "FAULTY",
  "confidence": 0.72,
  "fault_probability": 0.72,
  "health_probability": 0.28,
  "rul_hours": 8.2,
  "rul_status": "WARNING",
  "alert_level": "CRITICAL",
  "threshold": 0.5,
  "timestamp": "2024-04-05T14:23:45.123456",
  "spectrogram_shape": "(64, 64)"
}
```

### Response Field Meanings
| Field | Meaning |
|-------|---------|
| **status** | HEALTHY / FAULTY |
| **confidence** | Fault probability [0-1] |
| **fault_probability** | Same as confidence |
| **health_probability** | 1 - confidence |
| **rul_hours** | Estimated hours until failure |
| **rul_status** | CRITICAL (≤5h) / WARNING (5-20h) / GOOD (>20h) |
| **alert_level** | CRITICAL / NORMAL |
| **threshold** | Decision boundary (0.5) |
| **timestamp** | ISO 8601 timestamp |

---

## 🔍 Interpreting Results

### Fault Detection Interpretation
| Confidence | Status | Action |
|------------|--------|--------|
| 0.0 - 0.3 | Healthy | Monitor normally |
| 0.3 - 0.5 | Borderline | Increase monitoring frequency |
| 0.5 - 0.7 | Faulty | Plan maintenance |
| 0.7 - 1.0 | Faulty | Schedule immediate maintenance |

### RUL Interpretation
| RUL (hours) | Status | Action |
|-------------|--------|----|
| > 50 | GOOD | Normal operation |
| 20 - 50 | GOOD | Routine monitoring |
| 5 - 20 | WARNING | Schedule maintenance within days |
| ≤ 5 | CRITICAL | Schedule urgent maintenance |
| ≤ 0.5 | FAILURE | Stop operation immediately |

---

## 💡 Advantages of CNN-LSTM
1. **Spatial Feature Learning**: CNN extracts frequency patterns from spectrograms
2. **Temporal Dependency**: LSTM captures degradation trends over time
3. **Multi-task Learning**: Joint optimization of fault & RUL improves generalization
4. **Flexible Sequence Length**: Can handle variable-length time series
5. **Robustness**: Dual outputs provide complementary information

---

## ⚠️ Limitations & Considerations

1. **Training Data**: Requires diverse failure modes and degradation histories
2. **Sequence Length**: Must be consistent or padded/truncated
3. **Computational Cost**: More expensive than simple CNN (LSTM has more parameters)
4. **Inference Time**: ~100-200ms per prediction (vs ~50ms for simple CNN)
5. **RUL Uncertainty**: Predictions have larger variance near failure point

---

## 📚 References

### Relevant Research
- LSTMs for remaining useful life prediction: [IEEE Papers]
- CNN-LSTM hybrid architectures: [DeepLearning.com]
- Bearing fault detection: [NASA Prognostics Center]

### Dataset
- NASA IMS Bearing Run-to-Failure Dataset
- CWRU Bearing Dataset
- Company-specific bearing degradation data

---

## 🔧 Troubleshooting

### Problem: RUL predictions are always negative
**Solution**: Check ReLU activation in RUL head - should prevent negative outputs

### Problem: Fault and RUL predictions disagree
**Solution**: Check loss weights - balance between tasks using `loss_weights` parameter

### Problem: Model overfits easily
**Solution**: 
- Increase dropout rates
- Increase LSTM sequence length (more timesteps)
- Reduce model complexity
- Use data augmentation

### Problem: Training is slow
**Solution**:
- Use GPU acceleration
- Reduce sequence length
- Use smaller batch sizes
- Consider simpler architecture

---

**Model Version**: 1.0  
**Last Updated**: April 2024  
**Framework**: TensorFlow/Keras 2.14+

# CNN Technical Architecture & Data Flow

## 1. Signal Processing Pipeline

```
Raw .txt File (Multi-Channel, 20 kHz)
│
├─→ Load & Select Channel (default: channel 0)
│   └─→ 1-D Time Series Array (1,000,000+ samples)
│
├─→ PREPROCESSING BLOCK
│   ├─ Step 1: Handle NaN/Inf via interpolation
│   │   └─→ No discontinuities
│   │
│   ├─ Step 2: Butterworth Bandpass Filter
│   │   ├─ Cutoff: 500 Hz (low) — 9,000 Hz (high)
│   │   ├─ Order: 4
│   │   ├─ Method: filtfilt (zero-phase)
│   │   └─→ Isolates bearing fault frequencies
│   │
│   ├─ Step 3: Z-Score Normalization
│   │   ├─ Formula: (signal - mean) / std
│   │   └─→ Zero mean, unit variance
│   │
│   └─ Step 4: Outlier Clipping
│       ├─ Range: [-5σ, +5σ]
│       └─→ Remove noise extremes
│
├─→ WINDOWING BLOCK
│   ├─ Window Size: 2,048 samples (≈102 ms @ 20 kHz)
│   ├─ Hop Size: 1,024 samples (50% overlap)
│   └─→ Output: N × 2,048 windows
│
├─→ SPECTROGRAM GENERATION (STFT)
│   ├─ FFT Window: 256 samples
│   ├─ FFT Overlap: 128 samples (50%)
│   ├─ Window Function: Hann
│   │
│   ├─ Compute STFT Matrix (complex)
│   │   └─→ Shape: (129 freq bins, T time frames)
│   │
│   ├─ Power Spectrogram: |STFT|²
│   │   └─→ (129, T) array
│   │
│   ├─ Log Compression: log(1 + S)
│   │   └─→ Compress dynamic range
│   │
│   ├─ Min-Max Normalization: (S - min) / (max - min)
│   │   └─→ Values in [0, 1]
│   │
│   └─ Resize to 64×64 (bilinear interpolation)
│       └─→ Consistent input size for CNN
│
└─→ Final Output: 64×64 Grayscale Image (normalized)
    └─→ Shape: (64, 64) → Add channel (64, 64, 1)
```

---

## 2. CNN Detailed Architecture

### Layer-by-Layer Breakdown

```
┌──────────────────────────────────────────────────────┐
│ INPUT LAYER                                          │
│ Shape: (batch_size, 64, 64, 1)                       │
│ Type: Float32, normalized [0, 1]                    │
└──────────────────┬───────────────────────────────────┘

┌──────────────────┴───────────────────────────────────┐
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│ ┃ CONV BLOCK 1                                  ┃  │
│ ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  │
│ ┃ Conv2D(32, kernel_size=(3,3), padding='same')┃  │
│ ┃   Filters: 32                                  ┃  │
│ ┃   Kernel: 3×3                                 ┃  │
│ ┃   Stride: 1                                   ┃  │
│ ┃   Activation: ReLU                            ┃  │
│ ┃   Parameters: 32×(3×3×1 + 1) = 320           ┃  │
│ ┃   Output shape: (64, 64, 32)                  ┃  │
│ ┃                                                ┃  │
│ ┃ BatchNormalization()                           ┃  │
│ ┃   Normalizes across batch dimension            ┃  │
│ ┃   Learnable scale & shift (2×32 params)       ┃  │
│ ┃   Output shape: (64, 64, 32)                  ┃  │
│ ┃                                                ┃  │
│ ┃ MaxPooling2D(pool_size=(2,2))                 ┃  │
│ ┃   Stride: 2 (default)                         ┃  │
│ ┃   Output shape: (32, 32, 32)                  ┃  │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
└──────────────────┬───────────────────────────────────┘

┌──────────────────┴───────────────────────────────────┐
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│ ┃ CONV BLOCK 2                                  ┃  │
│ ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  │
│ ┃ Conv2D(64, kernel_size=(3,3), padding='same')┃  │
│ ┃   Filters: 64                                  ┃  │
│ ┃   Parameters: 64×(3×3×32 + 1) = 18,496       ┃  │
│ ┃   Output shape: (32, 32, 64)                  ┃  │
│ ┃                                                ┃  │
│ ┃ BatchNormalization()                           ┃  │
│ ┃   Output shape: (32, 32, 64)                  ┃  │
│ ┃                                                ┃  │
│ ┃ MaxPooling2D(pool_size=(2,2))                 ┃  │
│ ┃   Output shape: (16, 16, 64)                  ┃  │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
└──────────────────┬───────────────────────────────────┘

┌──────────────────┴───────────────────────────────────┐
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│ ┃ CONV BLOCK 3                                  ┃  │
│ ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  │
│ ┃ Conv2D(128, kernel_size=(3,3), padding='same')┃  │
│ ┃   Filters: 128                                 ┃  │
│ ┃   Parameters: 128×(3×3×64 + 1) = 73,856      ┃  │
│ ┃   Output shape: (16, 16, 128)                 ┃  │
│ ┃                                                ┃  │
│ ┃ BatchNormalization()                           ┃  │
│ ┃   Output shape: (16, 16, 128)                 ┃  │
│ ┃                                                ┃  │
│ ┃ MaxPooling2D(pool_size=(2,2))                 ┃  │
│ ┃   Output shape: (8, 8, 128)                   ┃  │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
└──────────────────┬───────────────────────────────────┘

┌──────────────────┴───────────────────────────────────┐
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│ ┃ FLATTEN LAYER                                 ┃  │
│ ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  │
│ ┃ Reshape: (8, 8, 128) → (8,192,)              ┃  │
│ ┃ Total units: 8 × 8 × 128 = 8,192            ┃  │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
└──────────────────┬───────────────────────────────────┘

┌──────────────────┴───────────────────────────────────┐
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│ ┃ DENSE LAYER (Feature Fusion)                 ┃  │
│ ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  │
│ ┃ Dense(256, activation='relu')                 ┃  │
│ ┃   Input units: 8,192                          ┃  │
│ ┃   Output units: 256                           ┃  │
│ ┃   Parameters: 8,192×256 + 256 = 2,097,408    ┃  │
│ ┃   Activation: ReLU (f(x) = max(0, x))        ┃  │
│ ┃                                                ┃  │
│ ┃ Dropout(rate=0.5)                             ┃  │
│ ┃   Randomly zeros 50% of units during training ┃  │
│ ┃   -> Prevents co-adaptation                   ┃  │
│ ┃   -> No effect during inference               ┃  │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
└──────────────────┬───────────────────────────────────┘

┌──────────────────┴───────────────────────────────────┐
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│ ┃ OUTPUT LAYER (Binary Classification)         ┃  │
│ ┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫  │
│ ┃ Dense(1, activation='sigmoid')                ┃  │
│ ┃   Input units: 256                            ┃  │
│ ┃   Output units: 1                             ┃  │
│ ┃   Parameters: 256 + 1 = 257                   ┃  │
│ ┃   Activation: Sigmoid (σ(x) = 1/(1+e^-x))    ┃  │
│ ┃   Output range: [0.0, 1.0]                   ┃  │
│ ┃   Interpretation: P(Faulty)                   ┃  │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
└──────────────────┬───────────────────────────────────┘

┌──────────────────┴───────────────────────────────────┐
│ OUTPUT: Probability ∈ [0.0, 1.0]                     │
│ Classification: p ≥ 0.5 → "Faulty" | p < 0.5 → "Healthy" │
└──────────────────────────────────────────────────────┘
```

### Parameter Summary

```
Layer                          Params         Output
─────────────────────────────────────────────────────────
Input                          0              (64, 64, 1)
Conv2D(32, 3×3)                320            (64, 64, 32)
BatchNorm                       128            (64, 64, 32)
MaxPool(2×2)                   0              (32, 32, 32)

Conv2D(64, 3×3)                18,496         (32, 32, 64)
BatchNorm                       256            (32, 32, 64)
MaxPool(2×2)                   0              (16, 16, 64)

Conv2D(128, 3×3)               73,856         (16, 16, 128)
BatchNorm                       512            (16, 16, 128)
MaxPool(2×2)                   0              (8, 8, 128)

Flatten                         0              (8192,)
Dense(256)                      2,097,408      (256,)
Dropout(0.5)                   0              (256,)
Dense(1, sigmoid)               257            (1,)

─────────────────────────────────────────────────────────
TOTAL TRAINABLE PARAMS: 2,191,233
TOTAL NON-TRAINABLE PARAMS: 896 (batch norm running stats)
```

---

## 3. Data Flow Diagram (End-to-End)

```
┌─────────────────────────────────────────────────┐
│ RAW ACOUSTIC FILE (bearing_data_001.txt)        │
│ 20 kHz sampling, multi-channel, variable length │
└────────────────┬────────────────────────────────┘
                 │ shape: (1000000, 4)
                 ↓
         Load & Select Channel 0
                 │
                 ↓ shape: (1000000,)
         ┌───────────────────────┐
         │  PREPROCESSING BLOCK  │
         ├───────────────────────┤
         │ 1. Handle NaN/Inf     │
         │ 2. Bandpass (500-9kHz)│
         │ 3. Z-norm [-5,+5]     │
         │ 4. Clip outliers      │
         └───────────────────────┘
                 │ shape: (1000000,)
                 ↓
        ┌─────────────────┐
        │ WINDOWING       │
        ├─────────────────┤
        │ Window: 2048    │
        │ Hop: 1024       │
        │ Overlap: 50%    │
        └─────────────────┘
                 │ shape: (976, 2048)
                 ↓
        ┌──────────────────────┐
        │ SPECTROGRAM BATCH    │
        ├──────────────────────┤
        │ Per window:          │
        │ - STFT (256→129,T)   │
        │ - Power |·|²         │
        │ - Log scale          │
        │ - Min-max norm [0,1] │
        │ - Resize to 64×64    │
        └──────────────────────┘
                 │ shape: (976, 64, 64)
                 ↓
        ┌──────────────────────┐
        │ ADD CHANNEL DIM      │
        ├──────────────────────┤
        │ Grayscale → RGB-like │
        │ (64,64) → (64,64,1)  │
        └──────────────────────┘
                 │ shape: (976, 64, 64, 1)
                 ↓
        ┌──────────────────────┐
        │ LABEL ASSIGNMENT     │
        ├──────────────────────┤
        │ File index 0-341:    │
        │ y = 0 (Healthy)      │
        │                      │
        │ File index 342-647:  │
        │ Excluded             │
        │                      │
        │ File index 648-976:  │
        │ y = 1 (Faulty)       │
        └──────────────────────┘
                 │ (X, y) pairs created
                 ↓
        ┌──────────────────────┐
        │ TRAIN/VAL/TEST SPLIT │
        ├──────────────────────┤
        │ Train: 70% (~3700)   │
        │ Val:   15% (~800)    │
        │ Test:  15% (~800)    │
        └──────────────────────┘
                 │
                 ↓
        ┌─────────────────────────────┐
        │ CNN TRAINING LOOP           │
        ├─────────────────────────────┤
        │ Optimizer: Adam             │
        │ Loss: Binary Crossentropy   │
        │ Batch Size: 32              │
        │ Max Epochs: 50              │
        │ Callbacks:                  │
        │ - EarlyStopping (pat=7)     │
        │ - ReduceLROnPlateau (pat=3) │
        │ - ModelCheckpoint           │
        └─────────────────────────────┘
                 │
                 ↓                   ↓
        ┌──────────────────┐  ┌──────────────────┐
        │ TRAINING CURVES  │  │ BEST WEIGHTS     │
        │ Accuracy/Loss    │  │ saved to disk    │
        └──────────────────┘  └──────────────────┘
                 │                   │
                 └──────────┬────────┘
                            ↓
                 ┌──────────────────────┐
                 │ INFERENCE ON TEST    │
                 ├──────────────────────┤
                 │ Load best weights    │
                 │ Forward pass         │
                 │ Output probabilities │
                 │ Compare vs y_true    │
                 └──────────────────────┘
                            │
                            ↓
                 ┌──────────────────────┐
                 │ EVALUATION METRICS   │
                 ├──────────────────────┤
                 │ Accuracy             │
                 │ Precision            │
                 │ Recall               │
                 │ AUC-ROC              │
                 │ Confusion Matrix     │
                 └──────────────────────┘
```

---

## 4. Feature Extraction Across Layers

```
Input Spectrogram (64×64)
│
├─→ Layer 1 (32 filters, 3×3 kernels)
│   └─→ Features: Edges, local texture
│       └─→ Output: 32 feature maps (64×64 each)
│           Edges in frequency domain, contours
│
├─→ MaxPool (2×2) → (32×32, 32 channels)
│
├─→ Layer 2 (64 filters, 3×3 kernels)
│   └─→ Features: Regions, patterns
│       └─→ Output: 64 feature maps (32×32 each)
│           Harmonic patterns, frequency bands
│
├─→ MaxPool (2×2) → (16×16, 64 channels)
│
├─→ Layer 3 (128 filters, 3×3 kernels)
│   └─→ Features: Degradation signatures
│       └─→ Output: 128 feature maps (16×16 each)
│           High-level fault indicators
│
├─→ MaxPool (2×2) → (8×8, 128 channels)
│
├─→ Flatten → 8,192 units
│
├─→ Dense(256) → Combines features
│   └─→ 256 neurons learning decision boundaries
│
└─→ Dense(1, sigmoid) → Binary output
    └─→ P(Faulty) = sigmoid(combined_features)
```

---

## 5. Training Dynamics

```
Forward Pass (Training)
────────────────────────────────────
Input spectrogram (64, 64, 1)
        ↓
 CNN forward pass
        ↓
 Sigmoid output: p ∈ [0, 1]
        ↓
 Loss = -[y·log(p) + (1-y)·log(1-p)]
        ↓
Backward Pass (Gradients)
────────────────────────────────────
dL/dw computed via chain rule
        ↓
Adam optimizer updates weights:
    - Momentum: m = β₁·m + (1-β₁)·g
    - Variance: v = β₂·v + (1-β₂)·g²
    - Update: w ← w - α·m/(√v + ε)
        ↓
Batch Normalization:
    - Running mean/variance tracked
    - Gradients smooth due to normalization
        ↓
 Epoch complete
```

---

## 6. Key Dimensions at Each Stage

### Image Path (Convolutional Blocks)

```
Input:      64 × 64 × 1
Conv1:      64 × 64 × 32  (320 params)
Pool1:      32 × 32 × 32  (max pool 2×2)
Conv2:      32 × 32 × 64  (18.5K params)
Pool2:      16 × 16 × 64  (max pool 2×2)
Conv3:      16 × 16 × 128 (73.9K params)
Pool3:      8 × 8 × 128   (max pool 2×2)
Flatten:    8,192          (8×8×128 = 8,192)
```

### Feature Path (Dense Layers)

```
Flatten:    8,192
Dense1:     256             (2.1M params)
Dropout:    256 (during training, 50% dropped)
Dense2:     1               (257 params)
Output:     Sigmoid → [0, 1]
```

---

## 7. Memory & Computation Requirements

### For Batch Size 32

```
Input:      32 × 64 × 64 × 1 = 131 KB
Conv1:      32 × 64 × 64 × 32 = 4.2 MB
Conv2:      32 × 32 × 32 × 64 = 2.1 MB
Conv3:      32 × 16 × 16 × 128 = 1.0 MB
Dense:      32 × 256 = 8 KB
Output:     32 × 1 = 32 bytes

Total forward activation: ~7.4 MB
Gradients + optimizer states: ~21.2 MB
Total per batch: ~30-40 MB (with overhead)
```

### Computation (FLOPs per batch)

```
Conv layers:  ~450M FLOPs
Dense layers: ~67M FLOPs
Total:        ~520M FLOPs per batch
```

---

## Summary: Architecture Strengths

1. **Progressive Feature Abstraction**: Early layers capture low-level spectral details; later layers learn high-level degradation patterns
2. **Computational Efficiency**: 286K parameters is lightweight for image classification
3. **Regularization**: BatchNorm + Dropout prevent overfitting
4. **Flexible Input Handling**: Accepts variable-length signals via windowing
5. **Interpretable Output**: Single probability output is easy to threshold and explain


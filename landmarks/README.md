# Facial Landmarks Model

This module implements a deep learning model for predicting empathy/valence scores from facial landmark sequences extracted from video frames. The model uses a 1D Convolutional Neural Network (CNN) architecture to learn temporal patterns in facial landmark movements.

## Overview

The model processes sequences of facial landmarks (68 points per frame) to predict continuous emotion/empathy scores. The architecture is designed to capture temporal dynamics in facial expressions by analyzing how landmark positions change over short time windows.

**Key Insight**: While facial landmarks are extracted from 2D images, the model processes them as **temporal sequences of coordinate vectors**, not as spatial image data. This is why Conv1D (for time series) is used instead of Conv2D (for images).

## Dependencies

- numpy
- keras
- tensorflow
- scikit-image
- pandas
- scipy
- dlib (for landmark detection)
- opencv-python (cv2)
- matplotlib

## Data Structure

### Input Data Format

- **68 facial landmarks** per frame, each with (x, y) coordinates
- **136 features per frame** (68 landmarks × 2 coordinates)
- Data is stored as CSV files with shape: `(num_frames, 136)`
- Each row represents one video frame with flattened landmark coordinates: `[x1, y1, x2, y2, ..., x68, y68]`

### Data Preprocessing Pipeline

1. **Video Frame Extraction**: Extract frames from video files
2. **Face Detection**: Detect faces in each frame (using dlib)
   - Actor face: left half of frame (0-1280 pixels)
   - Subject face: right half of frame (1280-2560 pixels)
3. **Landmark Detection**: Extract 68 facial landmarks per detected face
4. **Coordinate Extraction**: Flatten landmarks into 136-dimensional vectors
5. **Windowing**: Create temporal windows of 5 consecutive frames
6. **Normalization**: Standardize features (zero mean, unit variance)

### Final Input Shape

After preprocessing, the model receives:
- **Training/Validation**: `(num_samples, window_size=5, N_features=136)`
- Each sample is a sequence of 5 frames, each with 136 landmark features

## Model Architecture

### Current Model: `build_model()` - CNN-based

The model uses a **1D Convolutional Neural Network** to process temporal sequences:

```
Input: (5, 136)  # 5 frames × 136 features
  ↓
Conv1D(100 filters, kernel_size=2) + BatchNorm + ReLU
  ↓
Conv1D(100 filters, kernel_size=2) + ReLU
  ↓
Conv1D(160 filters, kernel_size=2) + ReLU
  ↓
Conv1D(160 filters, kernel_size=2) + ReLU
  ↓
GlobalAveragePooling1D()  # Aggregates temporal dimension
  ↓
Dense(32, ReLU)  # Feature compression
  ↓
Dense(1, linear)  # Regression output
```

#### Architecture Design Choices

1. **Conv1D Layers**: 
   - Operate along the **temporal dimension** (not spatial)
   - `kernel_size=2`: Captures patterns between consecutive frames
   - Increasing filters (100 → 160): Learns more complex temporal patterns in deeper layers

2. **BatchNormalization**: 
   - Applied after first convolution to stabilize training
   - Normalizes activations to prevent internal covariate shift

3. **GlobalAveragePooling1D**: 
   - Reduces temporal dimension to fixed-size vector
   - Provides temporal invariance (order-independent aggregation)
   - Common pattern in sequence-to-value tasks

4. **Dense Layers**: 
   - Maps aggregated features to single continuous value (empathy/valence score)
   - Final layer uses linear activation for regression

### Alternative Model: `build_model_LSTM()` - LSTM-based

An alternative LSTM-based architecture is available but currently commented out. It includes:
- LSTM layer for sequence modeling
- Optional attention mechanism (`AttentionWeightedAverage`)
- Similar dense layers for final prediction

**Note**: The LSTM model requires additional configuration parameters that are not currently defined in the codebase.

## Why Conv1D Instead of Conv2D?

This is a common point of confusion. Here's why Conv1D is correct:

### Data Representation
- **Landmarks are extracted from 2D images** → but stored as **1D coordinate vectors**
- The 2D spatial information (x, y coordinates) is **already flattened** into a feature vector
- Input shape: `(time_steps, features)` = `(5, 136)`, not `(height, width, channels)`

### Conv1D vs Conv2D
- **Conv1D**: Convolves along the **temporal dimension** (time axis)
  - Learns: "How do landmark patterns change over consecutive frames?"
  - Correct for: Sequences of feature vectors
  
- **Conv2D**: Convolves along **spatial dimensions** (height × width)
  - Would be used for: Raw image pixels or 2D spatial grids
  - Would misinterpret: `(5, 136)` as `(height=5, width=136)` ❌

### When Would Conv2D Be Appropriate?
- Processing raw face images: `(height, width, channels)`
- Creating 2D heatmaps from landmarks
- Using landmark positions to create spatial grid representations

## Training Configuration

### Hyperparameters (from `landmarks_main.py`)

- **Window Size**: 5 frames
- **Batch Size**: 512
- **Epochs**: 1000 (with early stopping)
- **Learning Rate**: 0.000001
- **Optimizer**: Adam (with decay=0.0)
- **Loss Function**: CCC Error (Concordance Correlation Coefficient)
- **Early Stopping**: Patience = 1000 epochs (monitors validation loss)

### Loss Function: CCC Error

The model uses **Concordance Correlation Coefficient (CCC)** as the loss function:

```
CCC = 2ρσxσy / (σx² + σy² + (μx - μy)²)
```

Where:
- `ρ`: Pearson correlation coefficient
- `σx, σy`: Standard deviations of true and predicted values
- `μx, μy`: Means of true and predicted values

CCC measures both correlation and agreement between predictions and ground truth, making it ideal for continuous emotion prediction tasks.

### Metrics

- **CCC (Concordance Correlation Coefficient)**: Primary evaluation metric
- **Pearson Correlation**: Secondary metric for correlation strength
- Model checkpoints are saved based on best validation CCC score

### Training/Validation Split

- **Training Subjects**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- **Training Stories**: [1, 4, 5, 8]
- **Validation Subjects**: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- **Validation Stories**: [2]

## Usage

### 1. Preprocessing: Extract Landmarks from Videos

Run the preprocessing script to extract facial landmarks from video files:

```bash
python landmarks_preprocessing.py
```

**Requirements**:
- Download `shape_predictor_68_face_landmarks.dat` from [dlib models](https://github.com/ageitgey/face_recognition_models/blob/master/face_recognition_models/models/shape_predictor_68_face_landmarks.dat)
- Place it in the `landmarks/` directory
- Configure input video paths in the script

**Output**:
- CSV files with landmark coordinates: `landmarksActor.csv`, `landmarksSubject.csv`
- Each CSV has shape: `(num_frames, 136)`

### 2. Training the Model

Train the model using the main script:

```bash
python landmarks_main.py
```

**Configuration**:
- Update `base_path_X` and `base_path_Y` in `landmarks_main.py` to point to your data directories
- Set `subject_data = True` or `actor_data = True` to choose which face to process
- Adjust hyperparameters as needed

**Output**:
- Creates experiment directory with timestamp: `experiment_YYYY-MM-DD_HH_MM_SS/`
- Saves best model weights: `best_model.h5`
- Saves model summary: `modelsummary.txt`
- TensorBoard logs: `logs/experiment_*/`

### 3. Making Predictions

After training, predictions are automatically saved if `save_predictions_test = True`:

```python
model.load_weights('best_model.h5')
save_predictions()
```

## File Structure

```
landmarks/
├── README.md                          # This file
├── landmarks_main.py                  # Main training script
├── landmarks_preprocessing.py         # Video → landmarks extraction
├── model.py                           # Model architecture definitions
├── utils.py                           # Data loading, preprocessing, metrics
├── shape_predictor_68_face_landmarks.dat  # dlib landmark predictor (download separately)
├── faces_extracted/                   # Preprocessed landmark data
│   └── training/
│       └── Subject_X_Story_Y/
│           ├── Actor_face_landmarks/
│           │   └── landmarksActor.csv
│           └── Subject_face_landmarks/
│               └── landmarksSubject.csv
└── tmp/                               # Temporary video files during preprocessing
```

## Key Concepts

### Temporal Convolution for Sequences

The model learns **temporal patterns** in facial landmark movements:
- Short-term: Frame-to-frame transitions (kernel_size=2)
- Medium-term: 3-4 frame sequences (deeper conv layers)
- Long-term: Full 5-frame window (global pooling)

### Feature Hierarchy

```
Frame-level features (136 dims: landmark coordinates)
    ↓ Conv1D
Short-term temporal patterns (2-frame transitions)
    ↓ Conv1D
Medium-term patterns (3-4 frame sequences)
    ↓ Conv1D
Longer-term patterns (full 5-frame window)
    ↓ GlobalAveragePooling
Aggregated temporal representation
    ↓ Dense
Continuous emotion/empathy prediction
```

### Why This Architecture?

1. **Efficiency**: Conv1D is faster than LSTM for short sequences
2. **Temporal Dynamics**: Captures how facial expressions evolve over time
3. **Robustness**: Global pooling provides temporal invariance
4. **Interpretability**: Convolutional filters can be visualized to understand learned patterns

## References & Learning Resources

### Conv1D Fundamentals
- [Keras Conv1D Documentation](https://keras.io/api/layers/convolution_layers/convolution1d/)
- Understanding 1D Convolutional Neural Networks for time series

### Temporal Modeling
- Temporal Convolutional Networks (TCN) papers
- Sequence modeling with CNNs (Stanford CS231n)

### BatchNormalization
- "Batch Normalization: Accelerating Deep Network Training" (Ioffe & Szegedy, 2015)

### Pooling Strategies
- Global Average Pooling (Lin et al., 2013 - Network in Network)

### Facial Landmark Analysis
- dlib facial landmark detection documentation
- Facial expression recognition papers using temporal CNNs

## Notes

- The model processes **sequences of coordinate vectors**, not raw images
- Spatial relationships between landmarks are encoded in the feature vector ordering
- The 68 landmarks follow a standard order: jaw (0-16), eyebrows (17-26), nose (27-35), eyes (36-47), mouth (48-67)
- Face detection uses caching (every 10 frames) for efficiency
- Missing face detections are handled by copying previous frame data

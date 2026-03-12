# EEG Sleep Stage Prediction with Deep Learning
Automated sleep-stage prediction from overnight EEG recordings using a CNN-based architecture.
This project aims to classify sleep stages from EEG signals using deep learning. The goal is to automate sleep-stage scoring, a critical step in sleep research and clinical diagnostics.

---
## Table of Contents
1. [Highlights / Key Results](#highlights--key-results)  
2. [Dataset](#dataset)  
3. [Preprocessing Pipeline](#preprocessing-pipeline)  
4. [Model Architecture and Parameters](#model-architecture-and-parameters)  
5. [Handling Class Imbalance](#handling-class-imbalance)  
6. [Training and Evaluation](#training-and-evaluation)  
7. [Results and Visualizations](#results-and-visualizations)  
8. [Random-Sample Inference](#random-sample-inference)  

---
## Highlights / Key Results
- **Initial run (imbalanced data):** Training/validation/test accuracy ≈ **0.85**. Predictions biased towards majority class.  
- **After downsampling majority class:**  
  - Best validation accuracy = **0.8904**  
  - Training loss = **0.3264**  
  - Validation loss = **0.3753**  
  - Test accuracy = **0.9167**
- Confusion matrices and learning curves are included:
### Learning curve (before)
- ![Learning curve (before)](Before_Optimization/Images/training_curves.png)
### Confusion matrix (before)
- ![Confusion matrix (before)](Before_Optimization/Images/confusion_matrix.png)
### Learning curve (after)
- ![Learning curve (after)](After_optimization/Images/training_curves.png)
### Confusion matrix (after)
- ![Confusion matrix (after)](After_optimization/Images/confusion_matrix.png)  

#### Additionally, **20 random test samples** were evaluated post-training and results are included.
---

## Dataset
- **Subjects:** 153 test subjects.  
- **Data:** Continuous overnight EEG recordings with labeled sleep-stage segments.  
- **Segmentation:** Each recording split into **30-second epochs** (standard in sleep research).  
- **Folder structure (recommended):**
```text
data/
raw_data/
subject01_X.npy
subject01_y.npy
...
processed_data/
subject01_X.npy
subject01_y.npy

```

---

## Preprocessing Pipeline
Steps performed:
1. Split continuous EEG into **30s segments**.  
2. Apply **bandpass filtering** (e.g., 0.5–30 Hz).  
3. Store processed segments into class-based folders.  
4. **Lazy loading** during training (memory-mapped `.npy` files).  
5. **Z-score normalization** applied per epoch on-the-fly.

---

## Model Architecture and Parameters
**Model type:** 1D-CNN with Squeeze-and-Excitation (SE) blocks.

| Layer | Type | Channels / Units | Kernel / Stride | Activation | Dropout | BatchNorm | Pooling | Output Shape |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Block 1** | Conv1d + SE | 64 | k=7, s=1 | ReLU | - | Yes | MaxPool(4) | (B, 64, T/4) |
| **Block 2** | Conv1d + SE | 128 | k=5, s=1 | ReLU | - | Yes | MaxPool(4) | (B, 128, T/16) |
| **Block 3** | Conv1d + SE | 256 | k=3, dil=2 | ReLU | - | Yes | None | (B, 256, T/16) |
| **Global** | AdaptiveAvgPool1d | - | - | - | - | - | - | (B, 256, 1) |
| **FC** | Linear | 128 → 5 | - | ReLU | 0.5 / 0.4 | - | - | (B, 5) |

- Total Parameters: ~1,192,261
- Loss: CrossEntropyLoss.
- Optimizer: Adam (lr=1e-3, weight_decay=1e-4).
- Batch size: 64.
- Epochs: 30.
- Seed: 42.

---
## Handling Class Imbalance
- **Initial approach:** WeightedRandomSampler → biased predictions.(Because of the highly imbalance data set)
- **Improved approach:** Downsampling majority class → balanced dataset.
  
- ![Downsampling](After_optimization/Images/Downsampling.png)  
### Comparison of Balancing Strategies

| Approach | Validation Acc | Test Acc | Comment |
| :--- | :--- | :--- | :--- |
| **WeightedRandomSampler** | ~0.85 | ~0.85 | Biased towards majority |
| **Downsampling** | **0.8904** | **0.9167** | Mostly Balanced, improved |


---

## Training and Evaluation
- **Split**: 70% train, 15% val, 15% test (subject-wise).
---
## Results and Visualizations
| Metric | Value |
| :--- | :--- | 
| Training Loss | 0.3264 |
| Val Loss | 0.3753 |
| Val Acc | 0.8904 |
| Test Acc | 0.9167 |

- confusion matrices and learning curvers are already added.

## Random-Sample Inference
- 20 random samples were tested post-training.
- ![Random 20](Test_random_data_wiith_latest_model/random20.png)  










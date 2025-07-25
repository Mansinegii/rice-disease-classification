# 🌾 Rice Leaf Disease Classification using Deep Learning

This project focuses on detecting and classifying four types of rice leaf diseases — **Bacterial Blight**, **Blast**, **Brown Spot**, and **Tungro** — using a **DenseNet121-based deep learning model** with the **Adam optimizer**. The approach involves data preprocessing, model training, evaluation, and visualizing results like accuracy/loss curves, confusion matrix, ROC, and metric histograms.

---

## 📁 Dataset

- Dataset Source: **Mendeley Data** and **Kaggle**
- Contains labeled rice leaf images of 4 disease classes:
  - Bacterial Blight
  - Blast
  - Brown Spot
  - Tungro
- Images were resized to 150x150 for model compatibility

---

## 🏗️ Model Architecture

The model is built using **DenseNet121 (pre-trained)** with a custom classification head:

- GlobalAveragePooling2D
- BatchNormalization
- Dropout (rate=0.4)
- Dense(256, activation='relu')
- Output Dense(4, activation='softmax')

Other tested models: CNN, MobileNet, VGG16.

---

## ⚙️ Training Configuration

- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Epochs:** 30
- **Metrics:** Accuracy, Precision, Recall, F1-Score
- **Data Augmentation:** Rotation, Flipping, Zoom, Rescale

---

## 🧪 Scripts Overview

| File | Purpose |
|------|---------|
| `data_loader.py` | Loads and augments the rice dataset |
| `model_builder.py` | Builds and compiles the DenseNet121 model |
| `train_model.py` | Trains the model and saves training history |
| `plot_results.py` | Plots accuracy, loss, confusion matrix, ROC, and metric histograms |
| `requirements.txt` | Lists all required Python libraries |

---

## 📊 Results (Graphs)

### 1. Accuracy vs Loss

![Accuracy and Loss](results/Screenshot%20(47).png)

---

### 2. Confusion Matrix

![Confusion Matrix](results/Screenshot%20(48).png)

---

### 3. ROC Curve

![ROC Curve](results/Screenshot%20(49)-Picsart-AiImageEnhancer.png)

---

### 4. Precision Histogram

![Precision](results/Screenshot%20(50).png)

---

### 5. Recall Histogram

![Recall](results/Screenshot%20(52).png)

---

### 6. F1-Score Histogram

![F1 Score](results/Screenshot%20(67).png)

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/Mansinegii/rice-disease-classification.git
cd rice-disease-classification

# Install dependencies
pip install -r requirements.txt

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

| File               | Purpose                                                  |
|--------------------|----------------------------------------------------------|
| `data_loader.py`   | Loads and augments the rice dataset                      |
| `model_builder.py` | Builds and compiles the DenseNet121 model                |
| `train_model.py`   | Trains the model and saves training history              |
| `plot_results.py`  | Plots accuracy, loss, confusion matrix, ROC, and metrics |
| `requirements.txt` | Lists all required Python libraries                      |

---

## 📊 Results (Graphs)

### 1. Training and Validation Accuracy

![Training and Validation Accuracy](results/Training%20and%20validation%20Accuracy.png)

---

### 2. Confusion Matrix

![Confusion Matrix](results/Confusion%20matrix.png)

---

### 3. ROC Curve

![ROC Curve](results/ROC%20curve.png)

---

### 4. Precision

![Precision](results/Precision.png)

---

### 5. Recall

![Recall](results/Recall.png)

---

### 6. F1-Score

![F1-Score](results/F1-score.png)

---

## 🚀 How to Run

```bash
# Clone the repository
git clone https://github.com/Mansinegii/rice-disease-classification.git
cd rice-disease-classification

# Install dependencies
pip install -r requirements.txt

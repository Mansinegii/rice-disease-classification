# 🌾 Rice Leaf Disease Classification using DenseNet121 with Adam Optimizer

This project focuses on the classification of rice leaf diseases—Bacterial Blight, Brown Spot, Leaf Blast, and Tungro—using the DenseNet121 deep learning model. The goal is to enhance classification accuracy under intra-class variability.

## 🧠 Model Overview

- **Architecture**: DenseNet121
- **Optimizer**: Adam
- **Dataset**: Kaggle + Mendeley Rice Leaf Disease Dataset
- **Framework**: TensorFlow/Keras
- **Loss Function**: Categorical Crossentropy

## 📊 Results

| Metric         | Score   |
|----------------|---------|
| Accuracy       | 95.2%   |
| Precision      | 94.9%   |
| Recall         | 94.6%   |
| F1-score       | 94.7%   |

### 📈 Performance Plots

- **[✓] Training and Validation Accuracy**  
- **[✓] Training and Validation Loss**  
- **[✓] Precision, Recall, F1-Score Bar Charts**  
- **[✓] Confusion Matrix**  
- **[✓] ROC Curve**

All results can be viewed in the [results section of this repo](#📈-performance-plots).

## 📂 Files

- `data_loader.py`: Loads and preprocesses the dataset
- `model_builder.py`: Builds the DenseNet121 model
- `train_model.py`: Trains and evaluates the model
- `plot_results.py`: Plots accuracy/loss and performance metrics
- `requirements.txt`: Lists dependencies for the project

## 📎 Dataset

- Kaggle: [Rice Leaf Disease Dataset](https://www.kaggle.com/datasets/minhhuy2810/rice-leaf-disease-dataset)
- Mendeley: [Mendeley Rice Leaf Dataset](https://data.mendeley.com/datasets/ny3bpzv3m2/1)

## 🔗 GitHub Repository

> This project has been developed as part of a research article submission to Cureus Journal.  
> **GitHub Link:** [https://github.com/Mansinegii/rice-disease-classification](https://github.com/Mansinegii/rice-disease-classification)


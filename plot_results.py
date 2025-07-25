import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from data_loader import val_data_gen

model = load_model("rice_model.h5")

# Prediction
y_true, y_pred = [], []
for images, labels in val_data_gen:
    preds = model.predict(images)
    y_true.extend(labels)
    y_pred.extend(np.argmax(preds, axis=1))
    if len(y_true) >= val_data_gen.samples:
        break

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=val_data_gen.class_indices.keys(), yticklabels=val_data_gen.class_indices.keys())
plt.title("Confusion Matrix")
plt.show()

# Metrics
TP = np.diag(conf_matrix)
FP = conf_matrix.sum(axis=0) - TP
FN = conf_matrix.sum(axis=1) - TP
TN = conf_matrix.sum() - (FP + FN + TP)

precision = TP / (TP + FP)
recall = TP / (TP + FN)
f_score = 2 * (precision * recall) / (precision + recall)

# Histograms
metrics = {"precision": precision, "recall": recall, "f1-score": f_score}
for metric_name, values in metrics.items():
    plt.bar(val_data_gen.class_indices.keys(), values)
    plt.title(f"{metric_name.capitalize()} per Class")
    plt.ylim(0, 1)
    plt.show()

# ROC Curve
y_true_bin = label_binarize(y_true, classes=list(range(len(val_data_gen.class_indices))))
y_pred_prob = model.predict(val_data_gen)

for i in range(len(val_data_gen.class_indices)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} AUC={roc_auc:.2f}')
plt.plot([0,1], [0,1], 'k--')
plt.title("ROC Curve")
plt.legend()
plt.show()

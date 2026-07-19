# src/utils.py

"""
utils.py

Modul ini berisi fungsi-fungsi bantu seperti visualisasi dan evaluasi metrik model . EDA (Exploratory Data Analysis).
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(6, 4), title="Confusion Matrix"):
    """
    Menampilkan confusion matrix sebagai heatmap.
    
    Parameters:
        y_true (array-like): Nilai target sebenarnya
        y_pred (array-like): Nilai prediksi
        labels (list): Label klasifikasi
        figsize (tuple): Ukuran gambar
        title (str): Judul grafik
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

def print_classification_report(y_true, y_pred):
    """
    Menampilkan classification report dalam format teks.
    
    Parameters:
        y_true (array-like): Nilai target sebenarnya
        y_pred (array-like): Nilai prediksi
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

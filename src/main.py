# src/main.py

"""
Main pipeline for Data Mining project.
Steps:
1. Load raw data
2. Simple preprocessing
3. Train model
4. Evaluate model
"""

from data_loader import load_raw_data
from model import train_model, evaluate_model
from utils import print_classification_report, plot_confusion_matrix

def main():
    # 1. Load data
    df = load_raw_data("data/raw/sample.csv")  # Ganti sesuai nama file dataset
    
    if df is None:
        print("Dataset tidak ditemukan.")
        return

    # 2. Preprocessing sederhana
    df = df.dropna()
    if "target" not in df.columns:
        print("Kolom 'target' tidak ditemukan dalam dataset.")
        return

    X = df.drop("target", axis=1)
    y = df["target"]

    # 3. Train & evaluate model
    model, y_test, y_pred = train_model(X, y)

    # 4. Evaluation
    print_classification_report(y_test, y_pred)
    plot_confusion_matrix(y_test, y_pred)

if __name__ == "__main__":
    main()

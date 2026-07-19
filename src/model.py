# src/model.py

"""
model.py

Modul ini digunakan untuk melatih dan mengevaluasi model Machine Learning.
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Memisahkan fitur dan target, lalu membagi data menjadi data latih dan data uji.

    Parameters:
        df (pd.DataFrame): Dataset lengkap
        target_column (str): Nama kolom target
        test_size (float): Proporsi data uji
        random_state (int): Seed random

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """
    Melatih model klasifikasi (default: Random Forest).

    Parameters:
        X_train: Fitur data latih
        y_train: Target data latih

    Returns:
        model terlatih
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi performa model pada data uji.

    Parameters:
        model: model yang sudah dilatih
        X_test: fitur data uji
        y_test: target data uji

    Returns:
        None (print hasil evaluasi)
    """
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

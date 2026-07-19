# src/model.py

"""
model.py

Modul ini digunakan untuk encoding label, vektorisasi TF-IDF,
penanganan imbalanced data (SMOTE), serta training dan evaluasi model Machine Learning
untuk klasifikasi sentimen (Logistic Regression, SVM, Naive Bayes).
"""

import pickle
import os
import numpy as np
import scipy.sparse
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE


# ─── Label Encoding ──────────────────────────────────────────────────────────

def encode_labels(df, label_col='label_manual'):
    """
    Mengubah label teks menjadi angka menggunakan LabelEncoder.

    Parameters:
        df (pd.DataFrame): DataFrame yang berisi kolom label
        label_col (str): Nama kolom label

    Returns:
        df (pd.DataFrame): DataFrame dengan kolom 'label_encoded' baru
        label_encoder (LabelEncoder): Objek encoder yang sudah di-fit
    """
    label_encoder = LabelEncoder()
    df = df.copy()
    df['label_encoded'] = label_encoder.fit_transform(df[label_col])

    print("[model] Mapping Label Encoding:")
    for i, kelas in enumerate(label_encoder.classes_):
        print(f"  {kelas:<10} → {i}")

    return df, label_encoder


# ─── TF-IDF Vectorization ────────────────────────────────────────────────────

def tfidf_vectorize(X_text, max_features=10000, ngram_range=(1, 2),
                    min_df=2, max_df=0.9, sublinear_tf=True):
    """
    Mengubah teks menjadi matriks TF-IDF.

    Parameters:
        X_text (pd.Series): Series teks bersih
        max_features (int): Jumlah maksimum fitur
        ngram_range (tuple): Range ngram (unigram + bigram = (1,2))
        min_df (int): Kata harus muncul minimal di N dokumen
        max_df (float): Buang kata yang muncul di > N% dokumen
        sublinear_tf (bool): Gunakan log scaling TF

    Returns:
        X_tfidf (sparse matrix): Matriks TF-IDF
        tfidf (TfidfVectorizer): Objek vectorizer yang sudah di-fit
    """
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf
    )

    X_tfidf = tfidf.fit_transform(X_text)

    print(f"[model] Bentuk matriks TF-IDF : {X_tfidf.shape}")
    print(f"  → {X_tfidf.shape[0]:,} baris (dokumen)")
    print(f"  → {X_tfidf.shape[1]:,} kolom (fitur kata/frasa unik)")

    fitur_nama = tfidf.get_feature_names_out()
    print(f"\n[model] Contoh 10 fitur pertama:")
    print(fitur_nama[:10])

    return X_tfidf, tfidf


# ─── Train-Test Split ────────────────────────────────────────────────────────

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


def split_data_stratified(X, y, test_size=0.2, random_state=42):
    """
    Membagi data latih dan data uji dengan stratifikasi (mempertahankan distribusi kelas).

    Parameters:
        X (sparse matrix atau array): Matriks fitur
        y (pd.Series atau array): Target label
        test_size (float): Proporsi data uji
        random_state (int): Seed random

    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[model] Data latih (train) : {X_train.shape[0]:,} baris")
    print(f"[model] Data uji (test)    : {X_test.shape[0]:,} baris")
    return X_train, X_test, y_train, y_test


# ─── SMOTE ───────────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train, k_neighbors=5, random_state=42):
    """
    Menerapkan SMOTE untuk menangani data yang tidak seimbang.

    Parameters:
        X_train: Data latih fitur
        y_train: Data latih target
        k_neighbors (int): Jumlah tetangga terdekat
        random_state (int): Seed random

    Returns:
        X_train_smote, y_train_smote: Data latih setelah SMOTE
    """
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"[model] Distribusi setelah SMOTE:")
    unique, counts = np.unique(y_train_smote, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Kelas {u}: {c:,} sampel")

    return X_train_smote, y_train_smote


# ─── Training Model ──────────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, C=1.0, max_iter=1000, random_state=42):
    """
    Melatih model Logistic Regression.

    Parameters:
        X_train: Fitur data latih
        y_train: Target data latih
        C (float): Parameter regularisasi
        max_iter (int): Jumlah iterasi maksimum
        random_state (int): Seed random

    Returns:
        model: Model Logistic Regression yang sudah dilatih
    """
    model = LogisticRegression(
        C=C, solver='lbfgs',
        class_weight='balanced',
        max_iter=max_iter,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("[model] Logistic Regression selesai dilatih.")
    return model


def train_svm(X_train, y_train, C=1.0, random_state=42):
    """
    Melatih model Support Vector Machine (SVM) dengan kernel linear.

    Parameters:
        X_train: Fitur data latih
        y_train: Target data latih
        C (float): Parameter regularisasi
        random_state (int): Seed random

    Returns:
        model: Model SVM yang sudah dilatih
    """
    model = SVC(
        kernel='linear', C=C,
        class_weight='balanced',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    print("[model] SVM selesai dilatih.")
    return model


def train_naive_bayes(X_train, y_train, alpha=1.0):
    """
    Melatih model Multinomial Naive Bayes.

    Parameters:
        X_train: Fitur data latih
        y_train: Target data latih
        alpha (float): Parameter Laplace smoothing

    Returns:
        model: Model Naive Bayes yang sudah dilatih
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    print("[model] Naive Bayes selesai dilatih.")
    return model


def train_model(X_train, y_train):
    """
    Melatih model klasifikasi default (Random Forest).
    Kompatibel dengan skeleton asli.

    Parameters:
        X_train: Fitur data latih
        y_train: Target data latih

    Returns:
        model: Model yang sudah dilatih
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train, X_train_smote, y_train_smote):
    """
    Melatih semua model (LR, SVM, NB) sebelum dan sesudah SMOTE.

    Parameters:
        X_train: Data latih sebelum SMOTE
        y_train: Target latih sebelum SMOTE
        X_train_smote: Data latih setelah SMOTE
        y_train_smote: Target latih setelah SMOTE

    Returns:
        dict: Dictionary berisi semua model yang sudah dilatih
    """
    print("\n[model] ===== MODELING SEBELUM SMOTE =====")
    model_lr_sbl  = train_logistic_regression(X_train, y_train)
    model_svm_sbl = train_svm(X_train, y_train)
    model_nb_sbl  = train_naive_bayes(X_train, y_train)

    print("\n[model] ===== MODELING SESUDAH SMOTE =====")
    model_lr_smt  = train_logistic_regression(X_train_smote, y_train_smote)
    model_svm_smt = train_svm(X_train_smote, y_train_smote)
    model_nb_smt  = train_naive_bayes(X_train_smote, y_train_smote)

    models = {
        'LR (Sebelum SMOTE)':  model_lr_sbl,
        'SVM (Sebelum SMOTE)': model_svm_sbl,
        'NB (Sebelum SMOTE)':  model_nb_sbl,
        'LR (Sesudah SMOTE)':  model_lr_smt,
        'SVM (Sesudah SMOTE)': model_svm_smt,
        'NB (Sesudah SMOTE)':  model_nb_smt,
    }
    return models


# ─── Evaluasi Model ──────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test):
    """
    Mengevaluasi performa model pada data uji.

    Parameters:
        model: Model yang sudah dilatih
        X_test: Fitur data uji
        y_test: Target data uji

    Returns:
        None (print hasil evaluasi)
    """
    y_pred = model.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# ─── Simpan & Load Model ─────────────────────────────────────────────────────

def save_models(models, tfidf, label_encoder, save_dir='output'):
    """
    Menyimpan semua model, vectorizer, dan label encoder ke file.

    Parameters:
        models (dict): Dictionary berisi semua model
        tfidf: Objek TfidfVectorizer
        label_encoder: Objek LabelEncoder
        save_dir (str): Direktori penyimpanan
    """
    os.makedirs(save_dir, exist_ok=True)

    for nama, model in models.items():
        nama_file = nama.replace(' ', '_').replace('(', '').replace(')', '') + '.pkl'
        path = os.path.join(save_dir, nama_file)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[model] Model disimpan: {path}")

    with open(os.path.join(save_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"[model] TF-IDF vectorizer disimpan: {os.path.join(save_dir, 'tfidf_vectorizer.pkl')}")

    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"[model] Label encoder disimpan: {os.path.join(save_dir, 'label_encoder.pkl')}")

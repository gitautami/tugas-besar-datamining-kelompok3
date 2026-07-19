# src/main.py

"""
Main pipeline untuk Sentimen Analisis Data Mining.

Pipeline (dijalankan via terminal: python src/main.py atau bash run.sh):
    1. EDA           – Exploratory Data Analysis pada dataset berlabel
    2. Preprocessing – Data cleaning pada dataset raw
    3. Encoding      – Load dataset_clean_manual.csv + TF-IDF vectorization
    4. Modeling      – Training LR, SVM, NB (sebelum & sesudah SMOTE)
    5. WordCloud     – Visualisasi kata penting per sentimen per model
    6. Evaluasi      – Accuracy, Precision, Recall, F1, Confusion Matrix

Cara menjalankan:
    cd <root-project>
    python src/main.py
"""

import os
import sys

# Tambahkan folder src ke path agar import modul bisa bekerja
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_raw_data, load_processed_data, load_kamus, save_processed_data
from preprocessing import (
    setup_stemmer, load_stopwords_github,
    build_stopword_final, build_kamus_gabungan, run_preprocessing
)
from model import (
    encode_labels, tfidf_vectorize,
    split_data_stratified, apply_smote,
    train_all_models, save_models
)
from utils import run_eda, plot_wordcloud_all_models, evaluate_all_models


# ─── Konfigurasi Path ────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR  = os.path.join(ROOT_DIR, 'output')


def main():
    print("\n" + "=" * 60)
    print("  PIPELINE SENTIMEN ANALISIS DATA MINING")
    print("  Kelompok 3 – Analisis Sentimen Pernyataan Prabowo")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # TAHAP 1: EDA – Exploratory Data Analysis
    # =========================================================================
    print("\n" + "─" * 60)
    print("  TAHAP 1: EXPLORATORY DATA ANALYSIS (EDA)")
    print("─" * 60)

    # Load dataset berlabel untuk EDA
    df_labeled = load_processed_data('dataset_clean_manual.csv')

    # Jalankan EDA
    run_eda(df_labeled, label_col='label_manual', save_dir=OUTPUT_DIR)

    # =========================================================================
    # TAHAP 2: PREPROCESSING – Data Cleaning pada Data Raw
    # =========================================================================
    print("\n" + "─" * 60)
    print("  TAHAP 2: PREPROCESSING – DATA CLEANING")
    print("─" * 60)

    # 2a. Load data raw
    df_raw  = load_raw_data('DatasetSentimen.csv')
    kamus_excel = load_kamus('kamuskatabaku.xlsx')

    # 2b. Setup komponen preprocessing
    stemmer         = setup_stemmer()
    stopword_github, _ = load_stopwords_github()
    stopword_final  = build_stopword_final(stopword_github)
    kamus_gabungan  = build_kamus_gabungan(kamus_excel)

    # 2c. Jalankan cleaning
    df_clean = run_preprocessing(
        df_raw, kamus_gabungan, stopword_final, stemmer, text_col='text'
    )

    # 2d. Simpan hasil cleaning (opsional, sebagai referensi)
    save_processed_data(df_clean[['text', 'text_clean']], 'dataset_clean_pipeline.csv')
    print("\n[main] INFO: Untuk pipeline selanjutnya, digunakan dataset_clean_manual.csv")
    print("             (hasil labeling manual yang sudah tersedia di data/processed/)")

    # =========================================================================
    # TAHAP 3: ENCODING TF-IDF
    # ─────────────────────────────────────────────────────────────────────────
    # PENTING: Load dari dataset_clean_manual.csv (sudah berlabel manual),
    # BUKAN dari hasil cleaning di tahap 2 (yang belum berlabel).
    # =========================================================================
    print("\n" + "─" * 60)
    print("  TAHAP 3: ENCODING TF-IDF")
    print("─" * 60)
    print("  [INFO] Loading dataset berlabel manual: dataset_clean_manual.csv")

    df_modeling = load_processed_data('dataset_clean_manual.csv')

    # Standarisasi label
    def standarkan_label(label):
        if str(label).strip() == '' or str(label).lower() == 'nan':
            return None
        label = str(label).strip().lower()
        mapping = {
            'positif': 'Positif', 'pos': 'Positif', 'p': 'Positif',
            'negatif': 'Negatif', 'neg': 'Negatif', 'n': 'Negatif',
            'netral':  'Netral',  'net': 'Netral',
        }
        return mapping.get(label, label)

    df_modeling['label_manual'] = df_modeling['label_manual'].apply(standarkan_label)
    df_modeling = df_modeling[
        df_modeling['label_manual'].isin(['Positif', 'Negatif', 'Netral'])
    ].reset_index(drop=True)

    print(f"\n[main] Distribusi label final:")
    print(df_modeling['label_manual'].value_counts())

    # Label encoding
    df_modeling, label_encoder = encode_labels(df_modeling, label_col='label_manual')

    # Fitur dan target
    X_text = df_modeling['text_clean']
    y      = df_modeling['label_encoded']

    print(f"\n[main] Jumlah data fitur (X) : {len(X_text):,}")
    print(f"[main] Jumlah data target (y): {len(y):,}")

    # TF-IDF Vectorization
    X_tfidf, tfidf = tfidf_vectorize(X_text)

    # =========================================================================
    # TAHAP 4: MODELING
    # =========================================================================
    print("\n" + "─" * 60)
    print("  TAHAP 4: MODELING (LR, SVM, NB – Sebelum & Sesudah SMOTE)")
    print("─" * 60)

    # Train-Test Split (stratified)
    X_train, X_test, y_train, y_test = split_data_stratified(X_tfidf, y)

    # Terapkan SMOTE pada data latih
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    # Training semua model
    models = train_all_models(X_train, y_train, X_train_smote, y_train_smote)

    # Simpan model
    save_models(models, tfidf, label_encoder, save_dir=OUTPUT_DIR)

    # =========================================================================
    # TAHAP 5: VISUALISASI WORDCLOUD
    # =========================================================================
    print("\n" + "─" * 60)
    print("  TAHAP 5: VISUALISASI WORDCLOUD")
    print("─" * 60)

    plot_wordcloud_all_models(models, tfidf, label_encoder, save_dir=OUTPUT_DIR)

    # =========================================================================
    # TAHAP 6: EVALUASI
    # =========================================================================
    print("\n" + "─" * 60)
    print("  TAHAP 6: EVALUASI MODEL")
    print("─" * 60)

    df_hasil = evaluate_all_models(models, X_test, y_test, label_encoder, save_dir=OUTPUT_DIR)

    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PIPELINE SELESAI!")
    print(f"  Output tersimpan di folder: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

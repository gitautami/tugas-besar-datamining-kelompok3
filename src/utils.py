# src/utils.py

"""
utils.py

Modul ini berisi fungsi-fungsi bantu untuk:
- Exploratory Data Analysis (EDA): distribusi label, statistik dataset
- Visualisasi WordCloud per kelas sentimen per model
- Evaluasi model: accuracy, precision, recall, F1, confusion matrix, classification report
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from wordcloud import WordCloud


# ─── EDA ─────────────────────────────────────────────────────────────────────

def run_eda(df, label_col='label_manual', save_dir='output'):
    """
    Menjalankan Exploratory Data Analysis pada dataset:
    - Statistik dasar dataset
    - Distribusi label (teks + pie/donut chart)

    Parameters:
        df (pd.DataFrame): DataFrame dataset (biasanya dataset_clean_manual.csv)
        label_col (str): Nama kolom label
        save_dir (str): Direktori penyimpanan grafik
    """
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 55)
    print("  STATISTIK DATASET")
    print("=" * 55)
    print(f"  Total baris          : {len(df):,}")
    print(f"  Total kolom          : {len(df.columns)}")
    print(f"  Kolom               : {df.columns.tolist()}")
    print(f"  Missing values      :")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            print(f"    - {col}: {n_missing:,} baris kosong")
    print()

    # Distribusi label
    distribusi = df[label_col].value_counts()
    total = len(df)

    print("=" * 55)
    print("  DISTRIBUSI LABEL")
    print("=" * 55)
    for label, jumlah in distribusi.items():
        persen = jumlah / total * 100
        print(f"  {label:<10} : {jumlah:>6,} ({persen:>5.1f}%)")
    print(f"\n  Total data : {total:,} baris")
    print("=" * 55)

    # Visualisasi Donut Chart
    labels = distribusi.index
    counts = distribusi.values
    warna_label = {'Negatif': '#D85A30', 'Netral': '#888780', 'Positif': '#1D9E75'}
    colors = [warna_label.get(label, '#4682B4') for label in labels]
    explode = [0.05 if label == 'Positif' else 0 for label in labels]

    plt.figure(figsize=(10, 7))
    plt.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=explode,
        shadow=True,
        pctdistance=0.85,
        textprops={'fontsize': 12, 'fontweight': 'bold'}
    )

    # Lingkaran tengah (donut effect)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title('Proporsi Sentimen – Dataset Label Manual', fontsize=16, fontweight='bold', pad=20)
    plt.legend(labels, title="Kategori", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.axis('equal')
    plt.tight_layout()

    save_path = os.path.join(save_dir, 'distribusi_label.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n[utils] Grafik distribusi disimpan: {save_path}")
    plt.show()


# ─── WordCloud ───────────────────────────────────────────────────────────────

def _ambil_bobot_kata(model, index_kelas, tipe_model):
    """
    Ambil array bobot kata untuk satu kelas tertentu dari model.

    Parameters:
        model: Model yang sudah dilatih (LR, SVM, atau NB)
        index_kelas (int): Indeks kelas (0, 1, atau 2)
        tipe_model (str): 'lr', 'svm', atau 'nb'

    Returns:
        np.ndarray: Array bobot 1D, satu nilai per fitur
    """
    if tipe_model in ('lr', 'svm'):
        bobot = model.coef_[index_kelas]
        if hasattr(bobot, 'toarray'):
            bobot = bobot.toarray()
        bobot = np.asarray(bobot).ravel()
        return bobot
    elif tipe_model == 'nb':
        bobot = np.asarray(model.feature_log_prob_[index_kelas]).ravel()
        return bobot
    else:
        raise ValueError(f"tipe_model tidak dikenal: {tipe_model}")


def _buat_wordcloud_dari_bobot(bobot, nama_fitur, judul, ax):
    """
    Membuat wordcloud dari array bobot kata (hanya bobot positif).

    Parameters:
        bobot (np.ndarray): Array bobot kata
        nama_fitur (list): Nama-nama fitur dari TF-IDF
        judul (str): Judul wordcloud
        ax: Matplotlib axes object
    """
    bobot = np.asarray(bobot).ravel()

    # Pasangkan kata dengan bobot
    pasangan = dict(zip(nama_fitur, bobot.tolist()))

    # Hanya ambil kata dengan bobot positif dan panjang >= 3
    kata_positif = {
        k: v for k, v in pasangan.items()
        if v > 0 and len(k) >= 3 and not any(c.isdigit() for c in k)
    }

    if len(kata_positif) == 0:
        ax.text(0.5, 0.5, 'Tidak ada kata dengan bobot positif',
                ha='center', va='center')
        ax.set_title(judul)
        ax.axis('off')
        return

    wc = WordCloud(
        width=500, height=350,
        background_color='white',
        colormap='viridis',
        max_words=50
    ).generate_from_frequencies(kata_positif)

    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(judul, fontsize=13, fontweight='bold')
    ax.axis('off')


def plot_wordcloud_all_models(models, tfidf, label_encoder, save_dir='output'):
    """
    Membuat dan menyimpan WordCloud untuk semua model dan semua kelas sentimen.

    Parameters:
        models (dict): Dictionary berisi semua model {'nama': model}
        tfidf: Objek TfidfVectorizer yang sudah di-fit
        label_encoder: Objek LabelEncoder yang sudah di-fit
        save_dir (str): Direktori penyimpanan gambar
    """
    os.makedirs(save_dir, exist_ok=True)

    nama_fitur = tfidf.get_feature_names_out()
    nama_kelas = label_encoder.classes_

    # Mapping tipe model
    tipe_map = {}
    for nama_model in models.keys():
        nama_lower = nama_model.lower()
        if 'lr' in nama_lower or 'logistic' in nama_lower:
            tipe_map[nama_model] = 'lr'
        elif 'svm' in nama_lower:
            tipe_map[nama_model] = 'svm'
        elif 'nb' in nama_lower or 'naive' in nama_lower or 'bayes' in nama_lower:
            tipe_map[nama_model] = 'nb'
        else:
            tipe_map[nama_model] = 'lr'  # default

    for nama_model, model in models.items():
        tipe = tipe_map[nama_model]
        print(f"\n[utils] Membuat WordCloud untuk: {nama_model}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, kelas in enumerate(nama_kelas):
            bobot = _ambil_bobot_kata(model, i, tipe)
            _buat_wordcloud_dari_bobot(
                bobot, nama_fitur,
                f'Sentimen {kelas}',
                axes[i]
            )

        fig.suptitle(f'WordCloud – {nama_model}', fontsize=16, fontweight='bold')
        plt.tight_layout()

        # Nama file aman
        nama_file = nama_model.replace(' ', '_').replace('(', '').replace(')', '') + '_wordcloud.png'
        save_path = os.path.join(save_dir, nama_file)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[utils] WordCloud disimpan: {save_path}")
        plt.show()


# ─── Evaluasi ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, labels=None, figsize=(6, 4),
                           title="Confusion Matrix", save_path=None):
    """
    Menampilkan confusion matrix sebagai heatmap.

    Parameters:
        y_true (array-like): Nilai target sebenarnya
        y_pred (array-like): Nilai prediksi
        labels (list): Label klasifikasi
        figsize (tuple): Ukuran gambar
        title (str): Judul grafik
        save_path (str|None): Path untuk menyimpan gambar (opsional)
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[utils] Confusion matrix disimpan: {save_path}")
    plt.show()


def print_classification_report(y_true, y_pred, target_names=None):
    """
    Menampilkan classification report dalam format teks.

    Parameters:
        y_true (array-like): Nilai target sebenarnya
        y_pred (array-like): Nilai prediksi
        target_names (list|None): Nama kelas label
    """
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=target_names))


def evaluate_all_models(models, X_test, y_test, label_encoder, save_dir='output'):
    """
    Mengevaluasi semua model dan menampilkan ringkasan performa.

    Parameters:
        models (dict): Dictionary berisi semua model {'nama': model}
        X_test: Fitur data uji
        y_test: Target data uji
        label_encoder: Objek LabelEncoder untuk nama kelas
        save_dir (str): Direktori penyimpanan confusion matrix

    Returns:
        pd.DataFrame: DataFrame ringkasan metrik semua model
    """
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)

    nama_kelas = label_encoder.classes_
    hasil = []

    print("\n" + "=" * 70)
    print("  RINGKASAN EVALUASI SEMUA MODEL")
    print("=" * 70)

    for nama_model, model in models.items():
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        hasil.append({
            'Model':     nama_model,
            'Accuracy':  acc,
            'Precision': prec,
            'Recall':    rec,
            'F1-Score':  f1
        })

        print(f"\n── {nama_model} ──")
        print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=nama_kelas))

        # Confusion Matrix
        nama_file = nama_model.replace(' ', '_').replace('(', '').replace(')', '') + '_cm.png'
        save_path = os.path.join(save_dir, nama_file)
        plot_confusion_matrix(
            y_test, y_pred,
            labels=list(range(len(nama_kelas))),
            title=f'Confusion Matrix – {nama_model}',
            save_path=save_path
        )

    # Tabel Ringkasan
    df_hasil = pd.DataFrame(hasil).set_index('Model')
    df_hasil = df_hasil.applymap(lambda x: f"{x:.4f}")

    print("\n" + "=" * 70)
    print("  TABEL PERBANDINGAN MODEL")
    print("=" * 70)
    print(df_hasil.to_string())
    print("=" * 70)

    return df_hasil

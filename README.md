# 🗳️ ANALISIS SENTIMEN PENGGUNA TIKTOK DAN YOUTUBE TERHADAP PERNYATAAN PRESIDEN PRABOWO BAHWA “ORANG DESA TIDAK PAKAI DOLAR”

> **Tugas Besar Data Mining – Kelompok 3**
> Klasifikasi sentimen berbahasa Indonesia menggunakan Machine Learning (Logistic Regression, SVM, Naive Bayes) dengan pipeline lengkap dari EDA hingga evaluasi model.

---

## 👥 Anggota Kelompok

| No | Nama | NIM |
|----|------|-----|
| 1  | [Resqi Audia Gita Utami] | [714230003] |
| 2  | [Muhammad Haitsam Izzuddin Azman] | [714230021] |
| 3  | [Hafidz Fakhri Muharram] | [714230031] |
| 4  | [Miqdam Syiam Nurrohman] | [714230043] |

---

## 📌 Deskripsi Kasus

* **Konteks Permasalahan:** Pelemahan nilai tukar Rupiah terhadap Dolar Amerika Serikat memicu dampak ekonomi makro yang sistemik (*imported inflation*). Hal ini secara langsung memengaruhi stabilitas harga kebutuhan pokok, pupuk, hingga pakan ternak di sektor pedesaan.
* **Pernyataan Kontroversial:** Di tengah kekhawatiran masyarakat kelas menengah ke bawah mengenai efek domino ekonomi tersebut, pemerintah berupaya meredam kepanikan publik lewat retorika populis. Presiden Prabowo mengeluarkan pernyataan bahwa *"orang desa tidak pakai Dolar"*, yang mengisyaratkan seolah fluktuasi mata uang asing tidak berdampak pada wilayah pedesaan.
* **Reaksi Diskursus Publik:** Narasi tersebut berbenturan dengan rasionalitas ekonomi di lapangan dan memicu gelombang opini kritis yang masif dari warganet. Respons digital mengalir deras pada platform media sosial berbasis video dinamis yang memiliki interaksi tinggi, khususnya TikTok dan YouTube.
* **Tujuan Penelitian:** Menggunakan pendekatan *Text Mining* dan algoritma *Machine Learning* untuk mengekstraksi dan memetakan puluhan ribu opini publik tidak terstruktur secara objektif ke dalam tiga kelas polaritas (Positif, Negatif, dan Netral). 
* **Temuan Utama Kasus:** Berdasarkan analisis, opini publik didominasi secara mutlak oleh sentimen **Negatif (81,1%)**. Hal ini menunjukkan adanya *gap* yang lebar antara komunikasi publik pemerintah dan realitas ekonomi yang dirasakan langsung oleh masyarakat di tingkat akar rumput.

---

## 📂 Sumber Dataset

* **Metode Akuisisi Data:** Data primer dikumpulkan secara mandiri menggunakan metode otomatisasi *Web Scraping* (melalui skrip pemrograman/API) lintas platform (*cross-platform*).
* **Platform Sumber:** Data diambil dari dua media sosial berbasis video utama, yaitu **TikTok** dan **YouTube**.
* **Karakteristik Data Teks:** Korpus data berupa teks komentar publik berbahasa Indonesia informal (ragam bahasa gaul, singkatan warganet, dan kata *slang*) yang secara spesifik merespons video terkait pernyataan presiden.
* **Periode Pengumpulan:** Proses pencarian dan penarikan data dilaksanakan selama rentang waktu **8 hingga 13 Juni 2026**.
* **Volume Ukuran Data:**
  * **Total Data Mentah (*Raw Data*):** Terkumpul sebanyak **11.163 baris komentar** (TikTok: 1.999 komentar; YouTube: 9.164 komentar).
  * **Total Data Valid:** Setelah melalui tahapan filtrasi ketat pada pra-pemrosesan (*text preprocessing*)—seperti menghapus *missing values*, komentar duplikat, dan teks di bawah 2 kata—diperoleh **10.295 baris observasi data valid**.
* **Format dan Atribut Penyimpanan:** Dataset gabungan akhir disimpan dalam ekstensi **CSV (*Comma Separated Values*)** dengan struktur dua kolom utama, yaitu:
  * `Text`: Berisi data teks komentar warganet yang telah dibersihkan.
  * `Label`: Berisi representasi target kelas sentimen hasil pelabelan (Negatif, Netral, atau Positif).
---

## 🔧 Langkah Preprocessing

Pipeline preprocessing teks Bahasa Indonesia dilakukan secara berurutan melalui modul [`src/preprocessing.py`](src/preprocessing.py):

| Tahap | Proses | Keterangan |
|-------|--------|------------|
| 1 | **Case Folding** | Mengubah semua teks menjadi huruf kecil (lowercase) |
| 2 | **Hapus URL** | Menghapus link `http://`, `https://`, `www.` |
| 3 | **Hapus Mention & Hashtag** | Menghapus `@username` dan `#hashtag` |
| 4 | **Hapus Emoji & Non-ASCII** | Membersihkan karakter non-ASCII dan emoji |
| 5 | **Hapus Tanda Baca & Angka** | Menghapus karakter selain huruf alfabet |
| 6 | **Normalisasi Spasi** | Menghapus spasi ganda dan whitespace berlebih |
| 7 | **Normalisasi Kata Tidak Baku** | Menggunakan kamus gabungan (Excel + manual) untuk menggantikan kata gaul/singkatan |
| 8 | **Stopword Removal** | Menghapus kata umum, **namun menjaga kata negasi** (`tidak`, `bukan`, `jangan`, dll.) |
| 9 | **Stemming** | Menggunakan **PySastrawi** untuk mengembalikan kata ke bentuk dasarnya |

**Output preprocessing:** kolom `text_clean` ditambahkan ke DataFrame, duplikat dan baris kosong dihapus.

---

## 🤖 Algoritma yang Digunakan

Model klasifikasi dilatih menggunakan representasi **TF-IDF** (unigram + bigram, max 10.000 fitur).

### Model

| Model | Keterangan |
|-------|------------|
| **Logistic Regression (LR)** | Menggunakan solver `lbfgs` dengan `class_weight='balanced'` |
| **Support Vector Machine (SVM)** | Kernel linear dengan `class_weight='balanced'` |
| **Multinomial Naive Bayes (NB)** | Dengan Laplace smoothing (`alpha=1.0`) |

### Penanganan Imbalanced Data

Semua model dilatih dalam **dua skenario**:
- **Sebelum SMOTE** – Training langsung pada data asli
- **Sesudah SMOTE** – Training setelah oversample kelas minoritas dengan [SMOTE](https://imbalanced-learn.org/)

### Total Model yang Dilatih

| # | Model |
|---|-------|
| 1 | LR (Sebelum SMOTE) |
| 2 | SVM (Sebelum SMOTE) |
| 3 | NB (Sebelum SMOTE) |
| 4 | LR (Sesudah SMOTE) |
| 5 | SVM (Sesudah SMOTE) |
| 6 | NB (Sesudah SMOTE) |

---

## 📊 Evaluasi & Hasil

Evaluasi dilakukan pada **data uji (20%)** menggunakan stratified split untuk menjaga distribusi kelas.

### Metrik Evaluasi

- **Accuracy** – Persentase prediksi yang benar
- **Precision** – Ketepatan prediksi per kelas (weighted average)
- **Recall** – Kelengkapan prediksi per kelas (weighted average)
- **F1-Score** – Harmonic mean dari Precision dan Recall

### Output Evaluasi

| Output | Lokasi |
|--------|--------|
| Confusion matrix (per model) | `output/<nama_model>_cm.png` |
| WordCloud (per model, per kelas) | `output/<nama_model>_wordcloud.png` |
| Distribusi label dataset | `output/distribusi_label.png` |
| Model tersimpan | `output/*.pkl` |
| TF-IDF vectorizer | `output/tfidf_vectorizer.pkl` |
| Label encoder | `output/label_encoder.pkl` |

> 📋 Hasil lengkap evaluasi (classification report & tabel perbandingan) dapat dilihat di notebook [`notebook/05_evaluation.ipynb`](notebook/05_evaluation.ipynb).

---

## 🚀 Cara Menjalankan

### ✅ 1. Persiapkan Environment

Install dependensi:
```bash
pip install -r requirements.txt
```

### ✅ 2. Jalankan Pipeline

#### 💻 Via Terminal:
```bash
bash run.sh
```

#### 📒 Via Jupyter Notebook:
Buka dan jalankan:
```text
src/main_notebook.ipynb
```

---


## 🗂️ Struktur Direktori

```
tugas-besar-datamining-kelompok3/
│
├── data/
│   ├── raw/                        ← Dataset mentah & kamus kata baku
│   └── processed/                  ← Dataset setelah preprocessing & labeling
│
├── notebook/
│   ├── 01_eda.ipynb                ← Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb      ← Data Cleaning Pipeline
│   ├── 03_modeling.ipynb           ← TF-IDF, SMOTE & Training Model
│   ├── 04_wordcloud.ipynb          ← Visualisasi WordCloud
│   └── 05_evaluation.ipynb         ← Evaluasi & Perbandingan Model
│
├── report/
│   ├── Sentimen_Analisis_Data_Mining.ipynb
│   ├── laporan-akhir_template.pdf
│   ├── lampiran_template.docx
│   └── struktur-lampiran.md
│
├── src/
│   ├── data_loader.py              ← Load & simpan dataset
│   ├── preprocessing.py            ← Pipeline cleaning teks Bahasa Indonesia
│   ├── model.py                    ← Training LR / SVM / NB + SMOTE
│   ├── utils.py                    ← EDA, WordCloud, evaluasi model
│   ├── main.py                     ← Orchestrator pipeline (via terminal)
│   └── main_notebook.ipynb         ← Versi notebook dari main.py
│
├── run.sh                          ← Script bash untuk menjalankan pipeline
├── requirements.txt                ← Daftar dependensi Python
└── README.md                       ← Dokumentasi ini
```

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan **Tugas Besar mata kuliah Data Mining** dan bersifat open-source untuk keperluan edukasi.
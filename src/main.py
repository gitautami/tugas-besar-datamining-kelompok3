"""
Main Orchestrator – Pipeline Sentimen Analisis Data Mining
Kelompok 3 – Analisis Sentimen Pernyataan Prabowo

Menjalankan 5 notebook pipeline secara berurutan via `jupyter nbconvert`:
    1. notebook/01_eda.ipynb           → EDA & distribusi label
    2. notebook/02_preprocessing.ipynb → Data cleaning
    3. notebook/03_modeling.ipynb      → TF-IDF + SMOTE + Training LR/SVM/NB
    4. notebook/04_wordcloud.ipynb     → Visualisasi WordCloud
    5. notebook/05_evaluation.ipynb    → Evaluasi metrik & confusion matrix

Cara menjalankan:
    cd <root-project>
    python src/main.py
    atau
    bash run.sh
"""

import os
import sys
import subprocess

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NOTEBOOKS = [
    os.path.join('notebook', '01_eda.ipynb'),
    os.path.join('notebook', '02_preprocessing.ipynb'),
    os.path.join('notebook', '03_modeling.ipynb'),
    os.path.join('notebook', '04_wordcloud.ipynb'),
    os.path.join('notebook', '05_evaluation.ipynb'),
]

TAHAP_NAMA = [
    'TAHAP 1 : EDA – Exploratory Data Analysis',
    'TAHAP 2 : PREPROCESSING – Data Cleaning',
    'TAHAP 3-4: ENCODING TF-IDF & MODELING',
    'TAHAP 5 : VISUALISASI WORDCLOUD',
    'TAHAP 6 : EVALUASI MODEL',
]

def run_notebook(nb_path, tahap_nama):
    """
    Menjalankan satu notebook via jupyter nbconvert --execute.

    Parameters:
        nb_path (str): Path relatif notebook dari root project
        tahap_nama (str): Nama tahap untuk ditampilkan di log
    """
    full_path = os.path.join(ROOT_DIR, nb_path)

    print(f"\n{'─' * 60}")
    print(f"  {tahap_nama}")
    print(f"  Notebook : {nb_path}")
    print(f"{'─' * 60}")

    result = subprocess.run(
        [
            sys.executable, '-m', 'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute', full_path,
            '--inplace',
            '--ExecutePreprocessor.timeout=600',
        ],
        capture_output=True,
        text=True,
        cwd=ROOT_DIR
    )

    if result.returncode != 0:
        print(f"\n[main] ❌ ERROR saat menjalankan: {nb_path}")
        print(result.stderr)
        sys.exit(1)

    print(f"[main] ✅ Selesai: {nb_path}")


def main():
    print("\n" + "=" * 60)
    print("  PIPELINE SENTIMEN ANALISIS DATA MINING")
    print("  Kelompok 3 – Analisis Sentimen Pernyataan Prabowo")
    print("=" * 60)

    for nb, nama in zip(NOTEBOOKS, TAHAP_NAMA):
        run_notebook(nb, nama)

    OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')

    print("\n" + "=" * 60)
    print("  PIPELINE SELESAI!")
    print(f"  Output tersimpan di folder: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

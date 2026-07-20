# src/data_loader.py

"""
data_loader.py

Module ini digunakan untuk memuat dataset dari direktori `data/` baik dari folder raw maupun processed.
"""

import pandas as pd
import os

RAW_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")


def load_csv(filename, processed=False):
    """
    Memuat file CSV dari folder data.

    Parameters:
        filename (str): Nama file (contoh: 'data.csv')
        processed (bool): Jika True, load dari folder processed, else dari raw.

    Returns:
        pd.DataFrame: Dataframe dari file yang dimuat
    """
    folder = PROCESSED_DATA_PATH if processed else RAW_DATA_PATH
    file_path = os.path.join(folder, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    return pd.read_csv(file_path)


def load_raw_data(filename):
    """
    Memuat file CSV dari folder data/raw/.

    Parameters:
        filename (str): Nama file (contoh: 'DatasetSentimen.csv')

    Returns:
        pd.DataFrame: Dataframe dari file yang dimuat
    """
    file_path = os.path.join(RAW_DATA_PATH, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[data_loader] Dataset '{filename}' dimuat: {len(df)} baris, {len(df.columns)} kolom.")
    return df


def load_processed_data(filename):
    """
    Memuat file CSV dari folder data/processed/.
    Digunakan untuk load dataset yang sudah dilabeling manual (dataset_clean_manual.csv).

    Parameters:
        filename (str): Nama file (contoh: 'dataset_clean_manual.csv')

    Returns:
        pd.DataFrame: Dataframe dari file yang dimuat
    """
    file_path = os.path.join(PROCESSED_DATA_PATH, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[data_loader] Processed dataset '{filename}' dimuat: {len(df)} baris, {len(df.columns)} kolom.")
    return df


def load_kamus(filename):
    """
    Memuat file kamus kata baku dari folder data/raw/.

    Parameters:
        filename (str): Nama file Excel kamus (contoh: 'kamuskatabaku.xlsx')

    Returns:
        dict: Dictionary {tidak_baku: kata_baku}
    """
    file_path = os.path.join(RAW_DATA_PATH, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File kamus tidak ditemukan: {file_path}")

    df_kamus = pd.read_excel(file_path)
    print(f"[data_loader] Kamus '{filename}' dimuat: {len(df_kamus)} entri.")
    print(f"             Kolom: {df_kamus.columns.tolist()}")

    # Konversi ke dictionary {tidak_baku: kata_baku}
    kamus = dict(zip(
        df_kamus.iloc[:, 0].astype(str).str.lower().str.strip(),
        df_kamus.iloc[:, 1].astype(str).str.lower().str.strip()
    ))
    return kamus


def save_processed_data(df, filename):
    """
    Menyimpan dataframe ke folder data/processed/.

    Parameters:
        df (pd.DataFrame): Dataframe yang akan disimpan
        filename (str): Nama file output (contoh: 'dataset_clean.csv')
    """
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    file_path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(file_path, index=False)
    print(f"[data_loader] Data disimpan ke: {file_path}")


def preview_data(df, rows=5):
    """
    Menampilkan preview awal dari dataframe

    Parameters:
        df (pd.DataFrame): Dataframe yang akan dipreview
        rows (int): Jumlah baris untuk ditampilkan
    """
    print(df.head(rows))

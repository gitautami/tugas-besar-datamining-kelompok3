# src/data_loader.py

"""
data_loader.py

Module ini digunakan untuk memuat dataset dari direktori `data/` baik dari folder raw maupun processed.
"""

import pandas as pd
import os

# Folder path relatif dari root repository
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

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

def preview_data(df, rows=5):
    """
    Menampilkan preview awal dari dataframe
    
    Parameters:
        df (pd.DataFrame): Dataframe yang akan dipreview
        rows (int): Jumlah baris untuk ditampilkan
    """
    print(df.head(rows))

# Contoh pemanggilan (hapus saat produksi):
if __name__ == "__main__":
    try:
        df = load_csv("your_dataset.csv")  # ganti nama file sesuai kebutuhan
        preview_data(df)
    except Exception as e:
        print(e)

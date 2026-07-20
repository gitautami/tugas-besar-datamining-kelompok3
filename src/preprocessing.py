
"""
preprocessing.py

Module ini berisi fungsi-fungsi untuk preprocessing teks Bahasa Indonesia:
- Setup stemmer (Sastrawi)
- Load dan susun stopword
- Build kamus kata baku gabungan
- Fungsi cleaning teks utama
- Jalankan full cleaning pipeline pada DataFrame
"""

import re
import requests
import pandas as pd


KAMUS_MANUAL = {
    # Negasi
    'gak':    'tidak', 'ga':     'tidak', 'ngak':   'tidak',
    'nggak':  'tidak', 'gk':     'tidak', 'gkk':    'tidak',
    'tdk':    'tidak', 'ndak':   'tidak',

    # Temporal
    'sdh':    'sudah', 'sblm':   'sebelum', 'spt':    'seperti',
    'krn':    'karena', 'karna':  'karena',

    # Kata ganti
    'gw':     'saya',  'gue':    'saya',  'lo':     'kamu',
    'lu':     'kamu',  'elu':    'kamu',

    # Entitas khusus dataset
    'indo':   'indonesia', 'ri':     'indonesia',
    'wowo':   'prabowo',   'pak':    'bapak',

    # Noise umum
    'wkwkwk': '', 'wkwk':   '', 'haha':   '',
    'hehe':   '', 'hihi':   '', 'lol':    '',
    'wtf':    '', 'amp':    '', 'br':     '', 'brbr':   '',
}

KATA_NEGASI_WAJIB = {
    'tidak', 'bukan', 'jangan', 'belum', 'tanpa',
    'tak', 'anti', 'non', 'tiada', 'mustahil'
}


STOPWORD_MANUAL = {
    # Partikel & konjungsi umum
    'yang', 'di', 'ke', 'dari', 'dan', 'dengan', 'untuk', 'pada',
    'ini', 'itu', 'adalah', 'atau', 'oleh', 'ada', 'juga',
    'akan', 'tapi', 'bisa', 'aja', 'nih', 'sih', 'deh', 'dong',
    'kah', 'lah', 'pun', 'nya',
    'ya', 'yah', 'iya', 'oke', 'ok', 'oh', 'ah', 'eh', 'wah',

    # Sapaan
    'hai', 'hei', 'kak', 'mas', 'bang', 'bro', 'sis',

    # Kata ganti (setelah normalisasi)
    'saya', 'aku', 'kamu', 'dia', 'mereka', 'kita', 'kami',

    # Demonstratif & lokatif
    'sana', 'sini', 'situ', 'tersebut',

    # Aspek waktu netral
    'pernah', 'sedang', 'lagi', 'sudah',

    # Konjungsi kondisional & kausal
    'jika', 'kalau', 'bila', 'maka', 'agar', 'supaya',
    'namun', 'tetapi', 'melainkan', 'padahal', 'meskipun',
    'karena', 'sebab', 'sehingga', 'akibat',

    # Kopula
    'jadi', 'menjadi', 'ialah', 'yaitu', 'yakni',
    'demikian', 'begitu', 'seperti', 'bagaikan',

    # Kata umum bermasalah di InSet Lexicon
    'siapa', 'banyak', 'semua', 'apa', 'coba',
    'ikut', 'usah', 'bikin', 'orang', 'ajar',
}


BASE_URL = "https://raw.githubusercontent.com/ArhamZaldin/Kamus-Custom-Preprocessing-Analisis-Sentimen-Indonesia/main/"


def setup_stemmer():
    """
    Menginisiasi stemmer Sastrawi untuk Bahasa Indonesia.

    Returns:
        stemmer: Objek stemmer Sastrawi yang siap digunakan
    """
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        print("[preprocessing] Stemmer Sastrawi siap.")
        return stemmer
    except ImportError:
        raise ImportError(
            "PySastrawi belum terinstall. Jalankan: pip install PySastrawi"
        )


def load_stopwords_github():
    """
    Ambil stopword dari GitHub repository.
    Jika gagal (offline/error), return set kosong.

    Returns:
        set: Set kata stopword dari GitHub
        set: Set kata negasi dari GitHub
    """
    def ambil_json(nama_file):
        try:
            resp = requests.get(BASE_URL + nama_file, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list):
                return set(data)
            elif isinstance(data, dict):
                return data
        except Exception as e:
            print(f"[preprocessing] [WARNING] Gagal ambil {nama_file} dari GitHub: {e}")
            return set()

    stopword_github = ambil_json('custom-stopwords.json')
    negasi_github   = ambil_json('negation-words.json')

    if stopword_github:
        print(f"[preprocessing] Stopword dari GitHub  : {len(stopword_github)} kata")
    if negasi_github:
        print(f"[preprocessing] Kata negasi (dijaga)  : {len(negasi_github)} kata")

    return stopword_github, negasi_github


def build_stopword_final(stopword_github=None):
    """
    Membangun daftar stopword final dengan memastikan kata negasi tidak masuk.

    Parameters:
        stopword_github (set|dict|None): Stopword dari GitHub (opsional)

    Returns:
        set: Stopword final yang aman (tanpa kata negasi)
    """

    if stopword_github is None:
        stopword_github_set = set()
    elif isinstance(stopword_github, dict):
        stopword_github_set = set(stopword_github.keys())
    else:
        stopword_github_set = set(stopword_github)

    stopword_gabungan = STOPWORD_MANUAL | stopword_github_set

    stopword_final = stopword_gabungan - KATA_NEGASI_WAJIB

    print(f"[preprocessing] Stopword manual       : {len(STOPWORD_MANUAL)} kata")
    print(f"[preprocessing] Stopword GitHub       : {len(stopword_github_set)} kata")
    print(f"[preprocessing] Stopword gabungan     : {len(stopword_gabungan)} kata")
    print(f"[preprocessing] Stopword final (aman) : {len(stopword_final)} kata")

    print("\n[preprocessing] Verifikasi — kata negasi masih ada di stopword?")
    for neg in sorted(KATA_NEGASI_WAJIB):
        status = 'BAHAYA' if neg in stopword_final else 'aman'
        print(f"  '{neg:10}' → {status}")

    return stopword_final


def build_kamus_gabungan(kamus_dari_excel=None):
    """
    Membangun kamus gabungan dari kamus Excel + kamus manual.

    Parameters:
        kamus_dari_excel (dict|None): Dictionary kamus dari file Excel
                                      (hasil data_loader.load_kamus())

    Returns:
        dict: Kamus gabungan {tidak_baku: kata_baku}
    """
    if kamus_dari_excel is None:
        kamus_dari_excel = {}

    kamus_gabungan = {**kamus_dari_excel, **KAMUS_MANUAL}
    print(f"[preprocessing] Total entri kamus gabungan: {len(kamus_gabungan)}")
    return kamus_gabungan


def bersihkan_teks(teks, kamus_gabungan, stopword_final, stemmer):
    """
    Membersihkan satu baris teks melalui tahapan lengkap:
    a. Guard (teks kosong/NaN)
    b. Case folding (lowercase)
    c. Hapus URL
    d. Hapus mention (@) dan hashtag (#)
    e. Hapus emoji & karakter non-ASCII
    f. Hapus tanda baca & angka
    g. Normalisasi spasi berlebih
    h. Normalisasi kata tidak baku
    i. Stopword removal (jaga kata negasi)
    j. Stemming (Sastrawi)

    Parameters:
        teks (str): Teks input
        kamus_gabungan (dict): Kamus normalisasi kata
        stopword_final (set): Set kata stopword final
        stemmer: Objek stemmer Sastrawi

    Returns:
        str: Teks yang sudah dibersihkan
    """
    if pd.isna(teks) or str(teks).strip() == '':
        return ''

    teks = str(teks).lower()

    teks = re.sub(r'http\S+|www\S+', '', teks)

    teks = re.sub(r'@\w+', '', teks)
    teks = re.sub(r'#\w+', '', teks)

    teks = teks.encode('ascii', 'ignore').decode('ascii')

    teks = re.sub(r'[^a-z\s]', '', teks)

    teks = re.sub(r'\s+', ' ', teks).strip()

    kata_kata = teks.split()
    kata_kata = [kamus_gabungan.get(k, k) for k in kata_kata]

    kata_kata = [
        k for k in kata_kata
        if k not in stopword_final and len(k) > 1
    ]

    kata_kata = [stemmer.stem(k) for k in kata_kata]

    return ' '.join(kata_kata)


def run_preprocessing(df, kamus_gabungan, stopword_final, stemmer, text_col='text'):
    """
    Menjalankan full preprocessing pipeline pada DataFrame.
    Meliputi: hapus duplikat → cleaning → hapus baris kosong → hapus baris 1 kata.

    Parameters:
        df (pd.DataFrame): DataFrame dengan kolom teks
        kamus_gabungan (dict): Kamus normalisasi kata
        stopword_final (set): Set kata stopword final
        stemmer: Objek stemmer Sastrawi
        text_col (str): Nama kolom teks yang akan dibersihkan

    Returns:
        pd.DataFrame: DataFrame setelah preprocessing dengan kolom 'text_clean' baru
    """
    sebelum_dup = len(df)
    df = df.drop_duplicates(subset=text_col).reset_index(drop=True)
    sesudah_dup = len(df)

    print(f"[preprocessing] Menjalankan cleaning pada {len(df)} baris...")
    df['text_clean'] = df[text_col].apply(
        lambda t: bersihkan_teks(t, kamus_gabungan, stopword_final, stemmer)
    )
    print("[preprocessing] Cleaning selesai.")

    sebelum_kosong = len(df)
    df = df[df['text_clean'].str.strip() != ''].reset_index(drop=True)
    sesudah_kosong = len(df)

    sebelum_1kata = len(df)
    df = df[df['text_clean'].str.split().str.len() >= 2].reset_index(drop=True)
    sesudah_1kata = len(df)

    print("\n" + "=" * 55)
    print("  LAPORAN DATA CLEANING")
    print("=" * 55)
    print(f"  Baris awal              : {sebelum_dup:,}")
    print(f"  Baris duplikat dihapus  : {sebelum_dup - sesudah_dup:,}")
    print(f"  Baris kosong dihapus    : {sebelum_kosong - sesudah_kosong:,}")
    print(f"  Baris 1 kata dihapus    : {sebelum_1kata - sesudah_1kata:,}")
    print(f"  Baris akhir (bersih)    : {sesudah_1kata:,}")
    print("=" * 55)

    print("\n  CONTOH HASIL (5 baris pertama):")
    print("-" * 55)
    for i, row in df.head(5).iterrows():
        print(f"  [{i+1}] SEBELUM : {str(row[text_col])[:65]}")
        print(f"       SESUDAH : {str(row['text_clean'])[:65]}")
        print()

    return df

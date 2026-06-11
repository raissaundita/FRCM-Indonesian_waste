import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform

DATA_PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"

METHOD_PATHS = {
    "kmeans": "D:/Raissa/Python/KLASTER DATA/KMEANS_K4",
    "fcm": "D:/Raissa/Python/KLASTER DATA/FCM_K4",
    "frcm": "D:/Raissa/Python/KLASTER DATA/FRCM_K4",
    "fafrcm": "D:/Raissa/Python/KLASTER DATA/FAFRCM_K3"
}

OUT_DIR = "D:/Raissa/Python/EVALUASI/Xie Beni"
N_RUNS = 100
M_FUZZY = 2   # fuzzifier, biasanya 2

os.makedirs(OUT_DIR, exist_ok=True)

NAMA_METODE = {
    "kmeans": "K-Means",
    "fcm": "FCM",
    "frcm": "FRCM",
    "fafrcm": "FA-FRCM"
}

URUTAN_METODE = {
    "K-Means": 1,
    "FCM": 2,
    "FRCM": 3,
    "FA-FRCM": 4
}

URUTAN_KOLOM = ["kmeans", "fcm", "frcm", "fafrcm"]
POLA_FILE = re.compile(r"(kmeans|fcm|frcm|fafrcm)_k(\d+)_run(\d+)\.csv", re.IGNORECASE)

# LOAD
data = pd.read_csv(DATA_PATH)
fitur = data.select_dtypes(include=[np.number]).copy()

if "Tahun" in fitur.columns:
    fitur = fitur.drop(columns=["Tahun"])

X = fitur.to_numpy()


# FUNGSI BANTU
def ambil_run(nama_file):
    hasil = re.search(r"run(\d+)", nama_file.lower())
    if hasil:
        return int(hasil.group(1))
    return np.nan

def ambil_k(nama_file):
    hasil = re.search(r"_k(\d+)", nama_file.lower())
    if hasil:
        return int(hasil.group(1))
    return np.nan

def ambil_label(path_file):
    df = pd.read_csv(path_file)
    df.columns = df.columns.str.strip()

    if "label" in df.columns:
        return df["label"].to_numpy()
    elif "cluster" in df.columns:
        return df["cluster"].to_numpy()
    else:
        raise ValueError(f"Kolom label/cluster tidak ditemukan di file: {os.path.basename(path_file)}")

def ambil_membership(path_file):
    """
    Membaca membership matrix U.
    Prioritas:
    1. kolom yang namanya diawali 'u_'
    2. kalau tidak ada, semua kolom numerik dipakai
    """
    df = pd.read_csv(path_file)
    df.columns = df.columns.str.strip()

    kolom_u = [c for c in df.columns if c.lower().startswith("u_")]

    if len(kolom_u) > 0:
        U = df[kolom_u].to_numpy(dtype=float)
    else:
        U = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)

    return U


# FUNGSI XIE-BENI
def hitung_centroid_fuzzy(X, U, m=2):
    """
    Menghitung centroid fuzzy:
    v_i = sum(u_ij^m * x_j) / sum(u_ij^m)
    """
    U_pangkat_m = U ** m
    pembilang = U_pangkat_m.T @ X
    penyebut = U_pangkat_m.sum(axis=0).reshape(-1, 1)

    centroid = pembilang / np.maximum(penyebut, 1e-12)
    return centroid


def hitung_xie_beni(X, U, m=2):
    # Semakin kecil nilainya, semakin baik.
    n = X.shape[0]
    # hitung centroid fuzzy
    centroid = hitung_centroid_fuzzy(X, U, m=m)

    # jarak kuadrat dari setiap data ke setiap centroid
    jarak_kuadrat = cdist(X, centroid, metric="sqeuclidean")

    # pembilang: compactness
    pembilang = np.sum((U ** m) * jarak_kuadrat)

    # penyebut: separation
    jarak_centroid = squareform(pdist(centroid, metric="sqeuclidean"))
    np.fill_diagonal(jarak_centroid, np.inf)

    minimum_jarak_centroid = np.min(jarak_centroid)
    penyebut = n * minimum_jarak_centroid

    if penyebut <= 1e-12 or np.isinf(penyebut):
        return np.nan

    xb = pembilang / penyebut
    return xb


# PROSES HITUNG
hasil_semua = []

for kode_metode, folder_metode in METHOD_PATHS.items():

    # skip kmeans karena bukan fuzzy
    if kode_metode == "kmeans":
       continue

    daftar_file = [
        f for f in os.listdir(folder_metode)
        if f.endswith(".csv")
        and "run" in f.lower()
        and "membership" not in f.lower()
        and "rekap" not in f.lower()
    ]

    daftar_file = sorted(daftar_file, key=ambil_run)

    print(f"\nMetode: {kode_metode}")
    print("File dibaca:", daftar_file)

    for nama_file in daftar_file:
        if not POLA_FILE.match(nama_file):
            continue

        path_file = os.path.join(folder_metode, nama_file)

        # file membership
        nama_file_membership = nama_file.replace("fcm_k", "fcm_membership_k")
        path_membership = os.path.join(folder_metode, nama_file_membership)

        if not os.path.exists(path_membership):
            print(f"[WARN] File membership tidak ditemukan: {nama_file_membership}")
            continue

        U = ambil_membership(path_membership)

        # cek jumlah baris membership harus sama dengan jumlah data
        if U.shape[0] != len(X):
            print(f"[WARN] Jumlah baris membership tidak cocok di {nama_file_membership}")
            continue

        try:
            nilai_xb = hitung_xie_beni(X, U, m=M_FUZZY)
        except Exception as e:
            print(f"[WARN] Gagal hitung XB untuk {nama_file}: {e}")
            continue

        hasil_semua.append({
            "Run": ambil_run(nama_file),
            "Metode": NAMA_METODE[kode_metode],
            "k": ambil_k(nama_file),
            "Xie-Beni": nilai_xb,
            "File": nama_file,
            "File_Membership": nama_file_membership
        })

hasil_long = pd.DataFrame(hasil_semua)

if hasil_long.empty:
    raise ValueError("Tidak ada hasil Xie-Beni yang berhasil dihitung.")

hasil_long["urutan"] = hasil_long["Metode"].map(URUTAN_METODE)
hasil_long = hasil_long.sort_values(["urutan", "k", "Run"]).drop(columns="urutan").reset_index(drop=True)

# TABEL WIDE
temp = hasil_long.copy()

temp["metode_kecil"] = temp["Metode"].replace({
    "K-Means": "kmeans",
    "FCM": "fcm",
    "FRCM": "frcm",
    "FA-FRCM": "fafrcm"
})

tabel_wide = temp.pivot_table(
    index="Run",
    columns="metode_kecil",
    values="Xie-Beni",
    aggfunc="mean"
)

tabel_wide = tabel_wide.reindex(columns=URUTAN_KOLOM)
tabel_wide = tabel_wide.rename(columns={
    "kmeans": "K-Means",
    "fcm": "FCM",
    "frcm": "FRCM",
    "fafrcm": "FA-FRCM"
})

tabel_wide = tabel_wide.reindex(range(1, N_RUNS + 1))
tabel_wide.index.name = "Run"


ringkasan = (
    hasil_long
    .groupby(["k", "Metode"])["Xie-Beni"]
    .agg(mean="mean", std="std", min="min", max="max", median="median")
    .reset_index()
)

ringkasan["urutan"] = ringkasan["Metode"].map(URUTAN_METODE)
ringkasan = ringkasan.sort_values(["urutan", "k"]).drop(columns="urutan").reset_index(drop=True)

# SIMPAN
path_long = os.path.join(OUT_DIR, "xie_beni_long.csv")
path_wide = os.path.join(OUT_DIR, "table_xie_beni.csv")
path_excel = os.path.join(OUT_DIR, "xie_beni_tables.xlsx")
path_boxplot = os.path.join(OUT_DIR, "boxplot_xie_beni.png")

hasil_long.to_csv(path_long, index=False)
tabel_wide.to_csv(path_wide)

with pd.ExcelWriter(path_excel, engine="openpyxl") as writer:
    hasil_long.to_excel(writer, sheet_name="LONG", index=False)
    tabel_wide.to_excel(writer, sheet_name="XB_Table")
    ringkasan.to_excel(writer, sheet_name="Summary_XB", index=False)

plt.figure(figsize=(8, 5))
hasil_long.boxplot(column="Xie-Beni", by="Metode")
plt.title("Boxplot Xie-Beni per Metode")
plt.suptitle("")
plt.xlabel("Metode")
plt.ylabel("Nilai Xie-Beni")
plt.tight_layout()
plt.savefig(path_boxplot, dpi=300)
plt.close()

print("\nSelesai!")
print("Output LONG :", path_long)
print("Output WIDE :", path_wide)
print("Output Excel:", path_excel)
print("Output Plot :", path_boxplot)
print("\nContoh tabel:")
print(tabel_wide.head(10))
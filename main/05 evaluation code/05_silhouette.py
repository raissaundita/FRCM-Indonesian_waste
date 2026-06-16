import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

DATA_PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"

METHOD_PATHS = {
    "kmeans": "D:/Raissa/Python/KLASTER DATA/KMEANS_K4",
    "fcm": "D:/Raissa/Python/KLASTER DATA/FCM_K4",
    "frcm": "D:/Raissa/Python/KLASTER DATA/FRCM_K4",
    "fafrcm": "D:/Raissa/Python/KLASTER DATA/FAFRCM_K3"
}

OUT_DIR = "D:/Raissa/Python/EVALUASI/Silhouette"
N_RUNS = 100

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
fitur = data.select_dtypes(include=[np.number]).copy()  # ambil kolom numerik saja

if "Tahun" in fitur.columns:
    fitur = fitur.drop(columns=["Tahun"])  # kalau ada kolom Tahun, buang

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

def hitung_silhouette(X, label):
    if len(np.unique(label)) < 2:
        return np.nan
    return silhouette_score(X, label, metric="euclidean")


# PROSES HITUNG
hasil_semua = []

for kode_metode, folder_metode in METHOD_PATHS.items():
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
        label = ambil_label(path_file)

        if len(label) != len(X):
            print(f"[WARN] Jumlah label tidak cocok: {nama_file}")
            continue

        nilai_silhouette = hitung_silhouette(X, label)

        hasil_semua.append({
            "Run": ambil_run(nama_file),
            "Metode": NAMA_METODE[kode_metode],
            "k": ambil_k(nama_file),
            "Silhouette": nilai_silhouette,
            "File": nama_file
        })

hasil_long = pd.DataFrame(hasil_semua)

if hasil_long.empty:
    raise ValueError("Tidak ada hasil silhouette yang berhasil dihitung.")

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
    values="Silhouette",
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
    .groupby(["k", "Metode"])["Silhouette"]
    .agg(mean="mean", std="std", min="min", max="max", median="median")
    .reset_index()
)

ringkasan["urutan"] = ringkasan["Metode"].map(URUTAN_METODE)
ringkasan = ringkasan.sort_values(["urutan", "k"]).drop(columns="urutan").reset_index(drop=True)

# SIMPAN
path_long = os.path.join(OUT_DIR, "silhouette_long.csv")
path_wide = os.path.join(OUT_DIR, "table_silhouette.csv")
path_excel = os.path.join(OUT_DIR, "silhouette_tables.xlsx")
path_boxplot = os.path.join(OUT_DIR, "boxplot_silhouette.png")

hasil_long.to_csv(path_long, index=False)
tabel_wide.to_csv(path_wide)

with pd.ExcelWriter(path_excel, engine="openpyxl") as writer:
    hasil_long.to_excel(writer, sheet_name="LONG", index=False)
    tabel_wide.to_excel(writer, sheet_name="Silhouette_Table")
    ringkasan.to_excel(writer, sheet_name="Summary_Silhouette", index=False)

plt.figure(figsize=(8, 5))
hasil_long.boxplot(column="Silhouette", by="Metode")
plt.title("Boxplot Silhouette per Metode")
plt.suptitle("")
plt.xlabel("Metode")
plt.ylabel("Nilai Silhouette")
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
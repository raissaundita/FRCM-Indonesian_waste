import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"

METHOD_PATHS = {
    "kmeans": "D:/Raissa/Python/KLASTER DATA/KMEANS_K4",
    "fcm": "D:/Raissa/Python/KLASTER DATA/FCM_K4",
    "frcm": "D:/Raissa/Python/KLASTER DATA/FRCM_K4",
    "fafrcm": "D:/Raissa/Python/KLASTER DATA/FAFRCM_K3"
}

OUT_DIR = "D:/Raissa/Python/EVALUASI/Symmetric Purity"
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


# FUNGSI PURITY DAN SP
def hitung_purity_satu_arah(label_p, label_q):
    """
    Menghitung Purity(P,Q)
    P dianggap sebagai hasil clustering yang sedang dicek,
    Q dianggap sebagai pembanding.
    Rumus:
    Purity(P,Q) = (1/n) * sum over cluster di P [ max irisan dengan cluster di Q ]
    """
    n = len(label_p)
    total_maksimum = 0

    cluster_p_unik = np.unique(label_p)
    cluster_q_unik = np.unique(label_q)

    for cp in cluster_p_unik:
        anggota_p = np.where(label_p == cp)[0]

        maksimum_irisan = 0
        for cq in cluster_q_unik:
            anggota_q = np.where(label_q == cq)[0]

            jumlah_irisan = len(np.intersect1d(anggota_p, anggota_q))
            if jumlah_irisan > maksimum_irisan:
                maksimum_irisan = jumlah_irisan

        total_maksimum += maksimum_irisan

    purity = total_maksimum / n
    return purity


def hitung_symmetric_purity(label_p, label_q):
    """
    Symmetric Purity:
    SP(P,Q) = Purity(P,Q) * Purity(Q,P)
    """
    purity_pq = hitung_purity_satu_arah(label_p, label_q)
    purity_qp = hitung_purity_satu_arah(label_q, label_p)

    sp = purity_pq * purity_qp
    return sp


def hitung_sp_rata_rata(daftar_label):
    hasil_sp = []

    for i in range(len(daftar_label)):
        nilai_perbandingan = []

        for j in range(len(daftar_label)):
            if i == j:
                continue

            sp = hitung_symmetric_purity(daftar_label[i], daftar_label[j])
            nilai_perbandingan.append(sp)

        if len(nilai_perbandingan) == 0:
            hasil_sp.append(np.nan)
        else:
            hasil_sp.append(np.mean(nilai_perbandingan))

    return hasil_sp


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

    data_run = []

    for nama_file in daftar_file:
        if not POLA_FILE.match(nama_file):
            continue

        path_file = os.path.join(folder_metode, nama_file)
        label = ambil_label(path_file)

        data_run.append({
            "Run": ambil_run(nama_file),
            "Metode": NAMA_METODE[kode_metode],
            "k": ambil_k(nama_file),
            "File": nama_file,
            "Label": label
        })

    if len(data_run) == 0:
        continue

    daftar_label = [item["Label"] for item in data_run]
    daftar_sp = hitung_sp_rata_rata(daftar_label)

    for item, nilai_sp in zip(data_run, daftar_sp):
        hasil_semua.append({
            "Run": item["Run"],
            "Metode": item["Metode"],
            "k": item["k"],
            "Symmetric Purity": nilai_sp,
            "File": item["File"]
        })

hasil_long = pd.DataFrame(hasil_semua)

if hasil_long.empty:
    raise ValueError("Tidak ada hasil Symmetric Purity yang berhasil dihitung.")

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
    values="Symmetric Purity",
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
    .groupby(["k", "Metode"])["Symmetric Purity"]
    .agg(mean="mean", std="std", min="min", max="max", median="median")
    .reset_index()
)

ringkasan["urutan"] = ringkasan["Metode"].map(URUTAN_METODE)
ringkasan = ringkasan.sort_values(["urutan", "k"]).drop(columns="urutan").reset_index(drop=True)

# SIMPAN FILE
path_long = os.path.join(OUT_DIR, "symmetric_purity_long.csv")
path_wide = os.path.join(OUT_DIR, "table_symmetric_purity.csv")
path_excel = os.path.join(OUT_DIR, "symmetric_purity_tables.xlsx")
path_boxplot = os.path.join(OUT_DIR, "boxplot_symmetric_purity.png")

hasil_long.to_csv(path_long, index=False)
tabel_wide.to_csv(path_wide)

with pd.ExcelWriter(path_excel, engine="openpyxl") as writer:
    hasil_long.to_excel(writer, sheet_name="LONG", index=False)
    tabel_wide.to_excel(writer, sheet_name="SP_Table")
    ringkasan.to_excel(writer, sheet_name="Summary_SP", index=False)

plt.figure(figsize=(8, 5))
hasil_long.boxplot(column="Symmetric Purity", by="Metode")
plt.title("Boxplot Symmetric Purity per Metode")
plt.suptitle("")
plt.xlabel("Metode")
plt.ylabel("Nilai Symmetric Purity")
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
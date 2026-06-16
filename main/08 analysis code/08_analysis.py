import os
import pandas as pd
import numpy as np

data_file = "D:/Raissa/Python/PREPROCE/04_aggregated_provinsi.csv"

# File label hasil FRCM run ke-6
label_file = "D:/Raissa/Python/KLASTER DATA/FRCM_K4/frcm_k4_run6.csv"
membership_file = "D:/Raissa/Python/KLASTER DATA/FRCM_K4/frcm_membership_k4_run6.csv"

output_file = "D:/Raissa/Python/VISUALISASI/analisis_cluster_FRCM_run6.xlsx"

# Nama kolom penting
col_tahun = "Tahun"
col_provinsi = "Provinsi"
col_label = "label"

nama_cluster = {
    0: "Klaster 1",
    1: "Klaster 2",
    2: "Klaster 3",
    3: "Klaster 4"
}

# ============================================================
# BACA DATA
data_df = pd.read_csv(data_file)
label_df = pd.read_csv(label_file)

print("=" * 60)
print("DATA BERHASIL DIBACA")
print("Ukuran data fitur :", data_df.shape)
print("Ukuran data label :", label_df.shape)

# Cek jumlah baris
if len(data_df) != len(label_df):
    raise ValueError("Jumlah baris data fitur dan label tidak sama.")

# Tambahkan label FRCM ke data
data_df[col_label] = label_df[col_label].values

# Tambahkan nama cluster
data_df["Cluster"] = data_df[col_label].map(nama_cluster)

# ============================================================
# TAMBAHKAN INFORMASI MEMBERSHIP
if os.path.exists(membership_file):
    membership_df = pd.read_csv(membership_file)

    print("\nFile membership ditemukan.")
    print("Kolom membership:", membership_df.columns.tolist())

    # Ambil hanya kolom numerik sebagai nilai membership
    membership_numeric = membership_df.select_dtypes(include=[np.number]).copy()

    # Jika ada kolom Tahun atau label ikut terbaca sebagai numerik, buang dulu
    membership_numeric = membership_numeric.drop(
        columns=[col_tahun, col_label],
        errors="ignore"
    )

    # Ambil nilai membership terbesar
    data_df["Max_Membership"] = membership_numeric.max(axis=1)

    # Keterangan lower / boundary region
    data_df["Status_Membership"] = np.where(
        data_df["Max_Membership"] >= 0.9999,
        "Lower Approximation",
        "Boundary Region"
    )

else:
    print("\nFile membership tidak ditemukan, bagian membership dilewati.")
    data_df["Max_Membership"] = np.nan
    data_df["Status_Membership"] = "-"

# ============================================================
# PILIH KOLOM FITUR NUMERIK
fitur_cols = data_df.drop(
    columns=[col_tahun, col_provinsi, col_label, "Cluster",
             "Max_Membership", "Status_Membership"],
    errors="ignore"
).columns.tolist()

# Pastikan semua fitur jadi numerik
for col in fitur_cols:
    data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

# ============================================================
# SHEET 1: DATA GABUNGAN
data_gabungan = data_df.copy()

# ============================================================
# SHEET 2: RINGKASAN PER CLUSTER
ringkasan_cluster = data_df.groupby("Cluster")[fitur_cols].mean().reset_index()

jumlah_anggota = data_df.groupby("Cluster").size().reset_index(name="n_data (provinsi-tahun)")

ringkasan_cluster = pd.merge(
    jumlah_anggota,
    ringkasan_cluster,
    on="Cluster",
    how="left"
)

# ============================================================
# SHEET 3: ANGGOTA PER CLUSTER DAN TAHUN
anggota_cluster_tahun = data_df.sort_values(
    by=["Cluster", col_tahun, col_provinsi]
).reset_index(drop=True)

# ============================================================
# SHEET 4: RANKING FITUR
# Untuk melihat fitur mana yang paling tinggi/rendah pada setiap cluster
mean_per_cluster = data_df.groupby("Cluster")[fitur_cols].mean()

ranking_list = []

for fitur in fitur_cols:
    nilai_fitur = mean_per_cluster[fitur]

    cluster_tertinggi = nilai_fitur.idxmax()
    cluster_terendah = nilai_fitur.idxmin()

    row = {
        "Fitur": fitur,
        "Cluster Tertinggi": cluster_tertinggi,
        "Nilai Tertinggi": nilai_fitur.max(),
        "Cluster Terendah": cluster_terendah,
        "Nilai Terendah": nilai_fitur.min(),
        "Selisih": nilai_fitur.max() - nilai_fitur.min()
    }

    for cluster in mean_per_cluster.index:
        row[cluster] = nilai_fitur.loc[cluster]

    ranking_list.append(row)

ranking_fitur = pd.DataFrame(ranking_list)

ranking_fitur = ranking_fitur.sort_values(
    by="Selisih",
    ascending=False
).reset_index(drop=True)

# ============================================================
# SHEET 5: JUMLAH ANGGOTA PER TAHUN
jumlah_per_tahun = data_df.pivot_table(
    index=col_tahun,
    columns="Cluster",
    values=col_provinsi,
    aggfunc="count",
    fill_value=0
).reset_index()

# ============================================================
# SHEET 6: DAFTAR PROVINSI PER TAHUN DAN CLUSTER
provinsi_per_tahun_cluster = data_df.groupby(
    [col_tahun, "Cluster"]
)[col_provinsi].apply(lambda x: ", ".join(x)).reset_index()

provinsi_per_tahun_cluster.columns = [
    "Tahun",
    "Cluster",
    "Daftar Provinsi"
]

# ============================================================
# SIMPAN KE EXCEL
output_folder = os.path.dirname(output_file)

if output_folder != "":
    os.makedirs(output_folder, exist_ok=True)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    data_gabungan.to_excel(writer, sheet_name="Data_Gabungan", index=False)
    ringkasan_cluster.to_excel(writer, sheet_name="Ringkasan_per_Cluster", index=False)
    anggota_cluster_tahun.to_excel(writer, sheet_name="Anggota_per_Cluster_Tahun", index=False)
    ranking_fitur.to_excel(writer, sheet_name="Ranking_Fitur", index=False)
    jumlah_per_tahun.to_excel(writer, sheet_name="Jumlah_per_Tahun", index=False)
    provinsi_per_tahun_cluster.to_excel(writer, sheet_name="Provinsi_per_Tahun", index=False)

print("\n" + "=" * 60)
print("FILE ANALISIS FRCM RUN 6 BERHASIL DIBUAT")
print(output_file)

print("\nJumlah anggota tiap cluster:")
print(data_df["Cluster"].value_counts())
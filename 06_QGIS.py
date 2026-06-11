import pandas as pd

label_file      = "D:/Raissa/Python/KLASTER DATA/FRCM_K4/frcm_k4_run6.csv"
membership_file = "D:/Raissa/Python/KLASTER DATA/FRCM_K4/frcm_membership_k4_run6.csv"
output_file     = "D:/Raissa/Python/VISUALISASI/PETA/frcm_run6_untuk_peta.csv"

# BACA FILE
df_label = pd.read_csv(label_file)
df_mem   = pd.read_csv(membership_file)

print("Kolom label file   :", df_label.columns.tolist())
print("Kolom membership   :", df_mem.columns.tolist())
print("Jumlah baris label :", len(df_label))
print("Jumlah baris mem   :", len(df_mem))

# GABUNGKAN LABEL + MEMBERSHIP
# Ambil kolom yang dibutuhkan dari membership
kolom_mem = ["Tahun", "Provinsi", "u_cluster_0", "u_cluster_1",
             "u_cluster_2", "u_cluster_3", "region_type"]

df_gabung = pd.merge(
    df_label,
    df_mem[kolom_mem],
    on=["Tahun", "Provinsi"],
    how="left"
)

# TAMBAH KOLOM NAMA KLASTER (lebih informatif di peta)
# label 0,1,2,3 → Klaster 1,2,3,4 (supaya lebih mudah dibaca)
df_gabung["nama_klaster"] = df_gabung["label"].apply(
    lambda x: f"Klaster {x + 1}"
)

# TAMBAH KOLOM MEMBERSHIP DOMINAN
# (seberapa yakin provinsi ini masuk klaster utamanya)
kolom_u = ["u_cluster_0", "u_cluster_1", "u_cluster_2", "u_cluster_3"]
df_gabung["membership_dominan"] = df_gabung[kolom_u].max(axis=1).round(4)

# Kalau membership = 1.0 artinya lower approximation (pasti masuk klaster itu)
# Kalau < 1.0 artinya boundary (agak tidak pasti)
df_gabung["kepastian"] = df_gabung["membership_dominan"].apply(
    lambda x: "Pasti (Lower)" if x == 1.0 else f"Boundary ({x:.2f})"
)

# TAMPILKAN SAMPEL HASIL
print("\n" + "=" * 60)
print("SAMPEL DATA GABUNGAN")
print(df_gabung[["Tahun", "Provinsi", "label", "nama_klaster",
                  "membership_dominan", "kepastian", "region_type"]].head(10))

print("\nJumlah data per tahun:")
print(df_gabung["Tahun"].value_counts().sort_index())

print("\nJumlah data per klaster:")
print(df_gabung["nama_klaster"].value_counts().sort_index())

# SIMPAN
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_gabung.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("File berhasil disimpan ke:")
print(output_file)
print("=" * 60)
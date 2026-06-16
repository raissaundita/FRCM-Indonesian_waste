import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

uji_statistik_file = "D:/Raissa/Python/UJI STATIS/HASIL_UJI_STATISTIKA.xlsx"
data_file = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"

eval_files = {
    "Silhouette": "D:/Raissa/Python/EVALUASI/Silhouette/table_silhouette.csv",
    "Dunn":       "D:/Raissa/Python/EVALUASI/Dunn/table_dunn.csv",
    "SP":         "D:/Raissa/Python/EVALUASI/Symmetric Purity/table_symmetric_purity.csv",
    "Xie-Beni":   "D:/Raissa/Python/EVALUASI/Xie Beni/table_xie_beni.csv",
}

label_dirs = {
    "kmeans": "D:/Raissa/Python/KLASTER DATA/KMEANS_K4",
    "fcm":    "D:/Raissa/Python/KLASTER DATA/FCM_K4",
    "frcm":   "D:/Raissa/Python/KLASTER DATA/FRCM_K4",
    "fafrcm": "D:/Raissa/Python/KLASTER DATA/FAFRCM_K3",
}

method_prefix = {
    "K-Means": "kmeans_k4",
    "FCM":     "fcm_k4",
    "FRCM":    "frcm_k4",
    "FA-FRCM": "fafrcm_k3",
}

# True = makin besar makin baik, False = makin kecil makin baik
arah = {
    "Silhouette": True,
    "Dunn":       True,
    "SP":         True,
    "Xie-Beni":   False,
}

# PARAMETER
sheet_final_rank  = "FinalRank"
col_nama          = "Provinsi"
col_label         = "label"

# Parameter PCA
pca_random_state  = 42

# Opsi visualisasi
show_labels       = False     # ubah ke True kalau ingin tampilkan nama provinsi
point_size        = 120
label_fontsize    = 8
figure_size_2d    = (14, 9)
figure_size_3d    = (14, 10)

# Output
output_pca_csv    = "D:/Raissa/Python/VISUALISASI/hasil_pca.csv"
output_pca_2d     = "D:/Raissa/Python/VISUALISASI/pca_2d.png"
output_pca_3d     = "D:/Raissa/Python/VISUALISASI/pca_3d.png"


# FUNGSI BANTU
def minmax_benefit(series):
    """Normalisasi untuk metrik yang makin besar makin baik."""
    smin, smax = series.min(), series.max()
    if smax == smin:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - smin) / (smax - smin)

def minmax_cost(series):
    """Normalisasi untuk metrik yang makin kecil makin baik."""
    smin, smax = series.min(), series.max()
    if smax == smin:
        return pd.Series([1.0] * len(series), index=series.index)
    return (smax - series) / (smax - smin)

# AMBIL METODE TERBAIK
final_rank_df = pd.read_excel(uji_statistik_file, sheet_name=sheet_final_rank)
final_rank_df = final_rank_df.sort_values("FinalRank").reset_index(drop=True)

best_method = final_rank_df.loc[0, "Method"]

print("=" * 60)
print("METODE TERBAIK BERDASARKAN HASIL UJI STATISTIK")
print(best_method)

# GABUNGKAN NILAI EVALUASI PER RUN
run_scores = None

for metric_name, file_path in eval_files.items():
    print("\nMembaca file evaluasi:", file_path)

    df = pd.read_csv(file_path)

    if "Run" not in df.columns:
        raise ValueError(f"Kolom 'Run' tidak ditemukan pada file: {file_path}")

    if best_method not in df.columns:
        raise ValueError(
            f"Kolom metode '{best_method}' tidak ditemukan pada file: {file_path}"
        )

    temp = df[["Run", best_method]].copy()
    temp.columns = ["Run", metric_name]

    if run_scores is None:
        run_scores = temp
    else:
        run_scores = pd.merge(run_scores, temp, on="Run", how="inner")

print("\n" + "=" * 60)
print("DATA EVALUASI GABUNGAN PER RUN UNTUK METODE TERBAIK")
print(run_scores.head())

# HITUNG SKOR GABUNGAN & TENTUKAN RUN TERBAIK
norm_cols = []

for metric_name in eval_files.keys():
    if metric_name in run_scores.columns:
        norm_col = metric_name + "_norm"

        if arah[metric_name]:
            run_scores[norm_col] = minmax_benefit(run_scores[metric_name])
        else:
            run_scores[norm_col] = minmax_cost(run_scores[metric_name])

        norm_cols.append(norm_col)

run_scores["final_score"] = run_scores[norm_cols].mean(axis=1)
run_scores = run_scores.sort_values("final_score", ascending=False).reset_index(drop=True)

best_run = int(run_scores.loc[0, "Run"])

print("\n" + "=" * 60)
print("RUN TERBAIK")
print(f"Run terbaik untuk metode {best_method} adalah run ke-{best_run}")
print("\n5 run teratas berdasarkan skor gabungan:")
print(run_scores.head())

# BACA FILE LABEL RUN TERBAIK
method_key_map = {
    "K-Means": "kmeans",
    "FCM":     "fcm",
    "FRCM":    "frcm",
    "FA-FRCM": "fafrcm",
}

if best_method not in method_key_map:
    raise ValueError(f"Method '{best_method}' tidak dikenali di mapping.")

method_key = method_key_map[best_method]
prefix     = method_prefix[best_method]
label_file = os.path.join(label_dirs[method_key], f"{prefix}_run{best_run}.csv")

print("\n" + "=" * 60)
print("FILE LABEL YANG AKAN DIBACA")
print(label_file)

labels_df = pd.read_csv(label_file)

print("\nKolom pada file label:")
print(labels_df.columns.tolist())

if col_label not in labels_df.columns:
    raise ValueError(
        f"Kolom label '{col_label}' tidak ditemukan pada file label.\n"
        f"Kolom yang tersedia: {labels_df.columns.tolist()}"
    )

# BACA DATA FITUR
data_df = pd.read_csv(data_file)

print("\n" + "=" * 60)
print("DATA FITUR")
print("Ukuran data fitur:", data_df.shape)
print("Kolom data fitur:", data_df.columns.tolist())

if len(data_df) != len(labels_df):
    raise ValueError(
        f"Jumlah baris data fitur ({len(data_df)}) tidak sama "
        f"dengan jumlah baris label ({len(labels_df)})."
    )

data_df["cluster"] = labels_df[col_label].values

print("\n" + "=" * 60)
print("JUMLAH ANGGOTA TIAP CLUSTER")
print(data_df["cluster"].value_counts().sort_index())

# AMBIL FITUR NUMERIK
X = data_df.drop(columns=[col_nama, "cluster"], errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce")

# Buang baris yang ada NaN (agar PCA tidak error)
valid_idx = X.dropna().index
X        = X.loc[valid_idx]
plot_df  = data_df.loc[valid_idx].copy()

print("\n" + "=" * 60)
print("DATA YANG DIGUNAKAN UNTUK PCA")
print("Ukuran X:", X.shape)

if X.shape[0] < 3:
    raise ValueError("Jumlah data valid terlalu sedikit untuk PCA.")


# JALANKAN PCA 2D
pca_2d    = PCA(n_components=2, random_state=pca_random_state)
X_pca_2d  = pca_2d.fit_transform(X)

# Simpan berapa persen varians yang berhasil dijelaskan oleh 2 komponen
var_2d = pca_2d.explained_variance_ratio_ * 100

print("\n" + "=" * 60)
print("HASIL PCA 2D")
print(f"Varians yang dijelaskan PC1: {var_2d[0]:.2f}%")
print(f"Varians yang dijelaskan PC2: {var_2d[1]:.2f}%")
print(f"Total varians (PC1+PC2)    : {sum(var_2d):.2f}%")

# JALANKAN PCA 3D
pca_3d    = PCA(n_components=3, random_state=pca_random_state)
X_pca_3d  = pca_3d.fit_transform(X)

# Simpan berapa persen varians yang berhasil dijelaskan oleh 3 komponen
var_3d = pca_3d.explained_variance_ratio_ * 100

print("\n" + "=" * 60)
print("HASIL PCA 3D")
print(f"Varians yang dijelaskan PC1: {var_3d[0]:.2f}%")
print(f"Varians yang dijelaskan PC2: {var_3d[1]:.2f}%")
print(f"Varians yang dijelaskan PC3: {var_3d[2]:.2f}%")
print(f"Total varians (PC1+PC2+PC3): {sum(var_3d):.2f}%")

# SIMPAN HASIL PCA KE CSV
pca_df = pd.DataFrame({
    "PC1"    : X_pca_2d[:, 0],
    "PC2"    : X_pca_2d[:, 1],
    "PC3"    : X_pca_3d[:, 2],  # komponen ke-3 dari PCA 3D
    "cluster": plot_df["cluster"].values,
})

if col_nama in plot_df.columns:
    pca_df[col_nama] = plot_df[col_nama].values

output_folder_csv = os.path.dirname(output_pca_csv)
if output_folder_csv != "":
    os.makedirs(output_folder_csv, exist_ok=True)

pca_df.to_csv(output_pca_csv, index=False)
print("\n" + "=" * 60)
print("Hasil PCA disimpan ke:")
print(output_pca_csv)

# VISUALISASI PCA 2D
plt.figure(figsize=figure_size_2d)

scatter_2d = plt.scatter(
    X_pca_2d[:, 0],
    X_pca_2d[:, 1],
    c=plot_df["cluster"].values,
    cmap="tab10",
    s=point_size,
    alpha=0.9,
)

# Label sumbu menyertakan persentase varians yang dijelaskan
plt.xlabel(f"PC1 ({var_2d[0]:.2f}% varians)")
plt.ylabel(f"PC2 ({var_2d[1]:.2f}% varians)")
plt.title(f"Visualisasi PCA 2D - {best_method} (Run {best_run})", fontsize=14)
plt.grid(alpha=0.3)

legend_2d = plt.legend(*scatter_2d.legend_elements(), title="Cluster")
plt.gca().add_artist(legend_2d)

# Tampilkan nama provinsi jika show_labels = True
if show_labels and col_nama in pca_df.columns:
    for _, row in pca_df.iterrows():
        plt.text(
            row["PC1"],
            row["PC2"],
            str(row[col_nama]),
            fontsize=label_fontsize,
            alpha=0.85,
        )

plt.tight_layout()

output_folder_2d = os.path.dirname(output_pca_2d)
if output_folder_2d != "":
    os.makedirs(output_folder_2d, exist_ok=True)

plt.savefig(output_pca_2d, dpi=300, bbox_inches="tight")
print("\nPlot PCA 2D disimpan ke:")
print(output_pca_2d)
plt.show()

# VISUALISASI PCA 3D
fig = plt.figure(figsize=figure_size_3d)

# ax = sumbu 3D
ax = fig.add_subplot(111, projection="3d")

scatter_3d = ax.scatter(
    X_pca_3d[:, 0],
    X_pca_3d[:, 1],
    X_pca_3d[:, 2],
    c=plot_df["cluster"].values,
    cmap="tab10",
    s=point_size,
    alpha=0.9,
)

# Label sumbu menyertakan persentase varians yang dijelaskan
ax.set_xlabel(f"PC1 ({var_3d[0]:.2f}%)")
ax.set_ylabel(f"PC2 ({var_3d[1]:.2f}%)")
ax.set_zlabel(f"PC3 ({var_3d[2]:.2f}%)")
ax.set_title(f"Visualisasi PCA 3D - {best_method} (Run {best_run})", fontsize=14)

legend_3d = ax.legend(*scatter_3d.legend_elements(), title="Cluster", loc="upper left")
ax.add_artist(legend_3d)

# Tampilkan nama provinsi jika show_labels = True
if show_labels and col_nama in pca_df.columns:
    for _, row in pca_df.iterrows():
        ax.text(
            row["PC1"],
            row["PC2"],
            row["PC3"],
            str(row[col_nama]),
            fontsize=label_fontsize,
            alpha=0.85,
        )

plt.tight_layout()

output_folder_3d = os.path.dirname(output_pca_3d)
if output_folder_3d != "":
    os.makedirs(output_folder_3d, exist_ok=True)

plt.savefig(output_pca_3d, dpi=300, bbox_inches="tight")
print("\nPlot PCA 3D disimpan ke:")
print(output_pca_3d)
plt.show()

print("\n" + "=" * 60)
print("SELESAI — PCA 2D dan 3D berhasil dibuat.")
print("=" * 60)
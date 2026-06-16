import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

uji_statistik_file = "D:/Raissa/Python/UJI STATIS/HASIL_UJI_STATISTIKA.xlsx"
data_file = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"

eval_files = {
    "Silhouette": "D:/Raissa/Python/EVALUASI/Silhouette/table_silhouette.csv",
    "Dunn":"D:/Raissa/Python/EVALUASI/Dunn/table_dunn.csv",
    "SP": "D:/Raissa/Python/EVALUASI/Symmetric Purity/table_symmetric_purity.csv",
    "Xie-Beni": "D:/Raissa/Python/EVALUASI/Xie Beni/table_xie_beni.csv"
}

label_dirs = {
    "kmeans": "D:/Raissa/Python/KLASTER DATA/KMEANS_K4",
    "fcm": "D:/Raissa/Python/KLASTER DATA/FCM_K4",
    "frcm": "D:/Raissa/Python/KLASTER DATA/FRCM_K4",
    "fafrcm": "D:/Raissa/Python/KLASTER DATA/FAFRCM_K3"
}

method_prefix = {
    "K-Means": "kmeans_k4",
    "FCM": "fcm_k4",
    "FRCM": "frcm_k4",
    "FA-FRCM": "fafrcm_k3"
}

# True  = makin besar makin baik
# False = makin kecil makin baik
arah = {
    "Silhouette": True,
    "Dunn": True,
    "SP": True,
    "Xie-Beni": False
}

# PARAMETER
sheet_final_rank = "FinalRank"
col_nama = "Provinsi"
col_label = "label"

# Parameter t-SNE
tsne_perplexity = 8
tsne_random_state = 42
tsne_max_iter = 1500
tsne_learning_rate = "auto"
tsne_init = "pca"

# Opsi visualisasi
show_labels = False          # ubah ke False kalau plot terlalu penuh
point_size = 120
label_fontsize = 8
figure_size = (14, 9)

save_tsne_result = True
output_tsne_csv = "D:/Raissa/Python/VISUALISASI/hasil_tsne.csv"
output_tsne_plot = "D:/Raissa/Python/VISUALISASI/2.png"


# FUNGSI BANTU
def minmax_benefit(series):
    smin = series.min()
    smax = series.max()

    if smax == smin:
        return pd.Series([1.0] * len(series), index=series.index)

    return (series - smin) / (smax - smin)

def minmax_cost(series):
    smin = series.min()
    smax = series.max()

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

# HITUNG SKOR GABUNGAN
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

# NORMALISASI NAMA METODE (BIAR KONSISTEN)
method_key_map = {
    "K-Means": "kmeans",
    "FCM": "fcm",
    "FRCM": "frcm",
    "FA-FRCM": "fafrcm"
}

if best_method not in method_key_map:
    raise ValueError(f"Method '{best_method}' tidak dikenali di mapping.")

method_key = method_key_map[best_method]

# BACA FILE LABEL RUN TERBAIK
prefix = method_prefix[best_method]
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
        f"Jumlah baris data fitur ({len(data_df)}) tidak sama dengan jumlah baris label ({len(labels_df)})."
    )

data_df["cluster"] = labels_df[col_label].values

print("\n" + "=" * 60)
print("JUMLAH ANGGOTA TIAP CLUSTER")
print(data_df["cluster"].value_counts().sort_index())

# AMBIL FITUR NUMERIK UNTUK t-SNE
X = data_df.drop(columns=[col_nama, "cluster"], errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce")

valid_idx = X.dropna().index
X = X.loc[valid_idx]
plot_df = data_df.loc[valid_idx].copy()

print("\n" + "=" * 60)
print("DATA YANG DIGUNAKAN UNTUK t-SNE")
print("Ukuran X:", X.shape)

if X.shape[0] < 3:
    raise ValueError("Jumlah data valid terlalu sedikit untuk t-SNE.")

# perplexity harus < jumlah sampel
if tsne_perplexity >= X.shape[0]:
    tsne_perplexity = max(2, X.shape[0] - 1)
    print(f"Perplexity disesuaikan otomatis menjadi {tsne_perplexity}")

# JALANKAN t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=tsne_perplexity,
    learning_rate=tsne_learning_rate,
    init=tsne_init,
    random_state=tsne_random_state,
    max_iter=tsne_max_iter
)

X_tsne = tsne.fit_transform(X)

tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "cluster": plot_df["cluster"].values
})

if col_nama in plot_df.columns:
    tsne_df[col_nama] = plot_df[col_nama].values

print("\n" + "=" * 60)
print("HASIL t-SNE")
print(tsne_df.head())

# SIMPAN HASIL t-SNE
if save_tsne_result:
    output_folder_csv = os.path.dirname(output_tsne_csv)
    if output_folder_csv != "":
        os.makedirs(output_folder_csv, exist_ok=True)

    tsne_df.to_csv(output_tsne_csv, index=False)
    print("\nHasil t-SNE disimpan ke:")
    print(output_tsne_csv)

# VISUALISASI t-SNE
plt.figure(figsize=figure_size)

unique_clusters = sorted(tsne_df["cluster"].unique())

scatter = plt.scatter(
    tsne_df["TSNE1"],
    tsne_df["TSNE2"],
    c=tsne_df["cluster"],
    cmap="tab10",
    s=point_size,
    alpha=0.9
)

plt.title(f"Visualisasi t-SNE - {best_method} (Run {best_run})", fontsize=14)
plt.xlabel("Dimensi t-SNE 1")
plt.ylabel("Dimensi t-SNE 2")
plt.grid(alpha=0.3)

legend = plt.legend(*scatter.legend_elements(), title="Cluster")
plt.gca().add_artist(legend)

if show_labels and col_nama in tsne_df.columns:
    for _, row in tsne_df.iterrows():
        plt.text(
            row["TSNE1"],
            row["TSNE2"],
            str(row[col_nama]),
            fontsize=label_fontsize,
            alpha=0.85
        )

plt.tight_layout()

# simpan plot
output_folder_plot = os.path.dirname(output_tsne_plot)
if output_folder_plot != "":
    os.makedirs(output_folder_plot, exist_ok=True)

plt.savefig(output_tsne_plot, dpi=300, bbox_inches="tight")
print("\nPlot t-SNE disimpan ke:")
print(output_tsne_plot)

plt.show()
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, studentized_range

files = {
    "Silhouette": "D:/Raissa/Python/EVALUASI/Silhouette/table_silhouette.csv",
    "Dunn":"D:/Raissa/Python/EVALUASI/Dunn/table_dunn.csv",
    "SP": "D:/Raissa/Python/EVALUASI/Symmetric Purity/table_symmetric_purity.csv",
    "Xie-Beni": "D:/Raissa/Python/EVALUASI/Xie Beni/table_xie_beni.csv",
}

# Arah evaluasi: True = makin besar makin baik, False = makin kecil makin baik
arah = {
    "Silhouette": True,
    "Dunn"      : True,
    "SP"        : True,
    "Xie-Beni"  : False,
}

output_dir = "D:/Raissa/Python/UJI STATIS"
os.makedirs(output_dir, exist_ok=True)

# KONFIGURASI METODE
METODE_4 = ["K-Means", "FCM", "FRCM", "FA-FRCM"]
METODE_FUZZY = ["FCM", "FRCM", "FA-FRCM"]

METRIK_UTAMA  = ["Silhouette", "Dunn", "SP"]   # 4 metode
METRIK_FUZZY  = ["Xie-Beni"]                   # 3 metode fuzzy saja

# FUNGSI BANTU
def buat_cd_diagram(avg_rank, cd, nama_metrik, output_dir, alpha=0.05):
    """
    Membuat Critical Difference Diagram sederhana.
    Semakin kecil average rank berarti metode semakin baik.
    """

    avg_rank = avg_rank.sort_values()
    methods = avg_rank.index.tolist()
    ranks = avg_rank.values

    fig, ax = plt.subplots(figsize=(9, 3))

    # Garis utama rank
    min_rank = 1
    max_rank = max(np.ceil(ranks.max()), len(methods))

    ax.hlines(y=0, xmin=min_rank, xmax=max_rank, color="black")
    ax.set_xlim(min_rank - 0.2, max_rank + 0.2)
    ax.set_ylim(-1.2, 1.6)

    # Titik dan nama metode
    for i, method in enumerate(methods):
        rank_value = avg_rank[method]
        ax.plot(rank_value, 0, marker="o", color="black")
        ax.text(rank_value, -0.25, method, ha="center", va="top", fontsize=10)
        ax.text(rank_value, 0.18, f"{rank_value:.3f}", ha="center", va="bottom", fontsize=9)

    # Garis Critical Difference
    cd_start = min_rank
    cd_end = min_rank + cd
    ax.hlines(y=1.0, xmin=cd_start, xmax=cd_end, color="black", linewidth=2)
    ax.vlines(x=cd_start, ymin=0.9, ymax=1.1, color="black")
    ax.vlines(x=cd_end, ymin=0.9, ymax=1.1, color="black")
    ax.text((cd_start + cd_end) / 2, 1.18, f"CD = {cd:.3f}", ha="center", fontsize=10)

    # Hubungkan metode yang tidak berbeda signifikan
    y_line = 0.55
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            diff = abs(avg_rank[methods[i]] - avg_rank[methods[j]])
            if diff <= cd:
                ax.hlines(
                    y=y_line,
                    xmin=avg_rank[methods[i]],
                    xmax=avg_rank[methods[j]],
                    color="gray",
                    linewidth=2
                )
                y_line += 0.12

    ax.set_title(f"Critical Difference Diagram - {nama_metrik}")
    ax.set_xlabel("Average Rank")
    ax.set_yticks([])
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()

    path_gambar = os.path.join(output_dir, f"cd_diagram_{nama_metrik}.png")
    plt.savefig(path_gambar, dpi=300, bbox_inches="tight")
    plt.close()

    return path_gambar

def load_table(path):
    df = pd.read_csv(path)
    if "Run" in df.columns:
        df = df.set_index("Run")
    return df.apply(pd.to_numeric, errors="coerce")


def friedman_nemenyi(df, metode, ascending, nama_metrik, alpha=0.05):
    df_clean = df[metode].dropna()
    n = len(df_clean)
    k = len(metode)

    # Friedman Test
    stat, p = friedmanchisquare(*[df_clean[col] for col in metode])

    friedman_dict = {
        "Metric"           : nama_metrik,
        "Metode_Dibandingkan": ", ".join(metode),
        "N_Run"            : n,
        "Statistic"        : round(stat, 4),
        "PValue"           : round(p, 6),
        "Significant_0.05" : "Ya" if p < alpha else "Tidak",
    }

    # Ranking per baris (run)
    rank = df_clean.rank(axis=1, ascending=ascending)
    rank.to_csv(os.path.join(output_dir, f"rank_{nama_metrik}.csv"))

    # Average rank
    avg_rank = rank.mean().sort_values()
    avg_rank_df = avg_rank.reset_index()
    avg_rank_df.columns = ["Method", "AverageRank"]
    avg_rank_df.insert(0, "Metric", nama_metrik)

    # Nemenyi post-hoc (hanya jika Friedman signifikan)
    nemenyi_rows = []
    if p < alpha:
        q = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
        cd = q * np.sqrt(k * (k + 1) / (6 * n))

             # Membuat Critical Difference Diagram
        path_cd = buat_cd_diagram(avg_rank, cd, nama_metrik, output_dir, alpha)
        print(f"  CD diagram disimpan di: {path_cd}")

        methods = avg_rank.index.tolist()
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                diff = abs(avg_rank[methods[i]] - avg_rank[methods[j]])
                nemenyi_rows.append({
                    "Metric"     : nama_metrik,
                    "Method1"    : methods[i],
                    "Method2"    : methods[j],
                    "RankDiff"   : round(diff, 4),
                    "CD"         : round(cd, 4),
                    "Significant": "Ya" if diff > cd else "Tidak",
                })

    return friedman_dict, avg_rank_df, pd.DataFrame(nemenyi_rows)

# ============================================================
# BAGIAN 1: FRIEDMAN UTAMA (4 METODE x 3 METRIK)
print("=" * 60)
print("BAGIAN 1: UJI STATISTIK UTAMA")
print("4 Metode x 3 Metrik (Silhouette, Dunn, SP)")
print("=" * 60)

all_friedman_utama = []
all_avg_rank_utama = []
all_nemenyi_utama  = []

for nama in METRIK_UTAMA:
    print(f"\nMetrik: {nama}")
    df = load_table(files[nama])

    friedman_dict, avg_rank_df, nemenyi_df = friedman_nemenyi(
        df       = df,
        metode   = METODE_4,
        ascending= not arah[nama],   # ascending=True berarti rank 1 = nilai terkecil
        nama_metrik = nama,
    )

    print(f"  Friedman: stat={friedman_dict['Statistic']}, "
          f"p={friedman_dict['PValue']}, "
          f"signifikan={friedman_dict['Significant_0.05']}")
    print(f"  Average rank:\n{avg_rank_df[['Method','AverageRank']].to_string(index=False)}")

    all_friedman_utama.append(pd.DataFrame([friedman_dict]))
    all_avg_rank_utama.append(avg_rank_df)
    all_nemenyi_utama.append(nemenyi_df)

    nemenyi_df.to_csv(
        os.path.join(output_dir, f"nemenyi_{nama}.csv"), index=False
    )

# Gabungkan
df_friedman_utama  = pd.concat(all_friedman_utama, ignore_index=True)
df_avg_rank_utama  = pd.concat(all_avg_rank_utama, ignore_index=True)
df_nemenyi_utama   = pd.concat(all_nemenyi_utama,  ignore_index=True)

# FinalRank: rata-rata average rank dari 3 metrik utama
summary_utama = (
    df_avg_rank_utama
    .groupby("Method")["AverageRank"]
    .mean()
    .sort_values()
    .reset_index()
)
summary_utama.columns = ["Method", "AverageRank_Mean"]
summary_utama["FinalRank"] = range(1, len(summary_utama) + 1)

print("\n" + "=" * 60)
print("FINAL RANK UTAMA (4 Metode, 3 Metrik: Sil + Dunn + SP)")
print("=" * 60)
print(summary_utama.to_string(index=False))

# ============================================================
# BAGIAN 2: FRIEDMAN TAMBAHAN — XIE-BENI (3 METODE FUZZY)
print("\n" + "=" * 60)
print("BAGIAN 2: ANALISIS TAMBAHAN — XIE-BENI (3 Metode Fuzzy)")
print("Catatan: K-Means dikecualikan karena bukan metode fuzzy.")
print("Friedman XB hanya membandingkan FCM, FRCM, dan FA-FRCM.")
print("=" * 60)

all_friedman_xb = []
all_avg_rank_xb = []
all_nemenyi_xb  = []

for nama in METRIK_FUZZY:
    print(f"\nMetrik: {nama}")
    df = load_table(files[nama])

    # Hanya ambil kolom fuzzy (buang K-Means yang NaN)
    df_fuzzy = df[METODE_FUZZY]

    friedman_dict, avg_rank_df, nemenyi_df = friedman_nemenyi(
        df       = df_fuzzy,
        metode   = METODE_FUZZY,
        ascending= not arah[nama],
        nama_metrik = nama,
    )

    print(f"  Friedman: stat={friedman_dict['Statistic']}, "
          f"p={friedman_dict['PValue']}, "
          f"signifikan={friedman_dict['Significant_0.05']}")
    print(f"  Average rank:\n{avg_rank_df[['Method','AverageRank']].to_string(index=False)}")

    if friedman_dict['Significant_0.05'] == 'Tidak':
        print("  → Tidak signifikan: tidak ada perbedaan bermakna antar metode fuzzy.")
        print("    Nemenyi tidak dilakukan.")
    else:
        print(f"  Nemenyi:\n{nemenyi_df.to_string(index=False)}")

    all_friedman_xb.append(pd.DataFrame([friedman_dict]))
    all_avg_rank_xb.append(avg_rank_df)
    all_nemenyi_xb.append(nemenyi_df)

    nemenyi_df.to_csv(
        os.path.join(output_dir, f"nemenyi_{nama}.csv"), index=False
    )

df_friedman_xb = pd.concat(all_friedman_xb, ignore_index=True)
df_avg_rank_xb = pd.concat(all_avg_rank_xb, ignore_index=True)
df_nemenyi_xb  = pd.concat(all_nemenyi_xb,  ignore_index=True)

# SIMPAN SEMUA OUTPUT KE EXCEL
excel_path = os.path.join(output_dir, "HASIL_UJI_STATISTIKA.xlsx")

with pd.ExcelWriter(excel_path) as writer:

    # Bagian 1: Utama
    df_friedman_utama.to_excel(
        writer, sheet_name="Friedman_Utama", index=False
    )
    df_avg_rank_utama.to_excel(
        writer, sheet_name="AverageRank_Utama", index=False
    )
    df_nemenyi_utama.to_excel(
        writer, sheet_name="Nemenyi_Utama", index=False
    )
    summary_utama.to_excel(
        writer, sheet_name="FinalRank", index=False
    )

    # Bagian 2: Tambahan XB
    df_friedman_xb.to_excel(
        writer, sheet_name="Friedman_XieBeni_Fuzzy", index=False
    )
    df_avg_rank_xb.to_excel(
        writer, sheet_name="AverageRank_XieBeni", index=False
    )
    df_nemenyi_xb.to_excel(
        writer, sheet_name="Nemenyi_XieBeni", index=False
    )

print("\n" + "=" * 60)
print("Semua hasil disimpan di:")
print(excel_path)
print("=" * 60)
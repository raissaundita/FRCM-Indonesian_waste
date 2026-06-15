import pandas as pd
import numpy as np
from scipy.stats import shapiro      # Shapiro-Wilk untuk uji normalitas
from sklearn.preprocessing import MinMaxScaler
from pyampute.exploration.mcar_statistical_tests import MCARTest

FILE_PATH = "D:/Raissa/Python/Dataset/data_sampah_raw.xlsx"
SHEET_NAME = "DATASET"
CUTOFF_YEAR = 2022

NEW_PROV = {"Papua Selatan", "Papua Tengah", "Papua Pegunungan", "Papua Barat Daya"}
MAP_PRE_2022 = {
    "Papua Selatan": "Papua",
    "Papua Tengah": "Papua",
    "Papua Pegunungan": "Papua",
    "Papua Barat Daya": "Papua Barat",
}

ID_COLS    = ["Tahun", "Provinsi", "Kabupaten/Kota"]
DROP_COLS  = ["Timbulan Sampah Tahunan(ton)"]

SENTINEL_VALUE = 99999.99
ROW_MISSING_THRESHOLD = 0.50  # baris dengan >50% kolom kosong di-drop

# FUNGSI BANTU:  berdasarkan uji normalitas Shapiro-Wilk (mean atau median)
def tentukan_nilai_isi(x, alpha=0.05):
    """
    Alur keputusan:
    len(x) == 0 : tidak ada data → tidak bisa diisi
    len(x) == 1 : hanya 1 data → langsung pakai nilai itu
                  (tidak perlu uji, karena mean = median = nilai itu sendiri)
    len(x) >= 2 : lakukan uji Shapiro-Wilk, toleransi kesalahan 5%
                  p >= 0.05 → normal → pakai mean
                  p <  0.05 → tidak normal → pakai median
    """
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return np.nan, "tidak bisa diisi (tidak ada data referensi)", "tidak diuji"

    elif len(x) == 1:
        fill   = float(x[0])
        method = "nilai tunggal (hanya 1 data tersedia, mean = median)"
        dist   = "tidak diuji"

    else:
        _, p = shapiro(x)
        dist = f"p={p:.4f}"

        if p >= alpha:
            fill   = float(np.mean(x))
            method = "mean (distribusi normal, Shapiro-Wilk)"
        else:
            fill   = float(np.median(x))
            method = "median (distribusi tidak normal, Shapiro-Wilk)"

    return fill, method, dist

# ============================================================
# TAHAP 1: LOAD DATA
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()
df["Provinsi"] = df["Provinsi"].astype(str).str.strip()

print("=" * 60)
print("TAHAP 1 — DATA UNDERSTANDING (SEBELUM PREPROCESSING)")
print("=" * 60)
print("Jumlah baris :", df.shape[0])
print("Jumlah kolom :", df.shape[1])

for c in ID_COLS:
    if c not in df.columns:
        raise ValueError(f"Kolom '{c}' tidak ditemukan. Cek nama kolom di Excel.")
print("\nContoh 5 baris pertama:")
print(df.head())
print("\nInfo struktur & tipe data:")
df.info()

dup = int(df.duplicated().sum())
print("\nJumlah baris duplikat:", dup)

mv     = df.isna().sum()
mv_pct = (mv / len(df) * 100).round(2)
missing_summary = pd.DataFrame(
    {"missing_count": mv, "missing_pct": mv_pct}
).sort_values("missing_count", ascending=False)
print("\nMissing value per kolom:")
print(missing_summary[missing_summary["missing_count"] > 0])

num_cols_all = df.select_dtypes(include="number").columns
scale_check  = df[num_cols_all].agg(["min", "max"]).T.sort_values("max", ascending=False)
print("\nSkala numerik (min/max) - sebelum normalisasi:")
print(scale_check)

# ============================================================
# TAHAP 2A: PENGECEKAN FORMAT & KELENGKAPAN DATA
print("\n" + "=" * 60)
print("TAHAP 2A — PENGECEKAN FORMAT & KELENGKAPAN DATA")
print("=" * 60)

percent_cols = [c for c in df.columns if "%" in c]
row_pct_sum  = df[percent_cols].sum(axis=1, skipna=True)

n_valid   = ((row_pct_sum >= 95) & (row_pct_sum <= 105)).sum()
n_zero    = (row_pct_sum == 0).sum()
n_partial = ((row_pct_sum > 0) & (row_pct_sum < 95)).sum()

print(f"Rata-rata total persen : {row_pct_sum.mean():.2f}")
print(f"Minimum total persen   : {row_pct_sum.min():.2f}")
print(f"Maksimum total persen  : {row_pct_sum.max():.2f}")
print(f"Standar deviasi        : {row_pct_sum.std():.2f}")
print(f"\nBaris total persen 95-105% (valid)  : {n_valid}")
print(f"Baris total persen = 0 (semua NaN) : {n_zero}")
print(f"Baris total persen > 0 tapi < 95%  : {n_partial}")
print("\nCATATAN: Baris total persen = 0 adalah baris yang seluruh kolom")
print("komposisinya tidak dilaporkan ke SIPSN — bukan typo atau kesalahan data.")

extreme_rows = df.loc[
    row_pct_sum == row_pct_sum.max(),
    ["Tahun", "Provinsi", "Kabupaten/Kota"] + percent_cols,
]
print(f"\nBaris dengan total persen maksimum ({row_pct_sum.max()}):")
print(extreme_rows)

if 95 <= row_pct_sum.mean() <= 105:
    print("\nKESIMPULAN: Komposisi persen secara umum mendekati 100%.")
else:
    print("\nCATATAN: Rata-rata < 95% karena banyak baris yang seluruh")
    print("kolom komposisinya kosong (tidak dilaporkan di SIPSN).")

# ============================================================
# TAHAP 2B: PEMBERSIHAN DATA
print("\n" + "=" * 60)
print("TAHAP 2B — PEMBERSIHAN DATA")
print("=" * 60)

anom = df[(df["Tahun"] < CUTOFF_YEAR) & (df["Provinsi"].isin(NEW_PROV))]
print(f"Anomali provinsi baru sebelum {CUTOFF_YEAR}: {len(anom)} baris")
df.loc[df["Tahun"] < CUTOFF_YEAR, "Provinsi"] = (
    df.loc[df["Tahun"] < CUTOFF_YEAR, "Provinsi"].replace(MAP_PRE_2022)
)
print("Relabeling provinsi baru selesai.")

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f"Kolom redundan di-drop: {DROP_COLS}")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols.remove("Tahun")

total_sentinel = 0
for col in numeric_cols:
    n = (df[col] == SENTINEL_VALUE).sum()
    if n > 0:
        df[col] = df[col].replace(SENTINEL_VALUE, np.nan)
        print(f"Sentinel {SENTINEL_VALUE} diganti NaN di '{col}': {n} nilai")
        total_sentinel += n
print(f"Total sentinel value diganti: {total_sentinel}")

expected_pairs = set(zip(df["Tahun"], df["Provinsi"]))
print(f"\nTotal pasangan (Tahun, Provinsi) yang harus ada: {len(expected_pairs)}")
print("Rincian per tahun:")
for tahun in sorted(df["Tahun"].unique()):
    n = sum(1 for t, _ in expected_pairs if t == tahun)
    print(f"  {tahun}: {n} provinsi")

is_placeholder    = df["Kabupaten/Kota"].isna()
missing_ratio_row = df[numeric_cols].isna().mean(axis=1)
is_too_empty      = (missing_ratio_row > ROW_MISSING_THRESHOLD) & (~is_placeholder)

n_before      = len(df)
n_placeholder = is_placeholder.sum()
n_drop        = is_too_empty.sum()
df = df[~is_too_empty].copy()

print(f"\nDrop baris kab/kota dengan >{ROW_MISSING_THRESHOLD*100:.0f}% kolom missing:")
print(f"  Sebelum     : {n_before} baris")
print(f"  Placeholder : {n_placeholder} baris (DIPROTEKSI)")
print(f"  Di-drop     : {n_drop} baris")
print(f"  Sesudah     : {len(df)} baris")

existing_pairs = set(zip(df["Tahun"], df["Provinsi"]))
missing_pairs  = expected_pairs - existing_pairs

if missing_pairs:
    restore_rows = []
    for (tahun, prov) in sorted(missing_pairs):
        row = {"Tahun": tahun, "Provinsi": prov, "Kabupaten/Kota": np.nan}
        for col in numeric_cols:
            row[col] = np.nan
        restore_rows.append(row)
    df_restore = pd.DataFrame(restore_rows)
    df = pd.concat([df, df_restore], ignore_index=True)
    print(f"\nRestore {len(restore_rows)} provinsi yang hilang akibat drop kab/kota:")
    for (tahun, prov) in sorted(missing_pairs):
        print(f"    {tahun} | {prov}")
else:
    print("\nTidak ada provinsi yang perlu di-restore.")


# ============================================================
# TAHAP 2C: UJI MCAR
print("\n" + "=" * 60)
print("TAHAP 2C — UJI MCAR")
print("=" * 60)

mcar_test = MCARTest(method="little")
p_value_mcar = mcar_test.little_mcar_test(df[numeric_cols])

print(f"P-value Little MCAR Test : {p_value_mcar:.5f}")

if p_value_mcar > 0.05:
    print("KESIMPULAN: Missing value bersifat MCAR")
else:
    print("KESIMPULAN: Missing value tidak MCAR")


# ============================================================
# TAHAP 2D: UJI NORMALITAS SHAPIRO-WILK & IMPUTASI (3 TINGKAT)
# Strategi imputasi bertahap dari paling spesifik ke paling umum:
#   TINGKAT 1 — Provinsi + Tahun yang sama
#   TINGKAT 2 — Provinsi yang sama, semua tahun (jika Tingkat 1 gagal)
#     Contoh: DKI Jakarta di kolom sumber sampah selalu kosong di
#     semua kab/kota tahun 2019, tapi ada datanya di tahun 2021 jadi
#     diisi dari data DKI Jakarta lintas tahun. 
#   TINGKAT 3 — Seluruh data nasional
print("\n" + "=" * 60)
print("TAHAP 2D — UJI NORMALITAS SHAPIRO-WILK &")
print("           IMPUTASI MISSING VALUE (3 TINGKAT)")
print("=" * 60)

alpha = 0.05
df_imp = df.copy()
numeric_cols = df_imp.select_dtypes(include="number").columns.tolist()
numeric_cols.remove("Tahun")

print(f"Uji normalitas : Shapiro-Wilk (valid untuk n = 2 hingga 50)")
print(f"Alpha          : {alpha} (tingkat signifikansi 5%)")
print(f"Jumlah fitur numerik: {len(numeric_cols)}")

# Hitung fallback nasional (Tingkat 3) sekali di awal
print("\nMenghitung nilai fallback nasional (Tingkat 3)...")
fallback_nasional = {}
for col in numeric_cols:
    fill, method, dist = tentukan_nilai_isi(df_imp[col].values, alpha=alpha)
    fallback_nasional[col] = {"fill": fill, "method": method, "dist": dist}

print("\nProses imputasi:\n")
imputation_log = []

for col in numeric_cols:
    idx_all_missing = df_imp.index[df_imp[col].isna()]
    if len(idx_all_missing) == 0:
        continue

    print(f"Kolom: '{col}' | total missing: {len(idx_all_missing)}")

    for idx in idx_all_missing:
        tahun    = df_imp.at[idx, "Tahun"]
        provinsi = df_imp.at[idx, "Provinsi"]

        # TINGKAT 1: Provinsi + Tahun yang sama
        mask_t1 = (
            (df_imp["Tahun"]    == tahun)    &
            (df_imp["Provinsi"] == provinsi) &
            (df_imp.index       != idx)
        )
        x_t1 = df_imp.loc[mask_t1, col].dropna().values

        if len(x_t1) > 0:
            fill, method, dist = tentukan_nilai_isi(x_t1, alpha=alpha)
            tingkat = 1
            sumber  = f"provinsi {provinsi}, tahun {tahun}"

        else:
            # TINGKAT 2: Provinsi yang sama, semua tahun
            mask_t2 = (
                (df_imp["Provinsi"] == provinsi) &
                (df_imp.index       != idx)
            )
            x_t2 = df_imp.loc[mask_t2, col].dropna().values

            if len(x_t2) > 0:
                fill, method, dist = tentukan_nilai_isi(x_t2, alpha=alpha)
                tingkat = 2
                sumber  = f"provinsi {provinsi}, semua tahun"

            else:
                # TINGKAT 3: Seluruh data nasional
                fill    = fallback_nasional[col]["fill"]
                method  = fallback_nasional[col]["method"]
                dist    = fallback_nasional[col]["dist"]
                tingkat = 3
                sumber  = "nasional (semua provinsi & tahun)"

        df_imp.at[idx, col] = fill

        imputation_log.append({
            "Tahun"         : tahun,
            "Provinsi"      : provinsi,
            "Kolom"         : col,
            "Tingkat"       : tingkat,
            "Sumber Data"   : sumber,
            "Uji Normalitas": dist,
            "Metode"        : method,
            "Nilai Isi"     : round(fill, 4) if not np.isnan(fill) else "NaN",
        })

        # print(f"  [{tahun} | {provinsi}] Tingkat {tingkat} ({sumber}) | {method} | isi={fill:.4f}")

sisa_missing = int(df_imp[numeric_cols].isna().sum().sum())
print(f"\nTotal missing setelah imputasi: {sisa_missing}")
if sisa_missing == 0:
    print("Tidak ada missing value tersisa.")
else:
    print("Masih ada missing — cek log_imputasi.csv untuk detailnya.")

# encoding="utf-8-sig" agar tanda seperti p=0.xxxx terbaca normal di Excel
pd.DataFrame(imputation_log).to_csv("log_imputasi.csv", index=False, encoding="utf-8-sig")
print("Log imputasi disimpan ke: log_imputasi.csv")

log_df = pd.DataFrame(imputation_log)
if len(log_df) > 0:
    print("\nRingkasan jumlah pengisian per tingkat imputasi:")
    ringkasan = log_df["Tingkat"].value_counts().sort_index()
    label_tingkat = {
        1: "Tingkat 1 — provinsi + tahun sama",
        2: "Tingkat 2 — provinsi sama, semua tahun",
        3: "Tingkat 3 — nasional (fallback terakhir)",
    }
    for t, jumlah in ringkasan.items():
        print(f"  {label_tingkat.get(t, t)}: {jumlah} pengisian")

# ============================================================
# TAHAP 2E: AGREGASI KAB/KOTA → PROVINSI (Median)
print("\n" + "=" * 60)
print("TAHAP 2E — AGREGASI KAB/KOTA → PROVINSI (Median)")
print("=" * 60)

df_prov = df_imp.groupby(["Tahun", "Provinsi"], as_index=False)[numeric_cols].median()

print(f"Shape setelah agregasi: {df_prov.shape}")
print("\nJumlah provinsi per tahun:")
for tahun in sorted(df_prov["Tahun"].unique()):
    n = len(df_prov[df_prov["Tahun"] == tahun])
    print(f"  {tahun}: {n} provinsi")

total_expected = len(expected_pairs)
total_actual   = len(df_prov)
status = "yes lengkap" if total_actual == total_expected else "oh no"
print(f"\nTotal baris: {total_actual} / {total_expected} {status}")

n_kabkota = df_imp.groupby(["Tahun", "Provinsi"]).size().reset_index(name="n_kabkota")
print("\nStatistik jumlah kab/kota per provinsi (setelah cleaning):")
print(n_kabkota["n_kabkota"].describe())
print("\nCATATAN: Median digunakan sebagai nilai representatif provinsi")
print("(robust terhadap outlier antar kab/kota dalam satu provinsi).")

# ============================================================
# TAHAP 2F: NORMALISASI DATA (Min-Max)
print("\n" + "=" * 60)
print("TAHAP 2F — NORMALISASI DATA (Min-Max Scaling)")
print("=" * 60)

scaler    = MinMaxScaler()
df_scaled = df_prov.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_prov[numeric_cols])

print("Rentang setelah normalisasi (semua fitur harus [0, 1]):")
print(df_scaled[numeric_cols].agg(["min", "max"]))

# SIMPAN OUTPUT
print("\n" + "=" * 60)
print("MENYIMPAN OUTPUT")
print("=" * 60)

# df.to_csv("02_after_admin_cleaned.csv", index=False)
# df_imp.to_csv("03_imputed_kabkota.csv", index=False)
# df_prov.to_csv("04_aggregated_provinsi.csv", index=False)
# df_scaled.to_csv("05_scaled_provinsi.csv", index=False)
# n_kabkota.to_csv("06_info_n_kabkota.csv", index=False)

print("02_after_admin_cleaned.csv  — setelah cleaning & standarisasi admin")
print("03_imputed_kabkota.csv      — setelah imputasi (level kab/kota)")
print("04_aggregated_provinsi.csv  — setelah agregasi median ke provinsi")
print("05_scaled_provinsi.csv      — setelah normalisasi Min-Max (INPUT CLUSTERING)")
print("06_info_n_kabkota.csv       — info jumlah kab/kota per provinsi-tahun")

print(f"\nPreprocessing selesai.")
print(f"   Input clustering: {df_scaled.shape[0]} baris x {len(numeric_cols)} fitur numerik")
print(f"   ({df_scaled['Tahun'].nunique()} tahun x rata-rata {df_scaled.shape[0]//df_scaled['Tahun'].nunique()} provinsi/tahun)")
import pandas as pd
import numpy as np
from scipy.stats import shapiro      # Shapiro-Wilk for normality testing
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
ROW_MISSING_THRESHOLD = 0.50  # rows with >50% missing columns are dropped

# HELPER FUNCTION: determine fill value based on Shapiro-Wilk normality test (mean or median)
def tentukan_nilai_isi(x, alpha=0.05):
    """
    Decision flow:
    len(x) == 0 : no data available → cannot be filled
    len(x) == 1 : only 1 data point → use that value directly
                  (no test needed, since mean = median = that single value)
    len(x) >= 2 : perform Shapiro-Wilk test with 5% significance level
                  p >= 0.05 → normal distribution → use mean
                  p <  0.05 → non-normal distribution → use median
    """
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return np.nan, "cannot be filled (no reference data available)", "not tested"

    elif len(x) == 1:
        fill   = float(x[0])
        method = "single value (only 1 data point available, mean = median)"
        dist   = "not tested"

    else:
        _, p = shapiro(x)
        dist = f"p={p:.4f}"

        if p >= alpha:
            fill   = float(np.mean(x))
            method = "mean (normal distribution, Shapiro-Wilk)"
        else:
            fill   = float(np.median(x))
            method = "median (non-normal distribution, Shapiro-Wilk)"

    return fill, method, dist

# ============================================================
# STAGE 1: LOAD DATA
df = pd.read_excel(FILE_PATH, sheet_name=SHEET_NAME)
df.columns = df.columns.str.strip()
df["Provinsi"] = df["Provinsi"].astype(str).str.strip()

print("=" * 60)
print("STAGE 1 — DATA UNDERSTANDING (BEFORE PREPROCESSING)")
print("=" * 60)
print("Number of rows :", df.shape[0])
print("Number of columns :", df.shape[1])

for c in ID_COLS:
    if c not in df.columns:
        raise ValueError(f"Column '{c}' not found. Check column names in Excel.")
print("\nFirst 5 rows:")
print(df.head())
print("\nStructure & data types:")
df.info()

dup = int(df.duplicated().sum())
print("\nNumber of duplicate rows:", dup)

mv     = df.isna().sum()
mv_pct = (mv / len(df) * 100).round(2)
missing_summary = pd.DataFrame(
    {"missing_count": mv, "missing_pct": mv_pct}
).sort_values("missing_count", ascending=False)
print("\nMissing values per column:")
print(missing_summary[missing_summary["missing_count"] > 0])

num_cols_all = df.select_dtypes(include="number").columns
scale_check  = df[num_cols_all].agg(["min", "max"]).T.sort_values("max", ascending=False)
print("\nNumerical scale (min/max) - before normalization:")
print(scale_check)

# ============================================================
# STAGE 2A: FORMAT & COMPLETENESS CHECK
print("\n" + "=" * 60)
print("STAGE 2A — FORMAT & COMPLETENESS CHECK")
print("=" * 60)

percent_cols = [c for c in df.columns if "%" in c]
row_pct_sum  = df[percent_cols].sum(axis=1, skipna=True)

n_valid   = ((row_pct_sum >= 95) & (row_pct_sum <= 105)).sum()
n_zero    = (row_pct_sum == 0).sum()
n_partial = ((row_pct_sum > 0) & (row_pct_sum < 95)).sum()

print(f"Average total percent : {row_pct_sum.mean():.2f}")
print(f"Minimum total percent  : {row_pct_sum.min():.2f}")
print(f"Maximum total percent  : {row_pct_sum.max():.2f}")
print(f"Standard deviation     : {row_pct_sum.std():.2f}")
print(f"\nRows with total percent 95-105% (valid)   : {n_valid}")
print(f"Rows with total percent = 0 (all NaN)     : {n_zero}")
print(f"Rows with total percent > 0 but < 95%     : {n_partial}")
print("\nNOTE: Rows with total percent = 0 are rows where all composition")
print("columns were not reported to SIPSN — not a typo or data error.")

extreme_rows = df.loc[
    row_pct_sum == row_pct_sum.max(),
    ["Tahun", "Provinsi", "Kabupaten/Kota"] + percent_cols,
]
print(f"\nRows with maximum total percent ({row_pct_sum.max()}):")
print(extreme_rows)

if 95 <= row_pct_sum.mean() <= 105:
    print("\nCONCLUSION: Percentage composition is generally close to 100%.")
else:
    print("\nNOTE: Average < 95% because many rows have all composition")
    print("columns empty (not reported in SIPSN).")

# ============================================================
# STAGE 2B: DATA CLEANING
print("\n" + "=" * 60)
print("STAGE 2B — DATA CLEANING")
print("=" * 60)

anom = df[(df["Tahun"] < CUTOFF_YEAR) & (df["Provinsi"].isin(NEW_PROV))]
print(f"Anomalous new provinces before {CUTOFF_YEAR}: {len(anom)} rows")
df.loc[df["Tahun"] < CUTOFF_YEAR, "Provinsi"] = (
    df.loc[df["Tahun"] < CUTOFF_YEAR, "Provinsi"].replace(MAP_PRE_2022)
)
print("Province relabeling complete.")

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
print(f"Redundant columns dropped: {DROP_COLS}")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols.remove("Tahun")

total_sentinel = 0
for col in numeric_cols:
    n = (df[col] == SENTINEL_VALUE).sum()
    if n > 0:
        df[col] = df[col].replace(SENTINEL_VALUE, np.nan)
        print(f"Sentinel {SENTINEL_VALUE} replaced with NaN in '{col}': {n} values")
        total_sentinel += n
print(f"Total sentinel values replaced: {total_sentinel}")

expected_pairs = set(zip(df["Tahun"], df["Provinsi"]))
print(f"\nTotal (Year, Province) pairs expected: {len(expected_pairs)}")
print("Breakdown per year:")
for tahun in sorted(df["Tahun"].unique()):
    n = sum(1 for t, _ in expected_pairs if t == tahun)
    print(f"  {tahun}: {n} provinces")

is_placeholder    = df["Kabupaten/Kota"].isna()
missing_ratio_row = df[numeric_cols].isna().mean(axis=1)
is_too_empty      = (missing_ratio_row > ROW_MISSING_THRESHOLD) & (~is_placeholder)

n_before      = len(df)
n_placeholder = is_placeholder.sum()
n_drop        = is_too_empty.sum()
df = df[~is_too_empty].copy()

print(f"\nDrop district/city rows with >{ROW_MISSING_THRESHOLD*100:.0f}% missing columns:")
print(f"  Before      : {n_before} rows")
print(f"  Placeholder : {n_placeholder} rows (PROTECTED)")
print(f"  Dropped     : {n_drop} rows")
print(f"  After       : {len(df)} rows")

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
    print(f"\nRestored {len(restore_rows)} provinces lost due to row dropping:")
    for (tahun, prov) in sorted(missing_pairs):
        print(f"    {tahun} | {prov}")
else:
    print("\nNo provinces need to be restored.")


# ============================================================
# STAGE 2C: MCAR TEST
print("\n" + "=" * 60)
print("STAGE 2C — MCAR TEST")
print("=" * 60)

mcar_test = MCARTest(method="little")
p_value_mcar = mcar_test.little_mcar_test(df[numeric_cols])

print(f"P-value Little MCAR Test : {p_value_mcar:.5f}")

if p_value_mcar > 0.05:
    print("CONCLUSION: Missing values are MCAR")
else:
    print("CONCLUSION: Missing values are NOT MCAR")


# ============================================================
# STAGE 2D: SHAPIRO-WILK NORMALITY TEST & IMPUTATION (3 LEVELS)
# Tiered imputation strategy from most specific to most general:
#   LEVEL 1 — Same Province + Same Year
#   LEVEL 2 — Same Province, all years (if Level 1 fails)
#     Example: DKI Jakarta's waste source columns are always empty across
#     all districts in 2019, but data exists in 2021, so it is filled
#     using DKI Jakarta data across all years.
#   LEVEL 3 — All national data
print("\n" + "=" * 60)
print("STAGE 2D — SHAPIRO-WILK NORMALITY TEST &")
print("           MISSING VALUE IMPUTATION (3 LEVELS)")
print("=" * 60)

alpha = 0.05
df_imp = df.copy()
numeric_cols = df_imp.select_dtypes(include="number").columns.tolist()
numeric_cols.remove("Tahun")

print(f"Normality test : Shapiro-Wilk (valid for n = 2 to 50)")
print(f"Alpha          : {alpha} (5% significance level)")
print(f"Number of numeric features: {len(numeric_cols)}")

# Compute national fallback values (Level 3) once upfront
print("\nComputing national fallback values (Level 3)...")
fallback_nasional = {}
for col in numeric_cols:
    fill, method, dist = tentukan_nilai_isi(df_imp[col].values, alpha=alpha)
    fallback_nasional[col] = {"fill": fill, "method": method, "dist": dist}

print("\nImputation process:\n")
imputation_log = []

for col in numeric_cols:
    idx_all_missing = df_imp.index[df_imp[col].isna()]
    if len(idx_all_missing) == 0:
        continue

    print(f"Column: '{col}' | total missing: {len(idx_all_missing)}")

    for idx in idx_all_missing:
        tahun    = df_imp.at[idx, "Tahun"]
        provinsi = df_imp.at[idx, "Provinsi"]

        # LEVEL 1: Same Province + Same Year
        mask_t1 = (
            (df_imp["Tahun"]    == tahun)    &
            (df_imp["Provinsi"] == provinsi) &
            (df_imp.index       != idx)
        )
        x_t1 = df_imp.loc[mask_t1, col].dropna().values

        if len(x_t1) > 0:
            fill, method, dist = tentukan_nilai_isi(x_t1, alpha=alpha)
            tingkat = 1
            sumber  = f"province {provinsi}, year {tahun}"

        else:
            # LEVEL 2: Same Province, all years
            mask_t2 = (
                (df_imp["Provinsi"] == provinsi) &
                (df_imp.index       != idx)
            )
            x_t2 = df_imp.loc[mask_t2, col].dropna().values

            if len(x_t2) > 0:
                fill, method, dist = tentukan_nilai_isi(x_t2, alpha=alpha)
                tingkat = 2
                sumber  = f"province {provinsi}, all years"

            else:
                # LEVEL 3: All national data
                fill    = fallback_nasional[col]["fill"]
                method  = fallback_nasional[col]["method"]
                dist    = fallback_nasional[col]["dist"]
                tingkat = 3
                sumber  = "national (all provinces & years)"

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

        # print(f"  [{tahun} | {provinsi}] Level {tingkat} ({sumber}) | {method} | fill={fill:.4f}")

sisa_missing = int(df_imp[numeric_cols].isna().sum().sum())
print(f"\nTotal missing after imputation: {sisa_missing}")
if sisa_missing == 0:
    print("No missing values remaining.")
else:
    print("Missing values still exist — check log_imputasi.csv for details.")

# encoding="utf-8-sig" so that characters like p=0.xxxx render correctly in Excel
pd.DataFrame(imputation_log).to_csv("log_imputasi.csv", index=False, encoding="utf-8-sig")
print("Imputation log saved to: log_imputasi.csv")

log_df = pd.DataFrame(imputation_log)
if len(log_df) > 0:
    print("\nSummary of fill counts per imputation level:")
    ringkasan = log_df["Tingkat"].value_counts().sort_index()
    label_tingkat = {
        1: "Level 1 — same province + same year",
        2: "Level 2 — same province, all years",
        3: "Level 3 — national fallback (last resort)",
    }
    for t, jumlah in ringkasan.items():
        print(f"  {label_tingkat.get(t, t)}: {jumlah} fills")

# ============================================================
# STAGE 2E: AGGREGATION FROM DISTRICT/CITY TO PROVINCE (Median)
print("\n" + "=" * 60)
print("STAGE 2E — AGGREGATION FROM DISTRICT/CITY TO PROVINCE (Median)")
print("=" * 60)

df_prov = df_imp.groupby(["Tahun", "Provinsi"], as_index=False)[numeric_cols].median()

print(f"Shape after aggregation: {df_prov.shape}")
print("\nNumber of provinces per year:")
for tahun in sorted(df_prov["Tahun"].unique()):
    n = len(df_prov[df_prov["Tahun"] == tahun])
    print(f"  {tahun}: {n} provinces")

total_expected = len(expected_pairs)
total_actual   = len(df_prov)
status = "yes complete" if total_actual == total_expected else "oh no"
print(f"\nTotal rows: {total_actual} / {total_expected} {status}")

n_kabkota = df_imp.groupby(["Tahun", "Provinsi"]).size().reset_index(name="n_kabkota")
print("\nStatistics of district/city count per province (after cleaning):")
print(n_kabkota["n_kabkota"].describe())
print("\nNOTE: Median is used as the representative value for each province")
print("(robust against outliers across districts/cities within a province).")

# ============================================================
# STAGE 2F: DATA NORMALIZATION (Min-Max)
print("\n" + "=" * 60)
print("STAGE 2F — DATA NORMALIZATION (Min-Max Scaling)")
print("=" * 60)

scaler    = MinMaxScaler()
df_scaled = df_prov.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df_prov[numeric_cols])

print("Range after normalization (all features must be in [0, 1]):")
print(df_scaled[numeric_cols].agg(["min", "max"]))

# SAVE OUTPUT
print("\n" + "=" * 60)
print("SAVING OUTPUT")
print("=" * 60)

# df.to_csv("02_after_admin_cleaned.csv", index=False)
# df_imp.to_csv("03_imputed_kabkota.csv", index=False)
# df_prov.to_csv("04_aggregated_provinsi.csv", index=False)
# df_scaled.to_csv("05_scaled_provinsi.csv", index=False)
# n_kabkota.to_csv("06_info_n_kabkota.csv", index=False)

print("02_after_admin_cleaned.csv  — after cleaning & administrative standardization")
print("03_imputed_kabkota.csv      — after imputation (district/city level)")
print("04_aggregated_provinsi.csv  — after median aggregation to province level")
print("05_scaled_provinsi.csv      — after Min-Max normalization (CLUSTERING INPUT)")
print("06_info_n_kabkota.csv       — district/city count info per province-year")

print(f"\nPreprocessing complete.")
print(f"   Clustering input: {df_scaled.shape[0]} rows x {len(numeric_cols)} numeric features")
print(f"   ({df_scaled['Tahun'].nunique()} years x avg {df_scaled.shape[0]//df_scaled['Tahun'].nunique()} provinces/year)")

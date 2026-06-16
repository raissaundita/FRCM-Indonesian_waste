import pandas as pd
import os

input_file = "D:/Raissa/Python/VISUALISASI/PETA/frcm_run6_untuk_peta.csv"
output_dir = "D:/Raissa/Python/VISUALISASI/PETA/per_tahun"

os.makedirs(output_dir, exist_ok=True)

# BACA FILE
df = pd.read_csv(input_file)

print("Total baris:", len(df))
print("Tahun yang tersedia:", sorted(df["Tahun"].unique().tolist()))
print("Kolom:", df.columns.tolist())

# SIMPAN SATU FILE PER TAHUN
for tahun in sorted(df["Tahun"].unique()):
    df_tahun = df[df["Tahun"] == tahun].copy()

    # Bulatkan membership_dominan biar lebih rapi di peta
    df_tahun["membership_dominan"] = df_tahun["membership_dominan"].round(2)

    # Tambah kolom label teks untuk tooltip di peta
    # Format: "Klaster 1 | Pasti (Lower)" atau "Klaster 2 | Boundary (0.54)"
    df_tahun["info_peta"] = (
        df_tahun["nama_klaster"] + " | " + df_tahun["kepastian"]
    )

    # Simpan
    out_path = os.path.join(output_dir, f"klaster_{tahun}.csv")
    df_tahun.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"Tahun {tahun}: {len(df_tahun)} provinsi → disimpan ke {out_path}")

print("\nSelesai! Semua file tersimpan di:")
print(output_dir)
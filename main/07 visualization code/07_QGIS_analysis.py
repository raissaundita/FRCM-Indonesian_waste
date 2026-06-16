import pandas as pd

# Baca file gabungan semua tahun
df = pd.read_csv("D:/Raissa/Python/VISUALISASI/PETA/frcm_run6_untuk_peta.csv")
data_asli = pd.read_csv("D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv")

# Gabungkan label klaster dengan data asli
# (asumsikan urutan baris sama)
data_asli["nama_klaster"] = df["nama_klaster"].values
data_asli["label"] = df["label"].values

# Hitung rata-rata tiap fitur per klaster
fitur_cols = [col for col in data_asli.columns 
              if col not in ["Tahun", "Provinsi", "nama_klaster", "label"]]

print("=" * 60)
print("RATA-RATA TIAP FITUR PER KLASTER")
print("(data sudah ternormalisasi 0-1)")
print("=" * 60)

ringkasan = data_asli.groupby("nama_klaster")[fitur_cols].mean().round(3)
print(ringkasan.T.to_string())  # .T supaya fitur jadi baris, klaster jadi kolom

print("\n" + "=" * 60)
print("JUMLAH PROVINSI PER KLASTER PER TAHUN")
print("=" * 60)
pivot = df.groupby(["Tahun", "nama_klaster"])["Provinsi"].count().unstack(fill_value=0)
print(pivot)

print("\n" + "=" * 60)
print("DAFTAR PROVINSI PER KLASTER (TAHUN 2025)")
print("=" * 60)
df_2025 = df[df["Tahun"] == 2025]
for klaster in sorted(df_2025["nama_klaster"].unique()):
    provs = df_2025[df_2025["nama_klaster"] == klaster]["Provinsi"].tolist()
    print(f"\n{klaster} ({len(provs)} provinsi):")
    for p in provs:
        print(f"  - {p}")
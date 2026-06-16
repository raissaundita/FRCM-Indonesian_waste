import pandas as pd

df = pd.read_csv("D:/Raissa/Python/VISUALISASI/PETA/frcm_run6_untuk_peta.csv")

print("=" * 60)
print("KONSISTENSI KLASTER PER PROVINSI (2019-2025)")
print("=" * 60)

# Hitung berapa tahun tiap provinsi masuk tiap klaster
pivot = df.groupby(["Provinsi", "nama_klaster"]).size().unstack(fill_value=0)
pivot["Total_Tahun"] = pivot.sum(axis=1)
pivot["Klaster_Dominan"] = pivot.drop("Total_Tahun", axis=1).idxmax(axis=1)
pivot["Tahun_Dominan"] = pivot.drop(["Total_Tahun", "Klaster_Dominan"], axis=1).max(axis=1)
pivot["Konsisten"] = pivot["Tahun_Dominan"] == pivot["Total_Tahun"]

print(pivot[["Klaster_Dominan", "Tahun_Dominan", "Total_Tahun", "Konsisten"]])

print("\n" + "=" * 60)
print("PROVINSI YANG SELALU KONSISTEN (sama setiap tahun)")
print("=" * 60)
konsisten = pivot[pivot["Konsisten"] == True]
print(f"Jumlah: {len(konsisten)} provinsi")
print(konsisten[["Klaster_Dominan", "Total_Tahun"]])

print("\n" + "=" * 60)
print("PROVINSI YANG TIDAK KONSISTEN (pernah pindah klaster)")
print("=" * 60)
tidak_konsisten = pivot[pivot["Konsisten"] == False]
print(f"Jumlah: {len(tidak_konsisten)} provinsi")
print(tidak_konsisten)

# Simpan untuk peta kesimpulan
output = pivot.reset_index()[["Provinsi", "Klaster_Dominan", "Tahun_Dominan", 
                               "Total_Tahun", "Konsisten"]]
output["Status"] = output["Konsisten"].apply(
    lambda x: "Konsisten" if x else "Berpindah Klaster"
)
output.to_csv(
    "D:/Raissa/Python/VISUALISASI/PETA/per_tahun/klaster_konsistensi.csv", 
    index=False
)
print("\nFile konsistensi disimpan!")
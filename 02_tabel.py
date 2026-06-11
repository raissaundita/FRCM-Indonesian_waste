import pandas as pd

df_elbow = pd.read_csv("D:/Raissa/elbow_numeric_tanpsi_all_methods.csv")
df_sil = pd.read_csv("D:/Raissa/silhouette_all_methods.csv")
df_dbi = pd.read_csv("D:/Raissa/dbi_all_methods_debug.csv")

# Merge berdasarkan k
comparison = df_elbow.merge(df_sil, on="k")
comparison = comparison.merge(df_dbi, on="k")

# Ranking
comparison["Rank_Sil_KMeans"] = comparison["Sil_KMeans"].rank(ascending=False)
comparison["Rank_DBI_KMeans"] = comparison["DBI_KMeans"].rank(ascending=True)

print("\n=== TABEL PERBANDINGAN METODE ===")
print(comparison)

comparison.to_excel("Tabel_Perbandingan_K.xlsx", index=False)
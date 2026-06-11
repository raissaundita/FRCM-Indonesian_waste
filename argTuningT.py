import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"

# Load data
df = pd.read_csv(PATH)
feature_cols = [col for col in df.columns if col not in ["Tahun", "Provinsi"]]
X = df[feature_cols].to_numpy(dtype=float)

# Hitung selisih |d1 - d2| dengan 50 inisialisasi acak, c=4
rng = np.random.default_rng(42)
all_diff = []

for _ in range(50):
    idx       = rng.choice(len(X), size=4, replace=False)
    centroids = X[idx]
    dists     = cdist(X, centroids, metric='euclidean')
    sorted_d  = np.sort(dists, axis=1)
    diff      = sorted_d[:, 1] - sorted_d[:, 0]  # selisih |d2 - d1|
    all_diff.extend(diff.tolist())

all_diff = np.array(all_diff)

print("Median selisih jarak |d1 - d2| :", round(np.median(all_diff), 6))
print("Persentil 25%                   :", round(np.percentile(all_diff, 25), 6))
print("Persentil 75%                   :", round(np.percentile(all_diff, 75), 6))

print("\nProporsi data masuk boundary region per nilai T:")
for T in [0.05, 0.07, 0.08, 0.09, 0.10, 0.20, 0.50]:
    prop = (all_diff <= T).mean() * 100
    print(f"  T = {T} -> {prop:.2f}%")
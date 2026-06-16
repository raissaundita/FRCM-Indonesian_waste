import os
import time
import numpy as np
import pandas as pd

PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"
OUT_DIR = "D:/Raissa/Python/KLASTER DATA/KMEANS_K4"

K = 4
N_RUNS = 100
SEED_BASE = 42
EPSILON = 1e-5
MAX_ITER = 200

year_col = "Tahun"
id_col = "Provinsi"

os.makedirs(OUT_DIR, exist_ok=True)

def euclidean_distance_matrix(X, C):
    # Menghitung jarak Euclidean semua data ke semua centroid
    return np.sqrt(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))

def compute_kmeans_objective(X, C, labels):
    # Menghitung fungsi objektif K-Means
    # yaitu total jarak kuadrat data ke centroid klasternya
    obj = 0.0

    for k in range(len(C)):
        anggota = X[labels == k]

        if len(anggota) > 0:
            obj += np.sum((anggota - C[k]) ** 2)

    return obj

def kmeans_manual(X, K, epsilon, max_iter, seed=None):
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # Inisialisasi centroid awal dari data
    init_idx = rng.choice(n, size=K, replace=False)
    C = X[init_idx].copy()

    n_iter_used = 0

    for t in range(1, max_iter + 1):
        n_iter_used = t
        C_prev = C.copy()

        # Step 1: hitung jarak Euclidean
        dist = euclidean_distance_matrix(X, C)

        # Step 2: assign ke centroid terdekat
        labels = np.argmin(dist, axis=1)

        # Step 3: update centroid
        for k in range(K):
            anggota = X[labels == k]

            if len(anggota) > 0:
                C[k] = anggota.mean(axis=0)
            else:
                # Jika ada klaster kosong, ambil ulang centroid dari data acak
                C[k] = X[rng.integers(0, n)]

        # Step 4: cek konvergensi
        if np.linalg.norm(C - C_prev) <= epsilon:
            break

    # label final
    dist = euclidean_distance_matrix(X, C)
    labels = np.argmin(dist, axis=1)

    return labels, C, n_iter_used

# LOAD DATA
df = pd.read_csv(PATH)

# Fitur clustering: buang Tahun dan Provinsi
X = df.drop(columns=[year_col, id_col])
X = X.select_dtypes(include=[np.number]).to_numpy(dtype=float)
X = np.nan_to_num(X)

log_rows = []
grand_start = time.perf_counter()

for r in range(1, N_RUNS + 1):
    seed = SEED_BASE + r

    start = time.perf_counter()
    labels, C, n_iter_used = kmeans_manual(X, K, EPSILON, MAX_ITER, seed=seed)
    end = time.perf_counter()

    final_objective = compute_kmeans_objective(X, C, labels)

    total_time = end - start
    avg_time_per_iter = total_time / n_iter_used if n_iter_used > 0 else np.nan

    # Simpan file label
    out = df[[year_col, id_col]].copy()
    out["label"] = labels

    out_path = os.path.join(OUT_DIR, f"kmeans_k{K}_run{r}.csv")
    out.to_csv(out_path, index=False)
    print("saved:", out_path)

    label_aktif = sorted(np.unique(labels).tolist())
    counts = np.bincount(labels, minlength=K)

    log_rows.append({
        "run": r,
        "seed": seed,
        "n_iter_internal": n_iter_used,
        "total_time_sec": total_time,
        "avg_time_per_iter_sec": avg_time_per_iter,
        "final_objective": final_objective,
        "n_label_aktif": len(label_aktif),
        "label_aktif": ",".join(map(str, label_aktif)),
        "n_label_0": int(counts[0]),
        "n_label_1": int(counts[1]),
        "n_label_2": int(counts[2]),
        "n_label_3": int(counts[3]),
    })

grand_end = time.perf_counter()
grand_total = grand_end - grand_start

rekap_df = pd.DataFrame(log_rows)

total_row = pd.DataFrame([{
    "run": "TOTAL_RUNTIME",
    "seed": "",
    "n_iter_internal": "",
    "total_time_sec": grand_total,
    "avg_time_per_iter_sec": "",
    "final_objective": "",
    "n_label_aktif": "",
    "label_aktif": "",
    "n_label_0": "",
    "n_label_1": "",
    "n_label_2": "",
    "n_label_3": "",
}])

rekap_df = pd.concat([rekap_df, total_row], ignore_index=True)

rekap_path = os.path.join(OUT_DIR, f"kmeans_k{K}_rekap_100run.csv")
rekap_df.to_csv(rekap_path, index=False)
print("saved:", rekap_path)
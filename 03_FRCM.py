# STEP 1: Inisialisasi / gunakan centroid saat ini
# STEP 2: Hitung jarak tiap objek ke centroid
# STEP 3: Tentukan lower approximation dan boundary region
# STEP 4: Perbarui derajat keanggotaan
# STEP 5: Hitung fungsi objektif FRCM
# STEP 6: Perbarui pusat klaster
# STEP 7: Cek konvergensi

import os
import time
import numpy as np
import pandas as pd

PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"
OUT_DIR = "D:/Raissa/Python/KLASTER DATA/FRCM_K4"

c = 4
N_RUNS = 100
SEED_BASE = 42
m = 2.0
T = 0.09  # hasil tuning
epsilon = 1e-5
tau = 200

year_col = "Tahun"
id_col = "Provinsi"

# Jika cluster kosong, centroid di-reinitialize dari data
REINIT_EMPTY_CLUSTER = True

SAVE_LABELS = True
SAVE_MEMBERSHIP = True
SAVE_OBJECTIVE_HISTORY = True

os.makedirs(OUT_DIR, exist_ok=True)

# FUNGSI BANTU
def compute_distances(X, V):  # hitung jarak Euclide
    return np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)

def compute_objective(X, V, U, m):
    """
    Fungsi objektif FRCM:
    J(U, V) = sum_j sum_i (u_ij^m) * ||x_j - v_i||^2
    """
    dist_sq = np.sum((X[:, None, :] - V[None, :, :]) ** 2, axis=2)
    return np.sum((U ** m) * dist_sq)

def initialize_centroids(X, c, rng):
    """
    Inisialisasi centroid awal dari data secara acak tanpa pengulangan.
    """
    n = X.shape[0]
    idx = rng.choice(n, size=c, replace=False)
    return X[idx].copy()

def validate_input_data(df, feature_cols):  # Validasi data sebelum clustering
    if len(feature_cols) == 0:
        raise ValueError("Tidak ada kolom fitur numerik yang bisa digunakan.")

    if df[feature_cols].isnull().sum().sum() > 0:
        na_info = df[feature_cols].isnull().sum()
        raise ValueError(
            "Masih ada missing value pada fitur numerik.\n"
            f"Rincian:\n{na_info[na_info > 0]}"
        )

def assign_regions_and_membership(X, V, c, m, T):
    """
    STEP FRCM:
    1. Hitung jarak
    2. Tentukan lower approximation dan boundary region
    3. Hitung membership U
    """
    n = X.shape[0]

    dist = compute_distances(X, V) + 1e-12
    q = np.argmin(dist, axis=1)

    in_lower = np.zeros((n, c), dtype=bool)
    in_boundary = np.zeros((n, c), dtype=bool)
    U = np.zeros((n, c), dtype=float)

    # Tentukan lower approximation dan boundary region
    for j in range(n):
        kandidat_boundary = []

        for p in range(c):
            if p == q[j]:
                continue
            if abs(dist[j, p] - dist[j, q[j]]) <= T:
                kandidat_boundary.append(p)

        if len(kandidat_boundary) > 0:
            in_boundary[j, q[j]] = True
            for p in kandidat_boundary:
                in_boundary[j, p] = True
        else:
            in_lower[j, q[j]] = True

    # Update membership
    for j in range(n):
        if in_lower[j].any():
            i0 = np.where(in_lower[j])[0][0]
            U[j, i0] = 1.0
        else:
            B = np.where(in_boundary[j])[0]
            if len(B) > 0:
                for i in B:
                    denom = 0.0
                    for k in B:
                        denom += (dist[j, i] / dist[j, k]) ** (2.0 / (m - 1.0))
                    U[j, i] = 1.0 / denom

    return dist, in_lower, in_boundary, U

def update_centroids(X, U, V_old, rng, reinit_empty_cluster=True):
    c = U.shape[1]
    Um = U ** m
    cluster_weights = Um.sum(axis=0)

    V_new = V_old.copy()

    for i in range(c):
        if cluster_weights[i] <= 1e-14:
            if reinit_empty_cluster:
                rand_idx = rng.integers(0, X.shape[0])
                V_new[i] = X[rand_idx].copy()
            else:
                V_new[i] = V_old[i].copy()
        else:
            V_new[i] = (Um[:, i][:, None] * X).sum(axis=0) / cluster_weights[i]

    return V_new


# FUNGSI UTAMA FRCM
def frcm(X, c, m, T, epsilon, tau, seed=None, reinit_empty_cluster=True):
    rng = np.random.default_rng(seed)

    # STEP 1: Inisialisasi pusat klaster
    V = initialize_centroids(X, c, rng)

    n_iter_used = 0
    objective_history = []

    # default output
    in_lower = np.zeros((X.shape[0], c), dtype=bool)
    in_boundary = np.zeros((X.shape[0], c), dtype=bool)
    U = np.zeros((X.shape[0], c), dtype=float)

    for t in range(1, tau + 1):
        n_iter_used = t
        V_prev = V.copy()

        # STEP 2-4: hitung jarak, tentukan lower/boundary, update membership
        dist, in_lower, in_boundary, U = assign_regions_and_membership(
            X=X, V=V, c=c, m=m, T=T
        )

        # STEP 5: Hitung objective pada state yang konsisten (U, V)
        obj = compute_objective(X, V, U, m)
        objective_history.append(obj)

        # STEP 6: Update centroid
        V = update_centroids(
            X=X,
            U=U,
            V_old=V,
            rng=rng,
            reinit_empty_cluster=reinit_empty_cluster
        )

        # STEP 7: Cek konvergensi
        if np.linalg.norm(V - V_prev) <= epsilon:
            break

    # Hitung ulang state final supaya output akhir sinkron dengan centroid final
    dist, in_lower, in_boundary, U = assign_regions_and_membership(
        X=X, V=V, c=c, m=m, T=T
    )

    return U, V, n_iter_used, in_lower, in_boundary, objective_history


# LOAD DATA
df = pd.read_csv(PATH)
feature_cols = [col for col in df.columns if col not in [year_col, id_col]]

X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
validate_input_data(pd.concat([df[[year_col, id_col]], X_df], axis=1), feature_cols)

X = X_df.to_numpy(dtype=float)

print("=" * 60)
print("INFO DATA")
print("Shape data:", X.shape)
print("Kolom fitur:")
print(feature_cols)

log_rows = []
grand_start = time.perf_counter()

for r in range(1, N_RUNS + 1):
    seed = SEED_BASE + r

    start = time.perf_counter()
    U, V, n_iter_used, in_lower, in_boundary, objective_history = frcm(
        X=X,
        c=c,
        m=m,
        T=T,
        epsilon=epsilon,
        tau=tau,
        seed=seed,
        reinit_empty_cluster=REINIT_EMPTY_CLUSTER
    )
    end = time.perf_counter()

    total_time = end - start
    avg_time_per_iter = total_time / n_iter_used if n_iter_used > 0 else np.nan

    labels = np.argmax(U, axis=1)
    counts = np.bincount(labels, minlength=c)

    # FILE LABEL
    if SAVE_LABELS:
        out = df[[year_col, id_col]].copy()
        out["label"] = labels

        out_path = os.path.join(OUT_DIR, f"frcm_k{c}_run{r}.csv")
        out.to_csv(out_path, index=False)
        print("saved:", out_path)

    # FILE MEMBERSHIP
    if SAVE_MEMBERSHIP:
        mem_out = df[[year_col, id_col]].copy()

        for i in range(c):
            mem_out[f"u_cluster_{i}"] = U[:, i]

        mem_out["label"] = labels
        mem_out["region_type"] = np.where(in_lower.any(axis=1), "lower", "boundary")
        mem_out["n_active_clusters"] = (U > 0).sum(axis=1)

        mem_path = os.path.join(OUT_DIR, f"frcm_membership_k{c}_run{r}.csv")
        mem_out.to_csv(mem_path, index=False)
        print("saved:", mem_path)

    # FILE RIWAYAT OBJEKTIF
    if SAVE_OBJECTIVE_HISTORY:
        obj_hist_df = pd.DataFrame({
            "iterasi": np.arange(1, len(objective_history) + 1),
            "objective_value": objective_history
        })
        obj_path = os.path.join(OUT_DIR, f"frcm_objective_k{c}_run{r}.csv")
        obj_hist_df.to_csv(obj_path, index=False)
        print("saved:", obj_path)

    # RINGKASAN RUN
    label_aktif = sorted(np.unique(labels).tolist())
    n_lower_objects = int(in_lower.any(axis=1).sum())
    n_boundary_objects = int(in_boundary.any(axis=1).sum())
    avg_active_clusters = float((U > 0).sum(axis=1).mean())
    min_cluster_size = int(counts.min())
    max_cluster_size = int(counts.max())
    final_objective = compute_objective(X, V, U, m)

    log_rows.append({
        "run": r,
        "seed": seed,
        "n_iter_internal": n_iter_used,
        "total_time_sec": total_time,
        "avg_time_per_iter_sec": avg_time_per_iter,
        "final_objective": final_objective,
        "n_label_aktif": len(label_aktif),
        "label_aktif": ",".join(map(str, label_aktif)),
        "n_lower_objects": n_lower_objects,
        "n_boundary_objects": n_boundary_objects,
        "avg_active_clusters_per_object": avg_active_clusters,
        "boundary_ratio": n_boundary_objects / X.shape[0],
        "max_cluster_ratio": max_cluster_size / X.shape[0],
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "n_label_0": int(counts[0]),
        "n_label_1": int(counts[1]),
        "n_label_2": int(counts[2]),
        "n_label_3": int(counts[3]),
        # "n_label_4": int(counts[4]),
        # "n_label_5": int(counts[5]),
    })

grand_end = time.perf_counter()
grand_total = grand_end - grand_start

rekap_df = pd.DataFrame(log_rows)
rekap_df["is_very_imbalanced"] = rekap_df["max_cluster_ratio"] > 0.80

total_row = pd.DataFrame([{
    "run": "TOTAL_RUNTIME",
    "seed": "",
    "n_iter_internal": "",
    "total_time_sec": grand_total,
    "avg_time_per_iter_sec": "",
    "final_objective": "",
    "n_label_aktif": "",
    "label_aktif": "",
    "n_lower_objects": "",
    "n_boundary_objects": "",
    "avg_active_clusters_per_object": "",
    "boundary_ratio": "",
    "max_cluster_ratio": "",
    "min_cluster_size": "",
    "max_cluster_size": "",
    "n_label_0": "",
    "n_label_1": "",
    "n_label_2": "",
    "n_label_3": "",
    # "n_label_4": "",
    # "n_label_5": "",
    "is_very_imbalanced": ""
}])

rekap_df = pd.concat([rekap_df, total_row], ignore_index=True)

rekap_path = os.path.join(OUT_DIR, f"frcm_k{c}_rekap_100run.csv")
rekap_df.to_csv(rekap_path, index=False)
print("saved:", rekap_path)

print("=" * 60)
print("SELESAI")
print("Total runtime (detik):", grand_total)
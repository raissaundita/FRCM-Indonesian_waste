# TUNING PARAMETER T — FRCM & FAFRCM
# Cara kerja: setiap nilai T dicoba sebanyak N_RUNS_PER_T kali
# (dengan seed berbeda) menggunakan FRCM saja — karena T adalah
# parameter milik FRCM dan efeknya pada FAFRCM akan mengikuti.
# Hasil terbaik (T dengan objective terendah dan klaster paling seimbang)
# digunakan sebagai nilai T tetap di semua code berikutnya.
#
# PARAMETER YANG TIDAK DITUNING DI SINI (sudah tetap):
#   m       = 2.0    (nilai standar fuzzy clustering)
#   epsilon = 1e-5   (ambang konvergensi)
#   tau     = 200    (iterasi maksimum)
# ============================================================

import numpy as np
import pandas as pd

PATH     = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"
OUT_PATH = "D:/Raissa/Python/TUNING/TUNING_T_summary.csv"

year_col = "Tahun"
id_col   = "Provinsi"

m                   = 2.0
epsilon             = 1e-5
tau                 = 200
T_values            = [0.05, 0.07, 0.08, 0.09, 0.10]
N_RUNS_PER_T        = 20
SEED_BASE           = 42
IMBALANCE_THRESHOLD = 0.80
c_for_tuning        = 4   # nilai tengah rentang k di elbow (2-7)

# LOAD DATA
df = pd.read_csv(PATH)
feature_cols = [col for col in df.columns if col not in [year_col, id_col]]
X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
if X_df.isnull().sum().sum() > 0:
    raise ValueError("Masih ada NaN di data!")
X = X_df.to_numpy(dtype=float)
print("Shape data:", X.shape)

# ============================================================
# FUNGSI BANTU
def hitung_jarak_euclidean(X, V):
    """
    Persamaan 2.2: d(x_j, v_i) = sqrt( sum_l (x_jl - v_il)^2 )
    """
    n = X.shape[0]
    c = V.shape[0]
    dist = np.zeros((n, c))
    for j in range(n):
        for i in range(c):
            dist[j, i] = np.sqrt(np.sum((X[j] - V[i]) ** 2))
    return dist

def hitung_objektif_frcm(X, V, U, m):
    """
    Persamaan 2.14: J(U,V) = sum_j sum_i (u_ij^m) * ||x_j - v_i||^2
    """
    n = X.shape[0]
    c = V.shape[0]
    J = 0.0
    for j in range(n):
        for i in range(c):
            J += (U[j, i] ** m) * np.sum((X[j] - V[i]) ** 2)
    return J

def tentukan_region_dan_keanggotaan(X, V, c, m, T):
    """
    Persamaan 2.12 dan 2.13: tentukan lower/boundary, hitung keanggotaan.
    """
    n    = X.shape[0]
    dist = hitung_jarak_euclidean(X, V) + 1e-12
    q    = np.argmin(dist, axis=1)

    in_lower    = np.zeros((n, c), dtype=bool)
    in_boundary = np.zeros((n, c), dtype=bool)
    U           = np.zeros((n, c), dtype=float)

    for j in range(n):
        kandidat = []
        for p in range(c):
            if p == q[j]:
                continue
            if abs(dist[j, p] - dist[j, q[j]]) <= T:
                kandidat.append(p)

        if len(kandidat) > 0:
            in_boundary[j, q[j]] = True
            for p in kandidat:
                in_boundary[j, p] = True
        else:
            in_lower[j, q[j]] = True

    for j in range(n):
        if in_lower[j].any():
            i0       = np.where(in_lower[j])[0][0]
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

def perbarui_centroid(X, U, V_lama, m, rng):
    """Persamaan 2.16: v_i = sum_j(u_ij^m * x_j) / sum_j(u_ij^m)"""
    c      = U.shape[1]
    Um     = U ** m
    bobot  = Um.sum(axis=0)
    V_baru = V_lama.copy()
    for i in range(c):
        if bobot[i] <= 1e-14:
            V_baru[i] = X[rng.integers(0, X.shape[0])].copy()
        else:
            V_baru[i] = (Um[:, i][:, None] * X).sum(axis=0) / bobot[i]
    return V_baru

def frcm(X, c, m, T, epsilon, tau, seed=None):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=c, replace=False)
    V   = X[idx].copy()

    n_iter_used = 0
    for t in range(1, tau + 1):
        n_iter_used = t
        V_prev      = V.copy()
        dist, in_lower, in_boundary, U = tentukan_region_dan_keanggotaan(
            X, V, c, m, T
        )
        V = perbarui_centroid(X, U, V, m, rng)
        if np.linalg.norm(V - V_prev) <= epsilon:
            break

    dist, in_lower, in_boundary, U = tentukan_region_dan_keanggotaan(
        X, V, c, m, T
    )
    final_obj = hitung_objektif_frcm(X, V, U, m)
    return U, V, n_iter_used, in_lower, in_boundary, final_obj


# UJI SETIAP NILAI T
summary_results = []
detail_results  = []

for T in T_values:
    print(f"\n=== Testing T = {T} ===")
    final_objectives  = []
    boundary_ratios   = []
    lower_ratios      = []
    imbalance_ratios  = []
    n_iters           = []
    n_imbalanced_runs = 0

    for r in range(1, N_RUNS_PER_T + 1):
        seed = SEED_BASE + r
        U, V, n_iter, in_lower, in_boundary, final_obj = frcm(
            X, c=c_for_tuning, m=m, T=T, epsilon=epsilon, tau=tau, seed=seed
        )
        labels         = np.argmax(U, axis=1)
        counts         = np.bincount(labels, minlength=c_for_tuning)
        max_ratio      = counts.max() / len(labels)
        boundary_ratio = in_boundary.any(axis=1).sum() / len(labels)
        lower_ratio    = in_lower.any(axis=1).sum() / len(labels)
        is_imbalanced  = max_ratio > IMBALANCE_THRESHOLD

        if is_imbalanced:
            n_imbalanced_runs += 1

        final_objectives.append(final_obj)
        boundary_ratios.append(boundary_ratio)
        lower_ratios.append(lower_ratio)
        imbalance_ratios.append(max_ratio)
        n_iters.append(n_iter)

        detail_results.append({
            "T": T, "run": r, "seed": seed, "n_iter": n_iter,
            "final_objective": final_obj, "boundary_ratio": boundary_ratio,
            "lower_ratio": lower_ratio, "max_cluster_ratio": max_ratio,
            "is_imbalanced": is_imbalanced,
        })
        print(
            f"  [run {r}] obj={final_obj:.6f} | boundary={boundary_ratio:.3f} | "
            f"lower={lower_ratio:.3f} | max_ratio={max_ratio:.3f}"
        )

    summary_results.append({
        "T"                    : T,
        "avg_final_objective"  : np.mean(final_objectives),
        "std_final_objective"  : np.std(final_objectives, ddof=1),
        "avg_boundary_ratio"   : np.mean(boundary_ratios),
        "avg_lower_ratio"      : np.mean(lower_ratios),
        "avg_max_cluster_ratio": np.mean(imbalance_ratios),
        "avg_n_iter"           : np.mean(n_iters),
        "n_imbalanced_runs"    : n_imbalanced_runs,
    })

summary_df = pd.DataFrame(summary_results)
detail_df  = pd.DataFrame(detail_results)
print("\nHASIL UJI T:")
print(summary_df)
summary_df.to_csv(OUT_PATH, index=False)
detail_df.to_csv(OUT_PATH.replace("summary", "detail"), index=False)
print(f"\nSummary disimpan ke: {OUT_PATH}")
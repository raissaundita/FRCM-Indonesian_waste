import os
import time
import numpy as np
import pandas as pd

PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"
OUT_DIR = "D:/Raissa/Python/KLASTER DATA/FAFRCM_K3"

c = 3
N_RUNS = 100
SEED_BASE = 42

# Parameter FRCM
m = 2.0
T = 0.09  # hasil tuning T
epsilon = 1e-5
tau_frcm = 200

# Parameter Firefly (hasil tuning FA)
N_FIREFLIES = 22
MAX_ITER_FA = 297
ALPHA0 = 1.0
THETA = 0.982
BETA0 = 1.312
GAMMA = 0.104

year_col = "Tahun"
id_col = "Provinsi"

os.makedirs(OUT_DIR, exist_ok=True)

# FUNGSI BANTU
def compute_distances(X, V):
    return np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)

def initialize_centroids(X, c, rng):
    idx = rng.choice(X.shape[0], size=c, replace=False)
    return X[idx].copy()

def compute_objective(X, V, U, m):
    dist_sq = np.sum((X[:, None, :] - V[None, :, :]) ** 2, axis=2)
    return float(np.sum((U ** m) * dist_sq))

def validate_input_data(df, feature_cols):
    if len(feature_cols) == 0:
        raise ValueError("Tidak ada kolom fitur numerik.")

    if df[feature_cols].isnull().sum().sum() > 0:
        na_info = df[feature_cols].isnull().sum()
        raise ValueError(
            "Masih ada missing value pada fitur numerik.\n"
            f"Rincian:\n{na_info[na_info > 0]}"
        )

# CORE FRCM
def frcm_regions_and_membership(X, V, m, T):
    n = X.shape[0]
    c = V.shape[0]

    dist = compute_distances(X, V) + 1e-12
    q = np.argmin(dist, axis=1)

    in_lower = np.zeros((n, c), dtype=bool)
    in_boundary = np.zeros((n, c), dtype=bool)

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

    U = np.zeros((n, c), dtype=float)

    for j in range(n):
        if in_lower[j].any():
            i0 = np.where(in_lower[j])[0][0]
            U[j, i0] = 1.0
            continue

        B = np.where(in_boundary[j])[0]
        if len(B) > 0:
            for i in B:
                denom = 0.0
                for k in B:
                    denom += (dist[j, i] / dist[j, k]) ** (2.0 / (m - 1.0))
                U[j, i] = 1.0 / denom

    return U, in_lower, in_boundary

def frcm_update_centers_safe(X, U, m, rng):
    Um = U ** m
    weights = Um.sum(axis=0)

    c = U.shape[1]
    d = X.shape[1]
    V_new = np.zeros((c, d), dtype=float)

    for i in range(c):
        if weights[i] <= 1e-14:
            # re-init cluster kosong
            rand_idx = rng.integers(0, X.shape[0])
            V_new[i] = X[rand_idx].copy()
        else:
            V_new[i] = (Um[:, i][:, None] * X).sum(axis=0) / weights[i]

    return V_new

def frcm_refine_with_initV(X, V_init, m, T, epsilon, tau, rng):
    V = V_init.copy()
    n_iter_used = 0
    objective_history = []

    U = np.zeros((X.shape[0], V.shape[0]), dtype=float)
    in_lower = np.zeros((X.shape[0], V.shape[0]), dtype=bool)
    in_boundary = np.zeros((X.shape[0], V.shape[0]), dtype=bool)

    for t in range(1, tau + 1):
        n_iter_used = t
        V_prev = V.copy()

        # STEP 1-3: hitung region dan membership berdasarkan centroid saat ini
        U, in_lower, in_boundary = frcm_regions_and_membership(X, V, m, T)

        # STEP 4: hitung objective pada state yang konsisten (U, V)
        obj = compute_objective(X, V, U, m)
        objective_history.append(obj)

        # STEP 5: update centroid
        V = frcm_update_centers_safe(X, U, m, rng)

        # STEP 6: cek konvergensi
        if np.linalg.norm(V - V_prev) <= epsilon:
            break

    # sinkronkan output akhir dengan centroid final
    U, in_lower, in_boundary = frcm_regions_and_membership(X, V, m, T)
    final_obj = compute_objective(X, V, U, m)

    return U, V, n_iter_used, in_lower, in_boundary, objective_history, final_obj

# FAFRCM
def fafrcm(X, c, m, T, epsilon, tau_frcm,
           N_FIREFLIES, MAX_ITER_FA, ALPHA0, THETA, BETA0, GAMMA,
           seed=None):
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # centroid awal
    V0 = initialize_centroids(X, c, rng)

    # sedikit refine awal agar start point lebih stabil
    U0, _, _ = frcm_regions_and_membership(X, V0, m, T)
    V_init = frcm_update_centers_safe(X, U0, m, rng)

    # populasi firefly
    fireflies = np.zeros((N_FIREFLIES, c, d), dtype=float)
    for i in range(N_FIREFLIES):
        noise = 0.05 * (rng.random((c, d)) - 0.5)
        fireflies[i] = V_init + noise

    def J_of(V):
        U_tmp, _, _ = frcm_regions_and_membership(X, V, m, T)
        return compute_objective(X, V, U_tmp, m)

    best_fa_objective_history = []

    for t in range(1, MAX_ITER_FA + 1):
        alpha_t = ALPHA0 * (THETA ** (t - 1))
        J_vals = np.array([J_of(fireflies[i]) for i in range(N_FIREFLIES)])

        best_fa_objective_history.append(float(J_vals.min()))

        for i in range(N_FIREFLIES):
            for j in range(N_FIREFLIES):
                if J_vals[j] < J_vals[i]:
                    rij = np.linalg.norm(
                        fireflies[i].reshape(-1) - fireflies[j].reshape(-1)
                    )
                    beta = BETA0 * np.exp(-GAMMA * (rij ** 2))

                    fireflies[i] = (
                        fireflies[i]
                        + beta * (fireflies[j] - fireflies[i])
                        + alpha_t * (rng.random((c, d)) - 0.5)
                    )

    # solusi terbaik dari FA
    J_vals = np.array([J_of(fireflies[i]) for i in range(N_FIREFLIES)])
    best_idx = int(np.argmin(J_vals))
    V_best = fireflies[best_idx]
    best_fa_objective = float(J_vals[best_idx])

    # refine pakai FRCM
    U_final, V_final, refine_iter, in_lower, in_boundary, refine_obj_hist, final_refined_objective = frcm_refine_with_initV(
        X, V_best, m, T, epsilon, tau_frcm, rng
    )

    return (
        U_final, V_final, refine_iter, in_lower, in_boundary,
        best_fa_objective, final_refined_objective,
        best_fa_objective_history, refine_obj_hist
    )

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
print("=" * 60)

# MAIN LOOP
log_rows = []
grand_start = time.perf_counter()

for r in range(1, N_RUNS + 1):
    seed = SEED_BASE + r

    start = time.perf_counter()
    (
        U, V, refine_iter, in_lower, in_boundary,
        best_fa_objective, final_refined_objective,
        best_fa_objective_history, refine_obj_hist
    ) = fafrcm(
        X, c, m, T, epsilon, tau_frcm,
        N_FIREFLIES, MAX_ITER_FA, ALPHA0, THETA, BETA0, GAMMA,
        seed=seed
    )
    end = time.perf_counter()

    total_time = end - start
    total_iter = MAX_ITER_FA + refine_iter
    avg_time_per_iter = total_time / total_iter if total_iter > 0 else np.nan

    labels = np.argmax(U, axis=1)
    counts = np.bincount(labels, minlength=c)

    label_aktif = sorted(np.unique(labels).tolist())
    n_lower_objects = int(in_lower.any(axis=1).sum())
    n_boundary_objects = int(in_boundary.any(axis=1).sum())

    max_cluster_size = int(counts.max())
    min_cluster_size = int(counts.min())
    max_cluster_ratio = max_cluster_size / len(labels)
    boundary_ratio = n_boundary_objects / len(labels)
    is_very_imbalanced = max_cluster_ratio > 0.80

    # save label file
    out = df[[year_col, id_col]].copy()
    out["label"] = labels

    out_path = os.path.join(OUT_DIR, f"fafrcm_k{c}_run{r}.csv")
    out.to_csv(out_path, index=False)
    print(f"[RUN {r}] saved label:", out_path)

    # save membership file
    mem_out = df[[year_col, id_col]].copy()
    for i in range(c):
        mem_out[f"u_cluster_{i}"] = U[:, i]
    mem_out["label"] = labels
    mem_out["region_type"] = np.where(in_lower.any(axis=1), "lower", "boundary")
    mem_out["n_active_clusters"] = (U > 0).sum(axis=1)

    mem_path = os.path.join(OUT_DIR, f"fafrcm_membership_k{c}_run{r}.csv")
    mem_out.to_csv(mem_path, index=False)
    print(f"[RUN {r}] saved membership:", mem_path)

    # save objective history
    fa_hist_df = pd.DataFrame({
        "fa_iter": np.arange(1, len(best_fa_objective_history) + 1),
        "best_fa_objective_so_far": best_fa_objective_history
    })
    fa_hist_path = os.path.join(OUT_DIR, f"fafrcm_fa_objective_k{c}_run{r}.csv")
    fa_hist_df.to_csv(fa_hist_path, index=False)

    refine_hist_df = pd.DataFrame({
        "frcm_iter": np.arange(1, len(refine_obj_hist) + 1),
        "refine_objective": refine_obj_hist
    })
    refine_hist_path = os.path.join(OUT_DIR, f"fafrcm_refine_objective_k{c}_run{r}.csv")
    refine_hist_df.to_csv(refine_hist_path, index=False)

    print(
        f"[RUN {r}] "
        f"aktif={len(label_aktif)} | "
        f"lower={n_lower_objects} | boundary={n_boundary_objects} ({boundary_ratio:.3f}) | "
        f"min_cluster={min_cluster_size} | max_cluster={max_cluster_size} ({max_cluster_ratio:.3f}) | "
        f"imbalanced={is_very_imbalanced} | "
        f"FA_obj={best_fa_objective:.6f} | final_obj={final_refined_objective:.6f}"
    )

    log_rows.append({
        "run": r,
        "seed": seed,
        "fa_iter": MAX_ITER_FA,
        "frcm_refine_iter": refine_iter,
        "total_iter_used": total_iter,
        "total_time_sec": total_time,
        "avg_time_per_iter_sec": avg_time_per_iter,
        "best_fa_objective": best_fa_objective,
        "final_refined_objective": final_refined_objective,
        "n_label_aktif": len(label_aktif),
        "label_aktif": ",".join(map(str, label_aktif)),
        "n_lower_objects": n_lower_objects,
        "n_boundary_objects": n_boundary_objects,
        "boundary_ratio": boundary_ratio,
        "min_cluster_size": min_cluster_size,
        "max_cluster_size": max_cluster_size,
        "max_cluster_ratio": max_cluster_ratio,
        "is_very_imbalanced": is_very_imbalanced,
        "n_label_0": int(counts[0]),
        "n_label_1": int(counts[1]),
        "n_label_2": int(counts[2]),
        # "n_label_3": int(counts[3]),
        # "n_label_4": int(counts[4]),
    })

grand_end = time.perf_counter()
grand_total = grand_end - grand_start

rekap_df = pd.DataFrame(log_rows)

total_row = pd.DataFrame([{
    "run": "TOTAL_RUNTIME",
    "seed": "",
    "fa_iter": "",
    "frcm_refine_iter": "",
    "total_iter_used": "",
    "total_time_sec": grand_total,
    "avg_time_per_iter_sec": "",
    "best_fa_objective": "",
    "final_refined_objective": "",
    "n_label_aktif": "",
    "label_aktif": "",
    "n_lower_objects": "",
    "n_boundary_objects": "",
    "boundary_ratio": "",
    "min_cluster_size": "",
    "max_cluster_size": "",
    "max_cluster_ratio": "",
    "is_very_imbalanced": "",
    "n_label_0": "",
    "n_label_1": "",
    "n_label_2": "",
    # "n_label_3": "",
    # "n_label_4": "",
}])

rekap_df = pd.concat([rekap_df, total_row], ignore_index=True)

rekap_path = os.path.join(OUT_DIR, f"fafrcm_k{c}_rekap_100run.csv")
rekap_df.to_csv(rekap_path, index=False)

print("=" * 60)
print("SELESAI")
print("saved rekap:", rekap_path)
print("Total runtime (detik):", grand_total)
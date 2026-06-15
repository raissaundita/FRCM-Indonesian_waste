import optuna
import os, time
import numpy as np
import pandas as pd

PATH    = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"
OUT_DIR = "D:/Raissa/Python/TUNING/tuning_FA_optuna"
os.makedirs(OUT_DIR, exist_ok=True)

year_col = "Tahun"
id_col   = "Provinsi"

T                   = 0.09  # hasil tuning T sebelumnya
m                   = 2.0
epsilon             = 1e-5
tau                 = 200
c_for_tuning        = 4
N_RUNS_PER_TRIAL    = 5     # berapa kali tiap kombinasi parameter dicoba
SEED_BASE           = 42
IMBALANCE_THRESHOLD = 0.80

# Jumlah trial Optuna
N_TRIALS = 100

# ==============================================================
# LOAD DATA
df = pd.read_csv(PATH)
feature_cols = [col for col in df.columns if col not in [year_col, id_col]]
X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
if X_df.isnull().sum().sum() > 0:
    raise ValueError("Masih ada NaN di data!")
X = X_df.to_numpy(dtype=float)
print("Shape data:", X.shape)


# FUNGSI BANTU
def hitung_jarak_euclidean(X, V):
    """Persamaan 2.2: d(x_j, v_i) = sqrt( sum_l (x_jl - v_il)^2 )"""
    n = X.shape[0]
    c = V.shape[0]
    dist = np.zeros((n, c))
    for j in range(n):
        for i in range(c):
            dist[j, i] = np.sqrt(np.sum((X[j] - V[i]) ** 2))
    return dist

def hitung_objektif_frcm(X, V, U, m):
    """Persamaan 2.14: J(U,V) = sum_j sum_i (u_ij^m) * ||x_j - v_i||^2"""
    n = X.shape[0]
    c = V.shape[0]
    J = 0.0
    for j in range(n):
        for i in range(c):
            J += (U[j, i] ** m) * np.sum((X[j] - V[i]) ** 2)
    return J

def tentukan_region_dan_keanggotaan(X, V, m, T):
    """Persamaan 2.12 dan 2.13."""
    n = X.shape[0]
    c = V.shape[0]
    dist = hitung_jarak_euclidean(X, V) + 1e-12
    q    = np.argmin(dist, axis=1)

    in_lower    = np.zeros((n, c), dtype=bool)
    in_boundary = np.zeros((n, c), dtype=bool)
    U           = np.zeros((n, c), dtype=float)

    for j in range(n):
        kandidat = []
        for p in range(c):
            if p == q[j]: continue
            if abs(dist[j, p] - dist[j, q[j]]) <= T:
                kandidat.append(p)
        if len(kandidat) > 0:
            in_boundary[j, q[j]] = True
            for p in kandidat: in_boundary[j, p] = True
        else:
            in_lower[j, q[j]] = True

    for j in range(n):
        if in_lower[j].any():
            U[j, np.where(in_lower[j])[0][0]] = 1.0
            continue
        B = np.where(in_boundary[j])[0]
        if len(B) > 0:
            for i in B:
                denom = sum(
                    (dist[j, i] / dist[j, k]) ** (2.0 / (m - 1.0)) for k in B
                )
                U[j, i] = 1.0 / denom

    return U, in_lower, in_boundary

def perbarui_centroid(X, U, m, rng):
    """Persamaan 2.16: v_i = sum_j(u_ij^m * x_j) / sum_j(u_ij^m)"""
    c      = U.shape[1]
    Um     = U ** m
    bobot  = Um.sum(axis=0)
    V_baru = np.zeros((c, X.shape[1]), dtype=float)
    for i in range(c):
        if bobot[i] <= 1e-14:
            V_baru[i] = X[rng.integers(0, X.shape[0])].copy()
        else:
            V_baru[i] = (Um[:, i][:, None] * X).sum(axis=0) / bobot[i]
    return V_baru

def frcm_refine(X, V_init, m, T, epsilon, tau, rng):
    V           = V_init.copy()
    n_iter_used = 0
    for t in range(1, tau + 1):
        n_iter_used = t
        V_prev      = V.copy()
        U, in_lower, in_boundary = tentukan_region_dan_keanggotaan(X, V, m, T)
        V = perbarui_centroid(X, U, m, rng)
        if np.linalg.norm(V - V_prev) <= epsilon:
            break
    U, in_lower, in_boundary = tentukan_region_dan_keanggotaan(X, V, m, T)
    return U, V, n_iter_used, in_lower, in_boundary, hitung_objektif_frcm(X, V, U, m)

def fafrcm(X, c, m, T, epsilon, tau,
           N_FIREFLIES, MAX_ITER_FA, ALPHA0, THETA, BETA0, GAMMA, seed=None):
    rng  = np.random.default_rng(seed)
    n, d = X.shape

    idx      = rng.choice(n, size=c, replace=False)
    V0       = X[idx].copy()
    U0, _, _ = tentukan_region_dan_keanggotaan(X, V0, m, T)
    V_init   = perbarui_centroid(X, U0, m, rng)

    fireflies = np.zeros((N_FIREFLIES, c, d), dtype=float)
    for i in range(N_FIREFLIES):
        fireflies[i] = V_init + 0.05 * (rng.random((c, d)) - 0.5)

    def J_of(V):
        U_tmp, _, _ = tentukan_region_dan_keanggotaan(X, V, m, T)
        return hitung_objektif_frcm(X, V, U_tmp, m)

    for t in range(1, MAX_ITER_FA + 1):
        # alpha_t = ALPHA0 * THETA^(t-1): langkah acak meluruh setiap iterasi
        alpha_t = ALPHA0 * (THETA ** (t - 1))
        J_vals  = np.array([J_of(fireflies[i]) for i in range(N_FIREFLIES)])

        for i in range(N_FIREFLIES):
            for j in range(N_FIREFLIES):
                if J_vals[j] < J_vals[i]:
                    # Persamaan 2.19: jarak antar kunang-kunang
                    rij = np.linalg.norm(
                        fireflies[i].reshape(-1) - fireflies[j].reshape(-1)
                    )
                    # Persamaan 2.18
                    beta = BETA0 * np.exp(-GAMMA * (rij ** 2))
                    # Persamaan 2.20
                    fireflies[i] = (
                        fireflies[i]
                        + beta * (fireflies[j] - fireflies[i])
                        + alpha_t * (rng.random((c, d)) - 0.5)
                    )

    J_vals   = np.array([J_of(fireflies[i]) for i in range(N_FIREFLIES)])
    best_idx = int(np.argmin(J_vals))
    V_best   = fireflies[best_idx]

    U_final, V_final, refine_iter, in_lower, in_boundary, final_obj = \
        frcm_refine(X, V_best, m, T, epsilon, tau, rng)

    return U_final, V_final, refine_iter, in_lower, in_boundary, final_obj


# ==============================================================
# FUNGSI OBJEKTIF UNTUK OPTUNA
# Tugas fungsi ini:
#   1. Minta Optuna pilihkan nilai parameter FA dalam rentang yang ditentukan
#   2. Jalankan FAFRCM sebanyak N_RUNS_PER_TRIAL kali
#   3. Kembalikan skor (makin kecil = makin baik) ke Optuna

def objective(trial):
    # suggest_int  : Optuna pilih bilangan bulat dalam rentang [low, high]
    # suggest_float: Optuna pilih bilangan desimal dalam rentang [low, high]
    N_FIREFLIES = trial.suggest_int  ("N_FIREFLIES", 20,   40  )
    MAX_ITER_FA = trial.suggest_int  ("MAX_ITER_FA", 100, 300  )
    THETA       = trial.suggest_float("THETA",       0.95, 0.99)
    BETA0       = trial.suggest_float("BETA0",       0.5,  2.0 )
    GAMMA       = trial.suggest_float("GAMMA",       0.1,  2.0 )
    ALPHA0      = 1.0  # ALPHA0 nilainya tetap, tidak di-tune

    # ----------------------------------------------------------
    # Jalankan FAFRCM sebanyak N_RUNS_PER_TRIAL kali
    final_objectives   = []
    max_cluster_ratios = []
    boundary_ratios    = []
    n_imbalanced_runs  = 0

    for r in range(1, N_RUNS_PER_TRIAL + 1):
        seed = SEED_BASE + r
        U, V, refine_iter, in_lower, in_boundary, final_obj = fafrcm(
            X, c_for_tuning, m, T, epsilon, tau,
            N_FIREFLIES, MAX_ITER_FA, ALPHA0,
            THETA, BETA0, GAMMA, seed=seed
        )
        labels         = np.argmax(U, axis=1)
        counts         = np.bincount(labels, minlength=c_for_tuning)
        max_ratio      = counts.max() / len(labels)
        boundary_ratio = in_boundary.any(axis=1).sum() / len(labels)
        is_imbalanced  = max_ratio > IMBALANCE_THRESHOLD

        if is_imbalanced: n_imbalanced_runs += 1
        final_objectives.append(final_obj)
        max_cluster_ratios.append(max_ratio)
        boundary_ratios.append(boundary_ratio)

    # Hitung skor (makin kecil skornya = kombinasi parameter makin baik)
    avg_final_objective   = np.mean(final_objectives)
    avg_max_cluster_ratio = np.mean(max_cluster_ratios)

    score = (
        avg_final_objective
        + 10 * avg_max_cluster_ratio
        + 5  * n_imbalanced_runs
    )

    # Simpan info/atribut tambahan ke dalam trial agar bisa dibaca nanti
    trial.set_user_attr("avg_final_objective",    avg_final_objective)
    trial.set_user_attr("std_final_objective",    np.std(final_objectives,   ddof=1))
    trial.set_user_attr("avg_max_cluster_ratio",  avg_max_cluster_ratio)
    trial.set_user_attr("std_max_cluster_ratio",  np.std(max_cluster_ratios, ddof=1))
    trial.set_user_attr("avg_boundary_ratio",     np.mean(boundary_ratios))
    trial.set_user_attr("n_imbalanced_runs",      n_imbalanced_runs)

    return score   # Optuna akan meminimalkan nilai ini


# JALANKAN OPTUNA
print(f"\nMemulai Optuna dengan {N_TRIALS} trial...\n")
grand_start = time.perf_counter()

# TPESampler: Optuna memilih parameter berikutnya berdasarkan hasil trial sebelumnya
study = optuna.create_study(
    direction  = "minimize",
    study_name = "tuning_FA_fafrcm",
    sampler    = optuna.samplers.TPESampler(seed=SEED_BASE),  # agar hasil bisa direproduksi
)

# Nonaktifkan log bawaan Optuna agar output lebih bersih
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Callback untuk cetak progress tiap trial selesai
def print_progress(study, trial):
    print(
        f"  [Trial {trial.number+1:>3}/{N_TRIALS}] "
        f"score={trial.value:.4f} | "
        f"N_FIREFLIES={trial.params['N_FIREFLIES']} | "
        f"MAX_ITER_FA={trial.params['MAX_ITER_FA']} | "
        f"THETA={trial.params['THETA']:.3f} | "
        f"BETA0={trial.params['BETA0']:.3f} | "
        f"GAMMA={trial.params['GAMMA']:.3f}"
    )
# start
study.optimize(objective, n_trials=N_TRIALS, callbacks=[print_progress])

# KUMPULKAN HASIL KE DATAFRAME
summary_rows = []
for t in study.trials:
    row = {
        "N_FIREFLIES"          : t.params["N_FIREFLIES"],
        "MAX_ITER_FA"          : t.params["MAX_ITER_FA"],
        "ALPHA0"               : 1.0,
        "THETA"                : t.params["THETA"],
        "BETA0"                : t.params["BETA0"],
        "GAMMA"                : t.params["GAMMA"],
        "avg_final_objective"  : t.user_attrs["avg_final_objective"],
        "std_final_objective"  : t.user_attrs["std_final_objective"],
        "avg_max_cluster_ratio": t.user_attrs["avg_max_cluster_ratio"],
        "std_max_cluster_ratio": t.user_attrs["std_max_cluster_ratio"],
        "avg_boundary_ratio"   : t.user_attrs["avg_boundary_ratio"],
        "n_imbalanced_runs"    : t.user_attrs["n_imbalanced_runs"],
        "score_select"         : t.value,
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("score_select").reset_index(drop=True)

summary_df.to_csv(os.path.join(OUT_DIR, "tuning_FA_summary.csv"), index=False)

print("\nTOP 10 PARAMETER FA TERBAIK")
print(summary_df.head(10).to_string())

best = study.best_trial
print("\n========== PARAMETER TERBAIK ==========")
print(f"  N_FIREFLIES : {best.params['N_FIREFLIES']}")
print(f"  MAX_ITER_FA : {best.params['MAX_ITER_FA']}")
print(f"  ALPHA0      : 1.0")
print(f"  THETA       : {best.params['THETA']:.3f}")
print(f"  BETA0       : {best.params['BETA0']:.3f}")
print(f"  GAMMA       : {best.params['GAMMA']:.3f}")
print(f"  Score       : {best.value:.4f}")
print(f"\nTotal runtime (detik): {time.perf_counter() - grand_start:.2f}")
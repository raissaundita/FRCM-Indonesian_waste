import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

DATA_PATH = "D:/Raissa/Dataset/05_scaled_provinsi.csv"
K_LIST = list(range(2, 8))
N_RUNS = 5

# FCM/FRCM settings
M_FUZZY = 2.0
T_BOUNDARY = 0.05
EPS = 1e-5
MAX_ITER_FCM = 200
MAX_ITER_FRCM = 200

# FA settings
N_FIREFLIES = 20
MAX_ITER_FA = 100
ALPHA0 = 1.0
THETA = 0.97
BETA0 = 1.0
GAMMA = 0.1
SEED = 42

# 1) LOAD DATA
df = pd.read_csv(DATA_PATH)
X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
n, d = X.shape
print("Data shape:", X.shape)
print("Numeric columns:", list(df.select_dtypes(include=[np.number]).columns))

xmin = X.min(axis=0)
xmax = X.max(axis=0)

# 2) UTILITIES
def euclidean_dist_matrix(X, V):
    """D[j, i] = ||x_j - v_i||"""
    return np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)

def clip_centroids(V):
    return np.clip(V, xmin[None, :], xmax[None, :])

def safe_silhouette(X, labels):
    """
    Silhouette butuh >=2 cluster dan label tidak boleh semuanya sama.
    Kadang algoritma fuzzy bisa menghasilkan cluster kosong (tidak kepilih).
    """
    labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return np.nan
    # silhouette_score akan error kalau jumlah cluster == n_samples
    if len(uniq) >= X.shape[0]:
        return np.nan
    try:
        return float(silhouette_score(X, labels))
    except Exception:
        return np.nan

# 3) K-MEANS (ambil silhouette)
def silhouette_kmeans(X, k_list, random_state=42, n_init=10, max_iter=300):
    rows = []
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
        labels = km.fit_predict(X)
        sil = safe_silhouette(X, labels)
        rows.append({"k": k, "silhouette": sil})
    return pd.DataFrame(rows)

# 4) FCM (return U, V, J)
def fcm_fit_full(X, c, m=2.0, eps=1e-5, max_iter=200, seed=42):
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # init U random, sum_i u_ij = 1
    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)

    last_V = None
    for _ in range(max_iter):
        Um = U ** m

        # Update centroid V
        V = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)

        if last_V is not None and np.linalg.norm(V - last_V) <= eps:
            break
        last_V = V

        # Update membership U
        D = euclidean_dist_matrix(X, V) + 1e-12
        power = 2.0 / (m - 1.0)
        ratio = (D[:, :, None] / D[:, None, :]) ** power
        U = 1.0 / ratio.sum(axis=2)

    # Objective J
    D2 = euclidean_dist_matrix(X, V) ** 2
    J = np.sum((U ** m) * D2)
    return U, V, float(J)

def silhouette_fcm_best_of_runs(X, k_list, m=2.0, eps=1e-5, max_iter=200, base_seed=42, n_runs=5):
    """
    Sesuai logika elbow kamu: untuk tiap k, jalankan n_runs lalu ambil solusi dgn J minimum,
    kemudian hitung silhouette dari label crisp (argmax U).
    """
    rows = []
    for k in k_list:
        best = {"J": np.inf, "labels": None}
        for r in range(n_runs):
            seed = base_seed + r
            U, V, J = fcm_fit_full(X, c=k, m=m, eps=eps, max_iter=max_iter, seed=seed)
            if J < best["J"]:
                labels = np.argmax(U, axis=1)
                best = {"J": J, "labels": labels}
        sil = safe_silhouette(X, best["labels"])
        rows.append({"k": k, "silhouette": sil, "J_min": float(best["J"])})
    return pd.DataFrame(rows)

# 5) FRCM (return U dari V)
def frcm_membership_from_V(X, V, m=2.0, T=0.05):
    D = euclidean_dist_matrix(X, V) + 1e-12

    order = np.argsort(D, axis=1)
    q1 = order[:, 0]
    q2 = order[:, 1]

    d1 = D[np.arange(X.shape[0]), q1]
    d2 = D[np.arange(X.shape[0]), q2]

    # boundary condition
    is_boundary = np.abs(d2 - d1) <= T

    U = np.zeros((X.shape[0], V.shape[0]), dtype=float)

    # lower -> hard
    idx_lower = np.where(~is_boundary)[0]
    U[idx_lower, q1[idx_lower]] = 1.0

    # boundary -> fuzzy (seperti FCM)
    idx_b = np.where(is_boundary)[0]
    if len(idx_b) > 0:
        Db = D[idx_b]
        power = 2.0 / (m - 1.0)
        ratio = (Db[:, :, None] / Db[:, None, :]) ** power
        Ub = 1.0 / ratio.sum(axis=2)
        U[idx_b] = Ub

    return U

def frcm_objective_from_V(X, V, m=2.0, T=0.05):
    U = frcm_membership_from_V(X, V, m=m, T=T)
    D2 = euclidean_dist_matrix(X, V) ** 2
    return float(np.sum((U ** m) * D2))

def frcm_fit_full(X, c, m=2.0, T=0.05, eps=1e-5, max_iter=200, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=c, replace=False)
    V = X[idx].copy()

    last_V = None
    for _ in range(max_iter):
        if last_V is not None and np.linalg.norm(V - last_V) <= eps:
            break
        last_V = V.copy()

        U = frcm_membership_from_V(X, V, m=m, T=T)
        Um = U ** m
        V = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)

    U = frcm_membership_from_V(X, V, m=m, T=T)
    J = frcm_objective_from_V(X, V, m=m, T=T)
    return U, V, float(J)

def silhouette_frcm_best_of_runs(X, k_list, m=2.0, T=0.05, eps=1e-5, max_iter=200, base_seed=42, n_runs=5):
    rows = []
    for k in k_list:
        best = {"J": np.inf, "labels": None}
        for r in range(n_runs):
            seed = base_seed + r
            U, V, J = frcm_fit_full(X, c=k, m=m, T=T, eps=eps, max_iter=max_iter, seed=seed)
            if J < best["J"]:
                labels = np.argmax(U, axis=1)
                best = {"J": J, "labels": labels}
        sil = safe_silhouette(X, best["labels"])
        rows.append({"k": k, "silhouette": sil, "J_min": float(best["J"])})
    return pd.DataFrame(rows)

# 6) FA–FRCM (return best V supaya bisa dihitung silhouette)
def fa_frcm_optimize_centroids_return_bestV(
    X, c,
    m=2.0, T=0.05,
    n_fireflies=20, max_iter=100,
    alpha0=1.0, theta=0.97,
    beta0=1.0, gamma=0.1,
    seed=42
):
    rng = np.random.default_rng(seed)

    fireflies = []
    for _ in range(n_fireflies):
        idx = rng.choice(X.shape[0], size=c, replace=False)
        fireflies.append(X[idx].copy())
    fireflies = np.array(fireflies)

    J = np.array([frcm_objective_from_V(X, V, m=m, T=T) for V in fireflies], dtype=float)

    for t in range(max_iter):
        alpha_t = alpha0 * (theta ** t)

        order = np.argsort(J)
        fireflies = fireflies[order]
        J = J[order]

        for i in range(n_fireflies):
            for j in range(n_fireflies):
                if J[j] < J[i]:
                    Xi = fireflies[i]
                    Xj = fireflies[j]

                    rij = np.linalg.norm((Xi - Xj).ravel())
                    beta = beta0 * np.exp(-gamma * (rij ** 2))

                    R = rng.random(size=Xi.shape)
                    Xi_new = Xi + beta * (Xj - Xi) + alpha_t * (R - 0.5)
                    Xi_new = clip_centroids(Xi_new)

                    J_new = frcm_objective_from_V(X, Xi_new, m=m, T=T)
                    if J_new < J[i]:
                        fireflies[i] = Xi_new
                        J[i] = J_new

    best_idx = int(np.argmin(J))
    V_best = fireflies[best_idx].copy()
    J_best = float(J[best_idx])
    return V_best, J_best

def silhouette_fa_frcm_best_of_runs(
    X, k_list,
    m=2.0, T=0.05,
    n_fireflies=20, max_iter=100,
    alpha0=1.0, theta=0.97,
    beta0=1.0, gamma=0.1,
    base_seed=42, n_runs=5
):
    rows = []
    for k in k_list:
        best = {"J": np.inf, "labels": None}
        for r in range(n_runs):
            seed = base_seed + r
            V_best, J_best = fa_frcm_optimize_centroids_return_bestV(
                X, c=k, m=m, T=T,
                n_fireflies=n_fireflies, max_iter=max_iter,
                alpha0=alpha0, theta=theta,
                beta0=beta0, gamma=gamma,
                seed=seed
            )
            if J_best < best["J"]:
                U = frcm_membership_from_V(X, V_best, m=m, T=T)
                labels = np.argmax(U, axis=1)
                best = {"J": J_best, "labels": labels}

        sil = safe_silhouette(X, best["labels"])
        rows.append({"k": k, "silhouette": sil, "J_min": float(best["J"])})
        print(f"[FA-FRCM] k={k} -> bestJ={best['J']:.6f} | silhouette={sil}")
    return pd.DataFrame(rows)

# 7) RUN ALL SILHOUETTE
df_sil_km = silhouette_kmeans(X, K_LIST, random_state=SEED, n_init=10, max_iter=300)

df_sil_fcm = silhouette_fcm_best_of_runs(
    X, K_LIST,
    m=M_FUZZY, eps=EPS, max_iter=MAX_ITER_FCM,
    base_seed=SEED, n_runs=N_RUNS
)

df_sil_frcm = silhouette_frcm_best_of_runs(
    X, K_LIST,
    m=M_FUZZY, T=T_BOUNDARY, eps=EPS, max_iter=MAX_ITER_FRCM,
    base_seed=SEED, n_runs=N_RUNS
)

df_sil_fa = silhouette_fa_frcm_best_of_runs(
    X, K_LIST,
    m=M_FUZZY, T=T_BOUNDARY,
    n_fireflies=N_FIREFLIES, max_iter=MAX_ITER_FA,
    alpha0=ALPHA0, theta=THETA, beta0=BETA0, gamma=GAMMA,
    base_seed=SEED, n_runs=N_RUNS
)

# 8) MERGE
merged = pd.DataFrame({"k": K_LIST})
merged["Sil_KMeans"] = df_sil_km["silhouette"].values
merged["Sil_FCM"] = df_sil_fcm["silhouette"].values
merged["Sil_FRCM"] = df_sil_frcm["silhouette"].values
merged["Sil_FAFRCM"] = df_sil_fa["silhouette"].values

print("\n=== SILHOUETTE (lebih besar lebih baik) ===")
print(merged)

def best_k_from_sil(series, k_list):
    arr = np.array(series, dtype=float)
    # abaikan nan
    mask = ~np.isnan(arr)
    if not mask.any():
        return None
    idx = np.argmax(arr[mask])
    return int(np.array(k_list)[mask][idx])

print("\n=== BEST k (silhouette max) ===")
print("KMeans  :", best_k_from_sil(merged["Sil_KMeans"], K_LIST))
print("FCM     :", best_k_from_sil(merged["Sil_FCM"], K_LIST))
print("FRCM    :", best_k_from_sil(merged["Sil_FRCM"], K_LIST))
print("FA-FRCM :", best_k_from_sil(merged["Sil_FAFRCM"], K_LIST))

# 9) PLOT (per metode)
def plot_sil(k, y, title):
    plt.figure()
    plt.plot(k, y, marker="o")
    plt.xlabel("Jumlah klaster (k)")
    plt.ylabel("Silhouette score")
    plt.title(title)
    plt.grid(True)
    plt.show()

k = merged["k"].tolist()
plot_sil(k, merged["Sil_KMeans"].tolist(), "Silhouette - KMeans")
plot_sil(k, merged["Sil_FCM"].tolist(), "Silhouette - FCM (best J dari runs)")
plot_sil(k, merged["Sil_FRCM"].tolist(), "Silhouette - FRCM (best J dari runs)")
plot_sil(k, merged["Sil_FAFRCM"].tolist(), "Silhouette - FA-FRCM (best J dari runs)")

# (opsional) simpan
merged.to_csv("silhouette_all_methods.csv", index=False)
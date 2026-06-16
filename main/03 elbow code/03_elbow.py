import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

DATA_PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"
K_LIST = list(range(2, 8))
N_RUNS = 5

# FCM/FRCM settings
M_FUZZY = 2.0
T_BOUNDARY = 0.09
EPS = 1e-5
MAX_ITER_FCM = 200
MAX_ITER_FRCM = 200

# FA settings (sesuai paper arXiv:2407.02537 dan hasil tuning)
N_FIREFLIES = 22          # dalam rentang 20-40
MAX_ITER_FA = 297         # dataset kecil, ambil batas bawah rekomendasi (100-1000)
ALPHA0 = 1.0
THETA = 0.982             # sesuai typical paper (alpha decay)
BETA0 = 1.312             # typical beta=1
GAMMA = 0.104             # typical gamma=0.1
SEED = 42

# LOAD DATA
df = pd.read_csv(DATA_PATH)
X = df.select_dtypes(include=[np.number]).to_numpy(dtype=float)
n, d = X.shape
print("Data shape:", X.shape)
print("Numeric columns:", list(df.select_dtypes(include=[np.number]).columns))

xmin = X.min(axis=0)
xmax = X.max(axis=0)

def euclidean_dist_matrix(X, V):
    return np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)


# (A) NUMERICAL ELBOW (tanpsi) - sesuai pseudocode paper arXiv
def tanpsi_from_curve(k_list, y_list):
    """
    Menghitung tanpsi(k) dari kurva y(k) dengan metode slope difference:
    slope1 = y(k) - y(k-1)
    slope2 = y(k+1) - y(k)
    Jika slope2 <= slope1 (drop makin besar / facing-downwards), tanpsi=0 (di-skip)
    Jika slope2 > slope1 (drop makin mengecil / mulai flatten), tanpsi dihitung:
        tanpsi = (slope1 - slope2) / (1 + slope1*slope2)
    """
    k_arr = np.array(k_list, dtype=int)
    y_arr = np.array(y_list, dtype=float)

    tanpsi = np.zeros_like(y_arr, dtype=float)

    # butuh minimal 3 titik untuk menghitung tanpsi
    if len(k_arr) < 3:
        return tanpsi

    for i in range(1, len(k_arr) - 1):
        slope1 = y_arr[i] - y_arr[i - 1]
        slope2 = y_arr[i + 1] - y_arr[i]

        # slope biasanya negatif untuk kurva menurun
        if slope2 <= slope1:
            tanpsi[i] = 0.0
        else:
            denom = 1.0 + (slope1 * slope2)
            # jaga-jaga denom mendekati 0
            if np.isclose(denom, 0.0):
                tanpsi[i] = 0.0
            else:
                tanpsi[i] = (slope1 - slope2) / denom

    tanpsi[0] = 0.0  #ujung kurva tidak dihitung
    tanpsi[-1] = 0.0  #ujung kurva tidak dihitung
    return tanpsi

def pick_k_by_tanpsi(k_list, y_list):
    tanpsi = tanpsi_from_curve(k_list, y_list)
    valid_idx = np.where(tanpsi < 0)[0]

    if len(valid_idx) == 0:
        diffs = np.abs(np.diff(y_list))
        best = int(k_list[int(np.argmax(diffs))])
        return best, tanpsi

    best_i = valid_idx[np.argmin(tanpsi[valid_idx])]
    return int(k_list[best_i]), tanpsi

# (B) KMeans curve (SSE / inertia)
def elbow_kmeans(X, k_list, random_state=42, n_init=10, max_iter=200):
    rows = []
    for k in k_list:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init, max_iter=max_iter)
        km.fit(X)
        rows.append({"k": k, "J": float(km.inertia_)})
    return pd.DataFrame(rows)

# (C) FCM
def fcm_fit(X, c, m=2.0, eps=1e-5, max_iter=200, seed=42):
    rng = np.random.default_rng(seed)
    n, d = X.shape

    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)

    last_V = None
    for _ in range(max_iter):
        Um = U ** m
        V = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)

        if last_V is not None and np.linalg.norm(V - last_V) <= eps:
            break
        last_V = V

        D = euclidean_dist_matrix(X, V) + 1e-12
        power = 2.0 / (m - 1.0)
        ratio = (D[:, :, None] / D[:, None, :]) ** power
        U = 1.0 / ratio.sum(axis=2)

    D2 = euclidean_dist_matrix(X, V) ** 2
    J = np.sum((U ** m) * D2)
    return float(J)

def elbow_fcm_min(X, k_list, m=2.0, eps=1e-5, max_iter=200, base_seed=42, n_runs=5):
    rows = []
    for k in k_list:
        Js = []
        for r in range(n_runs):
            seed = base_seed + r
            Js.append(fcm_fit(X, c=k, m=m, eps=eps, max_iter=max_iter, seed=seed))
        rows.append({"k": k, "J": float(np.min(Js)), "J_mean": float(np.mean(Js))})
    return pd.DataFrame(rows)

# (D) FRCM
def frcm_membership_from_V(X, V, m=2.0, T=0.09):
    D = euclidean_dist_matrix(X, V) + 1e-12

    order = np.argsort(D, axis=1)
    q1 = order[:, 0]
    q2 = order[:, 1]

    d1 = D[np.arange(X.shape[0]), q1]
    d2 = D[np.arange(X.shape[0]), q2]

    is_boundary = np.abs(d2 - d1) <= T
    U = np.zeros((X.shape[0], V.shape[0]), dtype=float)

    idx_lower = np.where(~is_boundary)[0]
    U[idx_lower, q1[idx_lower]] = 1.0

    idx_b = np.where(is_boundary)[0]
    if len(idx_b) > 0:
        Db = D[idx_b]
        power = 2.0 / (m - 1.0)
        ratio = (Db[:, :, None] / Db[:, None, :]) ** power
        Ub = 1.0 / ratio.sum(axis=2)
        U[idx_b] = Ub

    return U

def frcm_objective_from_V(X, V, m=2.0, T=0.09):
    U = frcm_membership_from_V(X, V, m=m, T=T)
    D2 = euclidean_dist_matrix(X, V) ** 2
    return float(np.sum((U ** m) * D2))

def frcm_fit(X, c, m=2.0, T=0.09, eps=1e-5, max_iter=200, seed=42):
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

    return frcm_objective_from_V(X, V, m=m, T=T)

def elbow_frcm_min(X, k_list, m=2.0, T=0.09, eps=1e-5, max_iter=200, base_seed=42, n_runs=5):
    rows = []
    for k in k_list:
        Js = []
        for r in range(n_runs):
            seed = base_seed + r
            Js.append(frcm_fit(X, c=k, m=m, T=T, eps=eps, max_iter=max_iter, seed=seed))
        rows.append({"k": k, "J": float(np.min(Js)), "J_mean": float(np.mean(Js))})
    return pd.DataFrame(rows)

# (E) FA–FRCM
def fa_frcm_optimize_centroids(
    X, c, m=2.0, T=0.09,
    n_fireflies=22, max_iter=297,
    alpha0=1.0, theta=0.982,
    beta0=1.312, gamma=0.104,
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

                    J_new = frcm_objective_from_V(X, Xi_new, m=m, T=T)
                    if J_new < J[i]:
                        fireflies[i] = Xi_new
                        J[i] = J_new

    return float(np.min(J))

def elbow_fa_frcm_min(
    X, k_list, m=2.0, T=0.09,
    n_fireflies=22, max_iter=297,
    alpha0=1.0, theta=0.982,
    beta0=1.312, gamma=0.104,
    base_seed=42, n_runs=5
):
    rows = []
    for k in k_list:
        Js = []
        for r in range(n_runs):
            seed = base_seed + r
            Js.append(
                fa_frcm_optimize_centroids(
                    X, c=k, m=m, T=T,
                    n_fireflies=n_fireflies, max_iter=max_iter,
                    alpha0=alpha0, theta=theta,
                    beta0=beta0, gamma=gamma,
                    seed=seed
                )
            )
        rows.append({"k": k, "J": float(np.min(Js)), "J_mean": float(np.mean(Js))})
        print(f"[FA-FRCM] k={k} -> minJ={np.min(Js):.6f} | meanJ={np.mean(Js):.6f}")
    return pd.DataFrame(rows)

# RUN: compute curves
df_km   = elbow_kmeans(X, K_LIST, random_state=SEED)
df_fcm  = elbow_fcm_min(X, K_LIST, m=M_FUZZY, eps=EPS, max_iter=MAX_ITER_FCM, base_seed=SEED, n_runs=N_RUNS)
df_frcm = elbow_frcm_min(X, K_LIST, m=M_FUZZY, T=T_BOUNDARY, eps=EPS, max_iter=MAX_ITER_FRCM, base_seed=SEED, n_runs=N_RUNS)
df_fa   = elbow_fa_frcm_min(X, K_LIST, m=M_FUZZY, T=T_BOUNDARY,
                            n_fireflies=N_FIREFLIES, max_iter=MAX_ITER_FA,
                            alpha0=ALPHA0, theta=THETA, beta0=BETA0, gamma=GAMMA,
                            base_seed=SEED, n_runs=N_RUNS)

merged = pd.DataFrame({"k": K_LIST})
merged["KMeans_J"]       = df_km["J"].values
merged["FCM_J_min"]      = df_fcm["J"].values
merged["FRCM_J_min"]     = df_frcm["J"].values
merged["FAFRCM_J_min"]   = df_fa["J"].values

print("\n=== MERGED (kurva objektif per k) ===")
print(merged)

# NUMERICAL ELBOW: pick k* per curve
k_km, tan_km     = pick_k_by_tanpsi(merged["k"].tolist(), merged["KMeans_J"].tolist())
k_fcm, tan_fcm   = pick_k_by_tanpsi(merged["k"].tolist(), merged["FCM_J_min"].tolist())
k_frcm, tan_frcm = pick_k_by_tanpsi(merged["k"].tolist(), merged["FRCM_J_min"].tolist())
k_fa, tan_fa     = pick_k_by_tanpsi(merged["k"].tolist(), merged["FAFRCM_J_min"].tolist())

print("\n=== NUMERICAL ELBOW (tanpsi min) ===")
print(f"KMeans  k* = {k_km}")
print(f"FCM     k* = {k_fcm}")
print(f"FRCM    k* = {k_frcm}")
print(f"FA-FRCM k* = {k_fa}")

merged["tanpsi_KMeans"]  = tan_km
merged["tanpsi_FCM"]     = tan_fcm
merged["tanpsi_FRCM"]    = tan_frcm
merged["tanpsi_FAFRCM"]  = tan_fa

print("\n=== TABEL TANPSI (untuk bukti elbow numerik) ===")
print(merged[["k", "tanpsi_KMeans", "tanpsi_FCM", "tanpsi_FRCM", "tanpsi_FAFRCM"]])

# PLOT: kurva + tanpsi
def plot_curve_and_tanpsi(k, y, tanpsi, title, ylabel):
    plt.figure()
    plt.plot(k, y, marker="o")
    plt.xlabel("Jumlah klaster (k)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(k, tanpsi, marker="o")
    plt.xlabel("Jumlah klaster (k)")
    plt.ylabel("tan(psi)")
    plt.title(title + " - Numerical Elbow (tanpsi)")
    plt.grid(True)
    plt.show()

k = merged["k"].tolist()
plot_curve_and_tanpsi(k, merged["KMeans_J"].tolist(), tan_km, "KMeans", "WCSS / inertia")
plot_curve_and_tanpsi(k, merged["FCM_J_min"].tolist(), tan_fcm, "FCM (min J)", "J")
plot_curve_and_tanpsi(k, merged["FRCM_J_min"].tolist(), tan_frcm, "FRCM (min J)", "J")
plot_curve_and_tanpsi(k, merged["FAFRCM_J_min"].tolist(), tan_fa, "FA-FRCM (min J)", "J")

# simpan hasil
merged.to_csv("elbow_numeric_tanpsi_all_methods.csv", index=False)
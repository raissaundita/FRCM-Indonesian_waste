import os
import time
import numpy as np
import pandas as pd

PATH = "D:/Raissa/Python/PREPROCE/05_scaled_provinsi.csv"
OUT_DIR = "D:/Raissa/Python/KLASTER DATA/FCM_K4"

c = 4
N_RUNS = 100
SEED_BASE = 42
m = 2.0
epsilon = 1e-5
tau = 200

year_col = "Tahun"
id_col = "Provinsi"

os.makedirs(OUT_DIR, exist_ok=True)

def compute_objective(X, V, U, m):
    dist_sq = np.sum((X[:, None, :] - V[None, :, :]) ** 2, axis=2)
    return np.sum((U ** m) * dist_sq)

def fcm(X, c, m, epsilon, tau, seed=None):
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # Inisialisasi membership awal
    U = rng.random((n, c))
    U = U / U.sum(axis=1, keepdims=True)

    V_prev = None
    n_iter_used = 0
    objective_history = []

    for t in range(1, tau + 1):
        n_iter_used = t

        # Persamaan 2.5: hitung pusat klaster
        Um = U ** m
        V = (Um.T @ X) / (Um.sum(axis=0)[:, None] + 1e-12)

        # Persamaan 2.6: update membership
        dist = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2) + 1e-12

        for j in range(n):
            for i in range(c):
                denom = 0.0
                for k in range(c):
                    denom += (dist[j, i] / dist[j, k]) ** (2.0 / (m - 1.0))
                U[j, i] = 1.0 / denom

        # Hitung objective
        obj = compute_objective(X, V, U, m)
        objective_history.append(obj)

        # Cek konvergensi
        if V_prev is not None:
            if np.linalg.norm(V - V_prev) <= epsilon:
                break

        V_prev = V.copy()

    return U, V, n_iter_used, objective_history

# LOAD DATA
df = pd.read_csv(PATH)

X = df.drop(columns=[year_col, id_col])
X = X.select_dtypes(include=[np.number]).to_numpy(dtype=float)
X = np.nan_to_num(X)

log_rows = []
grand_start = time.perf_counter()

for r in range(1, N_RUNS + 1):
    seed = SEED_BASE + r

    start = time.perf_counter()
    U, V, n_iter_used, objective_history = fcm(X, c, m, epsilon, tau, seed=seed)
    end = time.perf_counter()

    total_time = end - start
    avg_time_per_iter = total_time / n_iter_used if n_iter_used > 0 else np.nan

    labels = np.argmax(U, axis=1)

    # File label
    out = df[[year_col, id_col]].copy()
    out["label"] = labels

    out_path = os.path.join(OUT_DIR, f"fcm_k{c}_run{r}.csv")
    out.to_csv(out_path, index=False)
    print("saved:", out_path)

    # File membership
    mem_out = df[[year_col, id_col]].copy()
    for i in range(c):
        mem_out[f"u_cluster_{i}"] = U[:, i]
    mem_out["label"] = labels

    mem_path = os.path.join(OUT_DIR, f"fcm_membership_k{c}_run{r}.csv")
    mem_out.to_csv(mem_path, index=False)
    print("saved:", mem_path)

    # File objective history
    obj_out = pd.DataFrame({
        "iterasi": np.arange(1, len(objective_history) + 1),
        "objective_value": objective_history
    })

    obj_path = os.path.join(OUT_DIR, f"fcm_objective_k{c}_run{r}.csv")
    obj_out.to_csv(obj_path, index=False)
    print("saved:", obj_path)

    label_aktif = sorted(np.unique(labels).tolist())
    counts = np.bincount(labels, minlength=c)

    log_rows.append({
        "run": r,
        "seed": seed,
        "n_iter_internal": n_iter_used,
        "total_time_sec": total_time,
        "avg_time_per_iter_sec": avg_time_per_iter,
        "final_objective": objective_history[-1] if len(objective_history) > 0 else np.nan,
        "n_label_aktif": len(label_aktif),
        "label_aktif": ",".join(map(str, label_aktif)),
        "n_label_0": int(counts[0]),
        "n_label_1": int(counts[1]),
        "n_label_2": int(counts[2]),
        "n_label_3": int(counts[3]),
    #    "n_label_4": int(counts[4]),
    #    "n_label_5": int(counts[5]),
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
#    "n_label_4": "",
#    "n_label_5": "",
}])

rekap_df = pd.concat([rekap_df, total_row], ignore_index=True)

rekap_path = os.path.join(OUT_DIR, f"fcm_k{c}_rekap_100run.csv")
rekap_df.to_csv(rekap_path, index=False)
print("saved:", rekap_path)
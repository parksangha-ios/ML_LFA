import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

N_SAMPLES = 5000
N_POINTS  = 256
V_MAX_CUT = 0.98

L_MIN = 0.01
L_MAX = 6.0

ALPHA_MIN = 0.01      # [mm^2/s]
ALPHA_MAX = 2000.0    # [mm^2/s]


# ------------ 더 안정적인 Parker 수식 (odd n만 사용) ------------
def parker_curve(alpha_mm2s, L_mm, t_s, n_terms=200):
    """
    Parker 이론식: 
    V(t) = 1 - (8/π^2) Σ_{m=0}^{∞} (1/(2m+1)^2) exp(-(2m+1)^2 π^2 Fo)
    Fo = α t / L^2
    """
    Fo = alpha_mm2s * t_s[None, :] / (L_mm**2)   # shape: (1, T)
    m = np.arange(0, n_terms, dtype=np.float64)[:, None]   # (n_terms, 1)
    n = 2.0*m + 1.0                               # odd: 1,3,5,...
    expo = np.exp(-(n**2) * (np.pi**2) * Fo)      # (n_terms, T)
    terms = expo / (n**2)
    V = 1.0 - (8.0/(np.pi**2)) * terms.sum(axis=0)
    return V


def generate_parker_sample(alpha_mm2s, L_mm,
                           n_time_raw=2000,
                           v_max_cut=V_MAX_CUT):

    # 이론적인 t_1/2
    t_half = 0.1388 * (L_mm**2) / alpha_mm2s
    t_end  = 5.0 * t_half

    # t=0에서 시작하지 않고, 아주 작은 양수에서 시작
    t_start = t_end / n_time_raw   # 혹은 1e-6 같은 값
    t_raw   = np.linspace(t_start, t_end, n_time_raw, dtype=np.float64)

    V_raw = parker_curve(alpha_mm2s, L_mm, t_raw)

    # V >= v_max_cut 지점에서 잘라줌
    idx = np.where(V_raw >= v_max_cut)[0]
    if idx.size > 0:
        i_cut = idx[0]
        t_cut = t_raw[: i_cut + 1]
        V_cut = V_raw[: i_cut + 1]
    else:
        t_cut = t_raw
        V_cut = V_raw

    # τ = α t / L^2
    tau_raw = (alpha_mm2s * t_cut) / (L_mm**2)
    tau_max = float(tau_raw[-1])

    # 만약 tau_max가 너무 작으면(수치적 문제 방지용) 약간 올려줌
    if tau_max <= 0:
        tau_max = 1e-6

    tau_grid = np.linspace(0.0, tau_max, N_POINTS)
    V_resampled = np.interp(tau_grid, tau_raw, V_cut)

    return tau_grid, V_resampled, tau_max


rows = []

for i in range(N_SAMPLES):
    # α: log-uniform 샘플링
    log_a = rng.uniform(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX))
    alpha_mm2s = 10.0 ** log_a

    # L: uniform 샘플링
    L_mm = rng.uniform(L_MIN, L_MAX)

    tau_grid, V_vec, tau_max = generate_parker_sample(alpha_mm2s, L_mm)

    row = {f"V_{j}": V_vec[j] for j in range(N_POINTS)}
    row["alpha_mm2s"] = alpha_mm2s
    row["L_mm"]       = L_mm
    row["tau_max"]    = tau_max
    rows.append(row)

df = pd.DataFrame(rows)

csv_name = "C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/parker_dataset_alpha0.01_to_2000.csv"
df.to_csv(csv_name, index=False, encoding="utf-8")

print("CSV 생성 완료:", csv_name)
print("shape:", df.shape)
print(df.head())

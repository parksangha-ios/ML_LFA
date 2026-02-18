import numpy as np
import pandas as pd

# -----------------------------------
# Parker (rear-face) solution
# -----------------------------------
def parker_theta(Fo, n_terms=100):
    Fo = np.asarray(Fo)
    theta = np.ones_like(Fo, dtype=np.float64)
    coef = 8.0 / (np.pi ** 2)

    s = np.zeros_like(Fo, dtype=np.float64)
    for n in range(n_terms):
        m = 2 * n + 1
        s += np.exp(-(m ** 2) * (np.pi ** 2) * Fo / 4.0) / (m ** 2)
    theta = 1.0 - coef * s
    return theta   # 0~1 값

# -----------------------------------
# 데이터 생성 파라미터
# -----------------------------------
N_SAMPLES = 5000
N_TIME = 200
Fo_max = 1.0
rng = np.random.default_rng(42)

# alpha 범위 (log-uniform)
alpha_log_min = -2.0      # 10^-2 = 0.01
alpha_log_max = 3.5       # 약 3162

# 두께 범위(mm → m)
L_min = 0.5e-3
L_max = 5e-3

# 시간축(Fo)
Fo_grid = np.linspace(0, Fo_max, N_TIME)

# Parker V(Fo) 기본 커브
base_V = parker_theta(Fo_grid)

# -----------------------------------
# 데이터 생성 반복
# -----------------------------------
rows = []

for i in range(N_SAMPLES):
    # 1) alpha, L 샘플링
    log_alpha = rng.uniform(alpha_log_min, alpha_log_max)
    alpha = 10 ** log_alpha
    L = rng.uniform(L_min, L_max)

    # 2) t 계산: t = Fo * L^2 / alpha
    t = Fo_grid * (L ** 2) / alpha
    t_safe = np.where(t <= 0, 1e-12, t)
    t_log = np.log10(t_safe)

    # 3) V(Fo) + noise
    noise = rng.normal(0, 0.003, N_TIME)
    V = np.clip(base_V + noise, 0, 1)

    # 4) L을 log 스케일로 저장
    L_log = np.log10(L)

    # 5) Row record 만들기
    row = {}

    # V(t) 저장
    for k in range(N_TIME):
        row[f"V_{k}"] = V[k]

    # t_log 저장
    for k in range(N_TIME):
        row[f"tlog_{k}"] = t_log[k]

    # 두께 / 알파
    row["L_log"] = L_log
    row["alpha_log"] = log_alpha

    rows.append(row)

# -----------------------------------
# DataFrame → CSV 저장
# -----------------------------------
df = pd.DataFrame(rows)

csv_path = "C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/lfa_fullinput_dataset.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

print("CSV 파일 생성 완료:", csv_path)
print("데이터프레임 shape:", df.shape)

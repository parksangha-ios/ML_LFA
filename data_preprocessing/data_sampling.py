import numpy as np
import pandas as pd

# -----------------------------------
# Parker (rear-face) solution
# -----------------------------------
def parker_theta(Fo, n_terms=100):
    Fo = np.asarray(Fo)
    coef = 8.0 / (np.pi ** 2)

    s = np.zeros_like(Fo, dtype=np.float64)
    for n in range(n_terms):
        m = 2 * n + 1
        s += np.exp(-(m ** 2) * (np.pi ** 2) * Fo / 4.0) / (m ** 2)
    theta = 1.0 - coef * s
    return theta   # 0~1 값

# -----------------------------------
# 물리 파라미터 범위 (mm, mm^2/s 기준)
# -----------------------------------
ALPHA_MIN = 0.01      # [mm^2/s]
ALPHA_MAX = 2000.0    # [mm^2/s]
L_MIN     = 0.01      # [mm]
L_MAX     = 6.0       # [mm]

# -----------------------------------
# 그리드 해상도 설정 (100 x 100 = 10,000 샘플)
# -----------------------------------
N_ALPHA = 100   # log-alpha 방향 그리드 개수
N_L     = 100   # log-L 방향 그리드 개수
N_TIME  = 200   # 시간축 포인트 개수

# log10(alpha), log10(L)에서 균등 그리드
alpha_log_grid = np.linspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), N_ALPHA)
L_log_grid     = np.linspace(np.log10(L_MIN),     np.log10(L_MAX),     N_L)

# -----------------------------------
# 시간축(Fourier number) 설정
# -----------------------------------
Fo_max = 1.0
Fo_grid = np.linspace(0, Fo_max, N_TIME)

# Parker 기본 커브
base_V = parker_theta(Fo_grid)

rng = np.random.default_rng(42)
rows = []

# -----------------------------------
# 균등 그리드 기반 데이터 생성 루프
# -----------------------------------
for log_alpha in alpha_log_grid:
    alpha_mm2 = 10 ** log_alpha          # [mm^2/s]
    alpha_m2  = alpha_mm2 * 1e-6         # [m^2/s] 변환

    for log_L in L_log_grid:
        L_mm = 10 ** log_L               # [mm]
        L_m  = L_mm * 1e-3               # [m]

        # t = Fo * L^2 / alpha (모두 SI 단위)
        t = Fo_grid * (L_m ** 2) / alpha_m2
        t = np.where(t <= 0, 1e-12, t)
        tlog = np.log10(t)

        # V(Fo) + noise (필요 없으면 noise=0으로 바꿔도 됨)
        noise = rng.normal(0, 0.003, N_TIME)
        V = np.clip(base_V + noise, 0, 1)

        row = {}

        # V 저장
        for k in range(N_TIME):
            row[f"V_{k}"] = V[k]

        # log t 저장
        for k in range(N_TIME):
            row[f"tlog_{k}"] = tlog[k]

        # 두께 / 알파 (log10 값으로 저장)
        row["L_log"] = log_L
        row["alpha_log"] = log_alpha

        rows.append(row)

# -----------------------------------
# DataFrame → CSV 저장
# -----------------------------------
df = pd.DataFrame(rows)

csv_path = "C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/lfa_uniform_100x100_dataset.csv"
df.to_csv(csv_path, index=False, encoding="utf-8")

print("CSV 파일 생성 완료:", csv_path)
print("데이터프레임 shape:", df.shape)

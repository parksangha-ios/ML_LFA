import numpy as np
import pandas as pd
import time

# -----------------------------------
# 1. Parker Solution (Fo 기반, 기존 형식 그대로)
# -----------------------------------
def parker_theta(Fo, n_terms=100):
    Fo = np.asarray(Fo)
    coef = 8.0 / (np.pi ** 2)
    s = np.zeros_like(Fo, dtype=np.float64)
    for n in range(n_terms):
        m = 2 * n + 1
        s += np.exp(-(m ** 2) * (np.pi ** 2) * Fo / 4.0) / (m ** 2)
    theta = 1.0 - coef * s
    return theta   # 0~1 값 (이상적 Parker 곡선)

# -----------------------------------
# 2. 파라미터 및 그리드 설정
# -----------------------------------
ALPHA_MIN = 0.01      # [mm^2/s]
ALPHA_MAX = 2000.0    # [mm^2/s]
L_MIN     = 0.01      # [mm]
L_MAX     = 6.0       # [mm]

# log10 범위 계산 (클리핑용)
ALPHA_LOG_MIN = np.log10(ALPHA_MIN)
ALPHA_LOG_MAX = np.log10(ALPHA_MAX)
L_LOG_MIN     = np.log10(L_MIN)
L_LOG_MAX     = np.log10(L_MAX)

N_ALPHA = 25   # log-alpha 방향 등분 (grid center 개수)
N_L     = 20   # log-L 방향 등분
N_TIME  = 50   # 그래프 당 포인트 수

# ideal / heatloss 비율 (그래프 개수 기준)
N_IDEAL_PER     = 3   # 각 (alpha,L) 조합당 ideal 그래프 개수
N_HEATLOSS_PER  = 7   # 각 (alpha,L) 조합당 heatloss 그래프 개수

# Heat loss 계수 범위
LOSS_MIN = 0.1
LOSS_MAX = 3.0

# 노이즈 크기
NOISE_STD = 0.003

# ★ jitter 범위 (log10 공간에서)
#   예) ±0.05 → 선형 스케일에서는 약 ×10^±0.05 ≈ ×1.12 배
J_ALPHA = 0.05
J_L     = 0.05

# Log-Uniform Grid (중심점)
alpha_log_grid = np.linspace(ALPHA_LOG_MIN, ALPHA_LOG_MAX, N_ALPHA)
L_log_grid     = np.linspace(L_LOG_MIN,     L_LOG_MAX,     N_L)

# 시간축 (Fourier Number)
Fo_max = 1.0
Fo_grid = np.linspace(0, Fo_max, N_TIME)
base_V = parker_theta(Fo_grid)  # Fo에 대한 이상적 Parker 곡선 (alpha, L과 무관)

# -----------------------------------
# 3. 데이터 생성 루프 (Long Format, ideal + heatloss + jitter)
# -----------------------------------
total_graphs = N_ALPHA * N_L * (N_IDEAL_PER + N_HEATLOSS_PER)

print("데이터 생성 시작...")
print(f"- alpha log grid: {N_ALPHA}개")
print(f"- L     log grid: {N_L}개")
print(f"- 그래프당 포인트: {N_TIME}")
print(f"- ideal/heatloss 비율 (그래프 수 기준): {N_IDEAL_PER}:{N_HEATLOSS_PER}")
print(f"- 총 그래프 개수(ideal+heatloss): {total_graphs}")
print(f"- 예상 총 행 수: {total_graphs * N_TIME}")
start_time = time.time()

rng = np.random.default_rng(42)
all_data = []
experiment_id = 0

for log_alpha_center in alpha_log_grid:
    for log_L_center in L_log_grid:

        # --------- (1) ideal 그래프들 생성 (jitter 포함) ---------
        for _ in range(N_IDEAL_PER):
            # log-space jitter
            log_alpha_j = log_alpha_center + rng.uniform(-J_ALPHA, J_ALPHA)
            log_L_j     = log_L_center     + rng.uniform(-J_L,     J_L)

            # 전체 범위를 벗어나지 않게 클리핑
            log_alpha_j = np.clip(log_alpha_j, ALPHA_LOG_MIN, ALPHA_LOG_MAX)
            log_L_j     = np.clip(log_L_j,     L_LOG_MIN,     L_LOG_MAX)

            alpha_mm2 = 10 ** log_alpha_j
            alpha_m2  = alpha_mm2 * 1e-6

            L_mm = 10 ** log_L_j
            L_m  = L_mm * 1e-3

            # 물리 시간 t 계산 (Fo -> t)
            t = Fo_grid * (L_m ** 2) / alpha_m2
            t = np.where(t <= 0, 1e-12, t)
            tlog = np.log10(t)

            # 이상 Parker + 노이즈
            noise = rng.normal(0, NOISE_STD, N_TIME)
            V_raw = np.clip(base_V + noise, 0, None)

            # 곡선별 정규화 (max = 1)
            vmax = np.max(V_raw)
            if vmax > 0:
                V = V_raw / vmax
            else:
                V = V_raw

            ids         = np.full(N_TIME, experiment_id, dtype=int)
            time_idx    = np.arange(N_TIME, dtype=int)
            L_vals      = np.full(N_TIME, log_L_j,     dtype=np.float64)
            alpha_vals  = np.full(N_TIME, log_alpha_j, dtype=np.float64)
            loss_vals   = np.full(N_TIME, 0.0,         dtype=np.float64)  # ideal

            graph_data = np.column_stack(
                [ids, time_idx, V, tlog, L_vals, alpha_vals, loss_vals]
            )
            all_data.append(graph_data)
            experiment_id += 1

        # --------- (2) heat loss 그래프들 생성 (jitter 포함) ---------
        for _ in range(N_HEATLOSS_PER):
            # log-space jitter
            log_alpha_j = log_alpha_center + rng.uniform(-J_ALPHA, J_ALPHA)
            log_L_j     = log_L_center     + rng.uniform(-J_L,     J_L)

            log_alpha_j = np.clip(log_alpha_j, ALPHA_LOG_MIN, ALPHA_LOG_MAX)
            log_L_j     = np.clip(log_L_j,     L_LOG_MIN,     L_LOG_MAX)

            alpha_mm2 = 10 ** log_alpha_j
            alpha_m2  = alpha_mm2 * 1e-6

            L_mm = 10 ** log_L_j
            L_m  = L_mm * 1e-3

            # 물리 시간 t 계산 (Fo -> t)
            t = Fo_grid * (L_m ** 2) / alpha_m2
            t = np.where(t <= 0, 1e-12, t)
            tlog = np.log10(t)

            # heat loss factor 샘플링
            loss_factor = rng.uniform(LOSS_MIN, LOSS_MAX)

            decay_term = np.exp(-loss_factor * t)
            V_loss_raw = base_V * decay_term

            noise = rng.normal(0, NOISE_STD, N_TIME)
            V_raw = np.clip(V_loss_raw + noise, 0, None)

            vmax = np.max(V_raw)
            if vmax > 0:
                V = V_raw / vmax
            else:
                V = V_raw

            ids         = np.full(N_TIME, experiment_id, dtype=int)
            time_idx    = np.arange(N_TIME, dtype=int)
            L_vals      = np.full(N_TIME, log_L_j,     dtype=np.float64)
            alpha_vals  = np.full(N_TIME, log_alpha_j, dtype=np.float64)
            loss_vals   = np.full(N_TIME, loss_factor, dtype=np.float64)

            graph_data = np.column_stack(
                [ids, time_idx, V, tlog, L_vals, alpha_vals, loss_vals]
            )
            all_data.append(graph_data)
            experiment_id += 1

# -----------------------------------
# 4. CSV 저장
# -----------------------------------
final_array = np.vstack(all_data)

columns = [
    "Expt_ID",      # 그래프 ID
    "Time_Step",    # 0 ~ N_TIME-1
    "V",            # 정규화된 온도 신호 (0~1)
    "t_log",        # log10(t [s])
    "L_log",        # log10(L [mm])
    "alpha_log",    # log10(alpha [mm^2/s])
    "loss_factor"   # heat loss 계수 (ideal은 0)
]

df = pd.DataFrame(final_array, columns=columns)

df["Expt_ID"]   = df["Expt_ID"].astype(int)
df["Time_Step"] = df["Time_Step"].astype(int)

save_path = "C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/lfa_mixed_jitter_ideal_heatloss_50pt.csv"
df.to_csv(save_path, index=False, encoding="utf-8")

end_time = time.time()
print("------------------------------------------------")
print(f"CSV 파일 생성 완료: {save_path}")
print(f"총 소요 시간: {end_time - start_time:.2f}초")
print(f"데이터 Shape: {df.shape}")
print("ideal 그래프 개수:", N_ALPHA * N_L * N_IDEAL_PER)
print("heatloss 그래프 개수:", N_ALPHA * N_L * N_HEATLOSS_PER)
print("ideal : heatloss 비율 =", N_IDEAL_PER, ":", N_HEATLOSS_PER)
print("데이터 미리보기:")
print(df.head())

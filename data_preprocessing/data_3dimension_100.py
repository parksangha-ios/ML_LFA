import numpy as np
import pandas as pd
import time

# -----------------------------------
# 1. Parker Solution (물리 식)
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
# 2. 파라미터 및 그리드 설정
#    목표: 500개 그래프 x 200 포인트 = 100,000 행
# -----------------------------------
ALPHA_MIN = 0.01      # [mm^2/s]
ALPHA_MAX = 2000.0    # [mm^2/s]
L_MIN     = 0.01      # [mm]
L_MAX     = 6.0       # [mm]

# 500개의 조합을 만들기 위한 분할 (25 x 20 = 500)
N_ALPHA = 25   # log-alpha 방향 등분
N_L     = 20   # log-L 방향 등분
N_TIME  = 500  # 그래프 당 포인트 수

# Log-Uniform Grid 생성 (편차 없는 데이터 생성을 위함)
alpha_log_grid = np.linspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), N_ALPHA)
L_log_grid     = np.linspace(np.log10(L_MIN),     np.log10(L_MAX),     N_L)

# 시간축 (Fourier Number)
Fo_max = 1.0
Fo_grid = np.linspace(0, Fo_max, N_TIME)
base_V = parker_theta(Fo_grid) # 이상적인 기본 곡선

# -----------------------------------
# 3. 데이터 생성 루프 (Long Format)
# -----------------------------------
print(f"데이터 생성 시작... (목표: {N_ALPHA * N_L}개 그래프, 총 {N_ALPHA * N_L * N_TIME}행)")
start_time = time.time()

rng = np.random.default_rng(42)
all_data = []

experiment_id = 0 # 그래프별 고유 ID

for log_alpha in alpha_log_grid:
    # Alpha 값 변환
    alpha_mm2 = 10 ** log_alpha
    alpha_m2  = alpha_mm2 * 1e-6 

    for log_L in L_log_grid:
        # L 값 변환
        L_mm = 10 ** log_L
        L_m  = L_mm * 1e-3

        # (1) 시간 t 계산: t = Fo * L^2 / alpha
        t = Fo_grid * (L_m ** 2) / alpha_m2
        t = np.where(t <= 0, 1e-12, t) # log(0) 방지
        tlog = np.log10(t)

        # (2) 온도 V 계산 (노이즈 추가)
        noise = rng.normal(0, 0.003, N_TIME)
        V = np.clip(base_V + noise, 0, 1)

        # (3) 데이터를 묶어서 저장 (Batch 처리)
        # 루프를 200번 도는 것보다 numpy 배열로 한 번에 만드는 게 훨씬 빠름
        
        # 그래프 ID (이 200개는 같은 실험임)
        ids = np.full(N_TIME, experiment_id, dtype=int)
        
        # Time Index (0~199)
        time_idx = np.arange(N_TIME, dtype=int)
        
        # 상수값들 (L_log, alpha_log는 200번 동안 똑같음)
        L_vals = np.full(N_TIME, log_L)
        alpha_vals = np.full(N_TIME, log_alpha)

        # 하나의 그래프(200행)를 DataFrame으로 임시 생성 or 리스트에 추가
        # 여기서는 속도를 위해 리스트에 튜플/배열 형태로 저장
        # 컬럼 순서: [Expt_ID, Time_Idx, V, t_log, L_log, alpha_log]
        graph_data = np.column_stack((ids, time_idx, V, tlog, L_vals, alpha_vals))
        all_data.append(graph_data)

        experiment_id += 1

# -----------------------------------
# 4. CSV 저장
# -----------------------------------
# 전체 데이터를 하나의 거대한 배열로 합침
final_array = np.vstack(all_data)

columns = ["Expt_ID", "Time_Step", "V", "t_log", "L_log", "alpha_log"]
df = pd.DataFrame(final_array, columns=columns)

# ID와 Time_Step은 정수형으로 변환 (깔끔하게 보이기 위해)
df["Expt_ID"] = df["Expt_ID"].astype(int)
df["Time_Step"] = df["Time_Step"].astype(int)

save_path = "C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/ideal_time_500.csv"
df.to_csv(save_path, index=False, encoding="utf-8")

end_time = time.time()
print("------------------------------------------------")
print(f"CSV 파일 생성 완료: {save_path}")
print(f"총 소요 시간: {end_time - start_time:.2f}초")
print(f"데이터 Shape: {df.shape}")
print("------------------------------------------------")
print("데이터 미리보기:")
print(df.head())
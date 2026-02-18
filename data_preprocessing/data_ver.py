import numpy as np
import pandas as pd
import time
import os

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
# -----------------------------------
ALPHA_MIN = 0.01      # [mm^2/s]
ALPHA_MAX = 2000.0    # [mm^2/s]
L_MIN     = 0.01      # [mm]
L_MAX     = 6.0       # [mm]

# 목표: 500개 그래프 (25 x 20)
N_ALPHA = 25   
N_L     = 20   
N_TIME  = 200  

# 기본 Grid 생성
alpha_log_grid = np.linspace(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX), N_ALPHA)
L_log_grid     = np.linspace(np.log10(L_MIN),     np.log10(L_MAX),     N_L)

# Grid 간격 계산 (Jittering을 위해)
d_alpha = (alpha_log_grid[1] - alpha_log_grid[0]) if N_ALPHA > 1 else 0
d_L     = (L_log_grid[1] - L_log_grid[0]) if N_L > 1 else 0

Fo_max = 1.0
Fo_grid = np.linspace(0, Fo_max, N_TIME)
base_V = parker_theta(Fo_grid)

# 저장할 폴더 생성
output_dir = "C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/datasets_v10"
os.makedirs(output_dir, exist_ok=True)

# -----------------------------------
# 3. 10개 버전 데이터 생성 루프
# -----------------------------------
NUM_VERSIONS = 10

print(f"총 {NUM_VERSIONS}개의 데이터셋 생성을 시작합니다.")
total_start = time.time()

for version in range(1, NUM_VERSIONS + 1):
    print(f"\n[Version {version}/{NUM_VERSIONS}] 생성 중...")
    
    # 버전마다 다른 Seed 사용 -> 노이즈와 파라미터 값이 달라짐
    rng = np.random.default_rng(42 + version)
    
    all_data = []
    experiment_id = 0

    # Grid Loop
    for base_log_alpha in alpha_log_grid:
        for base_log_L in L_log_grid:
            
            # ---------------------------------------------------------
            # [핵심] Parameter Jittering (파라미터 미세 변동)
            # 격자(Grid)의 정규화된 분포는 유지하되, 값을 랜덤하게 살짝 흔듦
            # 모델이 특정 숫자를 암기하는 것을 방지
            # ---------------------------------------------------------
            jitter_alpha = rng.uniform(-d_alpha/3, d_alpha/3)
            jitter_L     = rng.uniform(-d_L/3, d_L/3)
            
            curr_log_alpha = base_log_alpha + jitter_alpha
            curr_log_L     = base_log_L + jitter_L
            
            # 범위 벗어나지 않게 Clip
            curr_log_alpha = np.clip(curr_log_alpha, np.log10(ALPHA_MIN), np.log10(ALPHA_MAX))
            curr_log_L     = np.clip(curr_log_L, np.log10(L_MIN), np.log10(L_MAX))

            # 실제 물리값 변환
            alpha_mm2 = 10 ** curr_log_alpha
            alpha_m2  = alpha_mm2 * 1e-6 
            
            L_mm = 10 ** curr_log_L
            L_m  = L_mm * 1e-3

            # (1) 시간 t 계산
            t = Fo_grid * (L_m ** 2) / alpha_m2
            t = np.where(t <= 0, 1e-12, t)
            tlog = np.log10(t)

            # (2) V 계산 (노이즈 추가 - 시드에 따라 달라짐)
            noise = rng.normal(0, 0.003, N_TIME)
            V = np.clip(base_V + noise, 0, 1)

            # (3) 데이터 묶기
            ids = np.full(N_TIME, experiment_id, dtype=int)
            time_idx = np.arange(N_TIME, dtype=int)
            L_vals = np.full(N_TIME, curr_log_L)
            alpha_vals = np.full(N_TIME, curr_log_alpha)

            # [Expt_ID, Time_Step, V, t_log, L_log, alpha_log]
            graph_data = np.column_stack((ids, time_idx, V, tlog, L_vals, alpha_vals))
            all_data.append(graph_data)

            experiment_id += 1

    # DataFrame 생성 및 저장
    final_array = np.vstack(all_data)
    columns = ["Expt_ID", "Time_Step", "V", "t_log", "L_log", "alpha_log"]
    df = pd.DataFrame(final_array, columns=columns)
    
    df["Expt_ID"] = df["Expt_ID"].astype(int)
    df["Time_Step"] = df["Time_Step"].astype(int)

    # 파일명: lfa_dataset_v1.csv, lfa_dataset_v2.csv ...
    file_name = f"lfa_dataset_v{version}.csv"
    save_path = os.path.join(output_dir, file_name)
    
    df.to_csv(save_path, index=False)
    print(f"  -> 저장 완료: {file_name} (Shape: {df.shape})")

total_end = time.time()
print("------------------------------------------------")
print(f"모든 작업 완료! 총 소요 시간: {total_end - total_start:.2f}초")
print(f"파일 저장 경로: {output_dir}")
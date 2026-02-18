import numpy as np
import pandas as pd
import time

# -----------------------------------
# 1. Parker Solution (Fo 기반)
# -----------------------------------
def parker_theta(Fo, n_terms=50):
    # 계산 속도를 위해 n_terms 약간 조정 (50이면 충분)
    Fo = np.asarray(Fo)
    coef = 8.0 / (np.pi ** 2)
    s = np.zeros_like(Fo, dtype=np.float64)
    
    # Fo가 0에 가까우면 계산이 불안정할 수 있으므로 클리핑
    # (Fo가 너무 작으면 theta=0)
    valid_mask = Fo > 1e-6
    Fo_valid = Fo[valid_mask]
    
    if len(Fo_valid) > 0:
        s_val = np.zeros_like(Fo_valid)
        for n in range(n_terms):
            m = 2 * n + 1
            s_val += np.exp(-(m ** 2) * (np.pi ** 2) * Fo_valid / 4.0) / (m ** 2)
        
        # 원래 배열에 값 할당
        out = np.zeros_like(Fo)
        out[valid_mask] = 1.0 - coef * s_val
        return np.clip(out, 0, 1) # 0~1 사이로 강제
    else:
        return np.zeros_like(Fo)

# -----------------------------------
# 2. 파라미터 설정
# -----------------------------------
# 물리적 물성 범위
ALPHA_MIN = 0.01       # [mm^2/s] (범위 넓힘)
ALPHA_MAX = 2000.0    # [mm^2/s]
L_MIN     = 0.01       # [mm]
L_MAX     = 6.0       # [mm]

# Log Scale Grid
ALPHA_LOG_MIN = np.log10(ALPHA_MIN)
ALPHA_LOG_MAX = np.log10(ALPHA_MAX)
L_LOG_MIN     = np.log10(L_MIN)
L_LOG_MAX     = np.log10(L_MAX)

# 생성할 데이터 규모
N_ALPHA = 30   # Alpha 그리드
N_L     = 20   # L 그리드
N_TIME  = 100  # 그래프 해상도를 좀 더 높임 (50 -> 100 권장)

# 그래프 개수 비율
N_IDEAL_PER    = 2
N_HEATLOSS_PER = 8  # 현실성을 위해 Heatloss 비율 높임

# Heat loss 계수
LOSS_MIN = 0.0
LOSS_MAX = 5.0 # 좀 더 강한 loss까지 포함

# 노이즈 및 Jitter
NOISE_STD = 0.01  # 노이즈를 조금 더 현실적으로 키움 (1%)
J_ALPHA = 0.05    # 물성 자체의 랜덤성
J_L     = 0.02

# -----------------------------------
# 3. 데이터 생성 루프
# -----------------------------------
alpha_log_grid = np.linspace(ALPHA_LOG_MIN, ALPHA_LOG_MAX, N_ALPHA)
L_log_grid     = np.linspace(L_LOG_MIN,     L_LOG_MAX,     N_L)

all_data = []
experiment_id = 0
rng = np.random.default_rng(42)

print("현실적인 데이터 생성 시작...")
start_time = time.time()

for log_alpha_c in alpha_log_grid:
    for log_L_c in L_log_grid:
        
        # 이 (alpha, L) 조합에 대해 몇 개의 그래프를 만들 것인가
        total_runs = N_IDEAL_PER + N_HEATLOSS_PER
        
        for i in range(total_runs):
            is_heatloss = (i >= N_IDEAL_PER)
            
            # 1. 물성 Jittering (샘플의 고유 속성)
            log_a = log_alpha_c + rng.uniform(-J_ALPHA, J_ALPHA)
            log_l = log_L_c     + rng.uniform(-J_L, J_L)
            
            # 물리량 변환
            alpha_val = 10 ** log_a # [mm^2/s]
            L_val     = 10 ** log_l # [mm]
            
            # SI 단위 변환 (계산용)
            alpha_si = alpha_val * 1e-6 # m^2/s
            L_si     = L_val * 1e-3     # m

            # ---------------------------------------------------------
            # ★ 핵심 수정: 현실적인 시간축(t_grid) 설정 ★
            # ---------------------------------------------------------
            # Half-rise time (이론적 반감기) = 1.38 * L^2 / (pi^2 * alpha)
            # 대략 t_char = L^2 / alpha 라고 볼 때,
            # 실험자가 측정 시간을 짧게 잡을 수도(0.5배), 길게 잡을 수도(3.0배) 있게 설정.
            # 이렇게 하면 t_max가 alpha와 '느슨한' 관계만 가지고, 정확한 비례식은 깨집니다.
            
            t_characteristic = (L_si ** 2) / alpha_si
            
            # "실험자가 설정한 측정 종료 시간" (랜덤성 부여)
            # 그래프가 다 올라가기도 전에 끊기거나(0.5), 아주 길게(3.0) 찍힐 수 있음
            time_factor = rng.uniform(0.5, 3.5) 
            t_max = t_characteristic * time_factor
            
            # 시간축 생성 (0 ~ t_max 등간격)
            t_grid = np.linspace(0, t_max, N_TIME)
            
            # t_grid에 0이 포함되면 log10에서 에러나므로 아주 작은 값으로 대체
            t_grid[0] = 1e-9 
            
            # ---------------------------------------------------------
            # 2. 온도(V) 계산
            # ---------------------------------------------------------
            # 현재 시간축에 맞는 Fourier Number 계산
            # Fo = alpha * t / L^2
            Fo_grid = alpha_si * t_grid / (L_si ** 2)
            
            # Parker 식 적용 (여기서 V가 결정됨)
            V_theoretical = parker_theta(Fo_grid)
            
            # Heat Loss 적용
            loss_factor = 0.0
            if is_heatloss:
                # Loss 계수 랜덤
                loss_factor = rng.uniform(LOSS_MIN, LOSS_MAX)
                # Cowan 등 모델에서의 Decay 항: exp(-loss * t_rel) 형태 모사
                # 간단히 시간 비례 감쇠 적용 (Cape-Lehman 근사)
                # t가 커질수록 신호가 떨어짐
                
                # 정규화된 시간 개념 도입 (Fo 기준)하여 감쇠 적용
                decay = np.exp(-loss_factor * Fo_grid)
                V_theoretical = V_theoretical * decay
            
            # 3. 노이즈 추가
            noise = rng.normal(0, NOISE_STD, N_TIME)
            V_noisy = V_theoretical + noise
            
            # 4. 정규화 (Normalization) - 중요!
            # 실험 데이터는 보통 최대값을 1로 맞추거나 전압 값을 그대로 씀.
            # 여기서는 Max Normalization 수행
            v_max_val = np.max(V_noisy)
            if v_max_val > 1e-3: # 신호가 너무 작으면(거의 0) 정규화 스킵
                V_final = V_noisy / v_max_val
            else:
                V_final = V_noisy # 그냥 0 근처 노이즈임
            
            # ---------------------------------------------------------
            # 4. 데이터 패키징
            # ---------------------------------------------------------
            # 모델 입력용 Log 값들
            t_log = np.log10(t_grid)
            L_log_repeat = np.full(N_TIME, np.log10(L_val))
            
            # 정답 (Target)
            alpha_log_repeat = np.full(N_TIME, np.log10(alpha_val))
            
            # 메타 데이터
            loss_repeat = np.full(N_TIME, loss_factor)
            id_repeat   = np.full(N_TIME, experiment_id)
            step_repeat = np.arange(N_TIME)
            
            # 저장 (ID, TimeStep, V, t_log, L_log, alpha_log, loss)
            stacked = np.column_stack((
                id_repeat,
                step_repeat,
                V_final,
                t_log,
                L_log_repeat,
                alpha_log_repeat,
                loss_repeat
            ))
            
            all_data.append(stacked)
            experiment_id += 1

# -----------------------------------
# 4. CSV 저장
# -----------------------------------
final_data = np.vstack(all_data)
df = pd.DataFrame(final_data, columns=[
    "Expt_ID", "Time_Step", "V", "t_log", "L_log", "alpha_log", "loss_factor"
])

# 데이터 타입 정리
df["Expt_ID"] = df["Expt_ID"].astype(int)
df["Time_Step"] = df["Time_Step"].astype(int)

# 파일명에 'realistic' 태그 붙임
save_path = "C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/lfa_realistic_fixed_time_100pt.csv"
df.to_csv(save_path, index=False)

end_time = time.time()
print("------------------------------------------------")
print(f"생성 완료: {save_path}")
print(f"총 소요 시간: {end_time - start_time:.2f}초")
print(f"총 그래프 개수: {experiment_id}")
print(f"데이터 Shape: {df.shape}")
print("------------------------------------------------")
print("데이터 예시:")
print(df.head())
import numpy as np
import pandas as pd
import time

# -----------------------------------
# 1. Parker Solution
# -----------------------------------
def parker_theta(Fo, n_terms=50):
    Fo = np.asarray(Fo)
    coef = 8.0 / (np.pi ** 2)
    valid_mask = Fo > 1e-6
    Fo_valid = Fo[valid_mask]
    if len(Fo_valid) > 0:
        s_val = np.zeros_like(Fo_valid)
        for n in range(n_terms):
            m = 2 * n + 1
            s_val += np.exp(-(m ** 2) * (np.pi ** 2) * Fo_valid / 4.0) / (m ** 2)
        out = np.zeros_like(Fo)
        out[valid_mask] = 1.0 - coef * s_val
        return np.clip(out, 0, 1)
    else:
        return np.zeros_like(Fo)

# -----------------------------------
# 2. 데이터 생성 함수 (Pure Random Sampling)
# -----------------------------------
def generate_and_save_csv(target_n_time):
    print(f"\n[작업 시작] N_TIME = {target_n_time} 포인트 데이터 생성 중...")
    
    # --- 파라미터 ---
    ALPHA_MIN, ALPHA_MAX = 0.01, 2000.0
    L_MIN, L_MAX         = 0.01, 6.0
    
    ALPHA_LOG_MIN = np.log10(ALPHA_MIN)
    ALPHA_LOG_MAX = np.log10(ALPHA_MAX)
    L_LOG_MIN     = np.log10(L_MIN)
    L_LOG_MAX     = np.log10(L_MAX)
    
    # 총 데이터 개수 목표 설정
    # (기존: 30 * 20 * 10 = 6000개 였으므로, 비슷하게 6000개로 설정)
    TOTAL_SAMPLES = 6000 
    
    N_TIME = target_n_time 
    
    # Ideal : Heatloss 비율 (2 : 8)
    PROB_HEATLOSS = 0.8 
    
    LOSS_MIN, LOSS_MAX = 0.0, 5.0
    NOISE_STD = 0.01

    rng = np.random.default_rng(42)
    start_time = time.time()
    
    all_data = []

    # --- ★ 핵심 변경: Grid Loop 제거 -> 통짜 랜덤 생성 ★ ---
    # 로그 공간에서 완벽하게 균일하게 6000개를 뿌림 (빈틈 없음)
    log_alpha_samples = rng.uniform(ALPHA_LOG_MIN, ALPHA_LOG_MAX, TOTAL_SAMPLES)
    log_L_samples     = rng.uniform(L_LOG_MIN,     L_LOG_MAX,     TOTAL_SAMPLES)
    
    # 각 샘플이 Heatloss를 가질지 말지 결정 (0 ~ 1 사이 난수 < 0.8 이면 Heatloss)
    is_heatloss_samples = rng.random(TOTAL_SAMPLES) < PROB_HEATLOSS
    
    # 루프: 미리 뽑은 랜덤 좌표 하나씩 처리
    for i in range(TOTAL_SAMPLES):
        experiment_id = i
        
        # 1. 물성 설정
        log_a = log_alpha_samples[i]
        log_l = log_L_samples[i]
        is_heatloss = is_heatloss_samples[i]
        
        alpha_val = 10 ** log_a
        L_val     = 10 ** log_l
        
        alpha_si = alpha_val * 1e-6
        L_si     = L_val * 1e-3
        
        # 2. 현실적인 시간축 설정
        t_characteristic = (L_si ** 2) / alpha_si
        time_factor = rng.uniform(0.5, 3.5)
        t_max = t_characteristic * time_factor
        
        t_grid = np.linspace(0, t_max, N_TIME)
        t_grid[0] = 1e-9
        
        # 3. Parker + Heatloss
        Fo_grid = alpha_si * t_grid / (L_si ** 2)
        V_theoretical = parker_theta(Fo_grid)
        
        loss_factor = 0.0
        if is_heatloss:
            loss_factor = rng.uniform(LOSS_MIN, LOSS_MAX)
            decay = np.exp(-loss_factor * Fo_grid)
            V_theoretical = V_theoretical * decay
        
        # 4. 노이즈 + 정규화
        noise = rng.normal(0, NOISE_STD, N_TIME)
        V_noisy = V_theoretical + noise
        
        v_max_val = np.max(V_noisy)
        if v_max_val > 1e-3:
            V_final = V_noisy / v_max_val
        else:
            V_final = V_noisy
        
        # 5. 패키징
        t_log = np.log10(t_grid)
        L_log_repeat = np.full(N_TIME, log_l) # 저장된 log_l 사용
        alpha_log_repeat = np.full(N_TIME, log_a) # 저장된 log_a 사용
        loss_repeat = np.full(N_TIME, loss_factor)
        id_repeat   = np.full(N_TIME, experiment_id)
        step_repeat = np.arange(N_TIME)
        
        stacked = np.column_stack((
            id_repeat, step_repeat, V_final, t_log, 
            L_log_repeat, alpha_log_repeat, loss_repeat
        ))
        all_data.append(stacked)

    # --- 저장 ---
    final_data = np.vstack(all_data)
    df = pd.DataFrame(final_data, columns=[
        "Expt_ID", "Time_Step", "V", "t_log", "L_log", "alpha_log", "loss_factor"
    ])
    
    df["Expt_ID"] = df["Expt_ID"].astype(int)
    df["Time_Step"] = df["Time_Step"].astype(int)
    
    save_path = f"C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/lfa_realistic_fixed_time_autojitter_{N_TIME}pt.csv"
    df.to_csv(save_path, index=False)
    
    end_time = time.time()
    print(f"-> 저장 완료: {save_path}")

# -----------------------------------
# 3. 실행
# -----------------------------------
target_points = [200, 100, 50, 30]

print(f"총 {len(target_points)}개의 CSV 파일을 생성합니다: {target_points}")
for pts in target_points:
    generate_and_save_csv(pts)
print("\n모든 파일 생성 완료!")
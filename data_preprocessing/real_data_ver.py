import numpy as np
import pandas as pd
import time

# -----------------------------------
# 1. Parker Solution (Fo 기반)
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
# 2. 데이터 생성 함수 정의
# -----------------------------------
def generate_and_save_csv(target_n_time):
    print(f"\n[작업 시작] N_TIME = {target_n_time} 포인트 데이터 생성 중...")
    
    # --- 파라미터 설정 (고정) ---
    ALPHA_MIN, ALPHA_MAX = 0.01, 2000.0
    L_MIN, L_MAX         = 0.01, 6.0
    
    ALPHA_LOG_MIN = np.log10(ALPHA_MIN)
    ALPHA_LOG_MAX = np.log10(ALPHA_MAX)
    L_LOG_MIN     = np.log10(L_MIN)
    L_LOG_MAX     = np.log10(L_MAX)
    
    N_ALPHA = 30
    N_L     = 20
    
    # ★ 여기서 함수 인자로 받은 target_n_time을 사용
    N_TIME = target_n_time 
    
    N_IDEAL_PER    = 2
    N_HEATLOSS_PER = 8
    
    LOSS_MIN, LOSS_MAX = 0.0, 5.0
    NOISE_STD = 0.01
    J_ALPHA, J_L = 0.05, 0.02
    
    alpha_log_grid = np.linspace(ALPHA_LOG_MIN, ALPHA_LOG_MAX, N_ALPHA)
    L_log_grid     = np.linspace(L_LOG_MIN,     L_LOG_MAX,     N_L)
    
    all_data = []
    experiment_id = 0
    rng = np.random.default_rng(42) # 시드 고정 (재현성)

    start_time = time.time()
    
    # --- 생성 루프 ---
    for log_alpha_c in alpha_log_grid:
        for log_L_c in L_log_grid:
            
            total_runs = N_IDEAL_PER + N_HEATLOSS_PER
            
            for i in range(total_runs):
                is_heatloss = (i >= N_IDEAL_PER)
                
                # 1. Jittering
                log_a = log_alpha_c + rng.uniform(-J_ALPHA, J_ALPHA)
                log_l = log_L_c     + rng.uniform(-J_L, J_L)
                
                # 물리량 변환
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
                
                # ★★★ [수정] loss_factor 초기화 (이 부분이 빠져서 에러 발생했음) ★★★
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
                L_log_repeat = np.full(N_TIME, np.log10(L_val))
                alpha_log_repeat = np.full(N_TIME, np.log10(alpha_val))
                
                # 여기서 loss_factor가 정의되어 있어야 함 (위에서 초기화 완료)
                loss_repeat = np.full(N_TIME, loss_factor)
                id_repeat   = np.full(N_TIME, experiment_id)
                step_repeat = np.arange(N_TIME)
                
                stacked = np.column_stack((
                    id_repeat, step_repeat, V_final, t_log, 
                    L_log_repeat, alpha_log_repeat, loss_repeat
                ))
                all_data.append(stacked)
                experiment_id += 1

    # --- 저장 ---
    final_data = np.vstack(all_data)
    df = pd.DataFrame(final_data, columns=[
        "Expt_ID", "Time_Step", "V", "t_log", "L_log", "alpha_log", "loss_factor"
    ])
    
    df["Expt_ID"] = df["Expt_ID"].astype(int)
    df["Time_Step"] = df["Time_Step"].astype(int)
    
    # 파일명에 포인트 수 자동 기입
    save_path = f"C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/lfa_realistic_fixed_time_{N_TIME}pt.csv"
    df.to_csv(save_path, index=False)
    
    end_time = time.time()
    print(f"-> 저장 완료: {save_path}")
    print(f"-> 소요 시간: {end_time - start_time:.2f}초")

# -----------------------------------
# 3. 실행 (한 번에 4개 생성)
# -----------------------------------
target_points = [200, 100, 50, 30]  # 생성하고 싶은 포인트 목록

print(f"총 {len(target_points)}개의 CSV 파일을 생성합니다: {target_points}")

for pts in target_points:
    generate_and_save_csv(pts)

print("\n 모든 파일 생성 완료!")
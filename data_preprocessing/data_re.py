import numpy as np
import pandas as pd

# 재현성용 시드
SEED = 42
rng = np.random.default_rng(SEED)

# 샘플 개수 / 포인트 수
N_SAMPLES = 5000      # 필요하면 10000 등으로 늘리면 됩니다
N_POINTS  = 256       # 곡선 한 개의 샘플링 포인트 수

# 시간축: 모든 샘플에서 공통으로 쓰는 t-grid [초]
T_MAX   = 1.0         # LFA에서 보고 싶은 최대 시간(초) - 필요시 조정
t_grid = np.linspace(0.0, T_MAX, N_POINTS, dtype=np.float64)

# LFA 장비 스펙 기반 범위
ALPHA_MIN = 0.01      # [mm^2/s]
ALPHA_MAX = 2000.0    # [mm^2/s]
L_MIN     = 0.01      # [mm]
L_MAX     = 6.0       # [mm]

def parker_curve_t(alpha_mm2s, L_mm, t_s, n_terms=500):
    """
    Parker 이상 모델 (adiabatic, 1D slab)의 후면 온도 상승 정규화 곡선.
    V(t) = 1 - (8/pi^2) * sum_{m=0}^{∞} [ 1/(2m+1)^2 * exp(-(2m+1)^2 pi^2 * alpha t / L^2) ]
    """
    # Fo = α t / L^2, 하지만 여기서는 내부에서만 사용 (외부로 노출 X)
    Fo = alpha_mm2s * t_s[None, :] / (L_mm**2)     # shape: (1, T)

    m = np.arange(0, n_terms, dtype=np.float64)[:, None]   # 0,1,2,... (n_terms x 1)
    n = 2.0*m + 1.0                                        # 1,3,5,... 홀수항

    expo  = np.exp(-(n**2) * (np.pi**2) * Fo)              # (n_terms x T)
    terms = expo / (n**2)

    V = 1.0 - (8.0 / (np.pi**2)) * terms.sum(axis=0)       # (T,)
    return V

rows = []

for i in range(N_SAMPLES):
    # α: 로그 스케일에서 균일하게 샘플링 (저/고 확산도 모두 고르게)
    log_a = rng.uniform(np.log10(ALPHA_MIN), np.log10(ALPHA_MAX))
    alpha = 10.0 ** log_a

    # L: 선형으로 샘플링
    L = rng.uniform(L_MIN, L_MAX)

    # Parker 곡선 계산 (t_grid 공통)
    V = parker_curve_t(alpha, L, t_grid, n_terms=500)

    # 수치적인 이유로 약간 클리핑
    V = np.clip(V, 0.0, 1.2)

    # 필요하다면 첫 포인트를 정확히 0으로 맞추고 싶을 때:
    # V = V - V[0]
    # V = np.clip(V, 0.0, 1.2)

    # 한 행(row) 구성
    row = {f"V_{j}": float(V[j]) for j in range(N_POINTS)}
    row["alpha_mm2s"] = float(alpha)
    row["L_mm"]       = float(L)

    rows.append(row)

# 데이터프레임으로 변환
df = pd.DataFrame(rows)

csv_path = "C:/Users/Heum/Desktop/Sang Ha Park/CODE/ML_LFA/parker_dataset_tgrid_alpha0.01_to_2000.csv"

df.to_csv(csv_path, index=False, encoding="utf-8")

print("CSV 생성 완료:", csv_path)
print("shape:", df.shape)
print(df.head())

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# --- 1. User님의 검증된 Parker 모델 (Ideal Rise) ---
def parker_theta(alpha, L_mm, t, n_terms=50):
    """
    User Provided Parker Model
    t: 시간 배열 (s)
    alpha: 열확산도 (mm^2/s)
    L_mm: 두께 (mm)
    """
    # Fourier Number 계산 (Fo = alpha * t / L^2)
    # L은 mm, alpha는 mm^2/s 이므로 단위 일치
    Fo = alpha * t / (L_mm ** 2)
    
    # Parker 수식 (Rising Curve 0 -> 1)
    # theta = 1 - (8/pi^2) * sum( exp(...) )
    
    coef = 8.0 / (np.pi ** 2)
    s = np.zeros_like(Fo, dtype=np.float64)
    
    for n in range(n_terms):
        m = 2 * n + 1
        # User 코드의 Fo/4.0 부분은 수식 정의에 따라 다르나, 
        # 일반적인 Parker 식(pi^2 * Fo)으로 맞추려면 아래가 표준입니다.
        # User님 코드를 존중하되, 물리적 거동 확인 후 표준형으로 조정했습니다.
        # 표준 Parker: exp( - m^2 * pi^2 * Fo )
        exponent = -(m ** 2) * (np.pi ** 2) * Fo 
        s += np.exp(exponent) / (m ** 2)
        
    theta = 1.0 - coef * s
    
    # 음수 값 보정 (초기 계산 오차)
    theta = np.maximum(theta, 0)
    return theta

# --- 2. 열손실이 적용된 LFA 데이터 생성기 ---
def generate_lfa_data(alpha, L_mm, loss_factor, t_max):
    """
    alpha: mm^2/s
    L_mm: mm
    loss_factor: 열손실 계수 (Clark-Taylor model concept)
                 0이면 Ideal, 값이 클수록 후반부에 뚝 떨어짐
    """
    # 시간 축 생성
    t = np.linspace(0.001, t_max, 500)
    
    # 1. 이상적인 상승 곡선 (Parker)
    T_ideal = parker_theta(alpha, L_mm, t)
    
    # 2. 열손실 적용 (Physics: Radiation/Convection Decay)
    # Clark-Taylor Model: T_measured = T_ideal * exp(-c * t)
    # 시간이 지날수록 열손실 효과가 누적되어 온도가 떨어짐
    decay_term = np.exp(-loss_factor * t)
    
    T_loss = T_ideal * decay_term
    
    # 정규화 (최대값을 1로 맞춤 - LFA 분석 시 필수)
    if np.max(T_loss) > 0:
        T_loss = T_loss / np.max(T_loss)
        
    return t, T_loss, T_ideal

# --- 3. Streamlit UI ---
st.title("LFA Data Generator (Corrected)")
st.markdown("Parker 식(Rising) & 열손실(Decay) 결합한 모델")

col1, col2 = st.columns([1, 2])

with col1:
    alpha_in = st.number_input("Alpha (mm²/s)", 0.01, 2000.0, 0.5)
    L_in = st.number_input("Thickness (mm)", 0.01, 6.0, 1.0)
    # 열손실 계수 조절
    loss_in = st.slider("Heat Loss Factor", 0.0, 5.0, 0.5, 
                        help="0: Ideal Parker (상승만 함)\n값이 클수록 뒤가 떨어짐")
    t_max_in = st.slider("Max Time (s)", 0.1, 2.0, 1.0)

with col2:
    t_data, T_loss, T_ideal = generate_lfa_data(alpha_in, L_in, loss_in, t_max_in)
    
    fig, ax = plt.subplots()
    # Ideal (Parker)
    ax.plot(t_data, T_ideal, label="Parker (Ideal Rise)", linestyle='--', color='gray', alpha=0.7)
    # Heat Loss
    ax.plot(t_data, T_loss, label=f"With Heat Loss (Factor={loss_in})", color='red', linewidth=2)
    
    ax.set_title("Generated LFA Curve")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Normalized Temperature")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
# LFA Formula Research Notes (Goal 2)

## 1) 당신이 공유한 현재 지표 해석

공유 지표:
- val_acc = 1.0
- val_macro_f1 = 1.0
- alpha_mape_mean = 0.190
- alpha_mape_median = 0.154
- alpha_mape_p90 = 0.409

해석:
- **분류 모델 성능은 매우 우수**합니다(현재 synthetic 분포 기준 완전 분리).
- **회귀(alpha)도 실무적으로 양호**합니다(특히 median 기준).
- 다만 `p90`가 mean/median 대비 높아 tail error가 존재하므로, 히스토그램/산점도 기반 확인이 필수입니다.

## 2) 핵심 식 정리

### Parker (Ideal, adiabatic)
\[
\alpha = 0.1388 \frac{L^2}{t_{1/2}}
\]

- `L`: thickness
- `t_{1/2}`: 후면 온도상승이 최대 상승의 1/2에 도달하는 시간

### Cowan / Cape-Lehman / Clark-Taylor

실무에서는 공통적으로 **Parker를 기반으로 보정 인자**를 곱하는 형태를 사용합니다.

\[
\alpha = \frac{L^2}{t_{1/2}} \cdot F(r)
\]

- `r`: 곡선 형상 비율(예: `t_0.75 / t_0.25`, pulse 관련 비율 등)
- `F(r)`: 방법별 다항식/보정 함수

> 중요한 점: Cowan, Cape-Lehman, Clark-Taylor의 계수는 사용하는 표준/문헌 판본/장비 구현(ASTM, vendor software)에 따라 달라질 수 있어, **확정 계수는 반드시 선택한 기준 문헌/장비 매뉴얼과 1:1 매칭**해야 합니다.

## 3) 코드 반영 원칙

이번 저장소에는 `lfa_corrections.py`를 추가해 다음을 구현했습니다.

1. Parker 확정식 함수
2. Ratio 기반 보정식의 공통 인터페이스
3. Cowan/Cape-Lehman/Clark-Taylor용 확장 포인트 (계수 외부 주입)

이렇게 하면, 장비(LFA 467)와 논문에서 확정한 계수를 넣는 즉시 재학습/재평가가 가능합니다.

## 4) LFA 467( NETZSCH ) 반영 가이드

- alpha/thickness min-max는 현재 설정 유지
- 다만 보정식 계수는 다음 우선순위로 확정 권장:
  1) NETZSCH LFA 467 application note
  2) ASTM E1461 관련 구현식
  3) 선택한 원전 논문의 식(판본 명시)

## 5) 논문 작성용 체크리스트

- 데이터셋 분리: synthetic-internal / synthetic-OOD / real-experiment
- 보고 지표: class별 macro-F1, alpha MAPE median/p90, outlier rate
- 시각화: confusion matrix + alpha error histogram + true-vs-pred(log-log)
- 재현성: seed 고정, config 파일(계수 버전 포함), commit hash 기록

## 6) 참고할 대표 문헌 키워드

- Parker flash method thermal diffusivity
- Cowan heat loss correction flash method
- Cape Lehman finite pulse correction
- Clark Taylor flash method correction
- ASTM E1461 laser flash thermal diffusivity

> 현재 실행 환경에서는 외부 사이트 직접 수집이 제한될 수 있으므로, 최종 계수는 사용 중인 장비 문서와 논문 PDF 원문으로 교차검증 후 확정하세요.

# ML_LFA

PyTorch 기반 Laser Flash Analysis(LFA) 실험 코드입니다.

## Goal 2용 멀티태스크 학습 스크립트

`mtl_heatloss_pytorch.py`는 다음을 한 번에 학습합니다.

- **분류(Classification)**: Parker / Cowan-like / Cape-Lehman-like (3-class, softmax+cross entropy)
- **회귀(Regression)**: `alpha_log = log10(alpha_mm2/s)` 예측
- **보정(Regression)**: heat-loss 강도(`loss_strength`) 예측

입력은 `(V(t), log10(t), log10(L))`이며, 출력은
`curve_type`, `alpha_log`, `loss_strength` 입니다.

### 학습 실행

```bash
uv run python mtl_heatloss_pytorch.py --epochs 25 --n-samples 24000 --n-points 128
```

빠른 검증:

```bash
uv run python mtl_heatloss_pytorch.py --epochs 3 --n-samples 4000 --n-points 96
```

- Jupyter/IPython에서도 `parse_known_args()`로 실행 가능
- 학습 로그는 `history_mtl_heatloss.json`
- 최고 모델은 `best_mtl_heatloss.pt`

## 진단(성능 + 데이터 밸런스 + 차원 리스크)

`goal2_diagnostics.py`는 아래를 자동 계산합니다.

1. **데이터 밸런스**
   - 클래스 카운트
   - 클래스 균형 비율(min/max)
   - `alpha_log`, `l_log` 히스토그램 CV
2. **차원성 리스크(차원의 저주 점검)**
   - 샘플/피처 비율
   - PCA 95% 누적 분산 필요 차원
   - Effective rank
3. **검증 성능**
   - Accuracy, Macro-F1, Confusion Matrix
   - Alpha MAPE(mean/median/p90)

실행:

```bash
uv run python goal2_diagnostics.py --n-samples 6000 --n-points 128 --checkpoint best_mtl_heatloss.pt
```

결과는 `goal2_diagnostic_report.json`로 저장됩니다.
예측 상세는 `goal2_predictions.csv`로 저장됩니다.


### 시각화 리포트 생성

```bash
uv run python goal2_visualize.py   --history history_mtl_heatloss.json   --report goal2_diagnostic_report.json   --pred goal2_predictions.csv   --out-dir artifacts
```

생성 파일:
- `artifacts/goal2_history_curves.png`
- `artifacts/goal2_confusion_matrix.png`
- `artifacts/goal2_alpha_error_hist.png`
- `artifacts/goal2_alpha_scatter.png`

## 연구 단계 권장 로드맵

1. Surrogate(Cowan-like/Cape-like)로 파이프라인 안정화
2. 실제 Cowan/Cape 해석식/수치해석 데이터 생성기로 교체
3. 실험 raw 시계열(장비/시편별)로 파인튜닝 + 외부 검증셋 분리
4. 논문 지표: per-material, per-thickness, per-temperature stratified error 보고


## 보정식 모듈화 (Cowan/Cape-Lehman/Clark-Taylor)

- `lfa_corrections.py`
  - `parker_alpha()` : 확정 Parker 식
  - `RatioCorrection` : 방법별 보정 함수를 다항식 형태로 주입하는 공통 인터페이스
- `LFA_FORMULA_RESEARCH_NOTES.md`
  - Goal2 지표 해석, 보정식 구조, 장비(LFA 467) 반영 원칙 정리

주의: Cowan/Cape-Lehman/Clark-Taylor의 수치 계수는 사용 표준/장비 구현에 따라 달라질 수 있으므로,
반드시 선택한 원문/장비 문서 기준으로 확정 후 주입하세요.

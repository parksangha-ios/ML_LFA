# ML_LFA

PyTorch 기반 Laser Flash Analysis(LFA) 실험 코드입니다.

## Goal 2용 멀티태스크 학습 스크립트

`mtl_heatloss_pytorch.py`는 다음을 한 번에 학습합니다.

- **분류(Classification)**: Parker / Cowan-like / Cape-Lehman-like (3-class, softmax+cross entropy)
- **회귀(Regression)**: `alpha_log = log10(alpha_mm2/s)` 예측
- **보정(Regression)**: heat-loss 강도(`loss_strength`) 예측

입력은 `(V(t), log10(t), log10(L))`이며, 출력은
`curve_type`, `alpha_log`, `loss_strength` 입니다.

### 실행

```bash
uv run python mtl_heatloss_pytorch.py --epochs 25 --n-samples 24000 --n-points 128
```

빠른 검증은 아래처럼 실행할 수 있습니다.

```bash
uv run python mtl_heatloss_pytorch.py --epochs 3 --n-samples 4000 --n-points 96
```

학습이 끝나면 최고 검증 성능 모델을 `best_mtl_heatloss.pt`로 저장합니다.

## 참고

- Cowan / Cape-Lehman은 문헌의 완전한 폐형식 해를 직접 구현하기보다,
  연구용 파이프라인 검증을 위해 **물리적으로 타당한 surrogate 생성기**로 구현했습니다.
- 실제 논문용 최종 모델에서는 실험 장비/샘플 조건(방사율, Biot 수, 펄스폭, 두께 편차 등)에 맞춘
  정식 해석식/수치해석 기반 데이터 생성기로 교체하는 것을 권장합니다.

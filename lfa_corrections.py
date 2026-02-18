from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# Canonical Parker half-time coefficient
PARKER_HALF_TIME_COEFF = 0.1388


@dataclass(frozen=True)
class RatioCorrection:
    """
    Generic ratio-based correction model.

    alpha = (L^2 / t_half) * poly(r)
    r can be e.g. t_0.75 / t_0.25 or other shape ratio depending on the method.

    NOTE:
    - Cowan/Cape-Lehman/Clark-Taylor 계수는 장비/표준/문헌 판본별로 다를 수 있으므로
      코드에 하드코딩하지 않고, 외부 계수 주입 방식으로 운용한다.
    """

    name: str
    ratio_name: str
    coeffs: tuple[float, ...]

    def factor(self, ratio: np.ndarray | float) -> np.ndarray:
        ratio_arr = np.asarray(ratio, dtype=np.float64)
        # np.polyval expects highest degree first
        return np.polyval(np.asarray(self.coeffs, dtype=np.float64), ratio_arr)

    def alpha(self, thickness_mm: np.ndarray | float, t_half_s: np.ndarray | float, ratio: np.ndarray | float) -> np.ndarray:
        l2 = np.asarray(thickness_mm, dtype=np.float64) ** 2
        t_half = np.asarray(t_half_s, dtype=np.float64)
        return (l2 / np.maximum(t_half, 1e-12)) * self.factor(ratio)


def parker_alpha(thickness_mm: np.ndarray | float, t_half_s: np.ndarray | float) -> np.ndarray:
    l2 = np.asarray(thickness_mm, dtype=np.float64) ** 2
    t_half = np.asarray(t_half_s, dtype=np.float64)
    return PARKER_HALF_TIME_COEFF * l2 / np.maximum(t_half, 1e-12)


def make_polynomial_correction(name: str, ratio_name: str, coeffs: Sequence[float]) -> RatioCorrection:
    if len(coeffs) == 0:
        raise ValueError("coeffs must not be empty")
    return RatioCorrection(name=name, ratio_name=ratio_name, coeffs=tuple(float(c) for c in coeffs))


# Example placeholders (must be replaced with verified values from your selected references)
# Keep these off by default to avoid silently using non-verified constants.
KNOWN_PLACEHOLDER_MODELS: dict[str, RatioCorrection] = {
    "cowan": make_polynomial_correction("cowan", "t_0.75/t_0.25", [PARKER_HALF_TIME_COEFF]),
    "cape_lehman": make_polynomial_correction("cape_lehman", "pulse_ratio", [PARKER_HALF_TIME_COEFF]),
    "clark_taylor": make_polynomial_correction("clark_taylor", "shape_ratio", [PARKER_HALF_TIME_COEFF]),
}

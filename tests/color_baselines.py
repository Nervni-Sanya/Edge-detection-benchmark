#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Цветовые baseline-детекторы границ для честного сравнения с vector_lighting.

Классические Sobel/Prewitt/Canny работают по grayscale и не видят
изолюминантные (хроматические) границы. Заявлять преимущество цветового
метода, сравниваясь только с ними, некорректно. Этот модуль добавляет
ПРАВИЛЬНЫЕ цветовые baseline'ы, которые изолюминантные границы видят:

  • Di Zenzo (1986) — многоканальный градиент через структурный тензор.
    Каноническая работа по цветовому выделению границ; именно её первым
    делом вспомнит рецензент. Magnitude = sqrt(λ_max) тензора, где
        E = ΣIk_x²,  F = ΣIk_x·Ik_y,  G = ΣIk_y²,
        λ_max = ((E+G) + sqrt((E−G)² + 4F²)) / 2.
  • Di Zenzo + NMS + гистерезис — «цветной Canny» (та же постобработка,
    что у Canny, но на цветовом тензоре). Самый сильный честный baseline.
  • Multichannel Sobel (L2) — наивный цветовой градиент: L2-норма
    поканальных градиентов Собеля.

Реализации намеренно сделаны корректно и без поддавков, чтобы сравнение
было честным по отношению к baseline'у, а не подыгрывало нашему методу.

Ссылка: S. Di Zenzo, "A note on the gradient of a multi-image",
Computer Vision, Graphics, and Image Processing, 33(1):116–125, 1986.
"""

from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
from scipy.signal import correlate2d

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=float)


def _normalize_u8(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _channel_gradients(image: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Поканальные градиенты по x и y. Возвращает массивы (H, W, C)."""
    if image.ndim == 2:
        image = image[..., None]
    chans = image.astype(float)
    if sigma > 0:
        chans = np.stack([gaussian_filter(chans[..., k], sigma=sigma)
                          for k in range(chans.shape[2])], axis=-1)
    gx = np.stack([correlate2d(chans[..., k], _SOBEL_X, mode='same', boundary='symm')
                   for k in range(chans.shape[2])], axis=-1)
    gy = np.stack([correlate2d(chans[..., k], _SOBEL_Y, mode='same', boundary='symm')
                   for k in range(chans.shape[2])], axis=-1)
    return gx, gy


def dizenzo_tensor(image: np.ndarray, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Структурный тензор Ди Дзензо. Возвращает (magnitude, direction).

    magnitude — sqrt(наибольшего собственного значения) тензора (не норм.);
    direction — ориентация максимального изменения, рад.
    """
    gx, gy = _channel_gradients(image, sigma)
    E = np.sum(gx * gx, axis=-1)
    F = np.sum(gx * gy, axis=-1)
    G = np.sum(gy * gy, axis=-1)
    tmp = np.sqrt(np.maximum((E - G) ** 2 + 4.0 * F * F, 0.0))
    lam_max = 0.5 * (E + G + tmp)
    magnitude = np.sqrt(np.maximum(lam_max, 0.0))
    direction = 0.5 * np.arctan2(2.0 * F, E - G)
    return magnitude, direction


def dizenzo(image: np.ndarray, sigma: float = 1.0, percentile: float = 95.0) -> np.ndarray:
    """Di Zenzo с percentile-бинаризацией (как Sobel/Prewitt в харнессе)."""
    magnitude, _ = dizenzo_tensor(image, sigma)
    mag = _normalize_u8(magnitude)
    thresh = np.percentile(mag, percentile)
    return (mag > thresh).astype(np.uint8) * 255


def _nms(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Подавление немаксимумов вдоль направления градиента (как в Canny)."""
    suppressed = np.zeros_like(magnitude)
    angle = (np.rad2deg(direction) + 90.0) % 180.0  # нормаль к границе
    mask_h = (angle < 22.5) | (angle >= 157.5)
    mask_d1 = (angle >= 22.5) & (angle < 67.5)
    mask_v = (angle >= 67.5) & (angle < 112.5)
    mask_d2 = (angle >= 112.5) & (angle < 157.5)
    below = np.roll(magnitude, -1, axis=0)
    above = np.roll(magnitude, 1, axis=0)
    right = np.roll(magnitude, -1, axis=1)
    left = np.roll(magnitude, 1, axis=1)
    br = np.roll(np.roll(magnitude, -1, axis=0), -1, axis=1)
    bl = np.roll(np.roll(magnitude, -1, axis=0), 1, axis=1)
    tr = np.roll(np.roll(magnitude, 1, axis=0), -1, axis=1)
    tl = np.roll(np.roll(magnitude, 1, axis=0), 1, axis=1)
    suppressed[mask_h] = magnitude[mask_h] * ((magnitude[mask_h] >= left[mask_h]) & (magnitude[mask_h] >= right[mask_h]))
    suppressed[mask_d1] = magnitude[mask_d1] * ((magnitude[mask_d1] >= br[mask_d1]) & (magnitude[mask_d1] >= tl[mask_d1]))
    suppressed[mask_v] = magnitude[mask_v] * ((magnitude[mask_v] >= above[mask_v]) & (magnitude[mask_v] >= below[mask_v]))
    suppressed[mask_d2] = magnitude[mask_d2] * ((magnitude[mask_d2] >= bl[mask_d2]) & (magnitude[mask_d2] >= tr[mask_d2]))
    suppressed[0, :] = suppressed[-1, :] = suppressed[:, 0] = suppressed[:, -1] = 0
    return suppressed


def _hysteresis(suppressed: np.ndarray, low: float, high: float) -> np.ndarray:
    strong = suppressed >= high
    weak = (suppressed >= low) & (suppressed < high)
    edges = strong.copy()
    while True:
        grown = binary_dilation(edges, structure=np.ones((3, 3))) & weak & ~edges
        if not np.any(grown):
            break
        edges |= grown
    return edges.astype(np.uint8) * 255


def dizenzo_canny(image: np.ndarray, sigma: float = 1.0,
                  low_percentile: float = 70.0, high_percentile: float = 90.0) -> np.ndarray:
    """«Цветной Canny»: тензор Ди Дзензо + NMS + гистерезис.

    Пороги задаются перцентилями ненулевой magnitude, чтобы метод
    автоматически подстраивался под контент (честный сильный baseline).
    """
    magnitude, direction = dizenzo_tensor(image, sigma)
    mag = _normalize_u8(magnitude).astype(float)
    suppressed = _nms(mag, direction)
    nz = suppressed[suppressed > 0]
    if nz.size == 0:
        return np.zeros(mag.shape, dtype=np.uint8)
    low = np.percentile(nz, low_percentile)
    high = np.percentile(nz, high_percentile)
    return _hysteresis(suppressed, low, high)


def multichannel_sobel(image: np.ndarray, sigma: float = 1.0, percentile: float = 95.0) -> np.ndarray:
    """Наивный цветовой градиент: L2-норма поканальных градиентов Собеля."""
    gx, gy = _channel_gradients(image, sigma)
    mag = np.sqrt(np.sum(gx * gx + gy * gy, axis=-1))
    mag = _normalize_u8(mag)
    thresh = np.percentile(mag, percentile)
    return (mag > thresh).astype(np.uint8) * 255


COLOR_BASELINES = {
    'DiZenzo': lambda img: dizenzo(img, sigma=1.0),
    'DiZenzo-Canny': lambda img: dizenzo_canny(img, sigma=1.0),
    'MC-Sobel': lambda img: multichannel_sobel(img, sigma=1.0),
}
